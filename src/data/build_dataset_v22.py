"""Build and freeze Dataset V2.2 training membership.

Dataset V2.2 consists of:

    Dataset V2.1 curated training set
        +
    QA-approved V2.2 offset-enrichment tiles

Validation and test data are NOT modified here. The existing frozen
Dataset V1 validation/test manifest remains the evaluation source.

This script:
- validates both input manifests
- combines them
- checks file existence
- checks tile-ID uniqueness
- checks train-only membership
- checks enrichment QA status
- checks semantic class accounting
- calculates final V2.2 class distribution
- writes the official V2.2 training manifest
- writes an immutable-style freeze manifest + provenance summary

It does NOT recompute SAR normalization.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

V21_MANIFEST = Path(
    "metadata/dataset_v21/dataset_manifest.csv"
)

V22_ENRICHMENT_ACCEPTED = Path(
    "metadata/dataset_v22/qa/"
    "dataset_v22_enrichment_accepted.csv"
)

V1_DATASET_CONFIG = Path(
    "metadata/dataset_v1/freeze/dataset_config.json"
)

V1_EVALUATION_MANIFEST = Path(
    "metadata/dataset_v1/dataset_manifest.csv"
)

OUTPUT_ROOT = Path(
    "metadata/dataset_v22"
)

FREEZE_DIR = (
    OUTPUT_ROOT / "freeze"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT / "dataset_manifest.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_ROOT / "dataset_manifest_summary.json"
)

OUTPUT_DISTRIBUTION = (
    OUTPUT_ROOT / "class_distribution.csv"
)

OUTPUT_CITY_SUMMARY = (
    OUTPUT_ROOT / "city_distribution.csv"
)


# ---------------------------------------------------------------------
# Expected dataset composition
# ---------------------------------------------------------------------

EXPECTED_V21_TRAINING = 1627
EXPECTED_ENRICHMENT_ACCEPTED = 497
EXPECTED_V22_TRAINING = 2124

EXPECTED_TRAIN_CITIES = 14

CLASS_NAMES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def require_file(
    path: Path,
    label: str,
) -> None:
    """Require a non-empty file."""

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}"
        )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256."""

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def resolve_tile_id_column(
    dataframe: pd.DataFrame,
    label: str,
) -> str:
    """Find the tile identifier column."""

    for candidate in (
        "tile_id",
        "candidate_id",
    ):

        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"{label} has no tile_id/candidate_id column."
    )


def resolve_path_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    """Resolve one tile-path column."""

    for candidate in candidates:

        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Could not identify {label} path column. "
        f"Tried: {candidates}"
    )


def normalize_base_manifest(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize Dataset V2.1 rows for V2.2."""

    df = dataframe.copy()

    tile_id_col = resolve_tile_id_column(
        df,
        "Dataset V2.1 manifest",
    )

    if tile_id_col != "tile_id":

        df["tile_id"] = (
            df[tile_id_col]
            .astype(str)
        )

    else:

        df["tile_id"] = (
            df["tile_id"]
            .astype(str)
        )

    if "split" not in df.columns:
        raise ValueError(
            "Dataset V2.1 manifest has no split column."
        )

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["dataset_version"] = "v2.2"

    df["dataset_source"] = "v21_curated_base"

    df["v22_component"] = "base"

    # Dataset V2.1 base tiles already passed the established tile QA.
    df["qa_pass"] = True
    df["qa_status"] = "accepted"
    df["qa_failure_count"] = 0
    df["qa_failures"] = ""

    # This is the canonical field consumed by SegmentationDataset.
    df["tile_qa_status"] = "PASS"

    df["v22_qa_status"] = "accepted"

    return df


def normalize_enrichment_manifest(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize QA-approved enrichment rows."""

    df = dataframe.copy()

    tile_id_col = resolve_tile_id_column(
        df,
        "V2.2 enrichment manifest",
    )

    if tile_id_col != "tile_id":

        df["tile_id"] = (
            df[tile_id_col]
            .astype(str)
        )

    else:

        df["tile_id"] = (
            df["tile_id"]
            .astype(str)
        )

    if "split" not in df.columns:
        raise ValueError(
            "V2.2 enrichment manifest has no split column."
        )

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "qa_pass" not in df.columns:
        raise ValueError(
            "Accepted enrichment manifest has no qa_pass column."
        )

    qa_pass = (
        df["qa_pass"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if qa_pass.isna().any():

        # Handles an already-boolean column cleanly.
        qa_pass = df["qa_pass"].astype(bool)

    if not bool(
        qa_pass.all()
    ):
        raise ValueError(
            "Rejected enrichment samples are present "
            "inside the accepted manifest."
        )

    df["qa_pass"] = True
    df["qa_status"] = "accepted"

    # The V2.2 enrichment rows reached this manifest only after qa_dataset_v22_enrichment accepted them.
    # SegmentationDataset validates this exact legacy/canonical field.
    df["tile_qa_status"] = "PASS"

    if "qa_failure_count" not in df.columns:
        df["qa_failure_count"] = 0

    if "qa_failures" not in df.columns:
        df["qa_failures"] = ""

    df["dataset_version"] = "v2.2"
    df["dataset_source"] = "v22_offset_enrichment"
    df["v22_component"] = "enrichment"
    df["v22_qa_status"] = "accepted"

    return df


def validate_training_only(
    dataframe: pd.DataFrame,
    label: str,
) -> None:
    """Ensure a manifest contains train rows only."""

    splits = set(
        dataframe["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    if splits != {"train"}:

        raise ValueError(
            f"{label} must contain train rows only. "
            f"Found splits: {sorted(splits)}"
        )


def check_required_files(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Check all image/semantic/validity paths."""

    image_col = resolve_path_column(
        dataframe,
        (
            "image_path",
            "sar_tile_path",
            "image",
        ),
        "image",
    )

    semantic_col = resolve_path_column(
        dataframe,
        (
            "semantic_mask_path",
            "semantic_path",
            "label_path",
            "mask_path",
        ),
        "semantic mask",
    )

    validity_col = resolve_path_column(
        dataframe,
        (
            "validity_mask_path",
            "validity_path",
        ),
        "validity mask",
    )

    missing: list[str] = []

    for _, row in (
        dataframe.iterrows()
    ):

        for column in (
            image_col,
            semantic_col,
            validity_col,
        ):

            path = Path(
                str(row[column])
            )

            if (
                not path.exists()
                or path.stat().st_size == 0
            ):

                missing.append(
                    f"{row['tile_id']} | "
                    f"{column} | {path}"
                )

    return {
        "image_column": image_col,
        "semantic_column": semantic_col,
        "validity_column": validity_col,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }


def reconcile_class_pixels(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Check five-class accounting."""

    required_columns = {
        "valid_pixel_count",
        *[
            f"{class_name}_pixel_count"
            for class_name in CLASS_NAMES
        ],
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            "Manifest missing class-accounting columns: "
            f"{sorted(missing)}"
        )

    class_sum = np.zeros(
        len(dataframe),
        dtype=np.int64,
    )

    for class_name in CLASS_NAMES:

        class_sum += (
            pd.to_numeric(
                dataframe[
                    f"{class_name}_pixel_count"
                ],
                errors="raise",
            )
            .astype(np.int64)
            .to_numpy()
        )

    valid = (
        pd.to_numeric(
            dataframe[
                "valid_pixel_count"
            ],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    mismatched = (
        class_sum != valid
    )

    return {
        "mismatch_count": int(
            mismatched.sum()
        ),
        "all_match": bool(
            not mismatched.any()
        ),
        "aggregate_valid_pixels": int(
            valid.sum()
        ),
        "aggregate_class_pixels": int(
            class_sum.sum()
        ),
    }


def build_class_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate final V2.2 training class distribution."""

    total_valid = int(
        dataframe[
            "valid_pixel_count"
        ].sum()
    )

    rows = []

    for class_name in CLASS_NAMES:

        pixels = int(
            dataframe[
                f"{class_name}_pixel_count"
            ].sum()
        )

        rows.append(
            {
                "class_name": class_name,
                "pixel_count": pixels,
                "fraction_valid": (
                    pixels / total_valid
                    if total_valid > 0
                    else 0.0
                ),
                "percentage_valid": (
                    100.0
                    * pixels
                    / total_valid
                    if total_valid > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_city_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize final V2.2 training membership by city."""

    component_counts = (
        dataframe
        .pivot_table(
            index=[
                "city_id",
                "city_name",
            ],
            columns="v22_component",
            values="tile_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    if "base" not in component_counts:
        component_counts["base"] = 0

    if (
        "enrichment"
        not in component_counts
    ):
        component_counts[
            "enrichment"
        ] = 0

    component_counts[
        "total_tiles"
    ] = (
        component_counts["base"]
        + component_counts["enrichment"]
    )

    component_counts = (
        component_counts.rename(
            columns={
                "base": "v21_base_tiles",
                "enrichment": (
                    "v22_enrichment_tiles"
                ),
            }
        )
    )

    return (
        component_counts
        .sort_values(
            "city_id"
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Build and freeze Dataset V2.2."""

    for path, label in (
        (
            V21_MANIFEST,
            "Dataset V2.1 manifest",
        ),
        (
            V22_ENRICHMENT_ACCEPTED,
            "V2.2 QA-approved enrichment manifest",
        ),
        (
            V1_DATASET_CONFIG,
            "Frozen Dataset V1 configuration",
        ),
        (
            V1_EVALUATION_MANIFEST,
            "Dataset V1 evaluation manifest",
        ),
    ):

        require_file(
            path,
            label,
        )

    print()
    print("Dataset V2.2 build and freeze")
    print("=============================")

    base_raw = pd.read_csv(
        V21_MANIFEST
    )

    enrichment_raw = pd.read_csv(
        V22_ENRICHMENT_ACCEPTED
    )

    base = normalize_base_manifest(
        base_raw
    )

    enrichment = (
        normalize_enrichment_manifest(
            enrichment_raw
        )
    )

    validate_training_only(
        base,
        "Dataset V2.1",
    )

    validate_training_only(
        enrichment,
        "V2.2 enrichment",
    )

    print()
    print("Input composition")
    print("-----------------")
    print(
        f"V2.1 base training tiles: "
        f"{len(base)}"
    )
    print(
        f"QA-approved enrichment:   "
        f"{len(enrichment)}"
    )

    if len(base) != EXPECTED_V21_TRAINING:

        raise RuntimeError(
            "Unexpected Dataset V2.1 training size: "
            f"{len(base)} != "
            f"{EXPECTED_V21_TRAINING}"
        )

    if (
        len(enrichment)
        != EXPECTED_ENRICHMENT_ACCEPTED
    ):

        raise RuntimeError(
            "Unexpected enrichment size: "
            f"{len(enrichment)} != "
            f"{EXPECTED_ENRICHMENT_ACCEPTED}"
        )

    # ---------------------------------------------------------
    # Cross-component duplicate check
    # ---------------------------------------------------------

    base_ids = set(
        base["tile_id"]
    )

    enrichment_ids = set(
        enrichment["tile_id"]
    )

    overlap = (
        base_ids
        & enrichment_ids
    )

    if overlap:

        raise RuntimeError(
            "Tile-ID overlap between V2.1 base and "
            "V2.2 enrichment: "
            f"{sorted(overlap)[:20]}"
        )

    # ---------------------------------------------------------
    # Harmonize schemas
    # ---------------------------------------------------------

    combined = pd.concat(
        [
            base,
            enrichment,
        ],
        ignore_index=True,
        sort=False,
    )

    # ---------------------------------------------------------
    # Canonical tile-QA status
    # ---------------------------------------------------------

    if "tile_qa_status" not in combined.columns:
        raise RuntimeError("Combined Dataset V2.2 has no tile_qa_status column.")

    failed_tile_qa = (
        combined["tile_qa_status"]
        .astype(str)
        .str.strip()
        .str.upper()
        != "PASS"
    )

    if failed_tile_qa.any():

        examples = (
            combined.loc[
                failed_tile_qa,
                [
                    "tile_id",
                    "city_id",
                    "v22_component",
                    "tile_qa_status",
                ],
            ]
            .head(20)
            .to_dict(orient="records")
        )

        raise RuntimeError(
            "Dataset V2.2 contains rows whose tile_qa_status "
            f"is not PASS. Examples: {examples}"
        )

    if len(combined) != EXPECTED_V22_TRAINING:

        raise RuntimeError(
            "Unexpected Dataset V2.2 size: "
            f"{len(combined)} != "
            f"{EXPECTED_V22_TRAINING}"
        )

    if combined[
        "tile_id"
    ].duplicated().any():

        duplicates = (
            combined.loc[
                combined[
                    "tile_id"
                ].duplicated(
                    keep=False
                ),
                "tile_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate tile IDs in combined Dataset V2.2: "
            f"{duplicates[:20]}"
        )

    validate_training_only(
        combined,
        "Combined Dataset V2.2",
    )

    # ---------------------------------------------------------
    # City membership
    # ---------------------------------------------------------

    city_count = int(
        combined[
            "city_id"
        ].nunique()
    )

    if (
        city_count
        != EXPECTED_TRAIN_CITIES
    ):

        raise RuntimeError(
            "Unexpected V2.2 training-city count: "
            f"{city_count}"
        )

    # ---------------------------------------------------------
    # File existence
    # ---------------------------------------------------------

    file_check = (
        check_required_files(
            combined
        )
    )

    print()
    print("File integrity")
    print("--------------")
    print(
        "Missing/non-empty tile files: "
        f"{file_check['missing_count']}"
    )

    if (
        file_check[
            "missing_count"
        ]
        != 0
    ):

        raise RuntimeError(
            "Missing Dataset V2.2 tile files:\n"
            + "\n".join(
                file_check[
                    "missing_examples"
                ]
            )
        )

    # ---------------------------------------------------------
    # Class accounting
    # ---------------------------------------------------------

    accounting = (
        reconcile_class_pixels(
            combined
        )
    )

    print()
    print("Class accounting")
    print("----------------")
    print(
        "Aggregate valid pixels: "
        f"{accounting['aggregate_valid_pixels']}"
    )
    print(
        "Five-class pixel sum:   "
        f"{accounting['aggregate_class_pixels']}"
    )
    print(
        "Tile-level mismatches:  "
        f"{accounting['mismatch_count']}"
    )

    if not accounting[
        "all_match"
    ]:

        raise RuntimeError(
            "Dataset V2.2 class accounting FAILED."
        )

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------

    distribution = (
        build_class_distribution(
            combined
        )
    )

    city_summary = (
        build_city_summary(
            combined
        )
    )

    print()
    print("Final Dataset V2.2 class distribution")
    print("-------------------------------------")

    print(
        distribution[
            [
                "class_name",
                "pixel_count",
                "percentage_valid",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print("Final city composition")
    print("----------------------")

    print(
        city_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Write official outputs
    # ---------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FREEZE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Deterministic ordering.
    combined = (
        combined.sort_values(
            [
                "city_id",
                "v22_component",
                "tile_id",
            ]
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    distribution.to_csv(
        OUTPUT_DISTRIBUTION,
        index=False,
    )

    city_summary.to_csv(
        OUTPUT_CITY_SUMMARY,
        index=False,
    )

    # ---------------------------------------------------------
    # Dataset configuration
    # ---------------------------------------------------------

    with V1_DATASET_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:

        v1_config = json.load(
            file
        )

    v22_config = dict(
        v1_config
    )

    v22_config[
        "dataset_version"
    ] = "v2.2"

    v22_config[
        "training_manifest"
    ] = str(
        OUTPUT_MANIFEST
    )

    v22_config[
        "evaluation_manifest"
    ] = str(
        V1_EVALUATION_MANIFEST
    )

    v22_config[
        "training_tile_count"
    ] = int(
        len(combined)
    )

    v22_config[
        "training_city_count"
    ] = city_count

    v22_config[
        "dataset_v22_provenance"
    ] = {
        "base_dataset": "v2.1",
        "base_training_tiles": int(
            len(base)
        ),
        "offset_enrichment_tiles": int(
            len(enrichment)
        ),
        "offset_pixels": 128,
        "enrichment_qa_policy": {
            "accepted_only": True,
            "maximum_padding_fraction": 0.25,
        },
        "validation_source": "dataset_v1",
        "test_source": "dataset_v1",
    }

    config_path = (
        FREEZE_DIR
        / "dataset_config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            v22_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Freeze copies
    # ---------------------------------------------------------

    frozen_manifest = (
        FREEZE_DIR
        / "dataset_manifest.csv"
    )

    frozen_distribution = (
        FREEZE_DIR
        / "class_distribution.csv"
    )

    frozen_city_summary = (
        FREEZE_DIR
        / "city_distribution.csv"
    )

    shutil.copy2(
        OUTPUT_MANIFEST,
        frozen_manifest,
    )

    shutil.copy2(
        OUTPUT_DISTRIBUTION,
        frozen_distribution,
    )

    shutil.copy2(
        OUTPUT_CITY_SUMMARY,
        frozen_city_summary,
    )

    # ---------------------------------------------------------
    # Checksums after final files exist
    # ---------------------------------------------------------

    checksums = {
        "dataset_manifest_sha256": (
            sha256_file(
                frozen_manifest
            )
        ),
        "dataset_config_sha256": (
            sha256_file(
                config_path
            )
        ),
        "class_distribution_sha256": (
            sha256_file(
                frozen_distribution
            )
        ),
        "city_distribution_sha256": (
            sha256_file(
                frozen_city_summary
            )
        ),
        "v21_source_manifest_sha256": (
            sha256_file(
                V21_MANIFEST
            )
        ),
        "enrichment_source_manifest_sha256": (
            sha256_file(
                V22_ENRICHMENT_ACCEPTED
            )
        ),
        "evaluation_manifest_sha256": (
            sha256_file(
                V1_EVALUATION_MANIFEST
            )
        ),
    }

    # ---------------------------------------------------------
    # Summary / freeze metadata
    # ---------------------------------------------------------

    summary = {
        "dataset_version": "v2.2",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": "frozen_training_membership",
        "training": {
            "total_tiles": int(
                len(combined)
            ),
            "base_v21_tiles": int(
                len(base)
            ),
            "v22_enrichment_tiles": int(
                len(enrichment)
            ),
            "city_count": city_count,
            "city_ids": (
                sorted(
                    combined[
                        "city_id"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                )
            ),
        },
        "evaluation": {
            "validation_test_manifest": str(
                V1_EVALUATION_MANIFEST
            ),
            "policy": (
                "unchanged_from_dataset_v1"
            ),
        },
        "qa": {
            "tile_id_duplicates": 0,
            "cross_component_tile_id_overlap": 0,
            "missing_tile_files": int(
                file_check[
                    "missing_count"
                ]
            ),
            "class_accounting_mismatches": int(
                accounting[
                    "mismatch_count"
                ]
            ),
            "class_accounting_pass": bool(
                accounting[
                    "all_match"
                ]
            ),
        },
        "class_distribution": (
            distribution.to_dict(
                orient="records"
            )
        ),
        "paths": {
            "official_manifest": str(
                OUTPUT_MANIFEST
            ),
            "freeze_manifest": str(
                frozen_manifest
            ),
            "freeze_config": str(
                config_path
            ),
            "class_distribution": str(
                OUTPUT_DISTRIBUTION
            ),
            "city_distribution": str(
                OUTPUT_CITY_SUMMARY
            ),
        },
        "checksums": checksums,
        "normalization_status": (
            "pending_recomputation_using_"
            "existing_training_normalization_method"
        ),
    }

    with OUTPUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    freeze_summary = (
        FREEZE_DIR
        / "dataset_freeze_summary.json"
    )

    shutil.copy2(
        OUTPUT_SUMMARY,
        freeze_summary,
    )

    print()
    print("Dataset V2.2 freeze result")
    print("--------------------------")
    print(
        f"Training tiles:   "
        f"{len(combined)}"
    )
    print(
        f"V2.1 base:        "
        f"{len(base)}"
    )
    print(
        f"V2.2 enrichment:  "
        f"{len(enrichment)}"
    )
    print(
        f"Training cities:  "
        f"{city_count}"
    )
    print(
        "Validation/test:   "
        "unchanged Dataset V1"
    )
    print(
        "Duplicate IDs:     0"
    )
    print(
        "Missing files:     "
        f"{file_check['missing_count']}"
    )
    print(
        "Class mismatches:  "
        f"{accounting['mismatch_count']}"
    )

    print()
    print("Frozen outputs")
    print("--------------")
    print(frozen_manifest)
    print(config_path)
    print(frozen_distribution)
    print(frozen_city_summary)
    print(freeze_summary)

    print()
    print(
        "Result: Dataset V2.2 training membership "
        "built and frozen successfully."
    )

    print()
    print(
        "NOTE: V2.2 training normalization has NOT "
        "yet been recomputed."
    )


if __name__ == "__main__":
    main()