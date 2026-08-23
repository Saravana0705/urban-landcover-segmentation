"""Quality assurance for Dataset V2.2 offset-enrichment tiles.

The script validates every materialized SAR / semantic / validity triplet
against the enrichment manifest.

Checks include:
- file existence
- raster dimensions
- expected band counts
- CRS equality
- affine-transform equality
- raster bounds equality
- finite SAR values
- allowed semantic labels
- allowed validity values
- validity/semantic consistency
- class-pixel reconciliation
- manifest-statistic reconciliation
- padding threshold

The script does NOT delete rejected tiles.

Instead, it creates:
1. full QA audit
2. accepted enrichment manifest
3. rejected enrichment manifest
4. Dataset V2.2 enrichment QA summary
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ENRICHMENT_MANIFEST = Path(
    "metadata/dataset_v22/enrichment/"
    "dataset_v22_enrichment_manifest.csv"
)

OUTPUT_DIR = Path(
    "metadata/dataset_v22/qa"
)


# ---------------------------------------------------------------------
# QA policy
# ---------------------------------------------------------------------

EXPECTED_TILE_SIZE = 256
EXPECTED_IMAGE_BANDS = 2

SEMANTIC_NODATA = 255

ALLOWED_SEMANTIC_VALUES = {
    0, 1, 2, 3, 4, 5, 255
}

ALLOWED_VALIDITY_VALUES = {
    0, 1
}

# We deliberately reject strongly padded edge tiles.
MAX_PADDING_FRACTION = 0.25

# Affine/bounds tolerance.
FLOAT_TOLERANCE = 1e-8


CLASS_IDS = {
    "buildings": 1,
    "roads": 2,
    "vegetation": 3,
    "bare_land": 4,
    "water": 5,
}


# ---------------------------------------------------------------------
# Helpers
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


def transforms_equal(
    left,
    right,
) -> bool:
    """Compare affine transforms numerically."""

    return bool(
        np.allclose(
            tuple(left),
            tuple(right),
            atol=FLOAT_TOLERANCE,
            rtol=0.0,
        )
    )


def bounds_equal(
    left,
    right,
) -> bool:
    """Compare raster bounds numerically."""

    return bool(
        np.allclose(
            tuple(left),
            tuple(right),
            atol=FLOAT_TOLERANCE,
            rtol=0.0,
        )
    )


def compare_integer(
    expected: Any,
    actual: int,
) -> bool:
    """Compare manifest integer against recomputed integer."""

    if pd.isna(expected):
        return True

    return int(expected) == int(actual)


def compare_float(
    expected: Any,
    actual: float,
    tolerance: float = 1e-9,
) -> bool:
    """Compare manifest floating value against recomputed value."""

    if pd.isna(expected):
        return True

    return abs(
        float(expected) - float(actual)
    ) <= tolerance


def add_failure(
    failures: list[str],
    condition: bool,
    message: str,
) -> None:
    """Append a failure message if condition is False."""

    if not condition:
        failures.append(message)


# ---------------------------------------------------------------------
# Tile QA
# ---------------------------------------------------------------------

def inspect_triplet(
    row: pd.Series,
) -> dict[str, Any]:
    """Inspect one materialized enrichment triplet."""

    failures: list[str] = []

    tile_id = str(row["tile_id"])

    image_path = Path(
        str(row["image_path"])
    )

    semantic_path = Path(
        str(row["semantic_mask_path"])
    )

    validity_path = Path(
        str(row["validity_mask_path"])
    )

    # ---------------------------------------------------------
    # File existence
    # ---------------------------------------------------------

    image_exists = (
        image_path.exists()
        and image_path.stat().st_size > 0
    )

    semantic_exists = (
        semantic_path.exists()
        and semantic_path.stat().st_size > 0
    )

    validity_exists = (
        validity_path.exists()
        and validity_path.stat().st_size > 0
    )

    add_failure(
        failures,
        image_exists,
        "missing_image",
    )

    add_failure(
        failures,
        semantic_exists,
        "missing_semantic_mask",
    )

    add_failure(
        failures,
        validity_exists,
        "missing_validity_mask",
    )

    # If anything is absent, raster-level QA cannot continue.
    if not (
        image_exists
        and semantic_exists
        and validity_exists
    ):
        return {
            "tile_id": tile_id,
            "qa_pass": False,
            "qa_status": "rejected",
            "qa_failure_count": len(failures),
            "qa_failures": ";".join(failures),
            "padding_fraction": (
                float(row["padding_fraction"])
                if "padding_fraction" in row
                else np.nan
            ),
        }

    # ---------------------------------------------------------
    # Open raster triplet
    # ---------------------------------------------------------

    with rasterio.open(
        image_path
    ) as image_ds, rasterio.open(
        semantic_path
    ) as semantic_ds, rasterio.open(
        validity_path
    ) as validity_ds:

        # -----------------------------------------------------
        # Structural checks
        # -----------------------------------------------------

        add_failure(
            failures,
            image_ds.width == EXPECTED_TILE_SIZE
            and image_ds.height == EXPECTED_TILE_SIZE,
            "invalid_image_dimensions",
        )

        add_failure(
            failures,
            semantic_ds.width == EXPECTED_TILE_SIZE
            and semantic_ds.height == EXPECTED_TILE_SIZE,
            "invalid_semantic_dimensions",
        )

        add_failure(
            failures,
            validity_ds.width == EXPECTED_TILE_SIZE
            and validity_ds.height == EXPECTED_TILE_SIZE,
            "invalid_validity_dimensions",
        )

        add_failure(
            failures,
            image_ds.count == EXPECTED_IMAGE_BANDS,
            "invalid_image_band_count",
        )

        add_failure(
            failures,
            semantic_ds.count == 1,
            "invalid_semantic_band_count",
        )

        add_failure(
            failures,
            validity_ds.count == 1,
            "invalid_validity_band_count",
        )

        # -----------------------------------------------------
        # Geospatial alignment
        # -----------------------------------------------------

        crs_match = (
            image_ds.crs == semantic_ds.crs
            and image_ds.crs == validity_ds.crs
        )

        transform_match = (
            transforms_equal(
                image_ds.transform,
                semantic_ds.transform,
            )
            and transforms_equal(
                image_ds.transform,
                validity_ds.transform,
            )
        )

        bounds_match = (
            bounds_equal(
                image_ds.bounds,
                semantic_ds.bounds,
            )
            and bounds_equal(
                image_ds.bounds,
                validity_ds.bounds,
            )
        )

        add_failure(
            failures,
            crs_match,
            "crs_mismatch",
        )

        add_failure(
            failures,
            transform_match,
            "transform_mismatch",
        )

        add_failure(
            failures,
            bounds_match,
            "bounds_mismatch",
        )

        # -----------------------------------------------------
        # Read arrays
        # -----------------------------------------------------

        image = image_ds.read()

        semantic = semantic_ds.read(1)

        validity = validity_ds.read(1)

        # -----------------------------------------------------
        # SAR checks
        # -----------------------------------------------------

        image_finite = bool(
            np.isfinite(image).all()
        )

        add_failure(
            failures,
            image_finite,
            "non_finite_sar_values",
        )

        # -----------------------------------------------------
        # Semantic labels
        # -----------------------------------------------------

        semantic_values = set(
            int(value)
            for value in np.unique(semantic)
        )

        semantic_values_valid = (
            semantic_values
            <= ALLOWED_SEMANTIC_VALUES
        )

        add_failure(
            failures,
            semantic_values_valid,
            "invalid_semantic_values",
        )

        # -----------------------------------------------------
        # Validity values
        # -----------------------------------------------------

        validity_values = set(
            int(value)
            for value in np.unique(validity)
        )

        validity_values_valid = (
            validity_values
            <= ALLOWED_VALIDITY_VALUES
        )

        add_failure(
            failures,
            validity_values_valid,
            "invalid_validity_values",
        )

        # -----------------------------------------------------
        # Recompute counts
        # -----------------------------------------------------

        valid_mask = validity == 1

        valid_pixel_count = int(
            np.count_nonzero(valid_mask)
        )

        tile_pixels = (
            EXPECTED_TILE_SIZE
            * EXPECTED_TILE_SIZE
        )

        valid_fraction = (
            valid_pixel_count
            / tile_pixels
        )

        class_counts: dict[str, int] = {}

        for class_name, class_id in (
            CLASS_IDS.items()
        ):

            class_counts[class_name] = int(
                np.count_nonzero(
                    valid_mask
                    & (semantic == class_id)
                )
            )

        total_class_pixels = sum(
            class_counts.values()
        )

        # In this dataset, every valid training pixel should belong
        # to exactly one of the five semantic classes.
        class_accounting_match = (
            total_class_pixels
            == valid_pixel_count
        )

        add_failure(
            failures,
            class_accounting_match,
            "valid_class_accounting_mismatch",
        )

        # -----------------------------------------------------
        # Semantic/validity logical consistency
        # -----------------------------------------------------

        # Valid pixels must not be semantic nodata.
        valid_pixels_are_not_nodata = not bool(
            np.any(
                valid_mask
                & (
                    semantic
                    == SEMANTIC_NODATA
                )
            )
        )

        add_failure(
            failures,
            valid_pixels_are_not_nodata,
            "valid_pixel_is_semantic_nodata",
        )

        # Valid pixels must be one of classes 1..5.
        valid_semantic_values = set(
            int(value)
            for value in np.unique(
                semantic[valid_mask]
            )
        )

        valid_labels_correct = (
            valid_semantic_values
            <= {1, 2, 3, 4, 5}
        )

        add_failure(
            failures,
            valid_labels_correct,
            "valid_pixel_has_nonclass_label",
        )

        # -----------------------------------------------------
        # Reconcile manifest statistics
        # -----------------------------------------------------

        manifest_valid_match = (
            compare_integer(
                row.get(
                    "valid_pixel_count"
                ),
                valid_pixel_count,
            )
        )

        add_failure(
            failures,
            manifest_valid_match,
            "manifest_valid_pixel_mismatch",
        )

        manifest_fraction_match = (
            compare_float(
                row.get(
                    "valid_fraction"
                ),
                valid_fraction,
            )
        )

        add_failure(
            failures,
            manifest_fraction_match,
            "manifest_valid_fraction_mismatch",
        )

        for class_name in CLASS_IDS:

            match = compare_integer(
                row.get(
                    f"{class_name}_pixel_count"
                ),
                class_counts[class_name],
            )

            add_failure(
                failures,
                match,
                (
                    "manifest_"
                    f"{class_name}_pixel_mismatch"
                ),
            )

        # -----------------------------------------------------
        # Padding policy
        # -----------------------------------------------------

        padding_fraction = float(
            row.get(
                "padding_fraction",
                0.0,
            )
        )

        padding_pass = (
            padding_fraction
            <= MAX_PADDING_FRACTION
        )

        add_failure(
            failures,
            padding_pass,
            "padding_fraction_exceeds_limit",
        )

        # -----------------------------------------------------
        # Result
        # -----------------------------------------------------

        qa_pass = (
            len(failures) == 0
        )

        return {
            "tile_id": tile_id,
            "city_id": str(
                row["city_id"]
            ),
            "city_name": str(
                row["city_name"]
            ),
            "candidate_id": str(
                row.get(
                    "candidate_id",
                    tile_id,
                )
            ),
            "qa_pass": qa_pass,
            "qa_status": (
                "accepted"
                if qa_pass
                else "rejected"
            ),
            "qa_failure_count": len(
                failures
            ),
            "qa_failures": ";".join(
                failures
            ),
            "image_exists": image_exists,
            "semantic_exists": (
                semantic_exists
            ),
            "validity_exists": (
                validity_exists
            ),
            "image_width": image_ds.width,
            "image_height": (
                image_ds.height
            ),
            "image_bands": image_ds.count,
            "semantic_bands": (
                semantic_ds.count
            ),
            "validity_bands": (
                validity_ds.count
            ),
            "crs": str(
                image_ds.crs
            ),
            "crs_match": crs_match,
            "transform_match": (
                transform_match
            ),
            "bounds_match": bounds_match,
            "sar_all_finite": image_finite,
            "semantic_values": (
                ",".join(
                    str(value)
                    for value in sorted(
                        semantic_values
                    )
                )
            ),
            "semantic_values_valid": (
                semantic_values_valid
            ),
            "validity_values": (
                ",".join(
                    str(value)
                    for value in sorted(
                        validity_values
                    )
                )
            ),
            "validity_values_valid": (
                validity_values_valid
            ),
            "padding_fraction": (
                padding_fraction
            ),
            "padding_pass": padding_pass,
            "valid_pixel_count_recomputed": (
                valid_pixel_count
            ),
            "valid_fraction_recomputed": (
                valid_fraction
            ),
            "buildings_pixel_count_recomputed": (
                class_counts["buildings"]
            ),
            "roads_pixel_count_recomputed": (
                class_counts["roads"]
            ),
            "vegetation_pixel_count_recomputed": (
                class_counts["vegetation"]
            ),
            "bare_land_pixel_count_recomputed": (
                class_counts["bare_land"]
            ),
            "water_pixel_count_recomputed": (
                class_counts["water"]
            ),
            "class_accounting_match": (
                class_accounting_match
            ),
            "manifest_valid_match": (
                manifest_valid_match
            ),
            "manifest_fraction_match": (
                manifest_fraction_match
            ),
        }


# ---------------------------------------------------------------------
# Distribution report
# ---------------------------------------------------------------------

def build_distribution(
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate accepted enrichment class distribution."""

    total_valid = int(
        accepted[
            "valid_pixel_count"
        ].sum()
    )

    rows = []

    for class_name in CLASS_IDS:

        column = (
            f"{class_name}_pixel_count"
        )

        pixels = int(
            accepted[column].sum()
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

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run V2.2 enrichment QA."""

    require_file(
        ENRICHMENT_MANIFEST,
        "V2.2 enrichment manifest",
    )

    manifest = pd.read_csv(
        ENRICHMENT_MANIFEST
    )

    if manifest.empty:
        raise RuntimeError(
            "V2.2 enrichment manifest is empty."
        )

    if manifest["tile_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate tile IDs found in enrichment manifest."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Dataset V2.2 enrichment QA")
    print("==========================")
    print(
        f"Materialized tiles: "
        f"{len(manifest)}"
    )
    print(
        "Maximum permitted padding fraction: "
        f"{MAX_PADDING_FRACTION:.2f}"
    )
    print()

    audit_rows: list[
        dict[str, Any]
    ] = []

    for index, row in (
        manifest.iterrows()
    ):

        result = inspect_triplet(
            row
        )

        audit_rows.append(
            result
        )

        if (
            (index + 1) % 50 == 0
            or (index + 1) == len(manifest)
        ):
            print(
                f"Checked "
                f"{index + 1}/{len(manifest)}"
            )

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path = (
        OUTPUT_DIR
        / "dataset_v22_enrichment_qa_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    # Attach QA result back onto the original manifest.
    qa_columns = audit[
        [
            "tile_id",
            "qa_pass",
            "qa_status",
            "qa_failure_count",
            "qa_failures",
        ]
    ]

    enriched_manifest = (
        manifest.merge(
            qa_columns,
            on="tile_id",
            how="left",
            validate="one_to_one",
        )
    )

    accepted = (
        enriched_manifest.loc[
            enriched_manifest[
                "qa_pass"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    rejected = (
        enriched_manifest.loc[
            ~enriched_manifest[
                "qa_pass"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    accepted_path = (
        OUTPUT_DIR
        / "dataset_v22_enrichment_accepted.csv"
    )

    rejected_path = (
        OUTPUT_DIR
        / "dataset_v22_enrichment_rejected.csv"
    )

    accepted.to_csv(
        accepted_path,
        index=False,
    )

    rejected.to_csv(
        rejected_path,
        index=False,
    )

    distribution = build_distribution(
        accepted
    )

    distribution_path = (
        OUTPUT_DIR
        / "dataset_v22_enrichment_accepted_distribution.csv"
    )

    distribution.to_csv(
        distribution_path,
        index=False,
    )

    failure_summary = (
        audit.loc[
            ~audit["qa_pass"],
            "qa_failures",
        ]
        .str.split(";")
        .explode()
        .value_counts()
        .rename_axis(
            "failure_reason"
        )
        .reset_index(
            name="tile_count"
        )
    )

    failure_summary_path = (
        OUTPUT_DIR
        / "dataset_v22_enrichment_failure_summary.csv"
    )

    failure_summary.to_csv(
        failure_summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # QA summary
    # ---------------------------------------------------------

    accepted_class_pixels = {
        class_name: int(
            accepted[
                f"{class_name}_pixel_count"
            ].sum()
        )
        for class_name in CLASS_IDS
    }

    accepted_valid_pixels = int(
        accepted[
            "valid_pixel_count"
        ].sum()
    )

    class_total = sum(
        accepted_class_pixels.values()
    )

    summary = {
        "dataset_version": "v2.2",
        "component": "offset_enrichment",
        "qa_created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "materialized_tile_count": int(
            len(manifest)
        ),
        "accepted_tile_count": int(
            len(accepted)
        ),
        "rejected_tile_count": int(
            len(rejected)
        ),
        "maximum_padding_fraction": (
            MAX_PADDING_FRACTION
        ),
        "accepted_city_count": int(
            accepted[
                "city_id"
            ].nunique()
        ),
        "accepted_valid_pixels": (
            accepted_valid_pixels
        ),
        "accepted_class_pixels": (
            accepted_class_pixels
        ),
        "accepted_class_pixel_sum": int(
            class_total
        ),
        "class_accounting_pass": bool(
            class_total
            == accepted_valid_pixels
        ),
        "all_accepted_tiles_qa_pass": bool(
            accepted[
                "qa_pass"
            ].all()
        ),
        "rejected_tiles": (
            rejected[
                [
                    "tile_id",
                    "city_id",
                    "city_name",
                    "padding_fraction",
                    "qa_failures",
                ]
            ].to_dict(
                orient="records"
            )
            if not rejected.empty
            else []
        ),
        "qa_audit": str(
            audit_path
        ),
        "accepted_manifest": str(
            accepted_path
        ),
        "rejected_manifest": str(
            rejected_path
        ),
        "accepted_distribution": str(
            distribution_path
        ),
        "failure_summary": str(
            failure_summary_path
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "dataset_v22_enrichment_qa_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Terminal report
    # ---------------------------------------------------------

    print()
    print("QA result")
    print("---------")
    print(
        f"Materialized tiles: {len(manifest)}"
    )
    print(
        f"Accepted tiles:     {len(accepted)}"
    )
    print(
        f"Rejected tiles:     {len(rejected)}"
    )
    print(
        f"Accepted cities:    "
        f"{accepted['city_id'].nunique()}"
    )

    print()
    print("Accepted enrichment class distribution")
    print("--------------------------------------")

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

    if not rejected.empty:

        print()
        print("Rejected tiles")
        print("--------------")

        print(
            rejected[
                [
                    "tile_id",
                    "city_id",
                    "city_name",
                    "padding_fraction",
                    "qa_failures",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("Class accounting")
    print("----------------")
    print(
        f"Accepted valid pixels: "
        f"{accepted_valid_pixels}"
    )
    print(
        f"Five-class pixel sum:  "
        f"{class_total}"
    )
    print(
        "Match: "
        f"{class_total == accepted_valid_pixels}"
    )

    print()
    print("Reports written")
    print("---------------")
    print(audit_path)
    print(accepted_path)
    print(rejected_path)
    print(distribution_path)
    print(failure_summary_path)
    print(summary_path)

    print()

    if (
        class_total
        != accepted_valid_pixels
    ):
        raise RuntimeError(
            "Dataset V2.2 enrichment QA FAILED: "
            "five-class accounting mismatch."
        )

    print(
        "Result: Dataset V2.2 enrichment QA "
        "completed successfully."
    )


if __name__ == "__main__":
    main()