"""Freeze Dataset Version 1 for reproducible model experiments.

This script creates a formal metadata freeze package. It does not copy or
modify the 2,880 image/mask tile files.

The freeze package records:
- Dataset version and creation time
- Authoritative manifest checksum
- Dataset statistics and split definitions
- Training-only normalization constants
- Candidate class weights
- Checksums of all metadata that defines Dataset V1
- A canonical dataset configuration for all model pipelines
- A human-readable freeze record

Outputs
-------
metadata/dataset_v1/freeze/
    DATASET_V1_FREEZE.json
    DATASET_V1_FREEZE.md
    dataset_config.json
    metadata_checksums.csv
    metadata_checksums.json
    FROZEN

By default, the script refuses to overwrite an existing freeze.
Use --force only when intentionally rebuilding the freeze package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "v1.0"

DEFAULT_METADATA_FILES = {
    "dataset_manifest": Path(
        "metadata/dataset_v1/dataset_manifest.csv"
    ),
    "dataset_manifest_summary": Path(
        "metadata/dataset_v1/dataset_manifest_summary.json"
    ),
    "dataset_city_summary": Path(
        "metadata/dataset_v1/dataset_city_summary.csv"
    ),
    "dataset_split_summary": Path(
        "metadata/dataset_v1/dataset_split_summary.csv"
    ),
    "dataset_class_distribution": Path(
        "metadata/dataset_v1/dataset_class_distribution.csv"
    ),
    "training_normalization": Path(
        "metadata/dataset_v1/normalization/"
        "training_normalization.json"
    ),
    "training_normalization_csv": Path(
        "metadata/dataset_v1/normalization/"
        "training_normalization.csv"
    ),
    "training_normalization_city_audit": Path(
        "metadata/dataset_v1/normalization/"
        "training_normalization_city_audit.csv"
    ),
    "dataset_statistics": Path(
        "metadata/dataset_v1/statistics/"
        "dataset_statistics.json"
    ),
    "dataset_statistics_csv": Path(
        "metadata/dataset_v1/statistics/"
        "dataset_statistics.csv"
    ),
    "class_distribution_detailed": Path(
        "metadata/dataset_v1/statistics/"
        "class_distribution_detailed.csv"
    ),
    "class_weights_candidates": Path(
        "metadata/dataset_v1/statistics/"
        "class_weights_candidates.json"
    ),
    "class_weights_candidates_csv": Path(
        "metadata/dataset_v1/statistics/"
        "class_weights_candidates.csv"
    ),
    "dataset_statistics_markdown": Path(
        "metadata/dataset_v1/statistics/"
        "DATASET_V1_STATISTICS.md"
    ),
    "cities_registry": Path("config/cities.csv"),
    "rasterization_config": Path("config/rasterization.yaml"),
    "road_widths_config": Path("config/road_widths.yaml"),
    "tiling_config": Path("config/tiling.yaml"),
    "class_mapping_config": Path("config/class_mapping.yaml"),
    "osm_class_mapping_config": Path(
        "config/osm_class_mapping.yaml"
    ),
    "preprocessing_config": Path("config/preprocessing.yaml"),
    "acquisition_config": Path("config/acquisition.yaml"),
}


def require_file(path: Path, label: str) -> None:
    """Require a non-empty file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object."""
    require_file(path, "JSON file")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def write_json(path: Path, payload: Any) -> None:
    """Write formatted UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Freeze Dataset Version 1 metadata."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v1/freeze"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild an existing freeze package. "
            "Use only intentionally."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Create and validate the Dataset V1 freeze package."""
    args = parse_arguments()

    freeze_marker = args.output_dir / "FROZEN"

    if freeze_marker.exists() and not args.force:
        raise RuntimeError(
            "Dataset V1 is already frozen. "
            "Refusing to overwrite. Use --force only if intentional."
        )

    for label, path in DEFAULT_METADATA_FILES.items():
        require_file(path, label)

    manifest_summary_path = DEFAULT_METADATA_FILES[
        "dataset_manifest_summary"
    ]
    statistics_path = DEFAULT_METADATA_FILES[
        "dataset_statistics"
    ]
    normalization_path = DEFAULT_METADATA_FILES[
        "training_normalization"
    ]
    weights_path = DEFAULT_METADATA_FILES[
        "class_weights_candidates"
    ]
    manifest_path = DEFAULT_METADATA_FILES[
        "dataset_manifest"
    ]

    manifest_summary = load_json(manifest_summary_path)
    statistics = load_json(statistics_path)
    normalization = load_json(normalization_path)
    class_weights = load_json(weights_path)

    mandatory_checks = statistics.get("mandatory_checks")

    if not isinstance(mandatory_checks, dict):
        raise RuntimeError(
            "Dataset statistics do not contain mandatory checks."
        )

    failed_checks = [
        name
        for name, passed in mandatory_checks.items()
        if passed is not True
    ]

    if failed_checks:
        raise RuntimeError(
            "Dataset cannot be frozen because mandatory checks failed: "
            f"{failed_checks}"
        )

    if statistics.get("ready_for_dataset_freeze") is not True:
        raise RuntimeError(
            "Dataset statistics do not mark the dataset as freeze-ready."
        )

    actual_manifest_sha256 = sha256_file(manifest_path)

    recorded_manifest_sha256 = statistics.get(
        "manifest_sha256"
    )

    if actual_manifest_sha256 != recorded_manifest_sha256:
        raise RuntimeError(
            "Dataset manifest checksum differs from the statistics record."
        )

    if (
        actual_manifest_sha256
        != manifest_summary.get("manifest_sha256")
    ):
        raise RuntimeError(
            "Dataset manifest checksum differs from the manifest summary."
        )

    if normalization.get("validation_and_test_used") is not False:
        raise RuntimeError(
            "Normalization metadata indicates validation/test leakage."
        )

    if normalization.get("training_city_count") != 14:
        raise RuntimeError(
            "Expected normalization from exactly 14 training cities."
        )

    if manifest_summary.get("city_count") != 20:
        raise RuntimeError("Expected exactly 20 cities.")

    if manifest_summary.get("tile_count") != 2880:
        raise RuntimeError("Expected exactly 2,880 tiles.")

    if manifest_summary.get("city_splits_disjoint") is not True:
        raise RuntimeError("City splits are not disjoint.")

    if manifest_summary.get("all_tile_qa_passed") is not True:
        raise RuntimeError("Not all tiles passed QA.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checksums_json_path = (
        args.output_dir / "metadata_checksums.json"
    )
    checksums_csv_path = (
        args.output_dir / "metadata_checksums.csv"
    )
    dataset_config_path = (
        args.output_dir / "dataset_config.json"
    )
    freeze_json_path = (
        args.output_dir / "DATASET_V1_FREEZE.json"
    )
    freeze_markdown_path = (
        args.output_dir / "DATASET_V1_FREEZE.md"
    )

    checksum_rows: list[dict[str, Any]] = []

    for role, path in DEFAULT_METADATA_FILES.items():
        checksum_rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    checksums_payload = {
        "dataset_version": VERSION,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "files": checksum_rows,
    }

    write_json(
        checksums_json_path,
        checksums_payload,
    )

    with checksums_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "role",
                "path",
                "size_bytes",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(checksum_rows)

    vv = normalization["bands"]["VV"][
        "db_clipped_p01_p99"
    ]
    vh = normalization["bands"]["VH"][
        "db_clipped_p01_p99"
    ]

    dataset_config = {
        "dataset_version": VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "input": {
            "channels": [
                "Sigma0_VV",
                "Sigma0_VH",
            ],
            "channel_count": 2,
            "source_domain": "linear_sigma0",
            "conversion": (
                "10 * log10(max(linear_sigma0, 1e-10))"
            ),
            "tile_size": int(
                statistics["tile_size"]
            ),
            "spatial_resolution_m": 10,
        },
        "normalization": {
            "scope": "training_cities_only",
            "validation_and_test_used": False,
            "VV": {
                "clip_lower_db": float(
                    vv["clip_lower_db"]
                ),
                "clip_upper_db": float(
                    vv["clip_upper_db"]
                ),
                "mean_db": float(vv["mean"]),
                "std_db": float(
                    vv["std_population"]
                ),
            },
            "VH": {
                "clip_lower_db": float(
                    vh["clip_lower_db"]
                ),
                "clip_upper_db": float(
                    vh["clip_upper_db"]
                ),
                "mean_db": float(vh["mean"]),
                "std_db": float(
                    vh["std_population"]
                ),
            },
        },
        "labels": {
            "semantic_nodata": 255,
            "validity_valid": 1,
            "validity_invalid": 0,
            "classes": [
                {
                    "class_id": 1,
                    "class_name": "buildings",
                },
                {
                    "class_id": 2,
                    "class_name": "roads",
                },
                {
                    "class_id": 3,
                    "class_name": "vegetation",
                },
                {
                    "class_id": 4,
                    "class_name": "bare_land",
                },
                {
                    "class_id": 5,
                    "class_name": "water",
                },
            ],
        },
        "splits": {
            "train_city_ids": manifest_summary[
                "split_city_ids"
            ]["train"],
            "validation_city_ids": manifest_summary[
                "split_city_ids"
            ]["val"],
            "test_city_ids": manifest_summary[
                "split_city_ids"
            ]["test"],
            "train_tile_count": manifest_summary[
                "split_tile_counts"
            ]["train"],
            "validation_tile_count": manifest_summary[
                "split_tile_counts"
            ]["val"],
            "test_tile_count": manifest_summary[
                "split_tile_counts"
            ]["test"],
        },
        "class_weight_candidates": {
            "scope": class_weights.get(
                "scope"
            ),
            "selection_status": class_weights.get(
                "selection_status"
            ),
            "classes": class_weights.get(
                "classes"
            ),
        },
        "training_policy": {
            "ignore_unlabeled_pixels": True,
            "ignore_semantic_nodata": True,
            "use_validity_mask": True,
            "normalization_source": (
                "Dataset V1 training split only"
            ),
            "augmentation": (
                "to be defined in model-training configuration"
            ),
            "loss_weighting": (
                "not yet selected; evaluate candidate schemes"
            ),
        },
    }

    write_json(
        dataset_config_path,
        dataset_config,
    )

    freeze_record = {
        "dataset_name": (
            "Sentinel-1 Multi-City Urban Land-Cover "
            "Segmentation Dataset"
        ),
        "dataset_version": VERSION,
        "freeze_status": "FROZEN",
        "frozen_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "city_count": int(
            manifest_summary["city_count"]
        ),
        "tile_count": int(
            manifest_summary["tile_count"]
        ),
        "train_city_count": len(
            manifest_summary["split_city_ids"]["train"]
        ),
        "validation_city_count": len(
            manifest_summary["split_city_ids"]["val"]
        ),
        "test_city_count": len(
            manifest_summary["split_city_ids"]["test"]
        ),
        "train_tile_count": int(
            manifest_summary[
                "split_tile_counts"
            ]["train"]
        ),
        "validation_tile_count": int(
            manifest_summary[
                "split_tile_counts"
            ]["val"]
        ),
        "test_tile_count": int(
            manifest_summary[
                "split_tile_counts"
            ]["test"]
        ),
        "zero_valid_tile_count": int(
            manifest_summary["zero_valid_tile_count"]
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "metadata_checksums_path": str(
            checksums_json_path
        ),
        "dataset_config_path": str(
            dataset_config_path
        ),
        "normalization_scope": (
            "training cities only"
        ),
        "validation_and_test_used_for_normalization": False,
        "class_imbalance": {
            "dominant_class": statistics[
                "training_class_imbalance"
            ]["dominant_class"],
            "rarest_class": statistics[
                "training_class_imbalance"
            ]["rarest_class"],
            "dominant_to_rarest_pixel_ratio": statistics[
                "training_class_imbalance"
            ]["dominant_to_rarest_pixel_ratio"],
        },
        "mandatory_checks": mandatory_checks,
        "all_checks_passed": all(
            value is True
            for value in mandatory_checks.values()
        ),
        "freeze_policy": {
            "do_not_regenerate_or_modify_tiles": True,
            "do_not_change_city_splits": True,
            "do_not_recompute_normalization": True,
            "new_data_requires_new_dataset_version": True,
            "future_models_must_use_dataset_config": True,
        },
    }

    write_json(
        freeze_json_path,
        freeze_record,
    )

    markdown_lines = [
        "# Dataset Version 1 Freeze Record",
        "",
        f"- Dataset version: **{VERSION}**",
        "- Status: **FROZEN**",
        f"- Frozen at: **{freeze_record['frozen_at_utc']}**",
        f"- Cities: **{freeze_record['city_count']}**",
        f"- Tiles: **{freeze_record['tile_count']}**",
        f"- Training cities / tiles: **"
        f"{freeze_record['train_city_count']} / "
        f"{freeze_record['train_tile_count']}**",
        f"- Validation cities / tiles: **"
        f"{freeze_record['validation_city_count']} / "
        f"{freeze_record['validation_tile_count']}**",
        f"- Test cities / tiles: **"
        f"{freeze_record['test_city_count']} / "
        f"{freeze_record['test_tile_count']}**",
        f"- Zero-valid tiles retained and tracked: **"
        f"{freeze_record['zero_valid_tile_count']}**",
        "",
        "## Definition of Dataset V1",
        "",
        "Dataset V1 is defined by the authoritative global manifest, "
        "the fixed city-level train/validation/test split, the approved "
        "five-class semantic masks, tile-QA status, and the training-only "
        "normalization parameters recorded in this freeze package.",
        "",
        "## Input",
        "",
        "- Sentinel-1 Sigma0 VV and VH",
        "- Linear Sigma0 converted to dB",
        "- Per-band clipping using training p1 and p99",
        "- Standardization using training-only clipped mean and standard deviation",
        "- Tile size: 256 × 256 pixels",
        "- Spatial resolution: 10 m",
        "",
        "## Semantic Classes",
        "",
        "1. Buildings",
        "2. Roads",
        "3. Vegetation",
        "4. Bare land",
        "5. Water",
        "",
        "Unlabeled and invalid pixels are excluded from loss and metric "
        "calculations using the semantic and validity masks.",
        "",
        "## Reproducibility Policy",
        "",
        "- Do not regenerate or edit Dataset V1 image or mask tiles.",
        "- Do not change the city-level split assignments.",
        "- Do not recompute normalization using validation or test data.",
        "- All models must use `dataset_config.json`.",
        "- Any label, imagery, split, or preprocessing change requires "
        "a new dataset version such as v1.1 or v2.0.",
        "",
        "## Core Checksums",
        "",
        f"- Dataset manifest SHA-256: `{actual_manifest_sha256}`",
        f"- Metadata checksum inventory: `{checksums_json_path}`",
        "",
        "## Mandatory Checks",
        "",
    ]

    for check_name, passed in mandatory_checks.items():
        markdown_lines.append(
            f"- `{check_name}`: **{passed}**"
        )

    markdown_lines.extend(
        [
            "",
            "All mandatory checks passed: "
            f"**{freeze_record['all_checks_passed']}**",
            "",
        ]
    )

    freeze_markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    marker_content = {
        "dataset_version": VERSION,
        "status": "FROZEN",
        "frozen_at_utc": freeze_record[
            "frozen_at_utc"
        ],
        "freeze_record": str(
            freeze_json_path
        ),
        "manifest_sha256": actual_manifest_sha256,
    }

    write_json(
        freeze_marker,
        marker_content,
    )

    print("\nDataset V1 freeze")
    print("-----------------")
    print(f"Version: {VERSION}")
    print("Status: FROZEN")
    print(
        f"Cities: {freeze_record['city_count']}"
    )
    print(
        f"Tiles: {freeze_record['tile_count']}"
    )
    print(
        "Split tiles: "
        f"train={freeze_record['train_tile_count']}, "
        f"val={freeze_record['validation_tile_count']}, "
        f"test={freeze_record['test_tile_count']}"
    )
    print(
        f"Manifest SHA-256: "
        f"{actual_manifest_sha256}"
    )

    print("\nMandatory checks")
    for name, passed in mandatory_checks.items():
        print(f"  {name}: {passed}")

    print(
        f"\nFreeze record: {freeze_json_path}"
    )
    print(
        f"Freeze document: {freeze_markdown_path}"
    )
    print(
        f"Dataset config: {dataset_config_path}"
    )
    print(
        f"Checksum inventory: {checksums_json_path}"
    )
    print(
        f"Freeze marker: {freeze_marker}"
    )

    print(
        "\nResult: Dataset Version 1 has been formally frozen."
    )


if __name__ == "__main__":
    main()
