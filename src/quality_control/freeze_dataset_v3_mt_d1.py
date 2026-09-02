"""Create the immutable metadata freeze package for Dataset V3-MT-D1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SPLIT_COUNTS = {"train": 974, "val": 351, "test": 339}
EXPECTED_CITY_COUNTS = {"train": 14, "val": 3, "test": 3}
EXPECTED_BANDS = [
    band
    for month in range(1, 13)
    for band in (f"VV_M{month:02d}", f"VH_M{month:02d}")
]
EXPECTED_MANIFEST_SHA256 = (
    "1daa51177702c256cb831b0dd8a00c12d5df869dce3526332beeb166e9535b78"
)
CLASSES = [
    (1, "buildings"),
    (2, "roads"),
    (3, "vegetation"),
    (4, "bare_land"),
    (5, "water"),
]


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON input")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/dataset_v3_mt_d1/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--tile-qa",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/qa/dataset_v3_mt_d1_tile_qa.json"
        ),
    )
    parser.add_argument(
        "--materialization-audit",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/materialization/"
            "dataset_v3_mt_d1_materialization_audit.json"
        ),
    )
    parser.add_argument(
        "--source-qa",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/source_qa/"
            "dataset_v3_mt_d1_source_qa.json"
        ),
    )
    parser.add_argument(
        "--normalization-provenance",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/normalization/"
            "normalization_provenance.json"
        ),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/normalization/"
            "training_normalization.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v3_mt_d1/freeze"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "manifest": args.manifest,
        "tile_qa": args.tile_qa,
        "materialization_audit": args.materialization_audit,
        "source_qa": args.source_qa,
        "normalization_provenance": args.normalization_provenance,
        "normalization": args.normalization,
    }
    for label, path in inputs.items():
        require_file(path, label)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite freeze package: {args.output_dir}"
        )

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tile_qa = load_json(args.tile_qa)
    materialization = load_json(args.materialization_audit)
    source_qa = load_json(args.source_qa)
    provenance = load_json(args.normalization_provenance)
    normalization = load_json(args.normalization)
    manifest_sha = sha256_file(args.manifest)

    split_counts = {
        split: sum(row["split"].strip().lower() == split for row in rows)
        for split in EXPECTED_SPLIT_COUNTS
    }
    split_city_ids = {
        split: sorted(
            {
                row["city_id"]
                for row in rows
                if row["split"].strip().lower() == split
            }
        )
        for split in EXPECTED_SPLIT_COUNTS
    }
    city_counts = {split: len(ids) for split, ids in split_city_ids.items()}
    all_city_ids = [value for ids in split_city_ids.values() for value in ids]
    valid_by_split = {
        split: sum(
            int(row["valid_pixel_count"])
            for row in rows
            if row["split"].strip().lower() == split
        )
        for split in EXPECTED_SPLIT_COUNTS
    }
    class_counts_by_split = {
        split: {
            name: sum(
                int(row[f"{name}_pixel_count"])
                for row in rows
                if row["split"].strip().lower() == split
            )
            for _, name in CLASSES
        }
        for split in EXPECTED_SPLIT_COUNTS
    }
    class_accounting = all(
        sum(class_counts_by_split[split].values()) == valid_by_split[split]
        for split in EXPECTED_SPLIT_COUNTS
    )
    training_city_ids = [f"DE{index:02d}" for index in range(1, 15)]
    excluded_city_ids = [f"DE{index:02d}" for index in range(15, 21)]

    checks = {
        "manifest_has_1664_unique_tiles": (
            len(rows) == 1664 and len({row["tile_id"] for row in rows}) == 1664
        ),
        "manifest_hash_is_authoritative": manifest_sha == EXPECTED_MANIFEST_SHA256,
        "split_tile_counts_match": split_counts == EXPECTED_SPLIT_COUNTS,
        "split_city_counts_match": city_counts == EXPECTED_CITY_COUNTS,
        "city_splits_disjoint": len(all_city_ids) == len(set(all_city_ids)) == 20,
        "training_city_ids_match": split_city_ids["train"] == training_city_ids,
        "evaluation_city_ids_match": (
            split_city_ids["val"] + split_city_ids["test"] == excluded_city_ids
        ),
        "tile_qa_passed": (
            tile_qa.get("status") == "PASS"
            and tile_qa.get("tile_count") == 1664
            and tile_qa.get("passed_tile_count") == 1664
            and tile_qa.get("failed_tile_count") == 0
            and tile_qa.get("failures") == []
            and tile_qa.get("parent_dataset") == "v3-mt"
        ),
        "tile_qa_manifest_hash_matches": (
            tile_qa.get("manifest_sha256") == manifest_sha
        ),
        "tile_qa_split_counts_match": (
            tile_qa.get("split_tile_counts") == split_counts
        ),
        "materialization_passed": (
            materialization.get("status") == "PASS"
            and materialization.get("dataset_variant") == "v3-mt-d1"
            and materialization.get("parent_dataset") == "v3-mt"
            and materialization.get("tile_count") == 1664
            and materialization.get("raster_file_count") == 4992
            and materialization.get("temporal_removed_valid_pixel_count") == 0
            and materialization.get("temporal_retained_fraction_parent_valid")
            == 1.0
        ),
        "materialization_manifest_hash_matches": (
            materialization.get("manifest_sha256") == manifest_sha
        ),
        "materialization_split_counts_match": (
            materialization.get("split_tile_counts") == split_counts
        ),
        "source_qa_passed": (
            source_qa.get("status") == "PASS"
            and source_qa.get("city_count") == 20
            and source_qa.get("expected_band_order") == EXPECTED_BANDS
            and float(source_qa.get("minimum_valid_fraction", 0)) >= 0.98
        ),
        "source_qa_hash_matches_materialization": (
            sha256_file(args.source_qa) == materialization.get("source_qa_sha256")
        ),
        "normalization_passed": (
            normalization.get("status") == "PASS"
            and normalization.get("dataset_variant") == "v3-mt-d1"
            and normalization.get("band_order") == EXPECTED_BANDS
            and normalization.get("training_city_ids") == training_city_ids
            and normalization.get("validation_and_test_used") is False
        ),
        "normalization_hash_matches_materialization": (
            sha256_file(args.normalization)
            == materialization.get("normalization_sha256")
        ),
        "normalization_provenance_passed": (
            provenance.get("status") == "PASS"
            and provenance.get("training_city_ids") == training_city_ids
            and provenance.get("excluded_city_ids") == excluded_city_ids
        ),
        "normalization_provenance_hash_matches_materialization": (
            sha256_file(args.normalization_provenance)
            == materialization.get("normalization_provenance_sha256")
        ),
        "class_pixels_reconcile_to_valid_pixels": class_accounting,
        "all_manifest_rows_pass_qa": all(
            row.get("tile_qa_status") == "PASS" for row in rows
        ),
        "all_manifest_rows_materialized": all(
            row.get("materialization_status") == "PASS" for row in rows
        ),
        "all_image_paths_are_v3_mt_d1": all(
            str(row["image_path"]).replace("/", "\\").startswith(
                "data\\processed\\dataset_v3_mt_d1\\tiles\\"
            )
            for row in rows
        ),
        "legacy_dataset_artifacts_unmodified": (
            materialization.get("legacy_dataset_artifacts_modified") is False
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "Dataset V3-MT-D1 freeze refused: " + ", ".join(failed)
        )

    bands: dict[str, dict[str, float]] = {}
    for name in EXPECTED_BANDS:
        model_input = normalization["bands"][name]["model_input"]
        constants = {
            "clip_lower_db": float(model_input["clip_lower_db"]),
            "clip_upper_db": float(model_input["clip_upper_db"]),
            "mean_db": float(model_input["mean_db"]),
            "std_db": float(model_input["std_db"]),
        }
        if not all(math.isfinite(value) for value in constants.values()):
            raise RuntimeError(f"Non-finite normalization constants: {name}")
        if constants["clip_lower_db"] >= constants["clip_upper_db"]:
            raise RuntimeError(f"Invalid clip bounds: {name}")
        if constants["std_db"] <= 0:
            raise RuntimeError(f"Invalid standard deviation: {name}")
        bands[name] = constants

    class_distribution = {
        split: {
            name: {
                "pixel_count": count,
                "fraction_valid": count / valid_by_split[split],
            }
            for name, count in class_counts_by_split[split].items()
        }
        for split in EXPECTED_SPLIT_COUNTS
    }
    dataset_config = {
        "dataset_version": "v3-mt-d1",
        "parent_dataset": "v3-mt",
        "parent_label_dataset": "v3",
        "freeze_status": "FROZEN",
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "input": {
            "channels": EXPECTED_BANDS,
            "channel_count": 24,
            "enforce_band_descriptions": True,
            "source_domain": "linear_sigma0",
            "conversion": "10 * log10(max(linear_sigma0, epsilon))",
            "tile_size": 256,
            "spatial_resolution_m": 10,
            "temporal_composites": [
                f"M{month:02d}_2025" for month in range(1, 13)
            ],
        },
        "normalization": {
            "scope": "full_rasters_of_training_cities_only",
            "validation_and_test_used": False,
            "epsilon": float(provenance.get("epsilon", 1e-10)),
            "bands": bands,
        },
        "labels": {
            "semantic_nodata": 255,
            "validity_valid": 1,
            "validity_invalid": 0,
            "classes": [
                {"class_id": class_id, "class_name": name}
                for class_id, name in CLASSES
            ],
            "reused_unchanged_from": "v3",
        },
        "splits": {
            "train_city_ids": split_city_ids["train"],
            "validation_city_ids": split_city_ids["val"],
            "test_city_ids": split_city_ids["test"],
            "train_tile_count": split_counts["train"],
            "validation_tile_count": split_counts["val"],
            "test_tile_count": split_counts["test"],
        },
        "class_distribution": class_distribution,
        "training_policy": {
            "ignore_index": 255,
            "use_validity_mask": True,
            "training_selection": "identical to frozen Dataset V3-MT",
            "validation_test_selection": "identical to frozen Dataset V3-MT",
            "test_split_locked": True,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_config_path = args.output_dir / "dataset_config.json"
    write_json(dataset_config_path, dataset_config)
    checksums_path = args.output_dir / "metadata_checksums.json"
    write_json(
        checksums_path,
        {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in inputs.items()
        },
    )

    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze_record = {
        "schema_version": "dataset-v3-mt-d1-freeze-0.1",
        "dataset_version": "v3-mt-d1",
        "parent_dataset": "v3-mt",
        "parent_label_dataset": "v3",
        "freeze_status": "FROZEN",
        "frozen_at_utc": frozen_at,
        "city_count": 20,
        "tile_count": 1664,
        "raster_file_count": 4992,
        "split_tile_counts": split_counts,
        "split_city_ids": split_city_ids,
        "valid_pixel_counts": valid_by_split,
        "class_distribution": class_distribution,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "dataset_config_path": str(dataset_config_path),
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "metadata_checksums_path": str(checksums_path),
        "mandatory_checks": checks,
        "all_checks_passed": True,
        "experimental_control": {
            "intervention": (
                "replace eight quarterly VV/VH channels with "
                "24 monthly VV/VH channels"
            ),
            "labels_tiles_splits_unchanged_from_v3_mt": True,
            "paired_baseline": "unet_v3_mt_e0_paired_50ep",
            "primary_metric": "validation mean IoU",
        },
        "freeze_policy": {
            "do_not_modify_or_regenerate_tiles": True,
            "do_not_change_city_splits": True,
            "do_not_recompute_normalization": True,
            "future_data_changes_require_a_new_dataset_version": True,
            "models_must_use_frozen_dataset_config": True,
            "test_split_remains_locked_until_final_evaluation": True,
        },
    }
    freeze_path = args.output_dir / "dataset_v3_mt_d1_freeze.json"
    write_json(freeze_path, freeze_record)
    write_json(
        args.output_dir / "FROZEN",
        {
            "dataset_version": "v3-mt-d1",
            "status": "FROZEN",
            "frozen_at_utc": frozen_at,
            "freeze_record": str(freeze_path),
            "freeze_record_sha256": sha256_file(freeze_path),
            "manifest_sha256": manifest_sha,
        },
    )

    lines = [
        "# Dataset V3-MT-D1 Freeze Record",
        "",
        "- Status: **FROZEN**",
        f"- Frozen at: **{frozen_at}**",
        "- Input: **24 monthly Sentinel-1 VV/VH bands**",
        "- Tiles: **1,664**",
        "- Split: **974 train / 351 validation / 339 test**",
        "- Labels, tiles and city split: **unchanged from Dataset V3-MT**",
        "- Independent tile QA: **1,664/1,664 PASS**",
        "",
        "## Mandatory checks",
        "",
    ]
    lines.extend(f"- `{name}`: **{passed}**" for name, passed in checks.items())
    (args.output_dir / "DATASET_V3_MT_D1_FREEZE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "dataset_version": "v3-mt-d1",
                "tile_count": 1664,
                "split_tile_counts": split_counts,
                "input_channels": 24,
                "next_gate": "local_unet_v3_mt_d1_integration_check",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
