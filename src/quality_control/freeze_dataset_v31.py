"""Create the immutable metadata freeze package for Dataset V3.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SPLIT_COUNTS = {"train": 1072, "val": 346, "test": 336}
EXPECTED_CITY_COUNTS = {"train": 14, "val": 3, "test": 3}
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
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
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
        default=Path("metadata/dataset_v3_1/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--tile-qa",
        type=Path,
        default=Path("metadata/dataset_v3_1/qa/dataset_v31_tile_qa.json"),
    )
    parser.add_argument(
        "--materialization-audit",
        type=Path,
        default=Path(
            "metadata/dataset_v3_1/materialization/"
            "dataset_v31_materialization_audit.json"
        ),
    )
    parser.add_argument(
        "--selection-audit",
        type=Path,
        default=Path(
            "metadata/dataset_v3_1/selection/dataset_v31_selection_audit.json"
        ),
    )
    parser.add_argument(
        "--normalization-provenance",
        type=Path,
        default=Path(
            "metadata/dataset_v3_1/normalization/normalization_provenance.json"
        ),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path(
            "metadata/dataset_v3_1/normalization/training_normalization.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v3_1/freeze"),
    )
    return parser.parse_args()


def selected_count(selection: dict[str, Any]) -> Any:
    for key in ("selected_tile_count", "selected_count", "selected_rows"):
        if key in selection:
            return selection[key]
    return None


def main() -> None:
    args = parse_args()
    inputs = {
        "manifest": args.manifest,
        "tile_qa": args.tile_qa,
        "materialization_audit": args.materialization_audit,
        "selection_audit": args.selection_audit,
        "normalization_provenance": args.normalization_provenance,
        "normalization": args.normalization,
    }
    for label, path in inputs.items():
        require_file(path, label)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite freeze package: {args.output_dir}")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tile_qa = load_json(args.tile_qa)
    materialization = load_json(args.materialization_audit)
    selection = load_json(args.selection_audit)
    norm_provenance = load_json(args.normalization_provenance)
    normalization = load_json(args.normalization)

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
    valid_by_split = {
        split: sum(
            int(row["valid_pixel_count"])
            for row in rows
            if row["split"].strip().lower() == split
        )
        for split in EXPECTED_SPLIT_COUNTS
    }
    class_counts_by_split: dict[str, dict[str, int]] = {}
    for split in EXPECTED_SPLIT_COUNTS:
        class_counts_by_split[split] = {
            name: sum(
                int(row[f"{name}_pixel_count"])
                for row in rows
                if row["split"].strip().lower() == split
            )
            for _, name in CLASSES
        }
    all_city_ids = [city for ids in split_city_ids.values() for city in ids]
    manifest_sha = sha256_file(args.manifest)
    class_accounting = all(
        sum(class_counts_by_split[split].values()) == valid_by_split[split]
        for split in EXPECTED_SPLIT_COUNTS
    )
    all_class_counts = {
        name: sum(class_counts_by_split[split][name] for split in EXPECTED_SPLIT_COUNTS)
        for _, name in CLASSES
    }

    checks = {
        "manifest_has_1754_unique_tiles": len(rows) == 1754
        and len({row["tile_id"] for row in rows}) == 1754,
        "split_tile_counts_match": split_counts == EXPECTED_SPLIT_COUNTS,
        "split_city_counts_match": city_counts == EXPECTED_CITY_COUNTS,
        "city_splits_disjoint": len(all_city_ids) == len(set(all_city_ids)) == 20,
        "all_tile_qa_passed": tile_qa.get("qa_status") == "PASS"
        and tile_qa.get("passed_tile_count") == 1754
        and tile_qa.get("failed_tile_count") == 0,
        "tile_qa_manifest_hash_matches": tile_qa.get("official_manifest_sha256")
        == manifest_sha,
        "tile_qa_counts_match_manifest": tile_qa.get("split_tile_counts")
        == split_counts
        and tile_qa.get("class_pixel_counts") == all_class_counts
        and tile_qa.get("valid_pixel_count") == sum(valid_by_split.values()),
        "materialization_passed": materialization.get("status") == "PASS"
        and materialization.get("tile_count") == 1754
        and materialization.get("raster_file_count") == 10524,
        "materialization_split_counts_match": materialization.get(
            "selected_by_split"
        )
        == split_counts,
        "selection_passed": selection.get("status") == "PASS"
        and selected_count(selection) == 1754,
        "normalization_provenance_passed": norm_provenance.get("status")
        == "PASS",
        "normalization_is_v31": norm_provenance.get("dataset_version") == "v3.1",
        "normalization_excludes_validation_and_test": norm_provenance.get(
            "validation_and_test_used"
        )
        is False,
        "normalization_copy_hash_matches": norm_provenance.get(
            "copied_normalization_sha256"
        )
        == sha256_file(args.normalization),
        "class_pixels_reconcile_to_valid_pixels": class_accounting,
        "all_manifest_rows_pass_qa": all(
            row.get("tile_qa_status") == "PASS" for row in rows
        ),
        "manual_visual_qa_not_required": tile_qa.get(
            "manual_visual_qa_required"
        )
        is False,
        "legacy_dataset_artifacts_unmodified": tile_qa.get(
            "legacy_dataset_artifacts_modified"
        )
        is False
        and materialization.get("parent_dataset_v3_modified") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Dataset V3.1 freeze refused: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    vv = normalization["bands"]["VV"]["db_clipped_p01_p99"]
    vh = normalization["bands"]["VH"]["db_clipped_p01_p99"]
    constants = (
        vv["clip_lower_db"],
        vv["clip_upper_db"],
        vv["mean"],
        vv["std_population"],
        vh["clip_lower_db"],
        vh["clip_upper_db"],
        vh["mean"],
        vh["std_population"],
    )
    if not all(math.isfinite(value) for value in constants):
        raise RuntimeError("Normalization contains non-finite constants.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
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
        "dataset_version": "v3.1",
        "parent_dataset_version": "v3",
        "freeze_status": "FROZEN",
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "input": {
            "channels": ["Sigma0_VV", "Sigma0_VH"],
            "channel_count": 2,
            "source_domain": "linear_sigma0",
            "conversion": "10 * log10(max(linear_sigma0, 1e-10))",
            "tile_size": 256,
            "spatial_resolution_m": 10,
        },
        "normalization": {
            "scope": "full_rasters_of_training_cities_only",
            "inheritance": (
                "Frozen V3 constants after V3.1 source-SAR SHA-256 verification"
            ),
            "validation_and_test_used": False,
            "VV": {
                "clip_lower_db": vv["clip_lower_db"],
                "clip_upper_db": vv["clip_upper_db"],
                "mean_db": vv["mean"],
                "std_db": vv["std_population"],
            },
            "VH": {
                "clip_lower_db": vh["clip_lower_db"],
                "clip_upper_db": vh["clip_upper_db"],
                "mean_db": vh["mean"],
                "std_db": vh["std_population"],
            },
        },
        "labels": {
            "semantic_nodata": 255,
            "validity_valid": 1,
            "validity_invalid": 0,
            "classes": [
                {"class_id": class_id, "class_name": name}
                for class_id, name in CLASSES
            ],
            "v31_policy": {
                "roads": (
                    "retain resolvable priority OSM roads; uncertain narrow or "
                    "low-priority roads become Ignore"
                ),
                "bare_land": (
                    "retain stable high-confidence bare/open land; ambiguous or "
                    "temporary construction becomes Ignore"
                ),
                "policy_applied_to_all_splits": True,
            },
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
            "normalization_source": "unchanged full training-city SAR rasters",
            "training_selection": "class-aware after coverage gate",
            "validation_test_selection": "coverage-only; no class filtering",
        },
    }
    dataset_config_path = args.output_dir / "dataset_config.json"
    write_json(dataset_config_path, dataset_config)

    input_checksums = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in inputs.items()
    }
    checksums_path = args.output_dir / "metadata_checksums.json"
    write_json(checksums_path, input_checksums)

    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze_record = {
        "schema_version": "dataset-v3.1-freeze-0.1",
        "dataset_version": "v3.1",
        "parent_dataset_version": "v3",
        "freeze_status": "FROZEN",
        "frozen_at_utc": frozen_at,
        "city_count": 20,
        "tile_count": 1754,
        "raster_file_count": 10524,
        "split_tile_counts": split_counts,
        "split_city_ids": split_city_ids,
        "valid_pixel_counts": valid_by_split,
        "class_distribution": class_distribution,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "dataset_config_path": str(dataset_config_path),
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "metadata_checksums_path": str(checksums_path),
        "normalization_policy": dataset_config["normalization"],
        "manual_visual_qa": {
            "required": False,
            "status": "WAIVED_BY_CONSERVATIVE_ADDITIVE_POLICY",
            "rationale": (
                "V3.1 adds no new positive labels: uncertain V3 road and bare-land "
                "pixels are changed only to Ignore. Automated alignment, encoding, "
                "pixel-accounting and checksum QA passed all 1,754 tiles."
            ),
            "automated_visual_sample_manifest": str(
                Path("metadata/dataset_v3_1/qa/dataset_v31_visual_qa_sample.csv")
            ),
            "sample_tile_count": tile_qa.get("visual_qa_tile_count"),
        },
        "mandatory_checks": checks,
        "all_checks_passed": True,
        "freeze_policy": {
            "do_not_modify_or_regenerate_v31_tiles": True,
            "do_not_change_city_splits": True,
            "do_not_recompute_normalization_from_selected_tiles": True,
            "future_label_or_selection_changes_require_a_new_dataset_version": True,
            "models_must_use_frozen_dataset_config": True,
            "test_split_remains_locked_until_final_evaluation": True,
        },
    }
    freeze_path = args.output_dir / "dataset_v31_freeze.json"
    write_json(freeze_path, freeze_record)
    marker = {
        "dataset_version": "v3.1",
        "status": "FROZEN",
        "frozen_at_utc": frozen_at,
        "freeze_record": str(freeze_path),
        "freeze_record_sha256": sha256_file(freeze_path),
        "manifest_sha256": manifest_sha,
    }
    write_json(args.output_dir / "FROZEN", marker)

    lines = [
        "# Dataset V3.1 Freeze Record",
        "",
        "- Status: **FROZEN**",
        f"- Frozen at: **{frozen_at}**",
        "- Cities: **20**",
        "- Tiles: **1,754**",
        "- Split: **1,072 train / 346 validation / 336 test**",
        "",
        "## Scientific controls",
        "",
        "- City-wise 14/3/3 geographic split preserved.",
        "- V3.1 label policy applied identically to train, validation and test cities.",
        "- Validation/test tile selection used coverage only, not class composition.",
        "- Unknown and deliberately excluded labels remain 255 and are excluded from loss/metrics.",
        "- Frozen V3 training-only normalization inherited after SAR SHA-256 verification.",
        "- Automated QA passed all 1,754 tile groups (10,524 GeoTIFFs).",
        "- Manual visual QA was not required because V3.1 adds no positive labels.",
        "",
        "## Mandatory checks",
        "",
    ]
    lines.extend(f"- `{name}`: **{passed}**" for name, passed in checks.items())
    (args.output_dir / "DATASET_V31_FREEZE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "dataset_version": "v3.1",
                "tile_count": 1754,
                "split_tile_counts": split_counts,
                "manual_visual_qa_required": False,
                "next_gate": "controlled_v3_vs_v31_gpu_experiment",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
