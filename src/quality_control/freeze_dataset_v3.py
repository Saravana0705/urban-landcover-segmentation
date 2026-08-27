"""Create the immutable metadata freeze package for Dataset V3."""

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
CLASSES = [(1, "buildings"), (2, "roads"), (3, "vegetation"), (4, "bare_land"), (5, "water")]


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
    parser.add_argument("--manifest", type=Path, default=Path("metadata/dataset_v3/dataset_manifest.csv"))
    parser.add_argument("--tile-qa", type=Path, default=Path("metadata/dataset_v3/qa/dataset_v3_tile_qa.json"))
    parser.add_argument("--materialization-audit", type=Path, default=Path("metadata/dataset_v3/materialization/dataset_v3_materialization_audit.json"))
    parser.add_argument("--selection-audit", type=Path, default=Path("metadata/dataset_v3/selection/dataset_v3_selection_audit.json"))
    parser.add_argument("--normalization-provenance", type=Path, default=Path("metadata/dataset_v3/normalization/normalization_provenance.json"))
    parser.add_argument("--normalization", type=Path, default=Path("metadata/dataset_v3/normalization/training_normalization.json"))
    parser.add_argument("--manual-qa", type=Path, default=Path("metadata/dataset_v3/qa/manual_visual_qa/dataset_v3_manual_visual_qa.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("metadata/dataset_v3/freeze"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "manifest": args.manifest,
        "tile_qa": args.tile_qa,
        "materialization_audit": args.materialization_audit,
        "selection_audit": args.selection_audit,
        "normalization_provenance": args.normalization_provenance,
        "normalization": args.normalization,
        "manual_visual_qa": args.manual_qa,
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
    manual_qa = load_json(args.manual_qa)

    split_counts = {split: sum(row["split"].lower() == split for row in rows) for split in EXPECTED_SPLIT_COUNTS}
    split_city_ids = {
        split: sorted({row["city_id"] for row in rows if row["split"].lower() == split})
        for split in EXPECTED_SPLIT_COUNTS
    }
    city_counts = {split: len(ids) for split, ids in split_city_ids.items()}
    valid_by_split = {
        split: sum(int(row["valid_pixel_count"]) for row in rows if row["split"].lower() == split)
        for split in EXPECTED_SPLIT_COUNTS
    }
    class_counts_by_split: dict[str, dict[str, int]] = {}
    for split in EXPECTED_SPLIT_COUNTS:
        class_counts_by_split[split] = {
            name: sum(int(row[f"{name}_pixel_count"]) for row in rows if row["split"].lower() == split)
            for _, name in CLASSES
        }
    all_city_ids = [city for ids in split_city_ids.values() for city in ids]
    manifest_sha = sha256_file(args.manifest)
    class_accounting = all(
        sum(class_counts_by_split[split].values()) == valid_by_split[split]
        for split in EXPECTED_SPLIT_COUNTS
    )

    checks = {
        "manifest_has_1664_unique_tiles": len(rows) == 1664 and len({r["tile_id"] for r in rows}) == 1664,
        "split_tile_counts_match": split_counts == EXPECTED_SPLIT_COUNTS,
        "split_city_counts_match": city_counts == EXPECTED_CITY_COUNTS,
        "city_splits_disjoint": len(all_city_ids) == len(set(all_city_ids)) == 20,
        "all_tile_qa_passed": tile_qa.get("qa_status") == "PASS" and tile_qa.get("failed_tile_count") == 0,
        "tile_qa_manifest_hash_matches": tile_qa.get("official_manifest_sha256") == manifest_sha,
        "materialization_passed": materialization.get("status") == "PASS" and materialization.get("tile_count") == 1664,
        "selection_passed": selection.get("status") == "PASS" and selection.get("selected_tile_count") == 1664,
        "normalization_provenance_passed": norm_provenance.get("status") == "PASS",
        "normalization_excludes_validation_and_test": norm_provenance.get("validation_and_test_used") is False,
        "normalization_copy_hash_matches": norm_provenance.get("copied_normalization_sha256") == sha256_file(args.normalization),
        "manual_visual_qa_passed": manual_qa.get("status") == "PASS" and manual_qa.get("reviewed_tile_count") == 16,
        "class_pixels_reconcile_to_valid_pixels": class_accounting,
        "all_manifest_rows_pass_qa": all(row.get("tile_qa_status") == "PASS" for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Dataset V3 freeze refused: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    vv = normalization["bands"]["VV"]["db_clipped_p01_p99"]
    vh = normalization["bands"]["VH"]["db_clipped_p01_p99"]
    if not all(math.isfinite(value) for value in (vv["mean"], vv["std_population"], vh["mean"], vh["std_population"])):
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
        "dataset_version": "v3",
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
            "inheritance": "V1/V2.2 constants after source-SAR hash verification",
            "validation_and_test_used": False,
            "VV": {"clip_lower_db": vv["clip_lower_db"], "clip_upper_db": vv["clip_upper_db"], "mean_db": vv["mean"], "std_db": vv["std_population"]},
            "VH": {"clip_lower_db": vh["clip_lower_db"], "clip_upper_db": vh["clip_upper_db"], "mean_db": vh["mean"], "std_db": vh["std_population"]},
        },
        "labels": {
            "semantic_nodata": 255,
            "validity_valid": 1,
            "validity_invalid": 0,
            "classes": [{"class_id": class_id, "class_name": name} for class_id, name in CLASSES],
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

    input_checksums = {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in inputs.items()}
    checksums_path = args.output_dir / "metadata_checksums.json"
    write_json(checksums_path, input_checksums)

    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze_record = {
        "schema_version": "dataset-v3-freeze-0.1",
        "dataset_version": "v3",
        "freeze_status": "FROZEN",
        "frozen_at_utc": frozen_at,
        "city_count": 20,
        "tile_count": 1664,
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
        "manual_visual_qa": {"status": manual_qa["status"], "evidence_document": manual_qa["evidence_document"], "evidence_document_sha256": manual_qa["evidence_document_sha256"]},
        "mandatory_checks": checks,
        "all_checks_passed": True,
        "freeze_policy": {
            "do_not_modify_or_regenerate_v3_tiles": True,
            "do_not_change_city_splits": True,
            "do_not_recompute_normalization_from_selected_tiles": True,
            "future_changes_require_a_new_dataset_version": True,
            "models_must_use_frozen_dataset_config": True,
        },
    }
    freeze_path = args.output_dir / "dataset_v3_freeze.json"
    write_json(freeze_path, freeze_record)
    marker = {
        "dataset_version": "v3",
        "status": "FROZEN",
        "frozen_at_utc": frozen_at,
        "freeze_record": str(freeze_path),
        "freeze_record_sha256": sha256_file(freeze_path),
        "manifest_sha256": manifest_sha,
    }
    write_json(args.output_dir / "FROZEN", marker)

    lines = [
        "# Dataset V3 Freeze Record", "", "- Status: **FROZEN**",
        f"- Frozen at: **{frozen_at}**", "- Cities: **20**", "- Tiles: **1,664**",
        "- Split: **974 train / 351 validation / 339 test**", "",
        "## Scientific controls", "",
        "- City-wise 14/3/3 geographic split preserved.",
        "- Validation/test selection used coverage only, not class composition.",
        "- Unknown pixels remain label 255 and are excluded from loss/metrics.",
        "- Normalization inherits unchanged full training-city SAR statistics after SHA-256 verification.",
        "- Automated tile QA passed for all 1,664 tiles.",
        "- Targeted manual visual QA passed for 16 tiles and two conflict overlays.", "",
        "## Mandatory checks", "",
    ]
    lines.extend(f"- `{name}`: **{passed}**" for name, passed in checks.items())
    (args.output_dir / "DATASET_V3_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "FROZEN",
        "dataset_version": "v3",
        "tile_count": 1664,
        "split_tile_counts": split_counts,
        "next_gate": "short_unet_sanity_run",
    }, indent=2))


if __name__ == "__main__":
    main()
