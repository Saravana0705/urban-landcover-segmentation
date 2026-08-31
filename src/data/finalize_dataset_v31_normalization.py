"""Validate and record Dataset V3.1 normalization inheritance.

Dataset V3.1 changes label policy and tile selection, but not the Sentinel-1
source rasters or the 14/3/3 city split.  It therefore inherits the frozen V3
training-only normalization after checking the source-raster hashes recorded
during V3.1 materialization.  Validation and test imagery are never used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TRAIN_IDS = [f"DE{index:02d}" for index in range(1, 15)]
EXPECTED_SPLIT_COUNTS = {"train": 1072, "val": 346, "test": 336}


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
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalized_path(value: str) -> str:
    return value.replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/dataset_v3_1/dataset_manifest.csv"),
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
        "--source-normalization",
        type=Path,
        default=Path("metadata/dataset_v3/normalization/training_normalization.json"),
    )
    parser.add_argument(
        "--source-normalization-csv",
        type=Path,
        default=Path("metadata/dataset_v3/normalization/training_normalization.csv"),
    )
    parser.add_argument(
        "--source-provenance",
        type=Path,
        default=Path("metadata/dataset_v3/normalization/normalization_provenance.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v3_1/normalization"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.manifest, "V3.1 official manifest"),
        (args.materialization_audit, "V3.1 materialization audit"),
        (args.source_normalization, "V3 normalization"),
        (args.source_provenance, "V3 normalization provenance"),
    ):
        require_file(path, label)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite normalization output: {args.output_dir}"
        )

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("V3.1 manifest is empty.")
    tile_ids = [row["tile_id"] for row in rows]
    split_counts = {
        split: sum(row["split"].strip().lower() == split for row in rows)
        for split in EXPECTED_SPLIT_COUNTS
    }
    train_ids = sorted(
        {row["city_id"] for row in rows if row["split"].strip().lower() == "train"}
    )

    materialization = load_json(args.materialization_audit)
    source_normalization = load_json(args.source_normalization)
    source_provenance = load_json(args.source_provenance)
    materialization_hashes = {
        normalized_path(str(path)): str(digest)
        for path, digest in materialization.get("source_sha256", {}).items()
    }
    normalization_by_city = {
        str(item["city_id"]): item
        for item in source_normalization.get("source_rasters", [])
    }

    source_hash_checks: list[dict[str, Any]] = []
    for city_id in EXPECTED_TRAIN_IDS:
        item = normalization_by_city.get(city_id)
        if item is None:
            source_hash_checks.append(
                {"city_id": city_id, "status": "MISSING_NORMALIZATION_SOURCE"}
            )
            continue
        source_path = normalized_path(str(item["path"]))
        expected_hash = str(item["sha256"])
        actual_hash = materialization_hashes.get(source_path)
        source_hash_checks.append(
            {
                "city_id": city_id,
                "path": source_path,
                "v3_normalization_sha256": expected_hash,
                "v31_materialization_source_sha256": actual_hash,
                "status": "PASS" if actual_hash == expected_hash else "FAIL",
            }
        )

    source_norm_sha = sha256_file(args.source_normalization)
    checks = {
        "manifest_has_1754_unique_tiles": len(rows) == 1754
        and len(set(tile_ids)) == 1754,
        "split_tile_counts_match": split_counts == EXPECTED_SPLIT_COUNTS,
        "training_city_ids_match": train_ids == EXPECTED_TRAIN_IDS,
        "materialization_status_pass": materialization.get("status") == "PASS",
        "materialization_tile_count_is_1754": materialization.get("tile_count")
        == 1754,
        "source_v3_normalization_provenance_pass": source_provenance.get("status")
        == "PASS",
        "source_v3_normalization_hash_matches_provenance": source_provenance.get(
            "copied_normalization_sha256"
        )
        == source_norm_sha,
        "normalization_training_city_count_is_14": source_normalization.get(
            "training_city_count"
        )
        == 14,
        "normalization_training_city_ids_match": source_normalization.get(
            "training_city_ids"
        )
        == EXPECTED_TRAIN_IDS,
        "validation_and_test_not_used": source_normalization.get(
            "validation_and_test_used"
        )
        is False,
        "all_training_sar_hashes_match": all(
            item.get("status") == "PASS" for item in source_hash_checks
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Dataset V3.1 normalization inheritance failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    copied_json = args.output_dir / "training_normalization.json"
    shutil.copy2(args.source_normalization, copied_json)
    copied_csv: Path | None = None
    if args.source_normalization_csv.is_file():
        copied_csv = args.output_dir / "training_normalization.csv"
        shutil.copy2(args.source_normalization_csv, copied_csv)

    provenance = {
        "schema_version": "dataset-v3.1-normalization-provenance-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_version": "v3.1",
        "parent_dataset_version": "v3",
        "policy": "inherit_unchanged_v3_full_training_city_sar_normalization",
        "scientific_rationale": (
            "V3.1 changes label confidence policy and candidate selection only. "
            "It uses the same fourteen training cities and unchanged Sentinel-1 "
            "source rasters as V3. Recomputing from the V3.1 selected-tile subset "
            "would confound the controlled V3 versus V3.1 comparison."
        ),
        "validation_and_test_used": False,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "source_normalization": str(args.source_normalization),
        "source_normalization_sha256": source_norm_sha,
        "source_v3_provenance": str(args.source_provenance),
        "source_v3_provenance_sha256": sha256_file(args.source_provenance),
        "copied_normalization": str(copied_json),
        "copied_normalization_sha256": sha256_file(copied_json),
        "copied_normalization_csv": str(copied_csv) if copied_csv else None,
        "split_tile_counts": split_counts,
        "training_city_ids": train_ids,
        "training_sar_hash_checks": source_hash_checks,
        "mandatory_checks": checks,
    }
    write_json(args.output_dir / "normalization_provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": "PASS",
                "policy": provenance["policy"],
                "training_sar_hashes_verified": len(source_hash_checks),
                "validation_and_test_used": False,
                "next": "python -m src.quality_control.freeze_dataset_v31",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
