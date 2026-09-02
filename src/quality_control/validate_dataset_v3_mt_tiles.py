"""Independently validate all materialized Dataset V3-MT tile triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


EXPECTED_BANDS = (
    "VV_Q1", "VH_Q1", "VV_Q2", "VH_Q2",
    "VV_Q3", "VH_Q3", "VV_Q4", "VH_Q4",
)
EXPECTED_SPLITS = {"train": 974, "val": 351, "test": 339}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("metadata/dataset_v3_mt/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--materialization-audit", type=Path,
        default=Path(
            "metadata/dataset_v3_mt/materialization/"
            "dataset_v3_mt_materialization_audit.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("metadata/dataset_v3_mt/qa/dataset_v3_mt_tile_qa.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value: Any) -> Path:
    return Path(str(value).replace("\\", os.sep).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.materialization_audit):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty: {path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    audit = json.loads(args.materialization_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("tile_count") != 1664:
        raise RuntimeError("Materialization audit has not passed")
    if audit.get("manifest_sha256") != sha256_file(args.manifest):
        raise RuntimeError("Manifest hash differs from materialization audit")

    frame = pd.read_csv(args.manifest)
    if len(frame) != 1664 or frame["tile_id"].duplicated().any():
        raise RuntimeError("Manifest must contain 1,664 unique tiles")
    counts = {str(k): int(v) for k, v in frame.groupby("split").size().items()}
    if counts != EXPECTED_SPLITS:
        raise RuntimeError(f"Split counts mismatch: {counts}")

    failures: list[dict[str, Any]] = []
    city_rows = []
    for position, row in frame.iterrows():
        tile_failures = []
        paths = {
            role: project_path(row[column])
            for role, column in {
                "image": "image_path", "semantic": "semantic_mask_path",
                "validity": "validity_mask_path",
            }.items()
        }
        if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
            tile_failures.append("missing or empty tile file")
        else:
            with rasterio.open(paths["image"]) as image_ds, \
                 rasterio.open(paths["semantic"]) as semantic_ds, \
                 rasterio.open(paths["validity"]) as validity_ds:
                image = image_ds.read()
                semantic = semantic_ds.read(1)
                validity = validity_ds.read(1)
                if image.shape != (8, 256, 256):
                    tile_failures.append(f"image shape {image.shape}")
                if semantic.shape != (256, 256) or validity.shape != (256, 256):
                    tile_failures.append("mask shape mismatch")
                if tuple(image_ds.descriptions) != EXPECTED_BANDS:
                    tile_failures.append("band description/order mismatch")
                if image_ds.dtypes != ("float32",) * 8:
                    tile_failures.append(f"image dtypes {image_ds.dtypes}")
                if image_ds.nodata != -9999.0:
                    tile_failures.append(f"image nodata {image_ds.nodata}")
                if not (
                    image_ds.transform == semantic_ds.transform == validity_ds.transform
                    and image_ds.crs == semantic_ds.crs == validity_ds.crs
                ):
                    tile_failures.append("triplet grid mismatch")
                if not set(np.unique(validity)).issubset({0, 1}):
                    tile_failures.append("invalid validity values")
                if not set(np.unique(semantic)).issubset({1, 2, 3, 4, 5, 255}):
                    tile_failures.append("invalid semantic values")
                valid = validity == 1
                if np.any(valid & ~np.all(np.isfinite(image) & (image > 0), axis=0)):
                    tile_failures.append("valid pixels contain invalid temporal values")
                if np.any(valid & ~np.isin(semantic, [1, 2, 3, 4, 5])):
                    tile_failures.append("valid pixels contain invalid semantic values")
                if np.any((~valid) & (semantic != 255)):
                    tile_failures.append("invalid pixels are not semantic Ignore")
                valid_count = int(valid.sum())
                if valid_count != int(row["valid_pixel_count"]):
                    tile_failures.append("valid count mismatch")
                for class_id, name in enumerate(
                    ("buildings", "roads", "vegetation", "bare_land", "water"),
                    start=1,
                ):
                    actual = int(((semantic == class_id) & valid).sum())
                    if actual != int(row[f"{name}_pixel_count"]):
                        tile_failures.append(f"{name} count mismatch")
                for role, hash_column in {
                    "image": "image_sha256", "semantic": "semantic_sha256",
                    "validity": "validity_sha256",
                }.items():
                    if sha256_file(paths[role]) != str(row[hash_column]):
                        tile_failures.append(f"{role} hash mismatch")
        if tile_failures:
            failures.append({"tile_id": row["tile_id"], "failures": tile_failures})
        city_rows.append({
            "city_id": row["city_id"], "split": row["split"],
            "valid_pixel_count": int(row["valid_pixel_count"]),
            "parent_valid_pixel_count": int(row["parent_v3_valid_pixel_count"]),
            "tile_pass": not tile_failures,
        })
        if (position + 1) % 100 == 0:
            print(f"Validated {position + 1}/1664 tiles")

    city_frame = pd.DataFrame(city_rows)
    city_summary = city_frame.groupby(["city_id", "split"], as_index=False).agg(
        tile_count=("tile_pass", "count"),
        passed_tile_count=("tile_pass", "sum"),
        valid_pixel_count=("valid_pixel_count", "sum"),
        parent_valid_pixel_count=("parent_valid_pixel_count", "sum"),
    )
    city_summary["retained_fraction_parent_valid"] = (
        city_summary["valid_pixel_count"] / city_summary["parent_valid_pixel_count"]
    )
    status = "PASS" if not failures else "FAIL"
    report = {
        "schema_version": "dataset-v3-mt-tile-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "tile_count": len(frame),
        "passed_tile_count": len(frame) - len(failures),
        "failed_tile_count": len(failures),
        "split_tile_counts": counts,
        "manifest_sha256": sha256_file(args.manifest),
        "failures": failures,
        "labels_reused_from": "v3",
        "validation_policy": "parent validity intersect all-eight-band validity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_path = args.output.parent / "dataset_v3_mt_city_qa_summary.csv"
    city_summary.to_csv(summary_path, index=False, lineterminator="\n")
    print(json.dumps({
        "status": status, "passed": len(frame) - len(failures),
        "failed": len(failures), "report": str(args.output),
        "city_summary": str(summary_path),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
