"""Materialize eight-band Dataset V3-MT tiles from the frozen V3 tile set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, transform as window_transform


EXPECTED_PARENT_MANIFEST_SHA256 = (
    "598863050580021efdf7ebc7bf094cc9e330942471ededc636ccd3c9423d247f"
)
EXPECTED_BANDS = (
    "VV_Q1", "VH_Q1", "VV_Q2", "VH_Q2",
    "VV_Q3", "VH_Q3", "VV_Q4", "VH_Q4",
)
EXPECTED_SPLITS = {"train": 974, "val": 351, "test": 339}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-manifest", type=Path,
        default=Path("metadata/dataset_v3/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("data/raw/sar_multitemporal_v3_mt"),
    )
    parser.add_argument(
        "--source-qa", type=Path,
        default=Path(
            "metadata/dataset_v3_mt/source_qa/dataset_v3_mt_source_qa.json"
        ),
    )
    parser.add_argument(
        "--normalization", type=Path,
        default=Path(
            "metadata/dataset_v3_mt/normalization/training_normalization.json"
        ),
    )
    parser.add_argument(
        "--normalization-provenance", type=Path,
        default=Path(
            "metadata/dataset_v3_mt/normalization/normalization_provenance.json"
        ),
    )
    parser.add_argument(
        "--tile-root", type=Path,
        default=Path("data/processed/dataset_v3_mt/tiles"),
    )
    parser.add_argument(
        "--output-manifest", type=Path,
        default=Path("metadata/dataset_v3_mt/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path(
            "metadata/dataset_v3_mt/materialization/"
            "dataset_v3_mt_materialization_audit.json"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def project_path(value: Any) -> Path:
    return Path(str(value).replace("\\", os.sep).replace("/", os.sep))


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_write_scope(path: Path, permitted: Path, label: str) -> None:
    if not path.resolve().is_relative_to(permitted.resolve()):
        raise ValueError(f"{label} must remain under {permitted}: {path}")


def output_paths(root: Path, row: pd.Series) -> dict[str, Path]:
    city = f"{row['city_id']}_{row['city_name']}"
    base = root / str(row["split"]) / city
    tile_id = str(row["tile_id"])
    return {
        "image": base / "images" / f"{tile_id}_image.tif",
        "semantic": base / "semantic_masks" / f"{tile_id}_semantic.tif",
        "validity": base / "validity_masks" / f"{tile_id}_validity.tif",
    }


def atomic_write(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    descriptions: tuple[str, ...],
    tags: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.tif")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output: {temporary}")
    try:
        with rasterio.open(temporary, "w", **profile) as output:
            if array.ndim == 3:
                output.write(array)
            else:
                output.write(array, 1)
            for index, description in enumerate(descriptions, start=1):
                output.set_band_description(index, description)
            output.update_tags(**tags)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def image_profile(source: Any, transform: Any, tile_size: int) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff", width=tile_size, height=tile_size, count=8,
        transform=transform, dtype="float32", nodata=-9999.0,
        compress="DEFLATE", predictor=3, tiled=True,
        blockxsize=tile_size, blockysize=tile_size, BIGTIFF="IF_SAFER",
    )
    return profile


def mask_profile(source: Any, transform: Any, tile_size: int, semantic: bool) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff", width=tile_size, height=tile_size, count=1,
        transform=transform, dtype="uint8", nodata=255 if semantic else 0,
        compress="DEFLATE", predictor=2, tiled=True,
        blockxsize=tile_size, blockysize=tile_size, BIGTIFF="IF_SAFER",
    )
    return profile


def validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    for path, label in (
        (args.parent_manifest, "frozen V3 manifest"),
        (args.source_qa, "V3-MT source QA"),
        (args.normalization, "V3-MT normalization"),
        (args.normalization_provenance, "normalization provenance"),
    ):
        require_file(path, label)
    actual_hash = sha256_file(args.parent_manifest)
    if actual_hash != EXPECTED_PARENT_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen V3 manifest hash mismatch: {actual_hash}"
        )
    source_qa = json.loads(args.source_qa.read_text(encoding="utf-8"))
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    provenance = json.loads(args.normalization_provenance.read_text(encoding="utf-8"))
    if source_qa.get("status") != "PASS" or source_qa.get("city_count") != 20:
        raise RuntimeError("20-city source QA has not passed")
    if float(source_qa.get("minimum_valid_fraction", 0)) < 0.98:
        raise RuntimeError("Source-QA coverage gate is below 0.98")
    if normalization.get("status") != "PASS":
        raise RuntimeError("Normalization has not passed")
    if normalization.get("validation_and_test_used") is not False:
        raise RuntimeError("Normalization leakage control failed")
    if tuple(normalization.get("band_order", [])) != EXPECTED_BANDS:
        raise RuntimeError("Unexpected normalization band order")
    for band_name in EXPECTED_BANDS:
        model_input = normalization["bands"][band_name]["model_input"]
        values = np.asarray([
            model_input["clip_lower_db"], model_input["clip_upper_db"],
            model_input["mean_db"], model_input["std_db"],
        ], dtype=np.float64)
        if not np.isfinite(values).all() or values[3] <= 0 or values[0] >= values[1]:
            raise RuntimeError(f"Invalid normalization constants: {band_name}")
    if provenance.get("training_city_ids") != [f"DE{i:02d}" for i in range(1, 15)]:
        raise RuntimeError("Normalization provenance training cities mismatch")
    qa_hashes = {
        row["city_id"]: row["raster"]["sha256"]
        for row in source_qa["cities"]
    }
    provenance_hashes = {
        row["city_id"]: row["sha256"]
        for row in provenance.get("source_rasters", [])
    }
    if provenance_hashes != {
        city_id: qa_hashes[city_id] for city_id in [f"DE{i:02d}" for i in range(1, 15)]
    }:
        raise RuntimeError("Normalization source hashes differ from accepted source QA")

    frame = pd.read_csv(args.parent_manifest)
    required = {
        "tile_id", "city_id", "city_name", "split", "row_offset",
        "col_offset", "tile_size", "image_path", "semantic_mask_path",
        "validity_mask_path", "valid_pixel_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Parent manifest missing columns: {sorted(missing)}")
    if len(frame) != 1664 or frame["tile_id"].duplicated().any():
        raise RuntimeError("Frozen V3 manifest must contain 1,664 unique tiles")
    counts = {str(k): int(v) for k, v in frame.groupby("split").size().items()}
    if counts != EXPECTED_SPLITS:
        raise RuntimeError(f"Frozen split counts mismatch: {counts}")
    return frame, source_qa


def existing_row(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(paths["semantic"]) as semantic_ds:
        semantic = semantic_ds.read(1)
    with rasterio.open(paths["validity"]) as validity_ds:
        validity = validity_ds.read(1)
    return semantic, validity


def main() -> None:
    args = parse_args()
    require_write_scope(args.tile_root, Path("data/processed/dataset_v3_mt"), "tile root")
    require_write_scope(args.output_manifest, Path("metadata/dataset_v3_mt"), "manifest")
    require_write_scope(args.audit, Path("metadata/dataset_v3_mt"), "audit")
    frame, source_qa = validate_inputs(args)
    frame = frame.sort_values(["split", "city_id", "tile_id"]).reset_index(drop=True)

    if args.tile_root.exists() and not args.resume:
        raise FileExistsError(f"Tile root exists; inspect and use --resume: {args.tile_root}")
    if (args.output_manifest.exists() or args.audit.exists()) and not args.resume:
        raise FileExistsError("V3-MT metadata exists; inspect and use --resume")

    source_hashes = {
        row["city_id"]: row["raster"]["sha256"]
        for row in source_qa["cities"]
    }
    output_rows: list[dict[str, Any]] = []
    written = 0
    resumed = 0

    for city_id, city_rows in frame.groupby("city_id", sort=True):
        first = city_rows.iloc[0]
        city_name = str(first["city_name"])
        source_path = args.source_root / f"{city_id}_{city_name}" / (
            f"{city_id}_{city_name}_S1_MT_Q1Q4_2025.tif"
        )
        require_file(source_path, f"{city_id} temporal source")
        if sha256_file(source_path) != source_hashes[city_id]:
            raise RuntimeError(f"{city_id} temporal source hash changed after QA")

        with rasterio.open(source_path) as temporal:
            if temporal.count != 8 or tuple(temporal.descriptions) != EXPECTED_BANDS:
                raise RuntimeError(f"{city_id} temporal band schema mismatch")
            for _, row in city_rows.iterrows():
                tile_size = int(row["tile_size"])
                if tile_size != 256:
                    raise RuntimeError(f"Unexpected tile size for {row['tile_id']}")
                paths = output_paths(args.tile_root, row)
                existence = [path.exists() for path in paths.values()]
                parent_semantic_path = project_path(row["semantic_mask_path"])
                parent_validity_path = project_path(row["validity_mask_path"])
                require_file(parent_semantic_path, "parent semantic tile")
                require_file(parent_validity_path, "parent validity tile")

                if any(existence):
                    if not args.resume or not all(existence):
                        raise FileExistsError(
                            f"Partial or unexpected group: {row['tile_id']}"
                        )
                    semantic, validity = existing_row(paths)
                    resumed += 1
                else:
                    window = Window(
                        int(row["col_offset"]), int(row["row_offset"]),
                        tile_size, tile_size,
                    )
                    transform = window_transform(window, temporal.transform)
                    image = temporal.read(
                        window=window, out_shape=(8, tile_size, tile_size),
                        boundless=True, fill_value=-9999.0,
                    ).astype(np.float32, copy=False)
                    with ExitStack() as stack:
                        semantic_ds = stack.enter_context(rasterio.open(parent_semantic_path))
                        validity_ds = stack.enter_context(rasterio.open(parent_validity_path))
                        if semantic_ds.transform != transform or validity_ds.transform != transform:
                            raise RuntimeError(f"Grid mismatch: {row['tile_id']}")
                        if semantic_ds.crs != temporal.crs or validity_ds.crs != temporal.crs:
                            raise RuntimeError(f"CRS mismatch: {row['tile_id']}")
                        semantic = semantic_ds.read(1).astype(np.uint8, copy=False)
                        parent_validity = validity_ds.read(1) == 1
                        if semantic.shape != (tile_size, tile_size):
                            raise RuntimeError(f"Semantic shape mismatch: {row['tile_id']}")

                        temporal_validity = np.all(
                            np.isfinite(image) & (image > 0), axis=0
                        )
                        class_validity = np.isin(semantic, [1, 2, 3, 4, 5])
                        actual_parent_valid = int(
                            (parent_validity & class_validity).sum()
                        )
                        if actual_parent_valid != int(row["valid_pixel_count"]):
                            raise RuntimeError(
                                f"Parent valid count mismatch: {row['tile_id']}"
                            )
                        validity_bool = parent_validity & temporal_validity & class_validity
                        semantic = semantic.copy()
                        semantic[~validity_bool] = 255
                        validity = validity_bool.astype(np.uint8)

                        if int(validity.sum()) == 0:
                            raise RuntimeError(f"Zero-valid output tile: {row['tile_id']}")
                        tags = {
                            "dataset_variant": "v3-mt",
                            "parent_label_dataset": "v3",
                            "tile_id": str(row["tile_id"]),
                            "city_id": str(city_id),
                            "split": str(row["split"]),
                        }
                        atomic_write(
                            paths["image"], image,
                            image_profile(temporal, transform, tile_size),
                            EXPECTED_BANDS, tags,
                        )
                        atomic_write(
                            paths["semantic"], semantic,
                            mask_profile(semantic_ds, transform, tile_size, True),
                            ("dataset_v3_semantic_temporal_ignore_255",), tags,
                        )
                        atomic_write(
                            paths["validity"], validity,
                            mask_profile(validity_ds, transform, tile_size, False),
                            ("dataset_v3_mt_training_validity",), tags,
                        )
                    written += 1

                parent_valid = int(row["valid_pixel_count"])
                joint_valid = int((validity == 1).sum())
                class_counts = {
                    name: int(((semantic == class_id) & (validity == 1)).sum())
                    for class_id, name in enumerate(
                        ("buildings", "roads", "vegetation", "bare_land", "water"),
                        start=1,
                    )
                }
                output_row = row.to_dict()
                output_row.update({
                    "parent_v3_image_path": row["image_path"],
                    "parent_v3_semantic_mask_path": row["semantic_mask_path"],
                    "parent_v3_validity_mask_path": row["validity_mask_path"],
                    "image_path": str(paths["image"]),
                    "semantic_mask_path": str(paths["semantic"]),
                    "validity_mask_path": str(paths["validity"]),
                    "parent_v3_valid_pixel_count": parent_valid,
                    "valid_pixel_count": joint_valid,
                    "temporal_removed_valid_pixel_count": parent_valid - joint_valid,
                    "temporal_retained_fraction_parent_valid": (
                        joint_valid / parent_valid if parent_valid else 0.0
                    ),
                    **{f"{name}_pixel_count": count for name, count in class_counts.items()},
                    "image_sha256": sha256_file(paths["image"]),
                    "semantic_sha256": sha256_file(paths["semantic"]),
                    "validity_sha256": sha256_file(paths["validity"]),
                    "materialization_status": "PASS",
                })
                output_rows.append(output_row)
        print(f"{city_id}: {len(city_rows)} temporal tiles complete")

    output = pd.DataFrame(output_rows)
    city_summary = output.groupby(["city_id", "city_name", "split"], as_index=False).agg(
        tile_count=("tile_id", "count"),
        parent_valid_pixel_count=("parent_v3_valid_pixel_count", "sum"),
        valid_pixel_count=("valid_pixel_count", "sum"),
        temporal_removed_valid_pixel_count=("temporal_removed_valid_pixel_count", "sum"),
    )
    city_summary["temporal_retained_fraction_parent_valid"] = (
        city_summary["valid_pixel_count"] / city_summary["parent_valid_pixel_count"]
    )
    de13 = city_summary.loc[city_summary["city_id"] == "DE13"].iloc[0]
    if float(de13["temporal_retained_fraction_parent_valid"]) < 0.95:
        raise RuntimeError(
            "DE13 retains less than 95% of parent-valid pixels after temporal intersection"
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.output_manifest.with_suffix(".partial.csv")
    output.to_csv(temporary_manifest, index=False, lineterminator="\n")
    os.replace(temporary_manifest, args.output_manifest)
    summary_path = args.audit.parent / "dataset_v3_mt_materialization_city_summary.csv"
    temporary_summary = summary_path.with_suffix(".partial.csv")
    city_summary.to_csv(temporary_summary, index=False, lineterminator="\n")
    os.replace(temporary_summary, summary_path)

    audit = {
        "schema_version": "dataset-v3-mt-materialization-audit-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_variant": "v3-mt",
        "parent_label_dataset": "v3",
        "parent_manifest": str(args.parent_manifest),
        "parent_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
        "source_qa_sha256": sha256_file(args.source_qa),
        "normalization_sha256": sha256_file(args.normalization),
        "normalization_provenance_sha256": sha256_file(args.normalization_provenance),
        "tile_count": len(output),
        "split_tile_counts": {
            str(k): int(v) for k, v in output.groupby("split").size().items()
        },
        "raster_file_count": len(output) * 3,
        "written_tile_count_this_run": written,
        "resumed_tile_count": resumed,
        "parent_valid_pixel_count": int(output["parent_v3_valid_pixel_count"].sum()),
        "valid_pixel_count": int(output["valid_pixel_count"].sum()),
        "temporal_removed_valid_pixel_count": int(
            output["temporal_removed_valid_pixel_count"].sum()
        ),
        "temporal_retained_fraction_parent_valid": float(
            output["valid_pixel_count"].sum()
            / output["parent_v3_valid_pixel_count"].sum()
        ),
        "de13_retained_fraction_parent_valid": float(
            de13["temporal_retained_fraction_parent_valid"]
        ),
        "manifest": str(args.output_manifest),
        "manifest_sha256": sha256_file(args.output_manifest),
        "city_summary": str(summary_path),
        "city_summary_sha256": sha256_file(summary_path),
        "legacy_dataset_artifacts_modified": False,
        "qa_status": "PENDING_INDEPENDENT_TILE_QA",
    }
    temporary_audit = args.audit.with_suffix(".partial.json")
    temporary_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_audit, args.audit)
    print(json.dumps({
        "status": "PASS", "tile_count": len(output), "written": written,
        "resumed": resumed,
        "temporal_retained_fraction": audit["temporal_retained_fraction_parent_valid"],
        "de13_retained_fraction": audit["de13_retained_fraction_parent_valid"],
        "next": "python -m src.quality_control.validate_dataset_v3_mt_tiles",
    }, indent=2))


if __name__ == "__main__":
    main()
