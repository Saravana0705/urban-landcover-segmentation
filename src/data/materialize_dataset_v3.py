"""Materialize only the selected Dataset V3 candidate tiles.

The script reads full-city V3 rasters and writes a new Dataset V3 tile tree.
It does not write to Dataset V1, V2, V2.1, or V2.2 paths. Use --resume after
an interrupted run; existing complete tile groups are recognized and skipped,
then independently checked by the required validation stage.
"""

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
import yaml
from rasterio.windows import Window, transform as window_transform


ROLES = {
    "image": ("source_sar_path", "images", "image", 0.0),
    "semantic": ("source_semantic_path", "semantic_masks", "semantic", 255),
    "validity": ("source_validity_path", "validity_masks", "validity", 0),
    "provenance": ("source_provenance_path", "provenance_masks", "provenance", 0),
    "support": ("source_support_path", "support_masks", "support", 0),
    "conflict": ("source_conflict_path", "conflict_masks", "conflict", 0),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: Any) -> Path:
    text = str(value).replace("\\", os.sep).replace("/", os.sep)
    return Path(text)


def require_within(path: Path, permitted_prefix: Path, label: str) -> None:
    if not path.resolve().is_relative_to(permitted_prefix.resolve()):
        raise ValueError(f"{label} must remain under {permitted_prefix}: {path}")


def same_grid(reference: Any, other: Any) -> bool:
    return (
        reference.width,
        reference.height,
        reference.crs,
        reference.transform,
    ) == (other.width, other.height, other.crs, other.transform)


def output_paths(root: Path, row: pd.Series) -> dict[str, Path]:
    city_folder = f"{row['city_id']}_{row['city_name']}"
    city_root = root / str(row["split"]) / city_folder
    tile_id = str(row["tile_id"])
    return {
        role: city_root / folder / f"{tile_id}_{suffix}.tif"
        for role, (_, folder, suffix, _) in ROLES.items()
    }


def make_profile(source: Any, role: str, transform: Any, config: dict[str, Any]) -> dict[str, Any]:
    tile_size = int(config["encoding"]["tile_size"])
    gtiff = config["geotiff"]
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        width=tile_size,
        height=tile_size,
        transform=transform,
        compress=str(gtiff["compress"]),
        tiled=bool(gtiff["tiled"]),
        blockxsize=int(gtiff["block_size"]),
        blockysize=int(gtiff["block_size"]),
        BIGTIFF=str(gtiff["bigtiff"]),
    )
    if role == "image":
        profile.update(count=2, dtype="float32", nodata=None, predictor=int(gtiff["predictor_sar"]))
    else:
        nodata = 255 if role == "semantic" else 0
        profile.update(count=1, dtype="uint8", nodata=nodata, predictor=int(gtiff["predictor_mask"]))
    return profile


def atomic_write(path: Path, array: np.ndarray, profile: dict[str, Any], tags: dict[str, str], descriptions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.tif")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output found: {temporary}")
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


def validate_selection(frame: pd.DataFrame, tile_size: int) -> None:
    required = {
        "tile_id", "city_id", "city_name", "split", "row_offset", "col_offset",
        "tile_size", "selected", *[spec[0] for spec in ROLES.values()],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Selected manifest is missing columns: {missing}")
    if frame.empty or frame["tile_id"].duplicated().any():
        raise ValueError("Selected manifest is empty or contains duplicate tile IDs")
    selected_text = frame["selected"].astype(str).str.lower()
    if not selected_text.isin({"true", "1"}).all():
        raise ValueError("Input contains rows that are not selected")
    if not (frame["tile_size"].astype(int) == tile_size).all():
        raise ValueError(f"Every selected row must use tile_size={tile_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_materialization.yaml"))
    parser.add_argument("--selected-manifest", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Development smoke-test only; do not use for final V3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    selected_path = args.selected_manifest or Path(config["input"]["selected_manifest"])
    tile_root = Path(config["output"]["tile_root"])
    metadata_dir = Path(config["output"]["materialization_metadata"])
    tile_size = int(config["encoding"]["tile_size"])
    require_within(
        tile_root,
        Path(config["safety"]["permitted_data_write_prefix"]),
        "Dataset V3 tile root",
    )
    require_within(
        metadata_dir,
        Path(config["safety"]["permitted_metadata_write_prefix"]),
        "Dataset V3 materialization metadata",
    )

    frame = pd.read_csv(selected_path)
    validate_selection(frame, tile_size)
    frame = frame.sort_values(["split", "city_id", "tile_id"]).reset_index(drop=True)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        frame = frame.head(args.limit).copy()

    if tile_root.exists() and not args.resume:
        raise FileExistsError(f"Dataset V3 tile root already exists. Use --resume after checking it: {tile_root}")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = metadata_dir / "dataset_v3_materialized_manifest.csv"
    report_path = metadata_dir / "dataset_v3_materialization_audit.json"
    if (manifest_path.exists() or report_path.exists()) and not args.resume:
        raise FileExistsError("Dataset V3 materialization metadata already exists; use --resume after review")

    rows: list[dict[str, Any]] = []
    written = 0
    skipped = 0
    source_hashes: dict[str, str] = {}

    for city_id, city_rows in frame.groupby("city_id", sort=True):
        first = city_rows.iloc[0]
        source_paths = {role: project_path(first[column]) for role, (column, _, _, _) in ROLES.items()}
        for role, path in source_paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{city_id} {role} source missing or empty: {path}")
            source_hashes[str(path)] = sha256(path)

        with ExitStack() as stack:
            sources = {role: stack.enter_context(rasterio.open(path)) for role, path in source_paths.items()}
            reference = sources["image"]
            if reference.count != 2:
                raise ValueError(f"{city_id} SAR source must have two bands")
            for role, dataset in sources.items():
                if role != "image" and dataset.count != 1:
                    raise ValueError(f"{city_id} {role} source must have one band")
                if not same_grid(reference, dataset):
                    raise ValueError(f"{city_id} {role} source is not aligned with SAR")

            for _, row in city_rows.iterrows():
                paths = output_paths(tile_root, row)
                existing = [path.exists() for path in paths.values()]
                if any(existing):
                    if not args.resume or not all(existing):
                        raise FileExistsError(f"Partial or unexpected existing output group for {row['tile_id']}")
                    skipped += 1
                else:
                    window = Window(int(row["col_offset"]), int(row["row_offset"]), tile_size, tile_size)
                    transform = window_transform(window, reference.transform)
                    tags = {
                        "dataset_version": "v3",
                        "city_id": str(row["city_id"]),
                        "city_name": str(row["city_name"]),
                        "split": str(row["split"]),
                        "tile_id": str(row["tile_id"]),
                    }
                    for role, dataset in sources.items():
                        fill = ROLES[role][3]
                        if role == "image":
                            array = dataset.read(
                                window=window, out_shape=(2, tile_size, tile_size),
                                boundless=True, fill_value=fill,
                            ).astype(np.float32, copy=False)
                            descriptions = ["Sigma0_VV", "Sigma0_VH"]
                        else:
                            array = dataset.read(
                                1, window=window, out_shape=(tile_size, tile_size),
                                boundless=True, fill_value=fill,
                            ).astype(np.uint8, copy=False)
                            descriptions = [{
                                "semantic": "dataset_v3_semantic_ignore_255",
                                "validity": "dataset_v3_training_validity",
                                "provenance": "source_bits_osm1_dynamic_world2_urban_atlas4",
                                "support": "agreeing_source_count_for_assigned_class",
                                "conflict": "nonzero_source_class_disagreement",
                            }[role]]
                        profile = make_profile(dataset, role, transform, config)
                        atomic_write(paths[role], array, profile, tags, descriptions)
                    written += 1

                output_row = row.to_dict()
                output_row.update({
                    "image_path": str(paths["image"]),
                    "semantic_mask_path": str(paths["semantic"]),
                    "validity_mask_path": str(paths["validity"]),
                    "provenance_mask_path": str(paths["provenance"]),
                    "support_mask_path": str(paths["support"]),
                    "conflict_mask_path": str(paths["conflict"]),
                    "materialization_status": "PASS",
                })
                rows.append(output_row)

        print(f"{city_id}: {len(city_rows)} selected tiles complete")

    materialized = pd.DataFrame(rows)
    temporary_manifest = manifest_path.with_suffix(".partial.csv")
    materialized.to_csv(temporary_manifest, index=False)
    os.replace(temporary_manifest, manifest_path)

    report = {
        "schema_version": "dataset-v3-materialization-audit-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_version": "v3",
        "selected_manifest": str(selected_path),
        "selected_manifest_sha256": sha256(selected_path),
        "tile_count": int(len(materialized)),
        "written_tile_count_this_run": written,
        "resumed_tile_count": skipped,
        "raster_file_count": int(len(materialized) * len(ROLES)),
        "selected_by_split": {k: int(v) for k, v in materialized.groupby("split").size().items()},
        "source_sha256": source_hashes,
        "materialized_manifest": str(manifest_path),
        "materialized_manifest_sha256": sha256(manifest_path),
        "write_scope": [str(tile_root), str(metadata_dir)],
        "legacy_dataset_artifacts_modified": False,
        "qa_status": "PENDING_INDEPENDENT_TILE_QA",
    }
    temporary_report = report_path.with_suffix(".partial.json")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(json.dumps({"status": "PASS", "tile_count": len(materialized), "written": written, "resumed": skipped, "next": "python -m src.quality_control.validate_dataset_v3"}, indent=2))


if __name__ == "__main__":
    main()
