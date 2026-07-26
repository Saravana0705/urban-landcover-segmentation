"""Rasterize five urban land-cover classes onto a Sentinel-1 grid.

Outputs
-------
1. Semantic label mask with values:
   0 = unlabeled
   1 = buildings
   2 = roads
   3 = vegetation
   4 = bare land
   5 = water
   255 = file-level NoData for invalid SAR pixels

2. Binary validity mask:
   0 = ignore
   1 = valid labelled pixel

Class priority, from lowest to highest:
bare_land -> vegetation -> water -> roads -> buildings
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize
from shapely.geometry import box


@dataclass(frozen=True)
class ClassInput:
    """One thematic class used during rasterization."""

    name: str
    path: Path
    layer: str | None
    class_id: int
    priority: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML structure: {path}")

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required rasterization configuration fields."""
    required_top_level = {
        "all_touched",
        "dtype",
        "semantic_nodata",
        "validity_nodata",
        "ignore_index",
        "class_priority",
        "class_ids",
        "output",
    }

    missing = required_top_level.difference(config)
    if missing:
        raise ValueError(
            f"Missing rasterization configuration keys: {sorted(missing)}"
        )

    required_classes = {
        "unlabeled",
        "buildings",
        "roads",
        "vegetation",
        "bare_land",
        "water",
    }

    missing_ids = required_classes.difference(config["class_ids"])
    missing_priorities = {
        "buildings",
        "roads",
        "vegetation",
        "bare_land",
        "water",
    }.difference(config["class_priority"])

    if missing_ids:
        raise ValueError(f"Missing class IDs: {sorted(missing_ids)}")
    if missing_priorities:
        raise ValueError(
            f"Missing class priorities: {sorted(missing_priorities)}"
        )

    dtype = str(config["dtype"]).lower()
    if dtype != "uint8":
        raise ValueError("This implementation currently expects dtype: uint8.")

    semantic_nodata = int(config["semantic_nodata"])
    validity_nodata = int(config["validity_nodata"])
    class_values = {int(value) for value in config["class_ids"].values()}

    if semantic_nodata in class_values:
        raise ValueError(
            "semantic_nodata must not duplicate a semantic class ID."
        )
    if not 0 <= semantic_nodata <= 255:
        raise ValueError("semantic_nodata must fit inside uint8.")
    if not 0 <= validity_nodata <= 255:
        raise ValueError("validity_nodata must fit inside uint8.")
    if int(config["class_ids"]["unlabeled"]) != 0:
        raise ValueError("The unlabeled class must use value 0.")

    output_config = config["output"]
    required_output_keys = {
        "compress",
        "tiled",
        "block_size",
        "predictor",
        "bigtiff",
    }
    missing_output = required_output_keys.difference(output_config)
    if missing_output:
        raise ValueError(
            f"Missing output configuration keys: {sorted(missing_output)}"
        )


def read_class_layer(
    class_input: ClassInput,
    reference_crs: rasterio.crs.CRS,
    reference_bounds: rasterio.coords.BoundingBox,
) -> gpd.GeoDataFrame:
    """Read, validate, reproject if needed, and clip one vector class."""
    if not class_input.path.exists():
        raise FileNotFoundError(
            f"{class_input.name} file not found: {class_input.path}"
        )

    print(f"Reading {class_input.name}: {class_input.path}")

    if class_input.layer:
        gdf = gpd.read_file(class_input.path, layer=class_input.layer)
    else:
        gdf = gpd.read_file(class_input.path)

    if gdf.empty:
        raise ValueError(f"{class_input.name} contains no features.")
    if gdf.crs is None:
        raise ValueError(f"{class_input.name} has no CRS.")

    if gdf.crs != reference_crs:
        print(
            f"  Reprojecting {class_input.name}: "
            f"{gdf.crs} -> {reference_crs}"
        )
        gdf = gdf.to_crs(reference_crs)

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    invalid_count = int((~gdf.geometry.is_valid).sum())
    if invalid_count > 0:
        print(f"  Repairing {invalid_count:,} invalid geometries.")
        gdf["geometry"] = gdf.geometry.make_valid()

    gdf = gdf.loc[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.is_valid
    ].copy()

    if gdf.empty:
        raise ValueError(
            f"No valid geometries remain for {class_input.name}."
        )

    raster_extent = box(
        reference_bounds.left,
        reference_bounds.bottom,
        reference_bounds.right,
        reference_bounds.top,
    )

    gdf = gdf.loc[gdf.geometry.intersects(raster_extent)].copy()
    if gdf.empty:
        raise ValueError(
            f"{class_input.name} does not intersect the reference raster."
        )

    gdf["geometry"] = gdf.geometry.intersection(raster_extent)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    print(f"  Features intersecting raster: {len(gdf):,}")
    return gdf


def build_class_inputs(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[ClassInput]:
    """Build class inputs from command-line arguments and YAML."""
    ids = config["class_ids"]
    priorities = config["class_priority"]

    inputs = [
        ClassInput(
            name="buildings",
            path=args.buildings,
            layer=args.buildings_layer,
            class_id=int(ids["buildings"]),
            priority=int(priorities["buildings"]),
        ),
        ClassInput(
            name="roads",
            path=args.roads,
            layer=args.roads_layer,
            class_id=int(ids["roads"]),
            priority=int(priorities["roads"]),
        ),
        ClassInput(
            name="vegetation",
            path=args.vegetation,
            layer=args.vegetation_layer,
            class_id=int(ids["vegetation"]),
            priority=int(priorities["vegetation"]),
        ),
        ClassInput(
            name="bare_land",
            path=args.bare_land,
            layer=args.bare_land_layer,
            class_id=int(ids["bare_land"]),
            priority=int(priorities["bare_land"]),
        ),
        ClassInput(
            name="water",
            path=args.water,
            layer=args.water_layer,
            class_id=int(ids["water"]),
            priority=int(priorities["water"]),
        ),
    ]

    return sorted(inputs, key=lambda item: item.priority)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize five vector classes onto a Sentinel-1 grid."
        )
    )

    parser.add_argument("--city-id", required=True)
    parser.add_argument("--reference-raster", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)

    parser.add_argument("--buildings", required=True, type=Path)
    parser.add_argument("--roads", required=True, type=Path)
    parser.add_argument("--vegetation", required=True, type=Path)
    parser.add_argument("--bare-land", required=True, type=Path)
    parser.add_argument("--water", required=True, type=Path)

    parser.add_argument("--buildings-layer")
    parser.add_argument("--roads-layer")
    parser.add_argument("--vegetation-layer")
    parser.add_argument("--bare-land-layer")
    parser.add_argument("--water-layer")

    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("metadata/rasterization"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete label-rasterization workflow."""
    args = parse_arguments()
    config = load_yaml(args.config)
    validate_config(config)

    if not args.reference_raster.exists():
        raise FileNotFoundError(
            f"Reference raster not found: {args.reference_raster}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    all_touched = bool(config["all_touched"])
    dtype = str(config["dtype"]).lower()
    unlabeled_id = int(config["class_ids"]["unlabeled"])
    semantic_nodata = int(config["semantic_nodata"])
    validity_nodata = int(config["validity_nodata"])

    class_inputs = build_class_inputs(args, config)

    print("\nFive-class label rasterization")
    print("------------------------------")
    print(f"City: {args.city_id}")
    print(f"Reference: {args.reference_raster}")
    print(f"All touched: {all_touched}")
    print(f"Semantic NoData: {semantic_nodata}")
    print(f"Validity NoData: {validity_nodata}")

    print("\nBurn order, low to high priority")
    for item in class_inputs:
        print(
            f"  {item.priority}: "
            f"{item.name} -> class {item.class_id}"
        )

    with rasterio.open(args.reference_raster) as reference:
        reference_crs = reference.crs
        if reference_crs is None:
            raise ValueError("Reference raster has no CRS.")

        height = int(reference.height)
        width = int(reference.width)
        transform = reference.transform
        bounds = reference.bounds

        semantic_mask = np.full(
            (height, width),
            fill_value=unlabeled_id,
            dtype=np.uint8,
        )

        class_reports: dict[str, dict[str, Any]] = {}

        for class_input in class_inputs:
            gdf = read_class_layer(
                class_input=class_input,
                reference_crs=reference_crs,
                reference_bounds=bounds,
            )

            shapes = (
                (geometry, class_input.class_id)
                for geometry in gdf.geometry
            )

            class_raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=all_touched,
                dtype=dtype,
            )

            class_pixel_count_before_priority = int(
                np.count_nonzero(
                    class_raster == class_input.class_id
                )
            )

            overwrite_mask = class_raster == class_input.class_id
            overwritten_pixel_count = int(
                np.count_nonzero(
                    overwrite_mask
                    & (semantic_mask != unlabeled_id)
                    & (semantic_mask != class_input.class_id)
                )
            )

            semantic_mask[overwrite_mask] = class_input.class_id

            pixel_area_m2 = float(abs(transform.a * transform.e))
            class_reports[class_input.name] = {
                "source_file": str(class_input.path),
                "source_layer": class_input.layer,
                "class_id": int(class_input.class_id),
                "priority": int(class_input.priority),
                "vector_feature_count_in_extent": int(len(gdf)),
                "raw_rasterized_pixel_count": (
                    class_pixel_count_before_priority
                ),
                "pixels_overwriting_lower_priority_classes": (
                    overwritten_pixel_count
                ),
                "raw_rasterized_area_m2": float(
                    class_pixel_count_before_priority * pixel_area_m2
                ),
            }

        sar_valid_mask = reference.dataset_mask() > 0
        reference_profile = reference.profile.copy()

    semantic_mask[~sar_valid_mask] = semantic_nodata

    labelled_mask = (
        (semantic_mask != unlabeled_id)
        & (semantic_mask != semantic_nodata)
        & sar_valid_mask
    )
    validity_mask = labelled_mask.astype(np.uint8)

    total_pixels = int(width * height)
    labelled_pixels = int(np.count_nonzero(labelled_mask))
    unlabelled_pixels = int(
        np.count_nonzero(semantic_mask == unlabeled_id)
    )
    semantic_nodata_pixels = int(
        np.count_nonzero(semantic_mask == semantic_nodata)
    )
    valid_training_pixels = int(
        np.count_nonzero(validity_mask == 1)
    )

    final_values, final_counts_array = np.unique(
        semantic_mask,
        return_counts=True,
    )
    final_counts = {
        int(value): int(count)
        for value, count in zip(
            final_values,
            final_counts_array,
            strict=True,
        )
    }

    output_config = config["output"]
    block_size = int(output_config["block_size"])

    semantic_path = args.output_dir / f"{args.city_id}_label_mask.tif"
    validity_path = args.output_dir / f"{args.city_id}_validity_mask.tif"

    base_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": reference_crs,
        "transform": transform,
        "dtype": dtype,
        "compress": str(output_config["compress"]),
        "tiled": bool(output_config["tiled"]),
        "blockxsize": block_size,
        "blockysize": block_size,
        "predictor": int(output_config["predictor"]),
        "BIGTIFF": str(output_config["bigtiff"]).upper(),
    }

    semantic_profile = base_profile.copy()
    semantic_profile["nodata"] = semantic_nodata

    with rasterio.open(semantic_path, "w", **semantic_profile) as output:
        output.write(semantic_mask, 1)
        output.set_band_description(1, "five_class_semantic_label")
        output.update_tags(
            city_id=args.city_id,
            class_0="unlabeled",
            class_1="buildings",
            class_2="roads",
            class_3="vegetation",
            class_4="bare_land",
            class_5="water",
            semantic_nodata=str(semantic_nodata),
            all_touched=str(all_touched).lower(),
            reference_raster=str(args.reference_raster),
        )

    validity_profile = base_profile.copy()
    validity_profile["nodata"] = validity_nodata

    with rasterio.open(validity_path, "w", **validity_profile) as output:
        output.write(validity_mask, 1)
        output.set_band_description(1, "training_validity_mask")
        output.update_tags(
            city_id=args.city_id,
            value_0="ignore",
            value_1="valid_label",
            validity_nodata=str(validity_nodata),
            reference_raster=str(args.reference_raster),
        )

    pixel_area_m2 = float(abs(transform.a * transform.e))
    class_id_to_name = {
        int(value): key
        for key, value in config["class_ids"].items()
    }

    final_class_statistics: dict[str, dict[str, Any]] = {}
    for class_id in sorted(class_id_to_name):
        class_name = class_id_to_name[class_id]
        pixel_count = int(final_counts.get(class_id, 0))
        final_class_statistics[class_name] = {
            "class_id": class_id,
            "pixel_count": pixel_count,
            "area_m2": float(pixel_count * pixel_area_m2),
            "area_km2": float(
                pixel_count * pixel_area_m2 / 1_000_000.0
            ),
            "percentage_of_total_grid": float(
                pixel_count / total_pixels * 100.0
            ),
        }

    report = {
        "city_id": args.city_id,
        "reference_raster": str(args.reference_raster),
        "reference_raster_sha256": sha256_file(args.reference_raster),
        "semantic_mask": str(semantic_path),
        "semantic_mask_sha256": sha256_file(semantic_path),
        "validity_mask": str(validity_path),
        "validity_mask_sha256": sha256_file(validity_path),
        "crs": str(reference_crs),
        "width_pixels": width,
        "height_pixels": height,
        "transform": list(transform)[:6],
        "pixel_area_m2": pixel_area_m2,
        "all_touched": all_touched,
        "semantic_nodata": semantic_nodata,
        "validity_nodata": validity_nodata,
        "ignore_index": int(config["ignore_index"]),
        "burn_order_low_to_high": [item.name for item in class_inputs],
        "total_pixels": total_pixels,
        "labelled_pixels": labelled_pixels,
        "unlabelled_pixels": unlabelled_pixels,
        "semantic_nodata_pixels": semantic_nodata_pixels,
        "valid_training_pixels": valid_training_pixels,
        "labelled_percentage": float(
            labelled_pixels / total_pixels * 100.0
        ),
        "unlabelled_percentage": float(
            unlabelled_pixels / total_pixels * 100.0
        ),
        "semantic_nodata_percentage": float(
            semantic_nodata_pixels / total_pixels * 100.0
        ),
        "reference_profile": {
            "driver": reference_profile.get("driver"),
            "dtype": reference_profile.get("dtype"),
            "count": reference_profile.get("count"),
            "nodata": reference_profile.get("nodata"),
        },
        "class_rasterization": class_reports,
        "final_class_statistics": final_class_statistics,
        "allowed_semantic_values": sorted(
            int(value) for value in config["class_ids"].values()
        ),
        "allowed_validity_values": [0, 1],
    }

    report_path = (
        args.report_dir / f"{args.city_id}_rasterization_report.json"
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nRasterization completed")
    print("-----------------------")
    print(f"Semantic mask: {semantic_path}")
    print(f"Validity mask: {validity_path}")
    print(f"Report: {report_path}")
    print(
        f"Labelled pixels: {labelled_pixels:,} "
        f"({report['labelled_percentage']:.4f}%)"
    )
    print(
        f"Unlabelled pixels: {unlabelled_pixels:,} "
        f"({report['unlabelled_percentage']:.4f}%)"
    )
    print(
        f"Semantic NoData pixels: {semantic_nodata_pixels:,} "
        f"({report['semantic_nodata_percentage']:.4f}%)"
    )

    print("\nFinal class distribution")
    for class_name, statistics in final_class_statistics.items():
        print(
            f"  {class_name:<12} "
            f"id={statistics['class_id']} "
            f"pixels={statistics['pixel_count']:,} "
            f"area={statistics['area_km2']:.4f} km² "
            f"grid={statistics['percentage_of_total_grid']:.4f}%"
        )

    print("\nResult: masks generated on the exact Sentinel-1 grid.")


if __name__ == "__main__":
    main()
