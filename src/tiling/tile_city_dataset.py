"""Tile one aligned SAR-image/semantic-mask/validity-mask triplet.

The script preserves exact spatial alignment and georeferencing for every tile.
Edge tiles are padded rather than discarded, so the complete city AOI is retained.
Padding is excluded from training through:
    semantic value 255
    validity value 0

No tiles are removed at this stage. Tile suitability and class composition are
recorded in a per-city manifest so filtering or sampling can be performed later
without destructively altering Dataset Version 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.windows import Window, bounds as window_bounds
from rasterio.windows import transform as window_transform


TRAINABLE_CLASSES = {
    1: "buildings",
    2: "roads",
    3: "vegetation",
    4: "bare_land",
    5: "water",
}


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required tiling settings."""
    required = {
        "tile_size",
        "stride",
        "edge_policy",
        "image_fill_value",
        "semantic_fill_value",
        "validity_fill_value",
        "output",
    }
    missing = required.difference(config)

    if missing:
        raise ValueError(
            f"Missing tiling configuration keys: {sorted(missing)}"
        )

    tile_size = int(config["tile_size"])
    stride = int(config["stride"])

    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive.")

    if str(config["edge_policy"]).lower() != "pad":
        raise ValueError(
            "Dataset Version 1 uses edge_policy: pad to preserve full AOI coverage."
        )

    output_required = {
        "compress",
        "tiled",
        "block_size",
        "predictor_image",
        "predictor_mask",
        "bigtiff",
    }
    missing_output = output_required.difference(config["output"])

    if missing_output:
        raise ValueError(
            f"Missing tiling output keys: {sorted(missing_output)}"
        )


def transforms_equal(
    transform_a: Affine,
    transform_b: Affine,
    tolerance: float = 1e-9,
) -> bool:
    """Compare affine transforms coefficient by coefficient."""
    return all(
        abs(float(a) - float(b)) <= tolerance
        for a, b in zip(
            tuple(transform_a)[:6],
            tuple(transform_b)[:6],
            strict=True,
        )
    )


def validate_alignment(
    image: rasterio.io.DatasetReader,
    semantic: rasterio.io.DatasetReader,
    validity: rasterio.io.DatasetReader,
) -> None:
    """Fail unless all three datasets share the exact same grid."""
    checks = {
        "crs": image.crs == semantic.crs == validity.crs,
        "width": image.width == semantic.width == validity.width,
        "height": image.height == semantic.height == validity.height,
        "transform_image_semantic": transforms_equal(
            image.transform,
            semantic.transform,
        ),
        "transform_image_validity": transforms_equal(
            image.transform,
            validity.transform,
        ),
        "semantic_band_count": semantic.count == 1,
        "validity_band_count": validity.count == 1,
        "image_band_count": image.count == 2,
    }

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise RuntimeError(
            f"Input image/masks are not tile-compatible. Failed: {failed}"
        )


def tile_starts(
    size: int,
    stride: int,
) -> list[int]:
    """Return tile start indices with padded coverage of the final edge."""
    if size <= 0:
        raise ValueError("Raster dimension must be positive.")

    return list(range(0, size, stride))


def class_counts(array: np.ndarray) -> dict[int, int]:
    """Count semantic class values in one tile."""
    values, counts = np.unique(array, return_counts=True)

    return {
        int(value): int(count)
        for value, count in zip(values, counts, strict=True)
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tile one aligned city image/mask dataset."
    )

    parser.add_argument("--city-id", required=True)
    parser.add_argument("--city-name", required=True)
    parser.add_argument("--split", required=True)

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Two-band Sentinel-1 VV/VH GeoTIFF.",
    )
    parser.add_argument(
        "--semantic-mask",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validity-mask",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/tiling.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/tiles"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("metadata/tile_manifests"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing tile files.",
    )

    return parser.parse_args()


def main() -> None:
    """Tile image, semantic mask, and validity mask."""
    args = parse_arguments()
    config = load_yaml(args.config)
    validate_config(config)

    for path, label in (
        (args.image, "image"),
        (args.semantic_mask, "semantic mask"),
        (args.validity_mask, "validity mask"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    tile_size = int(config["tile_size"])
    stride = int(config["stride"])
    image_fill = float(config["image_fill_value"])
    semantic_fill = int(config["semantic_fill_value"])
    validity_fill = int(config["validity_fill_value"])
    output_config = config["output"]

    city_folder = f"{args.city_id}_{args.city_name}"
    city_root = args.output_root / args.split / city_folder

    image_dir = city_root / "images"
    semantic_dir = city_root / "semantic_masks"
    validity_dir = city_root / "validity_masks"

    for directory in (image_dir, semantic_dir, validity_dir):
        directory.mkdir(parents=True, exist_ok=True)

    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        args.manifest_dir
        / f"{args.city_id}_tile_manifest.csv"
    )
    summary_path = (
        args.manifest_dir
        / f"{args.city_id}_tiling_summary.json"
    )

    print("\nCity tiling")
    print("-----------")
    print(f"City: {args.city_id} {args.city_name}")
    print(f"Split: {args.split}")
    print(f"Tile size: {tile_size}")
    print(f"Stride: {stride}")
    print("Edge policy: pad")
    print("Tile filtering: none")

    rows: list[dict[str, Any]] = []

    with rasterio.open(args.image) as image_ds, rasterio.open(
        args.semantic_mask
    ) as semantic_ds, rasterio.open(
        args.validity_mask
    ) as validity_ds:
        validate_alignment(image_ds, semantic_ds, validity_ds)

        row_starts = tile_starts(image_ds.height, stride)
        column_starts = tile_starts(image_ds.width, stride)

        expected_tiles = len(row_starts) * len(column_starts)

        print(
            f"Source dimensions: {image_ds.width} × {image_ds.height}"
        )
        print(
            f"Tile grid: {len(column_starts)} columns × "
            f"{len(row_starts)} rows = {expected_tiles} tiles"
        )

        image_profile = image_ds.profile.copy()
        semantic_profile = semantic_ds.profile.copy()
        validity_profile = validity_ds.profile.copy()

        for tile_row, row_off in enumerate(row_starts):
            for tile_col, col_off in enumerate(column_starts):
                tile_id = (
                    f"{args.city_id}_"
                    f"r{tile_row:03d}_c{tile_col:03d}"
                )

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=tile_size,
                    height=tile_size,
                )

                tile_transform = window_transform(
                    window,
                    image_ds.transform,
                )

                image_array = image_ds.read(
                    window=window,
                    out_shape=(
                        image_ds.count,
                        tile_size,
                        tile_size,
                    ),
                    boundless=True,
                    fill_value=image_fill,
                )

                semantic_array = semantic_ds.read(
                    1,
                    window=window,
                    out_shape=(tile_size, tile_size),
                    boundless=True,
                    fill_value=semantic_fill,
                )

                validity_array = validity_ds.read(
                    1,
                    window=window,
                    out_shape=(tile_size, tile_size),
                    boundless=True,
                    fill_value=validity_fill,
                )

                source_width = max(
                    0,
                    min(tile_size, image_ds.width - col_off),
                )
                source_height = max(
                    0,
                    min(tile_size, image_ds.height - row_off),
                )
                source_pixel_count = source_width * source_height
                tile_pixel_count = tile_size * tile_size
                padding_pixel_count = (
                    tile_pixel_count - source_pixel_count
                )

                semantic_array[
                    validity_array == validity_fill
                ] = np.where(
                    semantic_array[
                        validity_array == validity_fill
                    ] == semantic_fill,
                    semantic_fill,
                    semantic_array[
                        validity_array == validity_fill
                    ],
                )

                counts = class_counts(semantic_array)

                valid_pixel_count = int(
                    np.count_nonzero(validity_array == 1)
                )
                unlabeled_pixel_count = int(
                    np.count_nonzero(semantic_array == 0)
                )
                nodata_pixel_count = int(
                    np.count_nonzero(semantic_array == semantic_fill)
                )

                image_path = image_dir / f"{tile_id}_image.tif"
                semantic_path = (
                    semantic_dir
                    / f"{tile_id}_semantic.tif"
                )
                validity_path = (
                    validity_dir
                    / f"{tile_id}_validity.tif"
                )

                if not args.force:
                    existing = [
                        path
                        for path in (
                            image_path,
                            semantic_path,
                            validity_path,
                        )
                        if path.exists()
                    ]
                    if existing:
                        raise FileExistsError(
                            "Tile outputs already exist. "
                            "Use --force to overwrite: "
                            f"{existing[0]}"
                        )

                image_tile_profile = image_profile.copy()
                image_tile_profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=2,
                    transform=tile_transform,
                    compress=output_config["compress"],
                    tiled=bool(output_config["tiled"]),
                    blockxsize=int(output_config["block_size"]),
                    blockysize=int(output_config["block_size"]),
                    predictor=int(
                        output_config["predictor_image"]
                    ),
                    BIGTIFF=str(
                        output_config["bigtiff"]
                    ).upper(),
                )

                with rasterio.open(
                    image_path,
                    "w",
                    **image_tile_profile,
                ) as output:
                    output.write(image_array)
                    output.set_band_description(1, "Sigma0_VV")
                    output.set_band_description(2, "Sigma0_VH")
                    output.update_tags(
                        city_id=args.city_id,
                        city_name=args.city_name,
                        split=args.split,
                        tile_id=tile_id,
                        source_image=str(args.image),
                    )

                semantic_tile_profile = semantic_profile.copy()
                semantic_tile_profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=1,
                    dtype="uint8",
                    nodata=semantic_fill,
                    transform=tile_transform,
                    compress=output_config["compress"],
                    tiled=bool(output_config["tiled"]),
                    blockxsize=int(output_config["block_size"]),
                    blockysize=int(output_config["block_size"]),
                    predictor=int(
                        output_config["predictor_mask"]
                    ),
                    BIGTIFF=str(
                        output_config["bigtiff"]
                    ).upper(),
                )

                with rasterio.open(
                    semantic_path,
                    "w",
                    **semantic_tile_profile,
                ) as output:
                    output.write(semantic_array.astype(np.uint8), 1)
                    output.set_band_description(
                        1,
                        "five_class_semantic_label",
                    )
                    output.update_tags(
                        city_id=args.city_id,
                        city_name=args.city_name,
                        split=args.split,
                        tile_id=tile_id,
                        class_0="unlabeled",
                        class_1="buildings",
                        class_2="roads",
                        class_3="vegetation",
                        class_4="bare_land",
                        class_5="water",
                        semantic_nodata=str(semantic_fill),
                    )

                validity_tile_profile = validity_profile.copy()
                validity_tile_profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=1,
                    dtype="uint8",
                    nodata=validity_fill,
                    transform=tile_transform,
                    compress=output_config["compress"],
                    tiled=bool(output_config["tiled"]),
                    blockxsize=int(output_config["block_size"]),
                    blockysize=int(output_config["block_size"]),
                    predictor=int(
                        output_config["predictor_mask"]
                    ),
                    BIGTIFF=str(
                        output_config["bigtiff"]
                    ).upper(),
                )

                with rasterio.open(
                    validity_path,
                    "w",
                    **validity_tile_profile,
                ) as output:
                    output.write(validity_array.astype(np.uint8), 1)
                    output.set_band_description(
                        1,
                        "training_validity_mask",
                    )
                    output.update_tags(
                        city_id=args.city_id,
                        city_name=args.city_name,
                        split=args.split,
                        tile_id=tile_id,
                        value_0="ignore",
                        value_1="valid_label",
                    )

                left, bottom, right, top = window_bounds(
                    window,
                    image_ds.transform,
                )

                row = {
                    "tile_id": tile_id,
                    "city_id": args.city_id,
                    "city_name": args.city_name,
                    "split": args.split,
                    "tile_row": tile_row,
                    "tile_col": tile_col,
                    "row_offset": row_off,
                    "col_offset": col_off,
                    "tile_size": tile_size,
                    "stride": stride,
                    "source_width_pixels": source_width,
                    "source_height_pixels": source_height,
                    "padding_pixel_count": padding_pixel_count,
                    "padding_fraction": (
                        padding_pixel_count / tile_pixel_count
                    ),
                    "valid_pixel_count": valid_pixel_count,
                    "valid_fraction": (
                        valid_pixel_count / tile_pixel_count
                    ),
                    "unlabeled_pixel_count": unlabeled_pixel_count,
                    "semantic_nodata_pixel_count": nodata_pixel_count,
                    "buildings_pixel_count": counts.get(1, 0),
                    "roads_pixel_count": counts.get(2, 0),
                    "vegetation_pixel_count": counts.get(3, 0),
                    "bare_land_pixel_count": counts.get(4, 0),
                    "water_pixel_count": counts.get(5, 0),
                    "contains_buildings": counts.get(1, 0) > 0,
                    "contains_roads": counts.get(2, 0) > 0,
                    "contains_vegetation": counts.get(3, 0) > 0,
                    "contains_bare_land": counts.get(4, 0) > 0,
                    "contains_water": counts.get(5, 0) > 0,
                    "bounds_left": left,
                    "bounds_bottom": bottom,
                    "bounds_right": right,
                    "bounds_top": top,
                    "crs": str(image_ds.crs),
                    "image_path": str(image_path),
                    "semantic_mask_path": str(semantic_path),
                    "validity_mask_path": str(validity_path),
                }

                rows.append(row)

    if not rows:
        raise RuntimeError("No tiles were generated.")

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "city_id": args.city_id,
        "city_name": args.city_name,
        "split": args.split,
        "source_image": str(args.image),
        "source_semantic_mask": str(args.semantic_mask),
        "source_validity_mask": str(args.validity_mask),
        "source_image_sha256": sha256_file(args.image),
        "source_semantic_mask_sha256": sha256_file(
            args.semantic_mask
        ),
        "source_validity_mask_sha256": sha256_file(
            args.validity_mask
        ),
        "tile_size": tile_size,
        "stride": stride,
        "edge_policy": "pad",
        "tile_filtering": "none",
        "tile_count": len(rows),
        "tiles_with_padding": sum(
            row["padding_pixel_count"] > 0
            for row in rows
        ),
        "tiles_with_zero_valid_pixels": sum(
            row["valid_pixel_count"] == 0
            for row in rows
        ),
        "tiles_containing_each_class": {
            class_name: sum(
                row[f"contains_{class_name}"]
                for row in rows
            )
            for class_name in TRAINABLE_CLASSES.values()
        },
        "total_valid_pixels_across_tiles": sum(
            row["valid_pixel_count"]
            for row in rows
        ),
        "manifest": str(manifest_path),
    }

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

    print("\nTiling completed")
    print("----------------")
    print(f"Tiles generated: {len(rows)}")
    print(
        f"Tiles with padding: "
        f"{summary['tiles_with_padding']}"
    )
    print(
        f"Tiles with zero valid pixels: "
        f"{summary['tiles_with_zero_valid_pixels']}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print("\nResult: aligned city tiles generated successfully.")


if __name__ == "__main__":
    main()
