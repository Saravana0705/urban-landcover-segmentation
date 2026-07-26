"""Validate one city's tiled SAR dataset.

The script independently validates every image/semantic/validity tile triplet
and compares calculated statistics against the city tile manifest and tiling
summary.

Expected semantic encoding
--------------------------
0   = unlabeled
1   = buildings
2   = roads
3   = vegetation
4   = bare land
5   = water
255 = semantic NoData / padding

Expected validity encoding
--------------------------
0 = ignore
1 = valid labelled training pixel
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
import pandas as pd
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox


ALLOWED_SEMANTIC_VALUES = {0, 1, 2, 3, 4, 5, 255}
ALLOWED_VALIDITY_VALUES = {0, 1}
TRAINABLE_CLASS_VALUES = {1, 2, 3, 4, 5}

EXPECTED_IMAGE_BANDS = 2
EXPECTED_TILE_SIZE = 256
EXPECTED_PIXEL_SIZE_M = 10.0
SEMANTIC_NODATA = 255
VALIDITY_NODATA = 0
FLOAT_TOLERANCE = 1e-9

def convert_numpy(obj):
    """Recursively convert NumPy types into native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    return obj

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def approximately_equal(
    value_a: float,
    value_b: float,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    """Compare floating-point values."""
    return math.isclose(
        value_a,
        value_b,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def transforms_equal(
    transform_a: Affine,
    transform_b: Affine,
) -> bool:
    """Compare six affine coefficients."""
    return all(
        approximately_equal(value_a, value_b)
        for value_a, value_b in zip(
            tuple(transform_a)[:6],
            tuple(transform_b)[:6],
            strict=True,
        )
    )


def bounds_equal(
    bounds_a: BoundingBox,
    bounds_b: BoundingBox,
) -> bool:
    """Compare raster bounds."""
    values_a = (
        bounds_a.left,
        bounds_a.bottom,
        bounds_a.right,
        bounds_a.top,
    )
    values_b = (
        bounds_b.left,
        bounds_b.bottom,
        bounds_b.right,
        bounds_b.top,
    )

    return all(
        approximately_equal(value_a, value_b)
        for value_a, value_b in zip(
            values_a,
            values_b,
            strict=True,
        )
    )


def value_counts(array: np.ndarray) -> dict[int, int]:
    """Return integer value counts."""
    values, counts = np.unique(
        array,
        return_counts=True,
    )

    return {
        int(value): int(count)
        for value, count in zip(
            values,
            counts,
            strict=True,
        )
    }


def parse_bool(value: Any) -> bool:
    """Convert CSV-compatible truth values to bool."""
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def require_file(path: Path, label: str) -> None:
    """Require a non-empty file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def read_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the tile manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Tile manifest not found: {path}")

    dataframe = pd.read_csv(path)

    required_columns = {
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "tile_size",
        "stride",
        "source_width_pixels",
        "source_height_pixels",
        "padding_pixel_count",
        "padding_fraction",
        "valid_pixel_count",
        "valid_fraction",
        "unlabeled_pixel_count",
        "semantic_nodata_pixel_count",
        "buildings_pixel_count",
        "roads_pixel_count",
        "vegetation_pixel_count",
        "bare_land_pixel_count",
        "water_pixel_count",
        "image_path",
        "semantic_mask_path",
        "validity_mask_path",
    }

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Tile manifest is missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError("Tile manifest contains no rows.")

    if dataframe["tile_id"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["tile_id"].duplicated(),
            "tile_id",
        ].tolist()

        raise ValueError(
            f"Duplicate tile IDs in manifest: {duplicates[:10]}"
        )

    return dataframe


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate one city's tiled SAR dataset."
    )

    parser.add_argument("--city-id", required=True)

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--tiling-summary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/tile_qa"),
    )

    return parser.parse_args()


def main() -> None:
    """Run tile-level QA for one city."""
    args = parse_arguments()

    manifest = read_manifest(args.manifest)
    tiling_summary = load_json(args.tiling_summary)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_city_ids = set(
        manifest["city_id"].astype(str)
    )

    if manifest_city_ids != {args.city_id}:
        raise ValueError(
            f"Manifest city IDs do not equal {args.city_id}: "
            f"{sorted(manifest_city_ids)}"
        )

    expected_tile_count = int(
        tiling_summary["tile_count"]
    )

    if len(manifest) != expected_tile_count:
        raise ValueError(
            "Manifest row count does not match tiling summary: "
            f"{len(manifest)} != {expected_tile_count}"
        )

    print("\nTile Dataset QA")
    print("---------------")
    print(f"City: {args.city_id}")
    print(f"Manifest: {args.manifest}")
    print(f"Expected tiles: {expected_tile_count}")

    tile_results: list[dict[str, Any]] = []
    failed_tiles: list[str] = []

    total_valid_pixels = 0
    total_padding_pixels = 0
    total_semantic_nodata_pixels = 0

    aggregate_class_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        255: 0,
    }

    for index, row in manifest.iterrows():
        tile_id = str(row["tile_id"])

        image_path = Path(str(row["image_path"]))
        semantic_path = Path(
            str(row["semantic_mask_path"])
        )
        validity_path = Path(
            str(row["validity_mask_path"])
        )

        for path, label in (
            (image_path, "image tile"),
            (semantic_path, "semantic tile"),
            (validity_path, "validity tile"),
        ):
            require_file(path, f"{tile_id} {label}")

        with rasterio.open(
            image_path
        ) as image_ds, rasterio.open(
            semantic_path
        ) as semantic_ds, rasterio.open(
            validity_path
        ) as validity_ds:
            image = image_ds.read()
            semantic = semantic_ds.read(1)
            validity = validity_ds.read(1)

            semantic_counts = value_counts(semantic)
            validity_counts = value_counts(validity)

            for value, count in semantic_counts.items():
                aggregate_class_counts[value] = (
                    aggregate_class_counts.get(value, 0)
                    + count
                )

            actual_valid_count = int(
                np.count_nonzero(validity == 1)
            )
            actual_padding_count = int(
                row["padding_pixel_count"]
            )
            actual_semantic_nodata_count = int(
                np.count_nonzero(
                    semantic == SEMANTIC_NODATA
                )
            )

            total_valid_pixels += actual_valid_count
            total_padding_pixels += actual_padding_count
            total_semantic_nodata_pixels += (
                actual_semantic_nodata_count
            )

            semantic_is_class = np.isin(
                semantic,
                sorted(TRAINABLE_CLASS_VALUES),
            )
            semantic_is_unlabeled = semantic == 0
            semantic_is_nodata = (
                semantic == SEMANTIC_NODATA
            )
            validity_is_valid = validity == 1
            validity_is_ignore = validity == 0

            tile_size = int(row["tile_size"])
            source_width = int(
                row["source_width_pixels"]
            )
            source_height = int(
                row["source_height_pixels"]
            )

            expected_padding_mask = np.ones(
                (tile_size, tile_size),
                dtype=bool,
            )
            expected_padding_mask[
                :source_height,
                :source_width,
            ] = False

            checks = {
                "image_band_count_is_2":
                    image_ds.count == EXPECTED_IMAGE_BANDS,
                "semantic_band_count_is_1":
                    semantic_ds.count == 1,
                "validity_band_count_is_1":
                    validity_ds.count == 1,
                "image_dimensions_are_256": (
                    image_ds.width == EXPECTED_TILE_SIZE
                    and image_ds.height == EXPECTED_TILE_SIZE
                ),
                "semantic_dimensions_are_256": (
                    semantic_ds.width == EXPECTED_TILE_SIZE
                    and semantic_ds.height == EXPECTED_TILE_SIZE
                ),
                "validity_dimensions_are_256": (
                    validity_ds.width == EXPECTED_TILE_SIZE
                    and validity_ds.height == EXPECTED_TILE_SIZE
                ),
                "image_dtype_float32":
                    all(
                        dtype == "float32"
                        for dtype in image_ds.dtypes
                    ),
                "semantic_dtype_uint8":
                    semantic_ds.dtypes[0] == "uint8",
                "validity_dtype_uint8":
                    validity_ds.dtypes[0] == "uint8",
                "crs_match": (
                    image_ds.crs
                    == semantic_ds.crs
                    == validity_ds.crs
                ),
                "transform_match": (
                    transforms_equal(
                        image_ds.transform,
                        semantic_ds.transform,
                    )
                    and transforms_equal(
                        image_ds.transform,
                        validity_ds.transform,
                    )
                ),
                "bounds_match": (
                    bounds_equal(
                        image_ds.bounds,
                        semantic_ds.bounds,
                    )
                    and bounds_equal(
                        image_ds.bounds,
                        validity_ds.bounds,
                    )
                ),
                "pixel_width_is_10m":
                    approximately_equal(
                        image_ds.transform.a,
                        EXPECTED_PIXEL_SIZE_M,
                    ),
                "pixel_height_is_10m":
                    approximately_equal(
                        abs(image_ds.transform.e),
                        EXPECTED_PIXEL_SIZE_M,
                    ),
                "semantic_nodata_is_255":
                    semantic_ds.nodata
                    == SEMANTIC_NODATA,
                "validity_nodata_is_0":
                    validity_ds.nodata
                    == VALIDITY_NODATA,
                "semantic_values_allowed":
                    set(semantic_counts).issubset(
                        ALLOWED_SEMANTIC_VALUES
                    ),
                "validity_values_allowed":
                    set(validity_counts).issubset(
                        ALLOWED_VALIDITY_VALUES
                    ),
                "all_class_pixels_valid": (
                    np.count_nonzero(
                        semantic_is_class
                        & ~validity_is_valid
                    )
                    == 0
                ),
                "all_valid_pixels_are_classes": (
                    np.count_nonzero(
                        validity_is_valid
                        & ~semantic_is_class
                    )
                    == 0
                ),
                "unlabeled_pixels_are_ignored": (
                    np.count_nonzero(
                        semantic_is_unlabeled
                        & validity_is_valid
                    )
                    == 0
                ),
                "semantic_nodata_pixels_are_ignored": (
                    np.count_nonzero(
                        semantic_is_nodata
                        & validity_is_valid
                    )
                    == 0
                ),
                "padding_semantic_is_255": (
                    np.count_nonzero(
                        expected_padding_mask
                        & ~semantic_is_nodata
                    )
                    == 0
                ),
                "padding_validity_is_0": (
                    np.count_nonzero(
                        expected_padding_mask
                        & ~validity_is_ignore
                    )
                    == 0
                ),
                "padding_image_is_fill_value": (
                    np.count_nonzero(
                        expected_padding_mask
                        & np.any(
                            image != 0.0,
                            axis=0,
                        )
                    )
                    == 0
                ),
                "padding_count_matches_manifest": (
                    int(
                        np.count_nonzero(
                            expected_padding_mask
                        )
                    )
                    == int(row["padding_pixel_count"])
                ),
                "valid_count_matches_manifest": (
                    actual_valid_count
                    == int(row["valid_pixel_count"])
                ),
                "unlabeled_count_matches_manifest": (
                    semantic_counts.get(0, 0)
                    == int(
                        row["unlabeled_pixel_count"]
                    )
                ),
                "semantic_nodata_count_matches_manifest": (
                    semantic_counts.get(255, 0)
                    == int(
                        row[
                            "semantic_nodata_pixel_count"
                        ]
                    )
                ),
                "buildings_count_matches_manifest": (
                    semantic_counts.get(1, 0)
                    == int(
                        row["buildings_pixel_count"]
                    )
                ),
                "roads_count_matches_manifest": (
                    semantic_counts.get(2, 0)
                    == int(
                        row["roads_pixel_count"]
                    )
                ),
                "vegetation_count_matches_manifest": (
                    semantic_counts.get(3, 0)
                    == int(
                        row["vegetation_pixel_count"]
                    )
                ),
                "bare_land_count_matches_manifest": (
                    semantic_counts.get(4, 0)
                    == int(
                        row["bare_land_pixel_count"]
                    )
                ),
                "water_count_matches_manifest": (
                    semantic_counts.get(5, 0)
                    == int(
                        row["water_pixel_count"]
                    )
                ),
            }

            checks["all_tile_checks_passed"] = all(
                checks.values()
            )

            if not checks["all_tile_checks_passed"]:
                failed_tiles.append(tile_id)

            tile_results.append(
                {
                    "tile_id": tile_id,
                    "status": (
                        "PASS"
                        if checks[
                            "all_tile_checks_passed"
                        ]
                        else "FAIL"
                    ),
                    "checks": checks,
                    "semantic_value_counts": {
                        str(value): count
                        for value, count
                        in sorted(
                            semantic_counts.items()
                        )
                    },
                    "validity_value_counts": {
                        str(value): count
                        for value, count
                        in sorted(
                            validity_counts.items()
                        )
                    },
                    "image_sha256":
                        sha256_file(image_path),
                    "semantic_sha256":
                        sha256_file(semantic_path),
                    "validity_sha256":
                        sha256_file(validity_path),
                }
            )

        if (index + 1) % 25 == 0:
            print(
                f"Validated {index + 1}/{len(manifest)} tiles"
            )

    summary_checks = {
        "manifest_tile_count_matches_summary":
            len(manifest) == expected_tile_count,
        "all_tiles_passed":
            len(failed_tiles) == 0,
        "total_valid_pixels_match_summary": (
            total_valid_pixels
            == int(
                tiling_summary[
                    "total_valid_pixels_across_tiles"
                ]
            )
        ),
        "tiles_with_padding_match_summary": (
            int(
                (
                    manifest[
                        "padding_pixel_count"
                    ] > 0
                ).sum()
            )
            == int(
                tiling_summary["tiles_with_padding"]
            )
        ),
        "zero_valid_tiles_match_summary": (
            int(
                (
                    manifest[
                        "valid_pixel_count"
                    ] == 0
                ).sum()
            )
            == int(
                tiling_summary[
                    "tiles_with_zero_valid_pixels"
                ]
            )
        ),
    }

    summary_checks["all_summary_checks_passed"] = all(
        summary_checks.values()
    )

    overall_pass = (
        len(failed_tiles) == 0
        and summary_checks[
            "all_summary_checks_passed"
        ]
    )

    qa_report = {
        "city_id": args.city_id,
        "qa_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "manifest": str(args.manifest),
        "tiling_summary": str(
            args.tiling_summary
        ),
        "manifest_sha256":
            sha256_file(args.manifest),
        "tiling_summary_sha256":
            sha256_file(args.tiling_summary),
        "tile_count": len(manifest),
        "passed_tile_count":
            len(manifest) - len(failed_tiles),
        "failed_tile_count":
            len(failed_tiles),
        "failed_tiles": failed_tiles,
        "summary_checks": summary_checks,
        "aggregate_statistics": {
            "total_valid_pixels":
                total_valid_pixels,
            "total_padding_pixels":
                total_padding_pixels,
            "total_semantic_nodata_pixels":
                total_semantic_nodata_pixels,
            "semantic_value_counts": {
                str(value): count
                for value, count
                in sorted(
                    aggregate_class_counts.items()
                )
            },
        },
        "tiles": tile_results,
    }

    output_path = (
        args.output_dir
        / f"{args.city_id}_tile_qa.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            convert_numpy(qa_report),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTile QA summary")
    print("---------------")
    print(f"Tiles checked: {len(manifest)}")
    print(
        f"Passed tiles: "
        f"{len(manifest) - len(failed_tiles)}"
    )
    print(f"Failed tiles: {len(failed_tiles)}")
    print(
        f"Total valid pixels: "
        f"{total_valid_pixels:,}"
    )

    print("\nSummary checks")
    for check_name, passed in summary_checks.items():
        print(f"  {check_name}: {passed}")

    print("\nOverall tile QA result")
    print("----------------------")
    print("PASS" if overall_pass else "FAIL")
    print(f"QA report: {output_path}")

    if not overall_pass:
        raise RuntimeError(
            "Tile dataset QA failed. "
            f"Failed tiles: {failed_tiles[:10]}"
        )


if __name__ == "__main__":
    main()
