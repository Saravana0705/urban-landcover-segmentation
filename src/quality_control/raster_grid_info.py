"""Inspect a multiband Sentinel-1 GeoTIFF and save its reference grid.

The raster is expected to contain:
    Band 1: Sigma0_VV
    Band 2: Sigma0_VH

The grid metadata saved by this script becomes the authoritative
reference for label rasterization and subsequent tiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS


EXPECTED_BAND_COUNT = 2
EXPECTED_EPSG = 32632
EXPECTED_PIXEL_SIZE_M = 10.0
FLOAT_TOLERANCE = 1e-9


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def affine_to_dict(transform: Affine) -> dict[str, float]:
    """Convert an affine transform into named JSON fields."""
    return {
        "pixel_width": float(transform.a),
        "row_rotation": float(transform.b),
        "upper_left_x": float(transform.c),
        "column_rotation": float(transform.d),
        "pixel_height": float(transform.e),
        "upper_left_y": float(transform.f),
    }


def affine_to_gdal_list(transform: Affine) -> list[float]:
    """Return affine coefficients in GDAL geotransform order."""
    return [
        float(transform.c),
        float(transform.a),
        float(transform.b),
        float(transform.f),
        float(transform.d),
        float(transform.e),
    ]


def bounds_to_dict(
    bounds: BoundingBox,
) -> dict[str, float]:
    """Convert raster bounds into serializable fields."""
    return {
        "left": float(bounds.left),
        "bottom": float(bounds.bottom),
        "right": float(bounds.right),
        "top": float(bounds.top),
    }


def crs_to_dict(
    crs: CRS | None,
) -> dict[str, Any]:
    """Convert CRS metadata into JSON-compatible fields."""
    if crs is None:
        return {
            "string": None,
            "epsg": None,
            "wkt": None,
        }

    return {
        "string": str(crs),
        "epsg": crs.to_epsg(),
        "wkt": crs.to_wkt(),
    }


def approximately_equal(
    value_a: float,
    value_b: float,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    """Compare floating-point values using absolute tolerance."""
    return math.isclose(
        value_a,
        value_b,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def inspect_band(
    dataset: rasterio.io.DatasetReader,
    band_index: int,
) -> dict[str, Any]:
    """Calculate statistics for one raster band."""
    band = dataset.read(
        band_index,
        masked=True,
    )

    valid_values = band.compressed()
    valid_pixel_count = int(valid_values.size)

    total_pixel_count = int(
        dataset.width * dataset.height
    )
    masked_pixel_count = (
        total_pixel_count - valid_pixel_count
    )

    if valid_pixel_count == 0:
        return {
            "band_index": band_index,
            "description": dataset.descriptions[
                band_index - 1
            ],
            "dtype": dataset.dtypes[band_index - 1],
            "nodata": dataset.nodatavals[
                band_index - 1
            ],
            "total_pixel_count": total_pixel_count,
            "valid_pixel_count": 0,
            "masked_pixel_count": masked_pixel_count,
            "valid_pixel_percentage": 0.0,
            "finite_pixel_count": 0,
            "non_finite_pixel_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "percentile_1": None,
            "percentile_50": None,
            "percentile_99": None,
        }

    finite_mask = np.isfinite(valid_values)
    finite_values = valid_values[finite_mask]

    finite_pixel_count = int(finite_values.size)
    non_finite_pixel_count = (
        valid_pixel_count - finite_pixel_count
    )

    if finite_pixel_count == 0:
        minimum = None
        maximum = None
        mean = None
        standard_deviation = None
        percentile_1 = None
        percentile_50 = None
        percentile_99 = None
    else:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
        standard_deviation = float(
            np.std(finite_values)
        )
        percentile_1 = float(
            np.percentile(finite_values, 1)
        )
        percentile_50 = float(
            np.percentile(finite_values, 50)
        )
        percentile_99 = float(
            np.percentile(finite_values, 99)
        )

    return {
        "band_index": band_index,
        "description": dataset.descriptions[
            band_index - 1
        ],
        "dtype": dataset.dtypes[band_index - 1],
        "nodata": dataset.nodatavals[
            band_index - 1
        ],
        "total_pixel_count": total_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "masked_pixel_count": masked_pixel_count,
        "valid_pixel_percentage": round(
            valid_pixel_count
            / total_pixel_count
            * 100.0,
            6,
        ),
        "finite_pixel_count": finite_pixel_count,
        "non_finite_pixel_count":
            non_finite_pixel_count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "percentile_1": percentile_1,
        "percentile_50": percentile_50,
        "percentile_99": percentile_99,
    }


def validate_reference_grid(
    dataset: rasterio.io.DatasetReader,
) -> dict[str, bool]:
    """Run mandatory reference-grid validation checks."""
    pixel_width = float(dataset.transform.a)
    pixel_height = float(abs(dataset.transform.e))

    checks = {
        "has_crs": dataset.crs is not None,
        "projected_crs": (
            dataset.crs is not None
            and dataset.crs.is_projected
        ),
        "epsg_32632": (
            dataset.crs is not None
            and dataset.crs.to_epsg()
            == EXPECTED_EPSG
        ),
        "band_count_is_2": (
            dataset.count == EXPECTED_BAND_COUNT
        ),
        "width_positive": dataset.width > 0,
        "height_positive": dataset.height > 0,
        "pixel_width_is_10m": approximately_equal(
            pixel_width,
            EXPECTED_PIXEL_SIZE_M,
        ),
        "pixel_height_is_10m": approximately_equal(
            pixel_height,
            EXPECTED_PIXEL_SIZE_M,
        ),
        "north_up": (
            approximately_equal(
                dataset.transform.b,
                0.0,
            )
            and approximately_equal(
                dataset.transform.d,
                0.0,
            )
            and dataset.transform.e < 0
        ),
        "square_pixels": approximately_equal(
            pixel_width,
            pixel_height,
        ),
        "single_grid_for_both_bands": True,
    }

    checks["reference_grid_valid"] = all(
        checks.values()
    )

    return checks


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a two-band Sentinel-1 reference raster."
        )
    )

    parser.add_argument(
        "--city-id",
        required=True,
        help="City identifier, for example DE01.",
    )

    parser.add_argument(
        "--raster",
        required=True,
        type=Path,
        help="Path to the two-band VV/VH GeoTIFF.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/raster_grid"),
        help="Directory for reference-grid metadata.",
    )

    return parser.parse_args()


def main() -> None:
    """Inspect the raster and save authoritative grid metadata."""
    args = parse_arguments()

    if not args.raster.exists():
        raise FileNotFoundError(
            f"Raster not found: {args.raster}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSentinel-1 reference-grid inspection")
    print("------------------------------------")
    print(f"City: {args.city_id}")
    print(f"Raster: {args.raster}")

    with rasterio.open(args.raster) as dataset:
        checks = validate_reference_grid(dataset)

        pixel_width = float(dataset.transform.a)
        pixel_height = float(
            abs(dataset.transform.e)
        )

        band_statistics = [
            inspect_band(dataset, band_index)
            for band_index in range(
                1,
                dataset.count + 1,
            )
        ]

        metadata_tags = dataset.tags()

        output = {
            "city_id": args.city_id,
            "reference_raster": {
                "path": str(args.raster),
                "filename": args.raster.name,
                "file_size_bytes": int(
                    args.raster.stat().st_size
                ),
                "sha256": sha256_file(args.raster),
                "driver": dataset.driver,
                "band_count": int(dataset.count),
                "band_descriptions": list(
                    dataset.descriptions
                ),
                "width_pixels": int(dataset.width),
                "height_pixels": int(dataset.height),
                "total_pixel_count": int(
                    dataset.width * dataset.height
                ),
                "crs": crs_to_dict(dataset.crs),
                "transform": affine_to_dict(
                    dataset.transform
                ),
                "gdal_geotransform":
                    affine_to_gdal_list(
                        dataset.transform
                    ),
                "bounds": bounds_to_dict(
                    dataset.bounds
                ),
                "pixel_width_m": pixel_width,
                "pixel_height_m": pixel_height,
                "pixel_area_m2": (
                    pixel_width * pixel_height
                ),
                "dtypes": list(dataset.dtypes),
                "nodatavals": list(
                    dataset.nodatavals
                ),
                "compression":
                    dataset.profile.get("compress"),
                "tiled": bool(
                    dataset.profile.get(
                        "tiled",
                        False,
                    )
                ),
                "block_shapes": [
                    [int(rows), int(columns)]
                    for rows, columns
                    in dataset.block_shapes
                ],
                "metadata_tags": metadata_tags,
            },
            "band_statistics": band_statistics,
            "validation_checks": checks,
        }

    output_path = (
        args.output_dir
        / f"{args.city_id}_reference_grid.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    reference = output["reference_raster"]

    print("\nReference grid")
    print(
        f"  Dimensions: "
        f"{reference['width_pixels']:,} × "
        f"{reference['height_pixels']:,}"
    )
    print(
        f"  Bands: "
        f"{reference['band_count']}"
    )
    print(
        f"  Band descriptions: "
        f"{reference['band_descriptions']}"
    )
    print(
        f"  CRS: "
        f"{reference['crs']['string']}"
    )
    print(
        f"  Pixel size: "
        f"{reference['pixel_width_m']:.6f} × "
        f"{reference['pixel_height_m']:.6f} m"
    )
    print(
        f"  Pixel area: "
        f"{reference['pixel_area_m2']:.2f} m²"
    )
    print(
        f"  Bounds: "
        f"{reference['bounds']}"
    )
    print(
        f"  Transform: "
        f"{reference['transform']}"
    )
    print(
        f"  Data types: "
        f"{reference['dtypes']}"
    )
    print(
        f"  NoData values: "
        f"{reference['nodatavals']}"
    )

    print("\nBand statistics")

    for band in output["band_statistics"]:
        print(
            f"  Band {band['band_index']} "
            f"({band['description']}):"
        )
        print(
            f"    valid pixels: "
            f"{band['valid_pixel_count']:,} "
            f"({band['valid_pixel_percentage']:.4f}%)"
        )
        print(
            f"    range: "
            f"{band['minimum']} to "
            f"{band['maximum']}"
        )
        print(
            f"    mean: {band['mean']}"
        )
        print(
            f"    std: "
            f"{band['standard_deviation']}"
        )
        print(
            f"    p1 / p50 / p99: "
            f"{band['percentile_1']} / "
            f"{band['percentile_50']} / "
            f"{band['percentile_99']}"
        )
        print(
            f"    non-finite pixels: "
            f"{band['non_finite_pixel_count']:,}"
        )

    print("\nValidation checks")

    for check_name, passed in checks.items():
        print(f"  {check_name}: {passed}")

    print(
        f"\nSaved reference-grid metadata: "
        f"{output_path}"
    )

    if not checks["reference_grid_valid"]:
        raise RuntimeError(
            "The Sentinel-1 reference grid failed one "
            "or more mandatory checks. Do not proceed "
            "to label rasterization."
        )

    non_finite_total = sum(
        band["non_finite_pixel_count"]
        for band in output["band_statistics"]
    )

    if non_finite_total > 0:
        print(
            "\nWarning: non-finite raster values were "
            "detected. Review them before training."
        )

    print(
        "\nResult: reference grid is valid and ready "
        "for label rasterization."
    )


if __name__ == "__main__":
    main()