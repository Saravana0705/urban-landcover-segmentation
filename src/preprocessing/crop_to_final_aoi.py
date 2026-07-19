from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        )

    return config


def validate_configuration(
    config: dict[str, Any],
) -> None:
    required_sections = {
        "inputs",
        "outputs",
        "processing",
    }

    missing_sections = (
        required_sections - set(config.keys())
    )

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(sorted(missing_sections))
        )

    required_input_keys = {
        "city_registry_file",
        "city_aoi_directory",
    }

    missing_input_keys = (
        required_input_keys
        - set(config["inputs"].keys())
    )

    if missing_input_keys:
        raise ValueError(
            "Missing input keys: "
            + ", ".join(sorted(missing_input_keys))
        )

    required_output_keys = {
        "intermediate_directory",
        "final_directory",
        "crop_report_file",
    }

    missing_output_keys = (
        required_output_keys
        - set(config["outputs"].keys())
    )

    if missing_output_keys:
        raise ValueError(
            "Missing output keys: "
            + ", ".join(sorted(missing_output_keys))
        )

    required_processing_keys = {
        "final_pixel_size_m",
        "final_nodata_value",
        "final_compression",
        "skip_existing_valid",
        "overwrite_invalid",
        "pilot_cities",
    }

    missing_processing_keys = (
        required_processing_keys
        - set(config["processing"].keys())
    )

    if missing_processing_keys:
        raise ValueError(
            "Missing processing keys: "
            + ", ".join(sorted(missing_processing_keys))
        )


def load_city_registry(
    config: dict[str, Any],
) -> pd.DataFrame:
    registry_file = Path(
        config["inputs"]["city_registry_file"]
    )

    if not registry_file.exists():
        raise FileNotFoundError(
            f"City registry not found: {registry_file}"
        )

    cities = pd.read_csv(registry_file)

    required_columns = {
        "city_id",
        "city_name",
        "utm_epsg",
        "aoi_size_km",
    }

    missing_columns = (
        required_columns - set(cities.columns)
    )

    if missing_columns:
        raise ValueError(
            "City registry is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    if cities["city_id"].duplicated().any():
        duplicates = cities.loc[
            cities["city_id"].duplicated(),
            "city_id",
        ].tolist()

        raise ValueError(
            f"Duplicate city IDs: {duplicates}"
        )

    return cities.sort_values(
        by="city_id"
    ).reset_index(drop=True)


def get_aoi_file(
    row: pd.Series,
    config: dict[str, Any],
) -> Path:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    return (
        Path(
            config["inputs"]["city_aoi_directory"]
        )
        / f"{city_id}_{city_name}_aoi.gpkg"
    )


def get_buffered_raster(
    row: pd.Series,
    config: dict[str, Any],
) -> Path:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    return (
        Path(
            config["outputs"]["intermediate_directory"]
        )
        / f"{city_id}_{city_name}"
        / f"{city_id}_{city_name}_S1_TC_buffered.tif"
    )


def get_final_raster(
    row: pd.Series,
    config: dict[str, Any],
) -> Path:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    output_directory = (
        Path(config["outputs"]["final_directory"])
        / f"{city_id}_{city_name}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_directory
        / f"{city_id}_{city_name}_S1_VV_VH.tif"
    )


def read_exact_aoi_bounds(
    aoi_file: Path,
    expected_epsg: int,
) -> tuple[float, float, float, float]:
    if not aoi_file.exists():
        raise FileNotFoundError(
            f"AOI file not found: {aoi_file}"
        )

    aoi = gpd.read_file(
        aoi_file,
        layer="aoi",
    )

    if aoi.empty:
        raise ValueError(
            f"AOI contains no geometries: {aoi_file}"
        )

    if aoi.crs is None:
        raise ValueError(
            f"AOI has no CRS: {aoi_file}"
        )

    aoi = aoi.to_crs(
        epsg=expected_epsg
    )

    geometry = aoi.geometry.union_all()

    if geometry.is_empty:
        raise ValueError(
            f"AOI geometry is empty: {aoi_file}"
        )

    return geometry.bounds


def build_target_grid(
    bounds: tuple[float, float, float, float],
    pixel_size: float,
) -> tuple[
    rasterio.Affine,
    int,
    int,
    tuple[float, float, float, float],
]:
    min_x, min_y, max_x, max_y = bounds

    width_m = max_x - min_x
    height_m = max_y - min_y

    width_pixels = int(
        round(width_m / pixel_size)
    )

    height_pixels = int(
        round(height_m / pixel_size)
    )

    if width_pixels <= 0 or height_pixels <= 0:
        raise ValueError(
            "Calculated target raster dimensions "
            "are invalid."
        )

    # Reconstruct exact bounds from the rounded
    # raster dimensions so all pixels are exactly
    # pixel_size × pixel_size.
    exact_max_x = min_x + (
        width_pixels * pixel_size
    )

    exact_min_y = max_y - (
        height_pixels * pixel_size
    )

    transform = from_origin(
        min_x,
        max_y,
        pixel_size,
        pixel_size,
    )

    exact_bounds = (
        min_x,
        exact_min_y,
        exact_max_x,
        max_y,
    )

    return (
        transform,
        width_pixels,
        height_pixels,
        exact_bounds,
    )


def calculate_band_statistics(
    array: np.ndarray,
    nodata_value: float,
) -> dict[str, float]:
    valid = (
        np.isfinite(array)
        & (array != nodata_value)
    )

    valid_values = array[valid]

    if valid_values.size == 0:
        return {
            "minimum": float("nan"),
            "maximum": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "valid_percent": 0.0,
        }

    return {
        "minimum": float(
            np.min(valid_values)
        ),
        "maximum": float(
            np.max(valid_values)
        ),
        "mean": float(
            np.mean(valid_values)
        ),
        "std": float(
            np.std(valid_values)
        ),
        "p01": float(
            np.percentile(valid_values, 1)
        ),
        "p50": float(
            np.percentile(valid_values, 50)
        ),
        "p95": float(
            np.percentile(valid_values, 95)
        ),
        "p99": float(
            np.percentile(valid_values, 99)
        ),
        "valid_percent": float(
            100.0 * valid.mean()
        ),
    }


def validate_final_raster(
    path: Path,
    expected_epsg: int,
    expected_width: int,
    expected_height: int,
    expected_pixel_size: float,
    nodata_value: float,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "output_not_found"

    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 2:
                return (
                    False,
                    f"unexpected_band_count_{dataset.count}",
                )

            if dataset.crs is None:
                return False, "missing_crs"

            if dataset.crs.to_epsg() != expected_epsg:
                return (
                    False,
                    f"unexpected_epsg_"
                    f"{dataset.crs.to_epsg()}",
                )

            if dataset.width != expected_width:
                return (
                    False,
                    f"unexpected_width_{dataset.width}",
                )

            if dataset.height != expected_height:
                return (
                    False,
                    f"unexpected_height_{dataset.height}",
                )

            pixel_width = abs(
                dataset.transform.a
            )

            pixel_height = abs(
                dataset.transform.e
            )

            tolerance = 1e-6

            if (
                abs(
                    pixel_width
                    - expected_pixel_size
                )
                > tolerance
                or abs(
                    pixel_height
                    - expected_pixel_size
                )
                > tolerance
            ):
                return (
                    False,
                    (
                        "unexpected_pixel_size_"
                        f"{pixel_width}_"
                        f"{pixel_height}"
                    ),
                )

            if dataset.dtypes != (
                "float32",
                "float32",
            ):
                return (
                    False,
                    f"unexpected_dtypes_"
                    f"{dataset.dtypes}",
                )

            expected_descriptions = (
                "Sigma0_VV",
                "Sigma0_VH",
            )

            if (
                dataset.descriptions
                != expected_descriptions
            ):
                return (
                    False,
                    "unexpected_band_descriptions_"
                    f"{dataset.descriptions}",
                )

            for band_index in (1, 2):
                array = dataset.read(
                    band_index
                )

                valid_pixels = (
                    np.isfinite(array)
                    & (array != nodata_value)
                )

                if not valid_pixels.any():
                    return (
                        False,
                        f"band_{band_index}_"
                        "contains_no_valid_pixels",
                    )

    except rasterio.errors.RasterioIOError as error:
        return (
            False,
            f"rasterio_error_{error}",
        )

    return True, "valid"


def crop_city(
    row: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    expected_epsg = int(row["utm_epsg"])

    pixel_size = float(
        config["processing"][
            "final_pixel_size_m"
        ]
    )

    nodata_value = float(
        config["processing"][
            "final_nodata_value"
        ]
    )

    compression = str(
        config["processing"][
            "final_compression"
        ]
    ).upper()

    input_raster = get_buffered_raster(
        row=row,
        config=config,
    )

    aoi_file = get_aoi_file(
        row=row,
        config=config,
    )

    output_raster = get_final_raster(
        row=row,
        config=config,
    )

    if not input_raster.exists():
        return {
            "city_id": city_id,
            "city_name": city_name,
            "status": "input_missing",
            "input_raster": str(input_raster),
            "output_raster": str(output_raster),
            "aoi_file": str(aoi_file),
            "validation": "not_run",
            "error": (
                "Buffered SAR raster does not exist."
            ),
        }

    bounds = read_exact_aoi_bounds(
        aoi_file=aoi_file,
        expected_epsg=expected_epsg,
    )

    (
        target_transform,
        target_width,
        target_height,
        target_bounds,
    ) = build_target_grid(
        bounds=bounds,
        pixel_size=pixel_size,
    )

    skip_existing = bool(
        config["processing"][
            "skip_existing_valid"
        ]
    )

    overwrite_invalid = bool(
        config["processing"][
            "overwrite_invalid"
        ]
    )

    if output_raster.exists():
        is_valid, validation_message = (
            validate_final_raster(
                path=output_raster,
                expected_epsg=expected_epsg,
                expected_width=target_width,
                expected_height=target_height,
                expected_pixel_size=pixel_size,
                nodata_value=nodata_value,
            )
        )

        if is_valid and skip_existing:
            LOGGER.info(
                "Skipping valid final raster "
                "for %s.",
                city_id,
            )

            return {
                "city_id": city_id,
                "city_name": city_name,
                "status": "skipped_valid",
                "input_raster": str(
                    input_raster
                ),
                "output_raster": str(
                    output_raster
                ),
                "aoi_file": str(aoi_file),
                "utm_epsg": expected_epsg,
                "width": target_width,
                "height": target_height,
                "pixel_size_m": pixel_size,
                "validation": validation_message,
                "error": "",
            }

        if not is_valid:
            if overwrite_invalid:
                LOGGER.warning(
                    "Deleting invalid final raster "
                    "for %s: %s",
                    city_id,
                    validation_message,
                )

                output_raster.unlink(
                    missing_ok=True
                )
            else:
                return {
                    "city_id": city_id,
                    "city_name": city_name,
                    "status":
                        "invalid_existing_output",
                    "input_raster": str(
                        input_raster
                    ),
                    "output_raster": str(
                        output_raster
                    ),
                    "aoi_file": str(aoi_file),
                    "validation":
                        validation_message,
                    "error": (
                        "Existing output is invalid."
                    ),
                }

    LOGGER.info(
        "Cropping %s (%s) to "
        "%d × %d pixels in EPSG:%d.",
        city_id,
        city_name,
        target_width,
        target_height,
        expected_epsg,
    )

    started = datetime.now()

    with rasterio.open(
        input_raster
    ) as source:
        if source.count != 2:
            return {
                "city_id": city_id,
                "city_name": city_name,
                "status":
                    "unexpected_input_band_count",
                "input_raster": str(
                    input_raster
                ),
                "output_raster": str(
                    output_raster
                ),
                "aoi_file": str(aoi_file),
                "validation": "not_run",
                "error": (
                    f"Expected 2 input bands but "
                    f"found {source.count}."
                ),
            }

        if source.crs is None:
            return {
                "city_id": city_id,
                "city_name": city_name,
                "status": "input_missing_crs",
                "input_raster": str(
                    input_raster
                ),
                "output_raster": str(
                    output_raster
                ),
                "aoi_file": str(aoi_file),
                "validation": "not_run",
                "error": "Input raster has no CRS.",
            }

        destination = np.full(
            (
                2,
                target_height,
                target_width,
            ),
            nodata_value,
            dtype=np.float32,
        )

        for band_index in (1, 2):
            reproject(
                source=rasterio.band(
                    source,
                    band_index,
                ),
                destination=destination[
                    band_index - 1
                ],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=target_transform,
                dst_crs=(
                    f"EPSG:{expected_epsg}"
                ),
                dst_nodata=nodata_value,
                resampling=Resampling.bilinear,
                num_threads=2,
                init_dest_nodata=True,
            )

    output_profile = {
        "driver": "GTiff",
        "width": target_width,
        "height": target_height,
        "count": 2,
        "dtype": "float32",
        "crs": f"EPSG:{expected_epsg}",
        "transform": target_transform,
        "nodata": nodata_value,
        "compress": compression,
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
        "interleave": "band",
    }

    output_raster.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        output_raster,
        "w",
        **output_profile,
    ) as destination_dataset:
        destination_dataset.write(
            destination[0],
            1,
        )

        destination_dataset.write(
            destination[1],
            2,
        )

        destination_dataset.set_band_description(
            1,
            "Sigma0_VV",
        )

        destination_dataset.set_band_description(
            2,
            "Sigma0_VH",
        )

        destination_dataset.update_tags(
            project=(
                "Automated Urban Land-Cover "
                "Segmentation Using Sentinel-1 SAR"
            ),
            city_id=city_id,
            city_name=city_name,
            source_raster=str(input_raster),
            source_aoi=str(aoi_file),
            band_1="Sigma0_VV",
            band_2="Sigma0_VH",
            value_domain="linear_sigma0",
            pixel_size_m=str(pixel_size),
            processing_stage="exact_aoi_crop",
        )

        # Build internal overviews to improve
        # QGIS display performance.
        overview_levels = [
            level
            for level in (2, 4, 8, 16)
            if (
                target_width // level >= 1
                and target_height // level >= 1
            )
        ]

        destination_dataset.build_overviews(
            overview_levels,
            Resampling.average,
        )

        destination_dataset.update_tags(
            ns="rio_overview",
            resampling="average",
        )

    duration_seconds = (
        datetime.now() - started
    ).total_seconds()

    is_valid, validation_message = (
        validate_final_raster(
            path=output_raster,
            expected_epsg=expected_epsg,
            expected_width=target_width,
            expected_height=target_height,
            expected_pixel_size=pixel_size,
            nodata_value=nodata_value,
        )
    )

    band_1_statistics = (
        calculate_band_statistics(
            destination[0],
            nodata_value,
        )
    )

    band_2_statistics = (
        calculate_band_statistics(
            destination[1],
            nodata_value,
        )
    )

    if not is_valid:
        return {
            "city_id": city_id,
            "city_name": city_name,
            "status":
                "output_validation_failed",
            "input_raster": str(input_raster),
            "output_raster": str(output_raster),
            "aoi_file": str(aoi_file),
            "utm_epsg": expected_epsg,
            "width": target_width,
            "height": target_height,
            "pixel_size_m": pixel_size,
            "duration_seconds":
                duration_seconds,
            "validation":
                validation_message,
            "error": "",
        }

    LOGGER.info(
        "Successfully created final raster "
        "for %s.",
        city_id,
    )

    return {
        "city_id": city_id,
        "city_name": city_name,
        "status": "cropped_valid",
        "input_raster": str(input_raster),
        "output_raster": str(output_raster),
        "aoi_file": str(aoi_file),
        "utm_epsg": expected_epsg,
        "width": target_width,
        "height": target_height,
        "pixel_size_m": pixel_size,
        "left": target_bounds[0],
        "bottom": target_bounds[1],
        "right": target_bounds[2],
        "top": target_bounds[3],
        "duration_seconds": duration_seconds,
        "validation": validation_message,
        "band_1_name": "Sigma0_VV",
        "band_1_mean":
            band_1_statistics["mean"],
        "band_1_std":
            band_1_statistics["std"],
        "band_1_p01":
            band_1_statistics["p01"],
        "band_1_p50":
            band_1_statistics["p50"],
        "band_1_p95":
            band_1_statistics["p95"],
        "band_1_p99":
            band_1_statistics["p99"],
        "band_1_valid_percent":
            band_1_statistics[
                "valid_percent"
            ],
        "band_2_name": "Sigma0_VH",
        "band_2_mean":
            band_2_statistics["mean"],
        "band_2_std":
            band_2_statistics["std"],
        "band_2_p01":
            band_2_statistics["p01"],
        "band_2_p50":
            band_2_statistics["p50"],
        "band_2_p95":
            band_2_statistics["p95"],
        "band_2_p99":
            band_2_statistics["p99"],
        "band_2_valid_percent":
            band_2_statistics[
                "valid_percent"
            ],
        "error": "",
    }


def save_report(
    reports: list[dict[str, Any]],
    report_file: Path,
) -> tuple[Path, Path]:
    report = pd.DataFrame(reports)

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    timestamped_report = (
        report_file.parent
        / (
            f"{report_file.stem}_"
            f"{timestamp}"
            f"{report_file.suffix}"
        )
    )

    report.to_csv(
        timestamped_report,
        index=False,
    )

    report.to_csv(
        report_file,
        index=False,
    )

    return (
        timestamped_report,
        report_file,
    )


def run_cropping(
    config_path: Path,
    city_filter: str | None,
    pilot_only: bool,
) -> None:
    config = load_yaml(config_path)

    validate_configuration(config)

    cities = load_city_registry(config)

    if city_filter:
        cities = cities.loc[
            cities["city_id"] == city_filter
        ].copy()

        if cities.empty:
            raise ValueError(
                f"City not found: {city_filter}"
            )

    elif pilot_only:
        pilot_cities = set(
            config["processing"][
                "pilot_cities"
            ]
        )

        cities = cities.loc[
            cities["city_id"].isin(
                pilot_cities
            )
        ].copy()

        missing_pilots = (
            pilot_cities
            - set(cities["city_id"])
        )

        if missing_pilots:
            raise ValueError(
                "Pilot cities missing: "
                + ", ".join(
                    sorted(missing_pilots)
                )
            )

    reports: list[dict[str, Any]] = []

    for _, row in tqdm(
        cities.iterrows(),
        total=len(cities),
        desc="Exact SAR cropping",
    ):
        reports.append(
            crop_city(
                row=row,
                config=config,
            )
        )

    report_file = Path(
        config["outputs"][
            "crop_report_file"
        ]
    )

    (
        timestamped_report,
        latest_report,
    ) = save_report(
        reports=reports,
        report_file=report_file,
    )

    successful_statuses = {
        "cropped_valid",
        "skipped_valid",
    }

    failed = [
        item
        for item in reports
        if item["status"]
        not in successful_statuses
    ]

    print("\nExact SAR cropping summary")
    print("-" * 70)
    print(f"Requested: {len(reports)}")
    print(
        "Successful:",
        sum(
            item["status"]
            in successful_statuses
            for item in reports
        ),
    )
    print(f"Failed: {len(failed)}")
    print(
        f"Timestamped report: "
        f"{timestamped_report}"
    )
    print(
        f"Latest report: {latest_report}"
    )

    if failed:
        print("\nFailed cities:")

        for item in failed:
            print(
                f"  {item['city_id']} — "
                f"{item['status']} — "
                f"{item.get('validation', '')}"
            )

        raise SystemExit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop buffered Sentinel-1 rasters "
            "to exact city AOI grids."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/preprocessing.yaml"
        ),
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help=(
            "Crop one city, for example DE01."
        ),
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Crop only configured pilot cities."
        ),
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()

    args = parse_arguments()

    if args.city and args.pilot:
        raise ValueError(
            "Use either --city or --pilot, "
            "not both."
        )

    run_cropping(
        config_path=args.config,
        city_filter=args.city,
        pilot_only=args.pilot,
    )


if __name__ == "__main__":
    main()