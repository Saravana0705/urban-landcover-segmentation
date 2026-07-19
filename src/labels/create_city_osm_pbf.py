from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from pyrosm import OSM
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML file: {path}")

    return config


def load_cities(config: dict[str, Any]) -> pd.DataFrame:
    cities = pd.read_csv(
        config["inputs"]["city_registry_file"]
    )

    required = {
        "city_id",
        "city_name",
        "utm_epsg",
    }

    missing = required - set(cities.columns)

    if missing:
        raise ValueError(
            "Missing city columns: "
            + ", ".join(sorted(missing))
        )

    return cities.sort_values(
        "city_id"
    ).reset_index(drop=True)


def get_city_aoi_bounds(
    row: pd.Series,
    config: dict[str, Any],
) -> tuple[list[float], Path]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    aoi_path = (
        Path(config["inputs"]["city_aoi_directory"])
        / f"{city_id}_{city_name}_aoi.gpkg"
    )

    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOI file not found: {aoi_path}"
        )

    aoi = gpd.read_file(
        aoi_path,
        layer="aoi",
    )

    if aoi.empty:
        raise ValueError(
            f"AOI is empty: {aoi_path}"
        )

    if aoi.crs is None:
        raise ValueError(
            f"AOI has no CRS: {aoi_path}"
        )

    aoi_wgs84 = aoi.to_crs("EPSG:4326")

    min_x, min_y, max_x, max_y = (
        aoi_wgs84.total_bounds
    )

    bbox = [
        float(min_x),
        float(min_y),
        float(max_x),
        float(max_y),
    ]

    return bbox, aoi_path.resolve()


def crop_city(
    row: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    source_pbf = Path(
        config["osm"]["pbf_file"]
    )

    output_directory = Path(
        config["osm"]["city_pbf_directory"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_pbf = (
        output_directory
        / f"{city_id}_{city_name}.osm.pbf"
    )

    bbox, aoi_path = get_city_aoi_bounds(
        row,
        config,
    )

    if output_pbf.exists():
        if output_pbf.stat().st_size > 100_000:
            LOGGER.info(
                "Skipping existing city PBF for %s.",
                city_id,
            )

            return {
                "city_id": city_id,
                "city_name": city_name,
                "status": "skipped_valid",
                "output_pbf": str(output_pbf),
                "size_mb": (
                    output_pbf.stat().st_size
                    / 1024
                    / 1024
                ),
                "bbox": str(bbox),
                "aoi_file": str(aoi_path),
                "duration_seconds": 0.0,
                "error": "",
            }

        output_pbf.unlink()

    LOGGER.info(
        "Cropping Germany PBF for %s using bbox %s.",
        city_id,
        bbox,
    )

    started = time.perf_counter()

    osm = OSM(
        str(source_pbf),
        bounding_box=bbox,
        engine="out_of_core",
        workers=1,
    )

    osm.to_pbf(
        output_path=str(output_pbf),
        keep_relations=True,
    )

    duration = time.perf_counter() - started

    if (
        not output_pbf.exists()
        or output_pbf.stat().st_size <= 100_000
    ):
        return {
            "city_id": city_id,
            "city_name": city_name,
            "status": "failed",
            "output_pbf": str(output_pbf),
            "size_mb": 0.0,
            "bbox": str(bbox),
            "aoi_file": str(aoi_path),
            "duration_seconds": duration,
            "error": "City PBF was not created or is too small.",
        }

    size_mb = (
        output_pbf.stat().st_size
        / 1024
        / 1024
    )

    LOGGER.info(
        "Created city PBF for %s: %.2f MB.",
        city_id,
        size_mb,
    )

    return {
        "city_id": city_id,
        "city_name": city_name,
        "status": "created_valid",
        "output_pbf": str(output_pbf),
        "size_mb": size_mb,
        "bbox": str(bbox),
        "aoi_file": str(aoi_path),
        "duration_seconds": duration,
        "error": "",
    }


def save_report(
    reports: list[dict[str, Any]],
    report_path: Path,
) -> None:
    report = pd.DataFrame(reports)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    timestamped = (
        report_path.parent
        / f"{report_path.stem}_{timestamp}.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    report.to_csv(
        timestamped,
        index=False,
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/labels.yaml"),
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    config = load_yaml(args.config)
    cities = load_cities(config)

    if args.city:
        cities = cities.loc[
            cities["city_id"] == args.city
        ].copy()

        if cities.empty:
            raise ValueError(
                f"Unknown city: {args.city}"
            )

    reports = []

    for _, row in tqdm(
        cities.iterrows(),
        total=len(cities),
        desc="City PBF cropping",
    ):
        reports.append(
            crop_city(row, config)
        )

    save_report(
        reports,
        Path(
            config["osm"]["crop_report_file"]
        ),
    )

    failed = [
        item
        for item in reports
        if item["status"] == "failed"
    ]

    print("\nCity PBF crop summary")
    print("-" * 70)
    print(f"Requested: {len(reports)}")
    print(
        f"Successful: "
        f"{len(reports) - len(failed)}"
    )
    print(f"Failed: {len(failed)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()