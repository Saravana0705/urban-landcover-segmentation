"""Generate and validate derived labels for one city."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def run_stage(command: list[str], stage: str) -> float:
    print("\n" + "=" * 72)
    print(f"Stage: {stage}")
    print("=" * 72)
    print(" ".join(command))
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def load_city(cities_file: Path, city_id: str) -> dict[str, Any]:
    cities = pd.read_csv(cities_file)
    required = {"city_id", "city_name", "utm_epsg"}
    missing = required.difference(cities.columns)
    if missing:
        raise ValueError(f"Missing city columns: {sorted(missing)}")

    match = cities.loc[cities["city_id"] == city_id]
    if match.empty:
        raise ValueError(f"Unknown city ID: {city_id}")

    row = match.iloc[0]
    return {
        "city_id": str(row["city_id"]),
        "city_name": str(row["city_name"]),
        "utm_epsg": int(row["utm_epsg"]),
        "split": str(row.get("split", "")),
    }


def validate_reference_raster(
    raster_path: Path,
    expected_epsg: int,
) -> dict[str, Any]:
    with rasterio.open(raster_path) as dataset:
        checks = {
            "band_count_is_2": dataset.count == 2,
            "expected_epsg": (
                dataset.crs is not None
                and dataset.crs.to_epsg() == expected_epsg
            ),
            "pixel_width_is_10m": abs(dataset.transform.a - 10.0) < 1e-9,
            "pixel_height_is_10m": abs(abs(dataset.transform.e) - 10.0) < 1e-9,
            "north_up": (
                abs(dataset.transform.b) < 1e-9
                and abs(dataset.transform.d) < 1e-9
                and dataset.transform.e < 0
            ),
            "square_pixels": (
                abs(dataset.transform.a - abs(dataset.transform.e)) < 1e-9
            ),
        }
        checks["reference_grid_valid"] = all(checks.values())

        metadata = {
            "crs": str(dataset.crs),
            "epsg": dataset.crs.to_epsg() if dataset.crs else None,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "band_descriptions": list(dataset.descriptions),
            "pixel_width": float(dataset.transform.a),
            "pixel_height": float(abs(dataset.transform.e)),
            "transform": list(dataset.transform)[:6],
            "bounds": {
                "left": float(dataset.bounds.left),
                "bottom": float(dataset.bounds.bottom),
                "right": float(dataset.bounds.right),
                "top": float(dataset.bounds.top),
            },
            "checks": checks,
        }

    if not checks["reference_grid_valid"]:
        raise RuntimeError(
            f"Reference-raster validation failed: {checks}"
        )

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labels and QA outputs for one city."
    )
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--cities-file",
        type=Path,
        default=Path("config/cities.csv"),
    )
    parser.add_argument(
        "--road-config",
        type=Path,
        default=Path("config/road_widths.yaml"),
    )
    parser.add_argument(
        "--rasterization-config",
        type=Path,
        default=Path("config/rasterization.yaml"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing derived outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city = load_city(args.cities_file, args.city)

    city_id = city["city_id"]
    city_name = city["city_name"]
    folder_name = f"{city_id}_{city_name}"

    osm_dir = Path("data/vector/osm") / folder_name
    derived_dir = Path("data/vector/derived") / folder_name
    sar_dir = Path("data/processed/sar") / folder_name
    masks_dir = Path("data/processed/masks") / folder_name

    buildings = osm_dir / "buildings.gpkg"
    roads = osm_dir / "roads.gpkg"
    vegetation = osm_dir / "vegetation.gpkg"
    bare_land = osm_dir / "bare_land.gpkg"
    water = osm_dir / "water.gpkg"
    reference_raster = sar_dir / f"{folder_name}_S1_VV_VH.tif"

    for path, label in (
        (buildings, "buildings"),
        (roads, "roads"),
        (vegetation, "vegetation"),
        (bare_land, "bare land"),
        (water, "water"),
        (reference_raster, "reference raster"),
        (args.road_config, "road configuration"),
        (args.rasterization_config, "rasterization configuration"),
    ):
        require_file(path, label)

    roads_buffered = derived_dir / "roads_buffered.gpkg"
    semantic_mask = masks_dir / f"{city_id}_label_mask.tif"
    validity_mask = masks_dir / f"{city_id}_validity_mask.tif"

    rasterization_report = (
        Path("metadata/rasterization")
        / f"{city_id}_rasterization_report.json"
    )
    qa_report = (
        Path("metadata/raster_qa")
        / f"{city_id}_label_raster_qa.json"
    )
    grid_report = (
        Path("metadata/raster_grid")
        / f"{city_id}_reference_grid.json"
    )
    pipeline_report = (
        Path("metadata/city_label_pipeline")
        / f"{city_id}_pipeline_summary.json"
    )

    for directory in (
        derived_dir,
        masks_dir,
        grid_report.parent,
        pipeline_report.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if args.force:
        for path in (
            roads_buffered,
            semantic_mask,
            validity_mask,
            rasterization_report,
            qa_report,
            grid_report,
            pipeline_report,
        ):
            if path.exists():
                path.unlink()

    started_at = datetime.now(timezone.utc).isoformat()
    overall_started = time.perf_counter()
    stage_durations: dict[str, float] = {}

    try:
        grid_metadata = validate_reference_raster(
            reference_raster,
            city["utm_epsg"],
        )
        with grid_report.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "city_id": city_id,
                    "expected_epsg": city["utm_epsg"],
                    "reference_raster": str(reference_raster),
                    "reference_grid": grid_metadata,
                },
                file,
                indent=2,
            )

        if roads_buffered.exists() and not args.force:
            print(f"Skipping existing buffered roads: {roads_buffered}")
            stage_durations["road_buffering"] = 0.0
        else:
            stage_durations["road_buffering"] = run_stage(
                [
                    sys.executable,
                    "-m",
                    "src.labels.buffer_roads",
                    "--roads",
                    str(roads),
                    "--layer",
                    "roads",
                    "--config",
                    str(args.road_config),
                    "--city-id",
                    city_id,
                    "--output",
                    str(roads_buffered),
                ],
                "Road polygon generation",
            )

        require_file(roads_buffered, "buffered roads")

        stage_durations["rasterization"] = run_stage(
            [
                sys.executable,
                "-m",
                "src.labels.rasterize_labels",
                "--city-id",
                city_id,
                "--reference-raster",
                str(reference_raster),
                "--config",
                str(args.rasterization_config),
                "--buildings",
                str(buildings),
                "--buildings-layer",
                "buildings",
                "--roads",
                str(roads_buffered),
                "--roads-layer",
                "roads_buffered",
                "--vegetation",
                str(vegetation),
                "--vegetation-layer",
                "vegetation",
                "--bare-land",
                str(bare_land),
                "--bare-land-layer",
                "bare_land",
                "--water",
                str(water),
                "--water-layer",
                "water",
                "--output-dir",
                str(masks_dir),
            ],
            "Five-class rasterization",
        )

        for path, label in (
            (semantic_mask, "semantic mask"),
            (validity_mask, "validity mask"),
            (rasterization_report, "rasterization report"),
        ):
            require_file(path, label)

        stage_durations["raster_qa"] = run_stage(
            [
                sys.executable,
                "-m",
                "src.quality_control.validate_label_rasters",
                "--city-id",
                city_id,
                "--reference-raster",
                str(reference_raster),
                "--semantic-mask",
                str(semantic_mask),
                "--validity-mask",
                str(validity_mask),
                "--rasterization-report",
                str(rasterization_report),
            ],
            "Independent raster QA",
        )

        require_file(qa_report, "raster QA report")
        with qa_report.open("r", encoding="utf-8") as file:
            qa_data = json.load(file)

        if qa_data.get("qa_status") != "PASS":
            raise RuntimeError(f"Raster QA failed for {city_id}")

        status = "PASS"
        error = ""

    except Exception as exc:
        status = "FAIL"
        error = str(exc)

        summary = {
            "city_id": city_id,
            "city_name": city_name,
            "folder_name": folder_name,
            "expected_epsg": city["utm_epsg"],
            "split": city["split"],
            "status": status,
            "error": error,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - overall_started,
            "stage_durations_seconds": stage_durations,
        }
        with pipeline_report.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        raise

    summary = {
        "city_id": city_id,
        "city_name": city_name,
        "folder_name": folder_name,
        "expected_epsg": city["utm_epsg"],
        "split": city["split"],
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - overall_started,
        "stage_durations_seconds": stage_durations,
        "outputs": {
            "roads_buffered": str(roads_buffered),
            "reference_grid_report": str(grid_report),
            "semantic_mask": str(semantic_mask),
            "validity_mask": str(validity_mask),
            "rasterization_report": str(rasterization_report),
            "raster_qa_report": str(qa_report),
        },
    }
    with pipeline_report.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nCity label pipeline result")
    print("--------------------------")
    print(f"City: {city_id} {city_name}")
    print("Status: PASS")
    print(f"Summary: {pipeline_report}")


if __name__ == "__main__":
    main()
