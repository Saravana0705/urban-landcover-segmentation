"""Create an auditable 20-city V3 registry and an exact-grid GEE export script."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def norm_grid(report: dict) -> dict:
    transform = [float(x) for x in report["transform"]]
    width, height = int(report["width_pixels"]), int(report["height_pixels"])
    left, top = transform[2], transform[5]
    right = left + transform[0] * width
    bottom = top + transform[4] * height
    return {
        "crs": str(report["crs"]), "width": width, "height": height,
        "transform": transform, "bounds": [left, bottom, right, top],
        "reference_raster": str(report["reference_raster"]),
        "osm_semantic": str(report["semantic_mask"]),
        "osm_validity": str(report["validity_mask"]),
    }


def signature(grid: dict) -> tuple:
    return (grid["crs"], grid["width"], grid["height"], tuple(grid["transform"]))


def gee_script(registry: dict, drive_folder: str) -> str:
    cities = []
    for city in registry["cities"]:
        cities.append({
            "cityId": city["city_id"], "cityName": city["city_name"],
            "split": city["split"], "acquisitionUtc": city["acquisition_datetime_utc"],
            "startDate": city["dynamic_world_start_date"],
            "endDateExclusive": city["dynamic_world_end_date_exclusive"],
            "crs": city["grid"]["crs"], "transform": city["grid"]["transform"],
            "bounds": city["grid"]["bounds"],
        })
    city_json = json.dumps(cities, indent=2)
    return f"""/** Generated Dataset V3 Dynamic World evidence exports. Additive only. */
var DW_COLLECTION = 'GOOGLE/DYNAMICWORLD/V1';
var DRIVE_FOLDER = {json.dumps(drive_folder)};
// Pilot exports already passed QA; remove an ID only if that pilot file is unavailable.
var SKIP_CITY_IDS = ['DE03', 'DE14'];
var PROBABILITY_BANDS = ['water','trees','grass','flooded_vegetation','crops','shrub_and_scrub','built','bare','snow_and_ice'];
var CITIES = {city_json};
function prefixed(prefix) {{ return PROBABILITY_BANDS.map(function(n) {{ return prefix + n; }}); }}
function exportCity(city) {{
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var collection = ee.ImageCollection(DW_COLLECTION).filterBounds(region).filterDate(city.startDate, city.endDateExclusive);
  var labels = collection.select('label');
  var probs = collection.select(PROBABILITY_BANDS);
  var mode = labels.reduce(ee.Reducer.mode()).rename('label_mode');
  var mean = probs.mean().rename(prefixed('p_mean_'));
  var median = probs.median().rename(prefixed('p_median_'));
  var count = labels.count().rename('observation_count');
  var agreement = labels.map(function(image) {{ return image.eq(mode).rename('agreement'); }}).sum().divide(count).rename('temporal_agreement');
  var output = mode.toFloat().addBands(mean.toFloat()).addBands(median.toFloat()).addBands(count.toFloat()).addBands(agreement.toFloat()).clip(region).set({{
    dataset_version: 'v3-batch-0.1', city_id: city.cityId, city_name: city.cityName,
    split: city.split, sentinel1_acquisition_utc: city.acquisitionUtc,
    temporal_start: city.startDate, temporal_end_exclusive: city.endDateExclusive,
    source_collection: DW_COLLECTION, source_image_count: collection.size()
  }});
  print(city.cityId + ' source image count', collection.size());
  Export.image.toDrive({{image: output, description: city.cityId + '_dynamic_world_v3_2025', folder: DRIVE_FOLDER,
    fileNamePrefix: city.cityId + '_dynamic_world_v3_2025', region: region, crs: city.crs,
    crsTransform: city.transform, maxPixels: 100000000, fileFormat: 'GeoTIFF', formatOptions: {{cloudOptimized: true}}}});
}}
CITIES.filter(function(city) {{ return SKIP_CITY_IDS.indexOf(city.cityId) === -1; }}).forEach(exportCity);
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", type=Path, default=Path("config/cities.csv"))
    parser.add_argument("--selected-scenes", type=Path, default=Path("metadata/selected_scenes.csv"))
    parser.add_argument("--rasterization-reports", type=Path, default=Path("metadata/rasterization"))
    parser.add_argument("--registry", type=Path, default=Path("config/dataset_v3_city_registry.json"))
    parser.add_argument("--gee-script", type=Path, default=Path("scripts/gee/export_dynamic_world_v3_all_cities.js"))
    parser.add_argument("--drive-folder", default="urban_landcover_v3_batch")
    args = parser.parse_args()

    for path in (args.cities, args.selected_scenes, args.rasterization_reports):
        if not path.exists(): parser.error(f"Required input not found: {path}")
    collisions = [p for p in (args.registry, args.gee_script) if p.exists()]
    if collisions: raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, collisions)))

    cities = {row["city_id"]: row for row in read_csv(args.cities)}
    scenes = {row["city_id"]: row for row in read_csv(args.selected_scenes)}
    reports = defaultdict(list)
    for path in args.rasterization_reports.rglob("*.json"):
        try: report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError): continue
        required = {"city_id", "transform", "width_pixels", "height_pixels", "crs",
                    "reference_raster", "semantic_mask", "validity_mask"}
        if required.issubset(report):
            reports[report["city_id"]].append((path, norm_grid(report)))
    if set(cities) != set(scenes) or set(cities) != set(reports):
        raise ValueError(f"City mismatch: cities={sorted(cities)} scenes={sorted(scenes)} reports={sorted(reports)}")

    entries = []
    for city_id in sorted(cities):
        row, scene = cities[city_id], scenes[city_id]
        if row["split"] != scene["split"]: raise ValueError(f"Split mismatch for {city_id}")
        grids = reports[city_id]
        if len({signature(grid) for _, grid in grids}) != 1:
            raise ValueError(f"Duplicate rasterization reports disagree for {city_id}")
        acquired = datetime.fromisoformat(scene["acquisition_datetime"].replace("Z", "+00:00"))
        entries.append({
            "city_id": city_id, "city_name": row["city_name"], "split": row["split"],
            "acquisition_datetime_utc": scene["acquisition_datetime"],
            "dynamic_world_start_date": (acquired - timedelta(days=30)).date().isoformat(),
            "dynamic_world_end_date_exclusive": (acquired + timedelta(days=31)).date().isoformat(),
            "grid": grids[0][1], "report_candidates": [str(path) for path, _ in grids],
        })
    counts = {split: sum(x["split"] == split for x in entries) for split in ("train", "val", "test")}
    if counts != {"train": 14, "val": 3, "test": 3}: raise ValueError(f"Unexpected split counts: {counts}")
    registry = {"schema_version": "dataset-v3-city-registry-0.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_files": {"cities": str(args.cities), "selected_scenes": str(args.selected_scenes), "rasterization_reports": str(args.rasterization_reports)},
                "split_counts": counts, "cities": entries}
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.gee_script.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    args.gee_script.write_text(gee_script(registry, args.drive_folder), encoding="utf-8")
    print(json.dumps({"status": "PASS", "cities": len(entries), "split_counts": counts,
                      "registry": str(args.registry), "gee_script": str(args.gee_script)}, indent=2))


if __name__ == "__main__": main()
