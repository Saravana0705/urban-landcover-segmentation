"""Generate the two-city Dataset V3-MT-D2 cross-orbit pilot exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PILOT_CITY_IDS = ("DE01", "DE15")
EXPECTED_CHANNELS = tuple(
    f"{polarisation}_{pass_abbreviation}_Q{quarter}"
    for pass_abbreviation in ("ASC", "DESC")
    for quarter in range(1, 5)
    for polarisation in ("VV", "VH")
)
QUARTERS = (
    {"name": "Q1", "start": "2025-01-01", "end": "2025-04-01"},
    {"name": "Q2", "start": "2025-04-01", "end": "2025-07-01"},
    {"name": "Q3", "start": "2025-07-01", "end": "2025-10-01"},
    {"name": "Q4", "start": "2025-10-01", "end": "2026-01-01"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--acquisition-qa",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d2/acquisition/"
            "dataset_v3_mt_d2_acquisition_audit_qa.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scripts/gee/export_sentinel1_cross_orbit_v3_mt_d2_pilot.js"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pilot_contract(
    registry_path: Path, acquisition_qa_path: Path
) -> list[dict[str, Any]]:
    registry = require_json(registry_path, "city registry")
    acquisition_qa = require_json(acquisition_qa_path, "D2 acquisition QA")
    if acquisition_qa.get("status") != "PASS":
        raise RuntimeError("D2 acquisition QA has not passed")
    if acquisition_qa.get("image_exports_authorized") is not True:
        raise RuntimeError("D2 acquisition QA does not authorize image exports")
    if acquisition_qa.get("coverage_unit") != "same_day_slice_union":
        raise RuntimeError("D2 audit is not the mosaic-aware v0.2 contract")
    if tuple(acquisition_qa.get("expected_channel_order", ())) != EXPECTED_CHANNELS:
        raise RuntimeError("D2 acquisition QA channel order is unexpected")

    registry_cities = {city["city_id"]: city for city in registry["cities"]}
    qa_cities = {city["city_id"]: city for city in acquisition_qa["cities"]}
    output: list[dict[str, Any]] = []
    for city_id in PILOT_CITY_IDS:
        if city_id not in registry_cities or city_id not in qa_cities:
            raise RuntimeError(f"Missing pilot city contract: {city_id}")
        city = registry_cities[city_id]
        qa_city = qa_cities[city_id]
        if qa_city.get("status") != "PASS":
            raise RuntimeError(f"Pilot city acquisition QA failed: {city_id}")
        selected = qa_city["selected_orbits"]
        for pass_name in ("ASCENDING", "DESCENDING"):
            record = selected.get(pass_name)
            if record is None:
                raise RuntimeError(f"{city_id} has no selected {pass_name} orbit")
            if record["minimum_quarterly_observation_count"] < 2:
                raise RuntimeError(f"{city_id} {pass_name} has inadequate coverage")
        grid = city["grid"]
        if (grid["width"], grid["height"]) != (3000, 3000):
            raise RuntimeError(f"Unexpected frozen grid for {city_id}")
        output.append({
            "cityId": city_id,
            "cityName": city["city_name"],
            "split": city["split"],
            "crs": grid["crs"],
            "crsTransform": grid["transform"],
            "bounds": grid["bounds"],
            "ascendingOrbit": selected["ASCENDING"]["relative_orbit"],
            "descendingOrbit": selected["DESCENDING"]["relative_orbit"],
        })
    return output


def render(cities: list[dict[str, Any]]) -> str:
    return f"""// Dataset V3-MT-D2 16-channel cross-orbit pilot exporter.
// Generated only after the mosaic-aware acquisition audit passed.
var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_PILOT_2025';
var NODATA = -9999;
var GRID_INSET_METRES = 1;
var MINIMUM_COVERAGE_FRACTION = 0.98;

var QUARTERS = {json.dumps(QUARTERS, indent=2)};
var CITIES = {json.dumps(cities, indent=2)};

function baseCollection(region) {{
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filterDate('2025-01-01', '2026-01-01')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .map(function(image) {{
      return image.set('acquisition_date', image.date().format('YYYY-MM-dd'));
    }});
}}

function dbToLinear(image) {{
  return ee.Image(10).pow(image.divide(10))
    .copyProperties(image, image.propertyNames());
}}

function qualifyingDailyMosaics(collection, region, regionArea) {{
  var dates = ee.List(collection.aggregate_array(
    'acquisition_date')).distinct().sort();
  var images = dates.map(function(dateText) {{
    dateText = ee.String(dateText);
    var sameDaySlices = collection.filter(ee.Filter.eq(
      'acquisition_date', dateText));
    var coverage = sameDaySlices.geometry(ee.ErrorMargin(10))
      .intersection(region, ee.ErrorMargin(10))
      .area(ee.ErrorMargin(10))
      .divide(regionArea);
    var date = ee.Date.parse('YYYY-MM-dd', dateText);
    return sameDaySlices.map(dbToLinear).mosaic()
      .set('acquisition_date', dateText)
      .set('city_coverage_fraction', coverage)
      .set('system:time_start', date.millis());
  }});
  return ee.ImageCollection.fromImages(images)
    .filter(ee.Filter.gte(
      'city_coverage_fraction', MINIMUM_COVERAGE_FRACTION));
}}

function passImages(city, region, regionArea, passName, relativeOrbit,
                    abbreviation) {{
  var matched = baseCollection(region)
    .filter(ee.Filter.eq('orbitProperties_pass', passName))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', relativeOrbit));
  var daily = qualifyingDailyMosaics(matched, region, regionArea);
  return QUARTERS.map(function(quarter) {{
    var quarterCollection = daily.filterDate(quarter.start, quarter.end);
    print(city.cityId + ' ' + abbreviation + ' ' + quarter.name +
          ' qualifying acquisition count:', quarterCollection.size());
    return quarterCollection.median().rename([
      'VV_' + abbreviation + '_' + quarter.name,
      'VH_' + abbreviation + '_' + quarter.name
    ]);
  }});
}}

function exportCity(city) {{
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var exportRegion = ee.Geometry.Rectangle([
    city.bounds[0] + GRID_INSET_METRES,
    city.bounds[1] + GRID_INSET_METRES,
    city.bounds[2] - GRID_INSET_METRES,
    city.bounds[3] - GRID_INSET_METRES
  ], city.crs, false);
  var regionArea = region.area(ee.ErrorMargin(10));
  var ascending = passImages(
    city, region, regionArea, 'ASCENDING', city.ascendingOrbit, 'ASC');
  var descending = passImages(
    city, region, regionArea, 'DESCENDING', city.descendingOrbit, 'DESC');
  var output = ee.Image.cat(ascending.concat(descending))
    .toFloat()
    .unmask(NODATA, false)
    .clip(region);
  var description = city.cityId + '_' + city.cityName +
    '_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025';
  print(city.cityId + ' output bands:', output.bandNames());
  Export.image.toDrive({{
    image: output,
    description: description,
    folder: EXPORT_FOLDER,
    fileNamePrefix: description,
    region: exportRegion,
    crs: city.crs,
    crsTransform: city.crsTransform,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {{cloudOptimized: true, noData: NODATA}}
  }});
}}

CITIES.forEach(exportCity);
"""


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite exporter: {args.output}")
    cities = load_pilot_contract(args.registry, args.acquisition_qa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(cities), encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "GENERATED",
        "pilot_city_ids": list(PILOT_CITY_IDS),
        "image_exports_created": len(cities),
        "selected_orbits": {
            city["cityId"]: {
                "ASCENDING": city["ascendingOrbit"],
                "DESCENDING": city["descendingOrbit"],
            }
            for city in cities
        },
        "expected_channel_order": list(EXPECTED_CHANNELS),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
