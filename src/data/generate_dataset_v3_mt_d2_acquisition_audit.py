"""Generate the Earth Engine acquisition audit for Dataset V3-MT-D2.

The audit enumerates Sentinel-1 relative orbits for ASCENDING and DESCENDING
passes over each frozen Dataset V3 city. Coverage is measured for the merged
footprint of all GRD slices belonging to the same orbit and UTC acquisition
date. It deliberately creates only one table export and no image exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CITY_IDS = tuple(f"DE{number:02d}" for number in range(1, 21))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scripts/gee/"
            "audit_sentinel1_cross_orbit_v3_mt_d2_all_cities.js"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cities(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"City registry not found or empty: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    cities = sorted(registry["cities"], key=lambda item: item["city_id"])
    city_ids = tuple(city["city_id"] for city in cities)
    if city_ids != EXPECTED_CITY_IDS:
        raise RuntimeError(
            f"Expected frozen cities {EXPECTED_CITY_IDS}; found {city_ids}"
        )
    split_counts = {
        split: sum(city["split"] == split for city in cities)
        for split in ("train", "val", "test")
    }
    if split_counts != {"train": 14, "val": 3, "test": 3}:
        raise RuntimeError(f"Unexpected frozen split counts: {split_counts}")
    return cities


def js_literal(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True)


def render(cities: list[dict[str, Any]]) -> str:
    gee_cities = [
        {
            "cityId": city["city_id"],
            "cityName": city["city_name"],
            "split": city["split"],
            "crs": city["grid"]["crs"],
            "bounds": city["grid"]["bounds"],
        }
        for city in cities
    ]
    return f"""// Dataset V3-MT-D2 cross-orbit acquisition audit v0.2.
// Generated from the frozen Dataset V3 city registry.
// Coverage is evaluated after merging same-day GRD slice footprints.
// AUDIT ONLY: this script creates one CSV task and no image export tasks.
var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_AUDIT_2025';
var MINIMUM_SCENE_COVERAGE_FRACTION = 0.98;
var YEAR_START = '2025-01-01';
var YEAR_END = '2026-01-01';

var PASSES = ['ASCENDING', 'DESCENDING'];
var QUARTERS = [
  {{name: 'Q1', start: '2025-01-01', end: '2025-04-01'}},
  {{name: 'Q2', start: '2025-04-01', end: '2025-07-01'}},
  {{name: 'Q3', start: '2025-07-01', end: '2025-10-01'}},
  {{name: 'Q4', start: '2025-10-01', end: '2026-01-01'}}
];

var CITIES = {js_literal(gee_cities)};

function baseCollection(region) {{
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filterDate(YEAR_START, YEAR_END)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
}}

function candidateFeatures(city, passName) {{
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var regionArea = region.area(ee.ErrorMargin(10));
  var collection = baseCollection(region)
    .filter(ee.Filter.eq('orbitProperties_pass', passName))
    .map(function(image) {{
      return image.set(
        'acquisition_date', image.date().format('YYYY-MM-dd'));
    }});

  var relativeOrbits = ee.List(collection.aggregate_array(
    'relativeOrbitNumber_start')).distinct().sort();

  return ee.FeatureCollection(relativeOrbits.map(function(relativeOrbit) {{
    relativeOrbit = ee.Number(relativeOrbit);
    var matched = collection.filter(ee.Filter.eq(
      'relativeOrbitNumber_start', relativeOrbit));
    var acquisitionDates = ee.List(
      matched.aggregate_array('acquisition_date')).distinct().sort();
    var acquisitions = ee.FeatureCollection(acquisitionDates.map(
      function(acquisitionDate) {{
        acquisitionDate = ee.String(acquisitionDate);
        var sameDaySlices = matched.filter(ee.Filter.eq(
          'acquisition_date', acquisitionDate));
        var mergedFootprint = sameDaySlices.geometry(ee.ErrorMargin(10));
        var overlap = mergedFootprint
          .intersection(region, ee.ErrorMargin(10))
          .area(ee.ErrorMargin(10));
        var date = ee.Date.parse('YYYY-MM-dd', acquisitionDate);
        return ee.Feature(null, {{
          acquisition_date: acquisitionDate,
          'system:time_start': date.millis(),
          slice_count: sameDaySlices.size(),
          city_coverage_fraction: overlap.divide(regionArea)
        }});
      }}));
    var qualifying = acquisitions.filter(ee.Filter.gte(
      'city_coverage_fraction', MINIMUM_SCENE_COVERAGE_FRACTION));
    var qualifyingCount = qualifying.size();
    var properties = {{
      city_id: city.cityId,
      city_name: city.cityName,
      split: city.split,
      orbit_pass: passName,
      relative_orbit: relativeOrbit,
      coverage_unit: 'same_day_slice_union',
      raw_scene_count: matched.size(),
      raw_acquisition_date_count: acquisitions.size(),
      annual_observation_count: qualifyingCount,
      minimum_scene_coverage_fraction:
        ee.Algorithms.If(
          qualifyingCount.gt(0),
          qualifying.aggregate_min('city_coverage_fraction'),
          0),
      mean_scene_coverage_fraction:
        ee.Algorithms.If(
          qualifyingCount.gt(0),
          qualifying.aggregate_mean('city_coverage_fraction'),
          0)
    }};
    QUARTERS.forEach(function(quarter) {{
      properties[quarter.name.toLowerCase() + '_observation_count'] =
        qualifying.filterDate(quarter.start, quarter.end).size();
    }});
    return ee.Feature(null, properties);
  }}));
}}

var acquisitionAudit = ee.FeatureCollection([]);
CITIES.forEach(function(city) {{
  PASSES.forEach(function(passName) {{
    acquisitionAudit = acquisitionAudit.merge(
      candidateFeatures(city, passName));
  }});
}});

print('Dataset V3-MT-D2 cross-orbit candidates:', acquisitionAudit);
print('Candidate row count:', acquisitionAudit.size());

Export.table.toDrive({{
  collection: acquisitionAudit,
  description: 'dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025',
  folder: EXPORT_FOLDER,
  fileNamePrefix:
    'dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025',
  fileFormat: 'CSV'
}});
"""


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    cities = load_cities(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(cities), encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "GENERATED",
        "city_count": len(cities),
        "image_exports_created": 0,
        "table_exports_created": 1,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
