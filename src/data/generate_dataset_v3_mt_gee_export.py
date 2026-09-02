"""Generate the full 20-city Earth Engine Dataset V3-MT exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


QUARTERS = [
    {"name": "Q1", "start": "2025-01-01", "end": "2025-04-01"},
    {"name": "Q2", "start": "2025-04-01", "end": "2025-07-01"},
    {"name": "Q3", "start": "2025-07-01", "end": "2025-10-01"},
    {"name": "Q4", "start": "2025-10-01", "end": "2026-01-01"},
]


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
            "scripts/gee/export_sentinel1_multitemporal_v3_mt_all_cities.js"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.registry.is_file():
        raise FileNotFoundError(f"City registry not found: {args.registry}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite exporter: {args.output}")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source_cities = registry["cities"]
    if len(source_cities) != 20:
        raise RuntimeError(f"Expected 20 cities, found {len(source_cities)}")

    cities = []
    for city in source_cities:
        grid = city["grid"]
        if (grid["width"], grid["height"]) != (3000, 3000):
            raise RuntimeError(f"Unexpected grid for {city['city_id']}")
        cities.append({
            "cityId": city["city_id"],
            "cityName": city["city_name"],
            "split": city["split"],
            "anchorTime": city["acquisition_datetime_utc"],
            "crs": grid["crs"],
            "crsTransform": grid["transform"],
            "bounds": grid["bounds"],
        })

    run_ids = [
        city["cityId"]
        for city in cities
        if city["cityId"] not in {"DE01", "DE15"}
    ]

    template_path = Path(__file__).with_name("dataset_v3_mt_export_template.js.txt")
    if not template_path.is_file():
        raise FileNotFoundError(f"Exporter template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    output = (
        template.replace("__QUARTERS__", json.dumps(QUARTERS, indent=2))
        .replace("__CITIES__", json.dumps(cities, indent=2))
        .replace("__RUN_CITY_IDS__", json.dumps(run_ids, indent=2))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "PASS",
        "cities_in_audit": len(cities),
        "image_exports": len(run_ids),
        "skipped_completed": ["DE01", "DE15"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
