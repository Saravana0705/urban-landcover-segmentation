"""Generate the remaining 18 Dataset V3-MT-D2 image export tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.generate_dataset_v3_mt_d2_pilot_export import (
    EXPECTED_CHANNELS,
    PILOT_CITY_IDS,
    render,
)


EXPECTED_CITY_IDS = tuple(f"DE{number:02d}" for number in range(1, 21))


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
        "--pilot-qa",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d2/pilot/"
            "dataset_v3_mt_d2_pilot_qa.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scripts/gee/export_sentinel1_cross_orbit_v3_mt_d2_all_cities.js"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_remaining_cities(args: argparse.Namespace) -> list[dict[str, Any]]:
    registry = read_json(args.registry, "city registry")
    acquisition = read_json(args.acquisition_qa, "D2 acquisition QA")
    pilot = read_json(args.pilot_qa, "D2 pilot QA")

    if acquisition.get("status") != "PASS":
        raise RuntimeError("D2 acquisition QA has not passed")
    if acquisition.get("image_exports_authorized") is not True:
        raise RuntimeError("D2 acquisition QA does not authorize exports")
    if acquisition.get("coverage_unit") != "same_day_slice_union":
        raise RuntimeError("D2 acquisition QA is not mosaic-aware")
    if tuple(acquisition.get("expected_channel_order", ())) != EXPECTED_CHANNELS:
        raise RuntimeError("D2 acquisition channel contract is unexpected")
    if pilot.get("status") != "PASS":
        raise RuntimeError("D2 pilot QA has not passed")
    if tuple(pilot.get("expected_band_order", ())) != EXPECTED_CHANNELS:
        raise RuntimeError("D2 pilot band contract is unexpected")
    if tuple(pilot.get("pilot_city_ids", ())) != PILOT_CITY_IDS:
        raise RuntimeError("D2 pilot did not validate exactly DE01 and DE15")

    source_cities = sorted(registry["cities"], key=lambda item: item["city_id"])
    city_ids = tuple(city["city_id"] for city in source_cities)
    if city_ids != EXPECTED_CITY_IDS:
        raise RuntimeError(f"Expected DE01-DE20 registry; found {city_ids}")
    split_counts = {
        split: sum(city["split"] == split for city in source_cities)
        for split in ("train", "val", "test")
    }
    if split_counts != {"train": 14, "val": 3, "test": 3}:
        raise RuntimeError(f"Unexpected split counts: {split_counts}")

    qa_cities = {city["city_id"]: city for city in acquisition["cities"]}
    output: list[dict[str, Any]] = []
    for city in source_cities:
        city_id = city["city_id"]
        qa_city = qa_cities.get(city_id)
        if qa_city is None or qa_city.get("status") != "PASS":
            raise RuntimeError(f"Missing passed acquisition contract for {city_id}")
        selected = qa_city["selected_orbits"]
        for pass_name in ("ASCENDING", "DESCENDING"):
            record = selected.get(pass_name)
            if record is None or record["minimum_quarterly_observation_count"] < 2:
                raise RuntimeError(f"Inadequate selected {pass_name} orbit for {city_id}")
        grid = city["grid"]
        if (grid["width"], grid["height"]) != (3000, 3000):
            raise RuntimeError(f"Unexpected frozen grid for {city_id}")
        if city_id not in PILOT_CITY_IDS:
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
    if len(output) != 18:
        raise RuntimeError(f"Expected 18 remaining cities; found {len(output)}")
    return output


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite exporter: {args.output}")
    cities = load_remaining_cities(args)
    script = render(cities)
    pilot_header = "// Dataset V3-MT-D2 16-channel cross-orbit pilot exporter."
    pilot_folder = "var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_PILOT_2025';"
    if pilot_header not in script or pilot_folder not in script:
        raise RuntimeError("Shared D2 exporter template markers have changed")
    script = script.replace(
        pilot_header,
        "// Dataset V3-MT-D2 16-channel remaining-city exporter.",
        1,
    ).replace(
        pilot_folder,
        "var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_ALL_CITIES_2025';",
        1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "GENERATED",
        "cities_embedded": len(cities),
        "image_exports_created": len(cities),
        "skipped_completed_pilot_cities": list(PILOT_CITY_IDS),
        "run_city_ids": [city["cityId"] for city in cities],
        "expected_channel_order": list(EXPECTED_CHANNELS),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
