"""Validate all 20 full-resolution Dataset V3-MT-D2 source rasters."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.quality_control.validate_dataset_v3_mt_d2_pilot import (
    EXPECTED_BANDS,
    PILOT_CITY_IDS,
    validate_city,
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
        "--input-root",
        type=Path,
        default=Path("data/raw/sar_multitemporal_v3_mt_d2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d2/source_qa/"
            "dataset_v3_mt_d2_source_qa.json"
        ),
    )
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if not 0 < args.minimum_valid_fraction <= 1:
        raise ValueError("minimum-valid-fraction must be in (0, 1]")

    registry = read_json(args.registry, "city registry")
    acquisition = read_json(args.acquisition_qa, "D2 acquisition QA")
    pilot = read_json(args.pilot_qa, "D2 pilot QA")
    if acquisition.get("status") != "PASS":
        raise RuntimeError("D2 acquisition QA has not passed")
    if acquisition.get("image_exports_authorized") is not True:
        raise RuntimeError("D2 image exports are not authorized")
    if tuple(acquisition.get("expected_channel_order", ())) != EXPECTED_BANDS:
        raise RuntimeError("D2 acquisition channel contract is unexpected")
    if pilot.get("status") != "PASS":
        raise RuntimeError("D2 pilot QA has not passed")
    if tuple(pilot.get("pilot_city_ids", ())) != PILOT_CITY_IDS:
        raise RuntimeError("D2 pilot city contract is unexpected")
    if tuple(pilot.get("expected_band_order", ())) != EXPECTED_BANDS:
        raise RuntimeError("D2 pilot band contract is unexpected")

    cities = sorted(registry["cities"], key=lambda item: item["city_id"])
    city_ids = tuple(city["city_id"] for city in cities)
    if city_ids != EXPECTED_CITY_IDS:
        raise RuntimeError(f"Expected DE01-DE20 registry; found {city_ids}")
    results = [
        validate_city(city, args.input_root, args.minimum_valid_fraction)
        for city in cities
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    qa_cities = {city["city_id"]: city for city in acquisition["cities"]}
    report = {
        "schema_version": "dataset-v3-mt-d2-source-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_version": "v3-mt-d2",
        "city_count": len(cities),
        "expected_band_order": list(EXPECTED_BANDS),
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "coverage_unit": acquisition.get("coverage_unit"),
        "selected_orbits": {
            city_id: qa_cities[city_id]["selected_orbits"]
            for city_id in EXPECTED_CITY_IDS
        },
        "labels_modified": False,
        "split_assignments_modified": False,
        "cities": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "passed": sum(item["status"] == "PASS" for item in results),
        "failed": sum(item["status"] == "FAIL" for item in results),
        "report": str(args.output),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
