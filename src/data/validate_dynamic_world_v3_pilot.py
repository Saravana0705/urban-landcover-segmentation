"""Validate Dataset V3 Dynamic World pilot exports without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml


PROBABILITY_CLASSES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

EXPECTED_BANDS = (
    ["label_mode"]
    + [f"p_mean_{name}" for name in PROBABILITY_CLASSES]
    + [f"p_median_{name}" for name in PROBABILITY_CLASSES]
    + ["observation_count", "temporal_agreement"]
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_list(left: list[float], right: list[float]) -> bool:
    return bool(np.allclose(left, right, atol=1e-6, rtol=0.0))


def finite_range(array: np.ndarray) -> tuple[float | None, float | None]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def validate_export(
    raster_path: Path,
    city_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    city = config["cities"][city_id]
    failures: list[str] = []

    if not raster_path.exists() or raster_path.stat().st_size == 0:
        raise FileNotFoundError(f"Export not found or empty: {raster_path}")

    with rasterio.open(raster_path) as source:
        transform = list(source.transform)[:6]
        descriptions = list(source.descriptions)

        checks = {
            "width": source.width == int(city["width"]),
            "height": source.height == int(city["height"]),
            "crs": str(source.crs) == str(city["crs"]),
            "transform": close_list(transform, city["transform"]),
            "band_count": source.count == len(EXPECTED_BANDS),
        }

        for name, passed in checks.items():
            if not passed:
                failures.append(f"failed_{name}")

        band_statistics: dict[str, dict[str, float | None]] = {}

        for index, expected_name in enumerate(EXPECTED_BANDS, start=1):
            if index > source.count:
                break

            array = source.read(index, masked=True).filled(np.nan)
            minimum, maximum = finite_range(array)
            band_statistics[expected_name] = {
                "minimum": minimum,
                "maximum": maximum,
                "finite_pixel_count": int(np.isfinite(array).sum()),
            }

            if expected_name == "label_mode":
                if minimum is not None and (minimum < 0 or maximum > 8):
                    failures.append("label_mode_outside_0_8")
            elif expected_name.startswith(("p_mean_", "p_median_")):
                if minimum is not None and (minimum < -1e-6 or maximum > 1.000001):
                    failures.append(f"{expected_name}_outside_0_1")
            elif expected_name == "observation_count":
                if minimum is not None and minimum < 0:
                    failures.append("negative_observation_count")
            elif expected_name == "temporal_agreement":
                if minimum is not None and (minimum < -1e-6 or maximum > 1.000001):
                    failures.append("temporal_agreement_outside_0_1")

    return {
        "dataset_version": config["dataset"]["version"],
        "city_id": city_id,
        "raster_path": str(raster_path),
        "raster_sha256": sha256_file(raster_path),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_status": "PASS" if not failures else "FAIL",
        "failures": sorted(set(failures)),
        "expected_bands": EXPECTED_BANDS,
        "stored_band_descriptions": descriptions,
        "grid_checks": checks,
        "band_statistics": band_statistics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True, choices=["DE03", "DE14"])
    parser.add_argument("--raster", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dataset_v3_pilot.yaml"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("metadata/dataset_v3/pilot/dynamic_world"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    report = validate_export(args.raster, args.city_id, config)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{args.city_id}_dynamic_world_qa.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(json.dumps({
        "city_id": args.city_id,
        "qa_status": report["qa_status"],
        "failures": report["failures"],
        "report": str(report_path),
    }, indent=2))

    if report["qa_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
