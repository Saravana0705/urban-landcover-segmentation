"""Validate the two full-resolution Dataset V3-MT-D2 pilot rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine


EXPECTED_BANDS = tuple(
    f"{polarisation}_{pass_abbreviation}_Q{quarter}"
    for pass_abbreviation in ("ASC", "DESC")
    for quarter in range(1, 5)
    for polarisation in ("VV", "VH")
)
PILOT_CITY_IDS = ("DE01", "DE15")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        "--input-root",
        type=Path,
        default=Path("data/raw/sar_multitemporal_v3_mt_d2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d2/pilot/"
            "dataset_v3_mt_d2_pilot_qa.json"
        ),
    )
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")


def close_transform(actual: Affine, expected: Affine) -> bool:
    return bool(np.allclose(tuple(actual), tuple(expected), rtol=0, atol=1e-6))


def validate_city(city: dict[str, Any], root: Path, minimum: float) -> dict[str, Any]:
    city_id = city["city_id"]
    city_name = city["city_name"]
    path = (
        root / f"{city_id}_{city_name}" /
        f"{city_id}_{city_name}_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025.tif"
    )
    reference = Path(city["grid"]["reference_raster"])
    require_file(path, f"{city_id} D2 pilot raster")
    require_file(reference, f"{city_id} frozen reference SAR raster")

    grid = city["grid"]
    expected_transform = Affine(*grid["transform"])
    failures: list[str] = []
    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.crs.to_string() != grid["crs"]:
            failures.append(f"CRS is {dataset.crs}; expected {grid['crs']}")
        if (dataset.width, dataset.height) != (grid["width"], grid["height"]):
            failures.append(
                f"shape is {dataset.width}x{dataset.height}; expected "
                f"{grid['width']}x{grid['height']}"
            )
        if not close_transform(dataset.transform, expected_transform):
            failures.append("affine transform differs from frozen grid")
        if dataset.count != len(EXPECTED_BANDS):
            failures.append(
                f"band count is {dataset.count}; expected {len(EXPECTED_BANDS)}"
            )
        if any(dtype != "float32" for dtype in dataset.dtypes):
            failures.append(f"dtypes are {dataset.dtypes}; expected float32")
        if dataset.nodata != -9999:
            failures.append(f"NoData is {dataset.nodata}; expected -9999")

        descriptions = tuple(value or "" for value in dataset.descriptions)
        if any(descriptions) and descriptions != EXPECTED_BANDS:
            failures.append(
                f"band descriptions are {descriptions}; expected {EXPECTED_BANDS}"
            )

        band_stats: list[dict[str, Any]] = []
        for index, band_name in enumerate(
            EXPECTED_BANDS[: min(dataset.count, len(EXPECTED_BANDS))], start=1
        ):
            data = dataset.read(index, masked=True)
            values = data.compressed().astype(np.float64, copy=False)
            finite = values[np.isfinite(values)]
            positive = finite[finite > 0]
            valid_fraction = positive.size / (dataset.width * dataset.height)
            if finite.size != values.size:
                failures.append(f"{band_name} contains non-finite valid pixels")
            if valid_fraction < minimum:
                failures.append(
                    f"{band_name} positive valid fraction {valid_fraction:.6f} "
                    f"is below {minimum:.6f}"
                )
            band_stats.append({
                "band": band_name,
                "positive_valid_fraction": valid_fraction,
                "minimum_positive": float(positive.min()) if positive.size else None,
                "maximum_positive": float(positive.max()) if positive.size else None,
                "mean_positive": float(positive.mean()) if positive.size else None,
            })

        raster_record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "dtype": list(dataset.dtypes),
            "nodata": dataset.nodata,
            "band_descriptions": list(descriptions),
            "band_statistics": band_stats,
        }

    with rasterio.open(reference) as dataset:
        if dataset.crs is None or dataset.crs.to_string() != grid["crs"]:
            failures.append("frozen reference CRS disagrees with registry")
        if not close_transform(dataset.transform, expected_transform):
            failures.append("frozen reference transform disagrees with registry")
        if (dataset.width, dataset.height) != (grid["width"], grid["height"]):
            failures.append("frozen reference dimensions disagree with registry")

    return {
        "city_id": city_id,
        "city_name": city_name,
        "split": city["split"],
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "raster": raster_record,
    }


def main() -> None:
    args = parse_args()
    require_file(args.registry, "city registry")
    require_file(args.acquisition_qa, "D2 acquisition QA")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite QA report: {args.output}")
    if not 0 < args.minimum_valid_fraction <= 1:
        raise ValueError("minimum-valid-fraction must be in (0, 1]")

    acquisition_qa = json.loads(args.acquisition_qa.read_text(encoding="utf-8"))
    if acquisition_qa.get("status") != "PASS":
        raise RuntimeError("D2 acquisition QA has not passed")
    if acquisition_qa.get("image_exports_authorized") is not True:
        raise RuntimeError("D2 acquisition QA does not authorize image exports")
    if tuple(acquisition_qa.get("expected_channel_order", ())) != EXPECTED_BANDS:
        raise RuntimeError("D2 acquisition QA channel order is unexpected")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    cities = {
        city["city_id"]: city
        for city in registry["cities"]
        if city["city_id"] in PILOT_CITY_IDS
    }
    if set(cities) != set(PILOT_CITY_IDS):
        raise RuntimeError("Registry does not contain exactly DE01 and DE15")
    qa_cities = {city["city_id"]: city for city in acquisition_qa["cities"]}

    results = [
        validate_city(cities[city_id], args.input_root, args.minimum_valid_fraction)
        for city_id in PILOT_CITY_IDS
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    report = {
        "schema_version": "dataset-v3-mt-d2-pilot-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_version": "v3-mt-d2",
        "pilot_city_ids": list(PILOT_CITY_IDS),
        "expected_band_order": list(EXPECTED_BANDS),
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "coverage_unit": acquisition_qa.get("coverage_unit"),
        "selected_orbits": {
            city_id: qa_cities[city_id]["selected_orbits"]
            for city_id in PILOT_CITY_IDS
        },
        "labels_modified": False,
        "split_assignments_modified": False,
        "cities": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "cities": {item["city_id"]: item["status"] for item in results},
        "report": str(args.output),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
