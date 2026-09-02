"""Validate all 20 full-resolution Dataset V3-MT-D1 monthly rasters."""

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
    band
    for month in range(1, 13)
    for band in (f"VV_M{month:02d}", f"VH_M{month:02d}")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("data/raw/sar_multitemporal_v3_mt_d1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/source_qa/"
            "dataset_v3_mt_d1_source_qa.json"
        ),
    )
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def transforms_match(actual: Affine, expected: Affine) -> bool:
    return bool(np.allclose(tuple(actual), tuple(expected), rtol=0, atol=1e-6))


def validate_city(city: dict[str, Any], root: Path, minimum: float) -> dict[str, Any]:
    city_id = city["city_id"]
    city_name = city["city_name"]
    path = root / f"{city_id}_{city_name}" / (
        f"{city_id}_{city_name}_S1_MT_D1_MONTHLY_2025.tif"
    )
    failures: list[str] = []
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "city_id": city_id, "city_name": city_name, "split": city["split"],
            "status": "FAIL", "failures": [f"missing or empty: {path}"],
        }

    grid = city["grid"]
    expected_transform = Affine(*grid["transform"])
    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.crs.to_string() != grid["crs"]:
            failures.append(f"CRS {dataset.crs} != {grid['crs']}")
        expected_shape = (grid["width"], grid["height"])
        if (dataset.width, dataset.height) != expected_shape:
            failures.append(
                f"shape {dataset.width}x{dataset.height} != "
                f"{expected_shape[0]}x{expected_shape[1]}"
            )
        if not transforms_match(dataset.transform, expected_transform):
            failures.append("transform does not match frozen city grid")
        if dataset.count != 24:
            failures.append(f"band count {dataset.count} != 24")
        if any(dtype != "float32" for dtype in dataset.dtypes):
            failures.append(f"dtypes {dataset.dtypes} != float32")
        if dataset.nodata != -9999:
            failures.append(f"NoData {dataset.nodata} != -9999")
        descriptions = tuple(value or "" for value in dataset.descriptions)
        if descriptions != EXPECTED_BANDS:
            failures.append(f"band order {descriptions} != {EXPECTED_BANDS}")

        band_statistics = []
        available_bands = min(dataset.count, len(EXPECTED_BANDS))
        for index, name in enumerate(
            EXPECTED_BANDS[:available_bands], start=1
        ):
            values = dataset.read(index, masked=True).compressed()
            positive = values[np.isfinite(values) & (values > 0)]
            fraction = positive.size / (dataset.width * dataset.height)
            if fraction < minimum:
                failures.append(
                    f"{name} positive coverage {fraction:.6f} < {minimum:.6f}"
                )
            band_statistics.append({
                "band": name,
                "positive_valid_fraction": fraction,
                "minimum_positive": float(positive.min()) if positive.size else None,
                "maximum_positive": float(positive.max()) if positive.size else None,
                "mean_positive": float(positive.mean()) if positive.size else None,
            })
        raster = {
            "path": str(path), "sha256": sha256_file(path),
            "dtype": list(dataset.dtypes), "nodata": dataset.nodata,
            "band_descriptions": list(descriptions),
            "band_statistics": band_statistics,
        }
    return {
        "city_id": city_id, "city_name": city_name, "split": city["split"],
        "status": "PASS" if not failures else "FAIL", "failures": failures,
        "raster": raster,
    }


def main() -> None:
    args = parse_args()
    if not args.registry.is_file():
        raise FileNotFoundError(f"Registry not found: {args.registry}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    cities = sorted(registry["cities"], key=lambda value: value["city_id"])
    if len(cities) != 20:
        raise RuntimeError(f"Expected 20 cities, found {len(cities)}")
    results = [validate_city(city, args.input_root, args.minimum_valid_fraction)
               for city in cities]
    status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    report = {
        "schema_version": "dataset-v3-mt-d1-source-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "city_count": len(cities),
        "expected_band_order": list(EXPECTED_BANDS),
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "labels_modified": False, "split_assignments_modified": False,
        "cities": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "passed": sum(row["status"] == "PASS" for row in results),
        "failed": sum(row["status"] == "FAIL" for row in results),
        "report": str(args.output),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
