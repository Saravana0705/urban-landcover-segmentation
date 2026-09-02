"""Compute leakage-free 24-band Dataset V3-MT-D1 normalization constants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


EXPECTED_BANDS = tuple(
    band
    for month in range(1, 13)
    for band in (f"VV_M{month:02d}", f"VH_M{month:02d}")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class RunningStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values = values.astype(np.float64, copy=False)
        count = int(values.size)
        mean = float(values.mean())
        differences = values - mean
        m2 = float(np.dot(differences, differences))
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
        else:
            total = self.count + count
            delta = mean - self.mean
            self.m2 += m2 + delta * delta * self.count * count / total
            self.mean += delta * count / total
            self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid_pixel_count": self.count,
            "mean": self.mean,
            "std_population": math.sqrt(self.m2 / self.count),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--source-qa", type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/source_qa/"
            "dataset_v3_mt_d1_source_qa.json"
        ),
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("data/raw/sar_multitemporal_v3_mt_d1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("metadata/dataset_v3_mt_d1/normalization"),
    )
    parser.add_argument("--sample-per-city-per-band", type=int, default=200_000)
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def valid_values(dataset: rasterio.io.DatasetReader, band_index: int) -> np.ndarray:
    values = dataset.read(band_index, masked=True).compressed()
    values = values[np.isfinite(values) & (values > 0)]
    return values.astype(np.float32, copy=False)


def deterministic_sample(values: np.ndarray, size: int, seed: int) -> np.ndarray:
    if values.size <= size:
        return values.copy()
    generator = np.random.default_rng(seed)
    indices = generator.choice(values.size, size=size, replace=False)
    return values[indices]


def quantiles(values: np.ndarray) -> dict[str, float]:
    p01, p50, p99 = np.quantile(values.astype(np.float64), [0.01, 0.50, 0.99])
    return {"p01": float(p01), "p50": float(p50), "p99": float(p99)}


def main() -> None:
    args = parse_args()
    require_file(args.registry, "city registry")
    require_file(args.source_qa, "source QA report")
    if args.sample_per_city_per_band <= 0:
        raise ValueError("sample-per-city-per-band must be positive")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")

    outputs = [
        args.output_dir / "training_normalization.json",
        args.output_dir / "training_normalization.csv",
        args.output_dir / "normalization_provenance.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source_qa = json.loads(args.source_qa.read_text(encoding="utf-8"))
    if source_qa.get("status") != "PASS" or source_qa.get("city_count") != 20:
        raise RuntimeError("Full 20-city source QA has not passed")
    if tuple(source_qa.get("expected_band_order", [])) != EXPECTED_BANDS:
        raise RuntimeError("Source-QA band order does not match V3-MT-D1")
    if float(source_qa.get("minimum_valid_fraction", 0)) < 0.98:
        raise RuntimeError("Source-QA coverage gate is below 0.98")
    if source_qa.get("labels_modified") or source_qa.get("split_assignments_modified"):
        raise RuntimeError("Source QA unexpectedly records label/split changes")

    training_cities = sorted(
        (city for city in registry["cities"] if city["split"] == "train"),
        key=lambda city: city["city_id"],
    )
    if [city["city_id"] for city in training_cities] != [
        f"DE{number:02d}" for number in range(1, 15)
    ]:
        raise RuntimeError("Expected training cities DE01-DE14 exactly")

    linear_stats = {name: RunningStatistics() for name in EXPECTED_BANDS}
    db_stats = {name: RunningStatistics() for name in EXPECTED_BANDS}
    samples = {name: [] for name in EXPECTED_BANDS}
    source_records = []

    print("Pass 1/2: exact linear/dB statistics and deterministic quantile samples")
    for city_position, city in enumerate(training_cities):
        city_id, city_name = city["city_id"], city["city_name"]
        path = args.input_root / f"{city_id}_{city_name}" / (
            f"{city_id}_{city_name}_S1_MT_D1_MONTHLY_2025.tif"
        )
        require_file(path, f"{city_id} multitemporal raster")
        print(f"  {city_id} {city_name}")
        with rasterio.open(path) as dataset:
            if dataset.count != 24 or tuple(dataset.descriptions) != EXPECTED_BANDS:
                raise RuntimeError(f"Unexpected band schema: {path}")
            for band_index, band_name in enumerate(EXPECTED_BANDS, start=1):
                linear = valid_values(dataset, band_index)
                if linear.size == 0:
                    raise RuntimeError(f"No valid values: {city_id} {band_name}")
                db = 10.0 * np.log10(np.maximum(linear, args.epsilon))
                linear_stats[band_name].update(linear)
                db_stats[band_name].update(db)
                samples[band_name].append(deterministic_sample(
                    db,
                    args.sample_per_city_per_band,
                    args.seed + city_position * 100 + band_index,
                ))
        source_records.append({
            "city_id": city_id, "city_name": city_name,
            "path": str(path), "sha256": sha256_file(path),
        })

    db_quantiles = {
        name: quantiles(np.concatenate(samples[name])) for name in EXPECTED_BANDS
    }
    clipped_stats = {name: RunningStatistics() for name in EXPECTED_BANDS}

    print("Pass 2/2: exact clipped-dB mean and population standard deviation")
    for city in training_cities:
        city_id, city_name = city["city_id"], city["city_name"]
        path = args.input_root / f"{city_id}_{city_name}" / (
            f"{city_id}_{city_name}_S1_MT_D1_MONTHLY_2025.tif"
        )
        print(f"  {city_id} {city_name}")
        with rasterio.open(path) as dataset:
            for band_index, band_name in enumerate(EXPECTED_BANDS, start=1):
                linear = valid_values(dataset, band_index)
                db = 10.0 * np.log10(np.maximum(linear, args.epsilon))
                limits = db_quantiles[band_name]
                clipped_stats[band_name].update(
                    np.clip(db, limits["p01"], limits["p99"])
                )

    bands: dict[str, Any] = {}
    csv_rows = []
    for band_name in EXPECTED_BANDS:
        linear = linear_stats[band_name].as_dict()
        db = db_stats[band_name].as_dict()
        clipped = clipped_stats[band_name].as_dict()
        limits = db_quantiles[band_name]
        bands[band_name] = {
            "linear_sigma0": linear,
            "db": db,
            "sampled_db_quantiles": limits,
            "model_input": {
                "clip_lower_db": limits["p01"],
                "clip_upper_db": limits["p99"],
                "mean_db": clipped["mean"],
                "std_db": clipped["std_population"],
            },
        }
        csv_rows.append({
            "band": band_name,
            "valid_pixel_count": linear["valid_pixel_count"],
            "clip_lower_db": limits["p01"],
            "clip_upper_db": limits["p99"],
            "mean_db": clipped["mean"],
            "std_db": clipped["std_population"],
        })

    generated_at = datetime.now(timezone.utc).isoformat()
    normalization = {
        "schema_version": "dataset-v3-mt-d1-training-normalization-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "dataset_variant": "v3-mt-d1",
        "parent_label_dataset": "v3",
        "training_city_ids": [city["city_id"] for city in training_cities],
        "validation_and_test_used": False,
        "source_domain": "linear_sigma0",
        "conversion": f"10 * log10(max(linear_sigma0, {args.epsilon}))",
        "clipping": "training-only sampled p01/p99 per temporal band",
        "standardization": "exact clipped training-pixel mean/std per band",
        "band_order": list(EXPECTED_BANDS),
        "bands": bands,
    }
    provenance = {
        "schema_version": "dataset-v3-mt-d1-normalization-provenance-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "registry_path": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "source_qa_path": str(args.source_qa),
        "source_qa_sha256": sha256_file(args.source_qa),
        "training_city_ids": [city["city_id"] for city in training_cities],
        "excluded_city_ids": [f"DE{number:02d}" for number in range(15, 21)],
        "sample_per_city_per_band": args.sample_per_city_per_band,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "source_rasters": source_records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(outputs[1], index=False, lineterminator="\n")
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "training_cities": len(training_cities),
        "bands": len(EXPECTED_BANDS), "validation_and_test_used": False,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
