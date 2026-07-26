"""Compute training-only Sentinel-1 normalization statistics.

The script uses the full-resolution SAR rasters of cities assigned to the
training split in config/cities.csv. Validation and test cities are never read.

For each band (VV and VH), it computes:
1. Exact linear-Sigma0 count, mean, population standard deviation, min and max.
2. Approximate linear-Sigma0 p1, p50 and p99 from a deterministic sample.
3. Exact dB-domain count, mean, population standard deviation, min and max.
4. Approximate dB-domain p1, p50 and p99.
5. Exact mean and population standard deviation after clipping dB values to the
   sampled training p1/p99 limits.

Recommended model-input policy:
    linear Sigma0
    -> 10 * log10(max(value, epsilon))
    -> clip to training p1/p99
    -> z-score with clipped training mean/std
"""

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


EXPECTED_BANDS = {1: "VV", 2: "VH"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def combine_moments(
    count_a: int,
    mean_a: float,
    m2_a: float,
    count_b: int,
    mean_b: float,
    m2_b: float,
) -> tuple[int, float, float]:
    if count_b == 0:
        return count_a, mean_a, m2_a
    if count_a == 0:
        return count_b, mean_b, m2_b

    delta = mean_b - mean_a
    total = count_a + count_b
    mean = mean_a + delta * count_b / total
    m2 = m2_a + m2_b + delta * delta * count_a * count_b / total
    return total, mean, m2


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

        values64 = values.astype(np.float64, copy=False)
        batch_count = int(values64.size)
        batch_mean = float(values64.mean())
        differences = values64 - batch_mean
        batch_m2 = float(np.dot(differences, differences))

        self.count, self.mean, self.m2 = combine_moments(
            self.count,
            self.mean,
            self.m2,
            batch_count,
            batch_mean,
            batch_m2,
        )
        self.minimum = min(self.minimum, float(values64.min()))
        self.maximum = max(self.maximum, float(values64.max()))

    @property
    def population_std(self) -> float:
        if self.count == 0:
            return float("nan")
        return math.sqrt(self.m2 / self.count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid_pixel_count": int(self.count),
            "mean": float(self.mean),
            "std_population": float(self.population_std),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def deterministic_sample(
    values: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if values.size <= sample_size:
        return values.astype(np.float32, copy=True)

    generator = np.random.default_rng(seed)
    indices = generator.choice(
        values.size,
        size=sample_size,
        replace=False,
    )
    return values[indices].astype(np.float32, copy=False)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot calculate quantiles from an empty sample.")

    p1, p50, p99 = np.quantile(
        values.astype(np.float64, copy=False),
        [0.01, 0.50, 0.99],
    )
    return {
        "p01": float(p1),
        "p50": float(p50),
        "p99": float(p99),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute training-only Sentinel-1 normalization statistics."
    )
    parser.add_argument(
        "--cities-file",
        type=Path,
        default=Path("config/cities.csv"),
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=Path(
            "metadata/dataset_v1/dataset_manifest_summary.json"
        ),
    )
    parser.add_argument(
        "--sar-root",
        type=Path,
        default=Path("data/processed/sar"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v1/normalization"),
    )
    parser.add_argument(
        "--sample-per-city-per-band",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    require_file(args.cities_file, "city registry")
    require_file(args.dataset_summary, "dataset manifest summary")

    cities = pd.read_csv(args.cities_file)
    required_columns = {"city_id", "city_name", "split"}
    missing = required_columns.difference(cities.columns)

    if missing:
        raise ValueError(
            f"City registry is missing columns: {sorted(missing)}"
        )

    training_cities = (
        cities.loc[cities["split"].astype(str) == "train"]
        .sort_values("city_id")
        .reset_index(drop=True)
    )

    if len(training_cities) != 14:
        raise RuntimeError(
            f"Expected 14 training cities, found {len(training_cities)}."
        )

    with args.dataset_summary.open("r", encoding="utf-8") as file:
        dataset_summary = json.load(file)

    expected_training_ids = dataset_summary["split_city_ids"]["train"]
    actual_training_ids = training_cities["city_id"].astype(str).tolist()

    if actual_training_ids != expected_training_ids:
        raise RuntimeError(
            "Training-city registry does not match the frozen dataset summary."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    linear_stats = {
        name: RunningStatistics() for name in EXPECTED_BANDS.values()
    }
    db_stats = {
        name: RunningStatistics() for name in EXPECTED_BANDS.values()
    }
    linear_samples = {name: [] for name in EXPECTED_BANDS.values()}
    db_samples = {name: [] for name in EXPECTED_BANDS.values()}

    city_audit_rows: list[dict[str, Any]] = []
    source_rasters: list[dict[str, Any]] = []

    print("\nTraining-only SAR normalization")
    print("--------------------------------")
    print(f"Training cities: {len(training_cities)}")
    print(
        "Quantile sample limit per city and band: "
        f"{args.sample_per_city_per_band:,}"
    )

    for city_position, (_, city) in enumerate(
        training_cities.iterrows(),
        start=1,
    ):
        city_id = str(city["city_id"])
        city_name = str(city["city_name"])
        folder = f"{city_id}_{city_name}"
        raster_path = (
            args.sar_root
            / folder
            / f"{folder}_S1_VV_VH.tif"
        )
        require_file(raster_path, f"{city_id} training SAR raster")

        print(f"  [{city_position:02d}/14] {city_id} {city_name}")

        with rasterio.open(raster_path) as dataset:
            if dataset.count != 2:
                raise RuntimeError(f"{city_id} does not have two bands.")

            for band_index, band_name in EXPECTED_BANDS.items():
                array = dataset.read(band_index)
                valid_mask = np.isfinite(array) & (array > args.epsilon)
                values = array[valid_mask]

                if values.size == 0:
                    raise RuntimeError(
                        f"{city_id} {band_name} has no valid pixels."
                    )

                db_values = (
                    10.0
                    * np.log10(
                        np.maximum(
                            values.astype(np.float64, copy=False),
                            args.epsilon,
                        )
                    )
                )

                linear_stats[band_name].update(values)
                db_stats[band_name].update(db_values)

                sample_seed = (
                    args.seed + city_position * 100 + band_index
                )
                linear_sample = deterministic_sample(
                    values.reshape(-1),
                    args.sample_per_city_per_band,
                    sample_seed,
                )
                db_sample = (
                    10.0
                    * np.log10(
                        np.maximum(
                            linear_sample.astype(np.float64, copy=False),
                            args.epsilon,
                        )
                    )
                ).astype(np.float32)

                linear_samples[band_name].append(linear_sample)
                db_samples[band_name].append(db_sample)

                city_audit_rows.append(
                    {
                        "city_id": city_id,
                        "city_name": city_name,
                        "band": band_name,
                        "total_pixel_count": int(array.size),
                        "valid_pixel_count": int(values.size),
                        "invalid_or_nonpositive_count": int(
                            array.size - values.size
                        ),
                        "sample_pixel_count": int(linear_sample.size),
                        "linear_mean": float(
                            values.astype(np.float64, copy=False).mean()
                        ),
                        "linear_std_population": float(
                            values.astype(np.float64, copy=False).std(ddof=0)
                        ),
                        "db_mean": float(db_values.mean()),
                        "db_std_population": float(db_values.std(ddof=0)),
                    }
                )

        source_rasters.append(
            {
                "city_id": city_id,
                "city_name": city_name,
                "path": str(raster_path),
                "sha256": sha256_file(raster_path),
            }
        )

    linear_q = {}
    db_q = {}

    for band_name in EXPECTED_BANDS.values():
        linear_q[band_name] = quantiles(
            np.concatenate(linear_samples[band_name])
        )
        db_q[band_name] = quantiles(
            np.concatenate(db_samples[band_name])
        )

    clipped_db_stats = {
        name: RunningStatistics() for name in EXPECTED_BANDS.values()
    }

    print("\nSecond pass: exact clipped dB statistics")

    for _, city in training_cities.iterrows():
        city_id = str(city["city_id"])
        city_name = str(city["city_name"])
        folder = f"{city_id}_{city_name}"
        raster_path = (
            args.sar_root
            / folder
            / f"{folder}_S1_VV_VH.tif"
        )

        with rasterio.open(raster_path) as dataset:
            for band_index, band_name in EXPECTED_BANDS.items():
                array = dataset.read(band_index)
                valid_mask = np.isfinite(array) & (array > args.epsilon)
                values = array[valid_mask]
                db_values = (
                    10.0
                    * np.log10(
                        np.maximum(
                            values.astype(np.float64, copy=False),
                            args.epsilon,
                        )
                    )
                )
                clipped = np.clip(
                    db_values,
                    db_q[band_name]["p01"],
                    db_q[band_name]["p99"],
                )
                clipped_db_stats[band_name].update(clipped)

    output_json = args.output_dir / "training_normalization.json"
    output_csv = args.output_dir / "training_normalization.csv"
    audit_csv = (
        args.output_dir
        / "training_normalization_city_audit.csv"
    )

    band_results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for band_name in EXPECTED_BANDS.values():
        linear_result = linear_stats[band_name].as_dict()
        linear_result.update(linear_q[band_name])

        db_result = db_stats[band_name].as_dict()
        db_result.update(db_q[band_name])

        clipped_result = clipped_db_stats[band_name].as_dict()
        clipped_result.update(
            {
                "clip_lower_db": db_q[band_name]["p01"],
                "clip_upper_db": db_q[band_name]["p99"],
            }
        )

        band_results[band_name] = {
            "linear_sigma0": linear_result,
            "db_unclipped": db_result,
            "db_clipped_p01_p99": clipped_result,
        }

        for domain_name, statistics in (
            ("linear_sigma0", linear_result),
            ("db_unclipped", db_result),
            ("db_clipped_p01_p99", clipped_result),
        ):
            csv_rows.append(
                {
                    "band": band_name,
                    "domain": domain_name,
                    **statistics,
                }
            )

    result = {
        "dataset_version": "1.0-pre-freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "training_cities_only",
        "training_city_count": len(training_cities),
        "training_city_ids": actual_training_ids,
        "validation_and_test_used": False,
        "source_domain": "linear_sigma0",
        "db_conversion": (
            "10 * log10(max(linear_sigma0, epsilon))"
        ),
        "epsilon": args.epsilon,
        "quantile_method": {
            "type": "deterministic stratified sample by city and band",
            "sample_per_city_per_band": args.sample_per_city_per_band,
            "maximum_sample_per_band": (
                args.sample_per_city_per_band * len(training_cities)
            ),
            "base_seed": args.seed,
        },
        "recommended_model_input_policy": {
            "step_1": "convert linear Sigma0 to dB",
            "step_2": "clip each band to its training p01 and p99",
            "step_3": (
                "standardize using clipped training mean and "
                "population standard deviation"
            ),
            "padding_handling": (
                "exclude pixels where validity mask equals 0"
            ),
        },
        "bands": band_results,
        "source_rasters": source_rasters,
        "dataset_manifest_summary": str(args.dataset_summary),
        "dataset_manifest_summary_sha256": sha256_file(
            args.dataset_summary
        ),
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    pd.DataFrame(city_audit_rows).to_csv(audit_csv, index=False)

    print("\nTraining normalization summary")
    print("------------------------------")

    for band_name in EXPECTED_BANDS.values():
        stats = band_results[band_name]["db_clipped_p01_p99"]
        print(f"{band_name}:")
        print(
            f"  clip dB: {stats['clip_lower_db']:.6f} "
            f"to {stats['clip_upper_db']:.6f}"
        )
        print(f"  clipped mean: {stats['mean']:.6f}")
        print(
            f"  clipped std: {stats['std_population']:.6f}"
        )
        print(
            f"  valid pixels: {stats['valid_pixel_count']:,}"
        )

    mandatory_checks = {
        "training_city_count_is_14": len(training_cities) == 14,
        "validation_and_test_not_used": True,
        "vv_statistics_finite": all(
            math.isfinite(value)
            for value in (
                band_results["VV"]["db_clipped_p01_p99"]["mean"],
                band_results["VV"]["db_clipped_p01_p99"][
                    "std_population"
                ],
            )
        ),
        "vh_statistics_finite": all(
            math.isfinite(value)
            for value in (
                band_results["VH"]["db_clipped_p01_p99"]["mean"],
                band_results["VH"]["db_clipped_p01_p99"][
                    "std_population"
                ],
            )
        ),
        "vv_std_positive": (
            band_results["VV"]["db_clipped_p01_p99"][
                "std_population"
            ]
            > 0
        ),
        "vh_std_positive": (
            band_results["VH"]["db_clipped_p01_p99"][
                "std_population"
            ]
            > 0
        ),
    }

    print("\nMandatory checks")
    for name, passed in mandatory_checks.items():
        print(f"  {name}: {passed}")

    print(f"\nJSON: {output_json}")
    print(f"CSV: {output_csv}")
    print(f"City audit: {audit_csv}")

    if not all(mandatory_checks.values()):
        raise RuntimeError(
            "Training normalization failed mandatory checks."
        )

    print(
        "\nResult: training-only normalization metadata "
        "generated successfully."
    )


if __name__ == "__main__":
    main()
