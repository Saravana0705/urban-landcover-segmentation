"""Build training-only normalization for D2 orbit-specific cross-ratios.

The frozen 16-channel Dataset V3-MT-D2 remains immutable. This script reads
only its training tiles, derives four ascending and four descending
``VH_dB - VV_dB`` channels, and writes a separate 24-channel model-facing
configuration for the D2-F1 ablation.
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

from src.data.v3_mt_derived_features import (
    D2_F1_BANDS,
    D2_RATIO_BANDS,
    D2_SOURCE_BANDS,
    build_d2_model_input_db,
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
        values = values[np.isfinite(values)].astype(np.float64, copy=False)
        if values.size == 0:
            return
        count = int(values.size)
        mean = float(values.mean())
        centered = values - mean
        m2 = float(np.dot(centered, centered))
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

    def model_input(self, lower: float, upper: float) -> dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty running statistics")
        return {
            "clip_lower_db": float(lower),
            "clip_upper_db": float(upper),
            "mean_db": float(self.mean),
            "std_db": float(math.sqrt(self.m2 / self.count)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/dataset_v3_mt_d2/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("metadata/dataset_v3_mt_d2/freeze/dataset_config.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d2_features/cross_orbit_ratio"
        ),
    )
    parser.add_argument("--sample-per-tile-per-ratio", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def resolve_path(value: Any) -> Path:
    return Path(str(value).replace("\\", "/"))


def ratio_arrays(image: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    model_db = build_d2_model_input_db(
        image, epsilon=epsilon, include_cross_ratio=True
    )
    ratios = model_db[len(D2_SOURCE_BANDS):]
    return {
        name: ratios[index]
        for index, name in enumerate(D2_RATIO_BANDS)
    }


def deterministic_sample(
    values: np.ndarray,
    size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= size:
        return values.copy()
    indices = generator.choice(values.size, size=size, replace=False)
    return values[indices]


def main() -> None:
    args = parse_args()
    require_file(args.manifest, "D2 manifest")
    require_file(args.parent_config, "frozen D2 dataset config")
    if args.sample_per_tile_per_ratio <= 0:
        raise ValueError("sample-per-tile-per-ratio must be positive")

    outputs = (
        args.output_dir / "dataset_config.json",
        args.output_dir / "training_ratio_normalization.json",
        args.output_dir / "normalization_provenance.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    parent = json.loads(args.parent_config.read_text(encoding="utf-8"))
    if parent.get("freeze_status") != "FROZEN":
        raise RuntimeError("Parent Dataset V3-MT-D2 config is not frozen")
    if parent.get("dataset_version") != "v3-mt-d2":
        raise RuntimeError("Unexpected parent dataset version")
    if tuple(parent["input"]["channels"]) != D2_SOURCE_BANDS:
        raise RuntimeError("Unexpected parent D2 source band order")
    if parent["normalization"].get("validation_and_test_used") is not False:
        raise RuntimeError("Parent normalization provenance is not leakage-free")
    epsilon = float(parent["normalization"].get("epsilon", 1e-10))

    manifest = pd.read_csv(args.manifest)
    training = manifest.loc[
        manifest["split"].astype(str).str.lower() == "train"
    ].sort_values(["city_id", "tile_id"])
    if len(training) != int(parent["splits"]["train_tile_count"]):
        raise RuntimeError("Training tile count does not match frozen config")

    samples = {name: [] for name in D2_RATIO_BANDS}
    generator = np.random.default_rng(args.seed)
    print("Pass 1/2: deterministic training-only D2 ratio quantile samples")
    for position, row in enumerate(training.itertuples(index=False), start=1):
        image_path = resolve_path(getattr(row, "image_path"))
        validity_path = resolve_path(getattr(row, "validity_mask_path"))
        require_file(image_path, "training image tile")
        require_file(validity_path, "training validity tile")
        with rasterio.open(image_path) as dataset:
            descriptions = tuple(
                str(value).strip() if value is not None else ""
                for value in dataset.descriptions
            )
            if dataset.count != 16 or descriptions != D2_SOURCE_BANDS:
                raise RuntimeError(f"Unexpected D2 source schema: {image_path}")
            ratios = ratio_arrays(dataset.read().astype(np.float32), epsilon)
        with rasterio.open(validity_path) as dataset:
            valid = dataset.read(1) == 1
        for name, values in ratios.items():
            samples[name].append(
                deterministic_sample(
                    values[valid], args.sample_per_tile_per_ratio, generator
                )
            )
        if position % 100 == 0:
            print(f"  sampled {position}/{len(training)} tiles")

    limits: dict[str, tuple[float, float]] = {}
    for name in D2_RATIO_BANDS:
        combined = np.concatenate(samples[name]).astype(np.float64, copy=False)
        if combined.size == 0:
            raise RuntimeError(f"No sampled values for {name}")
        lower, upper = np.quantile(combined, [0.01, 0.99])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise RuntimeError(f"Invalid quantiles for {name}")
        limits[name] = (float(lower), float(upper))

    stats = {name: RunningStatistics() for name in D2_RATIO_BANDS}
    print("Pass 2/2: exact clipped D2 ratio mean and population std")
    for position, row in enumerate(training.itertuples(index=False), start=1):
        image_path = resolve_path(getattr(row, "image_path"))
        validity_path = resolve_path(getattr(row, "validity_mask_path"))
        with rasterio.open(image_path) as dataset:
            ratios = ratio_arrays(dataset.read().astype(np.float32), epsilon)
        with rasterio.open(validity_path) as dataset:
            valid = dataset.read(1) == 1
        for name, values in ratios.items():
            lower, upper = limits[name]
            selected = values[valid & np.isfinite(values)]
            stats[name].update(np.clip(selected, lower, upper))
        if position % 100 == 0:
            print(f"  processed {position}/{len(training)} tiles")

    ratio_bands = {
        name: stats[name].model_input(*limits[name])
        for name in D2_RATIO_BANDS
    }
    if any(entry["std_db"] <= 0 for entry in ratio_bands.values()):
        raise RuntimeError("Ratio normalization contains a non-positive std")

    generated_at = datetime.now(timezone.utc).isoformat()
    feature_config = json.loads(json.dumps(parent))
    feature_config.update({
        "dataset_version": "v3-mt-d2-f1",
        "parent_dataset_version": "v3-mt-d2",
        "freeze_status": "DERIVED_FROM_FROZEN_PARENT",
        "feature_config_generated_at_utc": generated_at,
    })
    feature_config["input"].update({
        "source_channels": list(D2_SOURCE_BANDS),
        "source_channel_count": len(D2_SOURCE_BANDS),
        "channels": list(D2_F1_BANDS),
        "channel_count": len(D2_F1_BANDS),
        "derived_features": {
            "cross_ratio": {
                "enabled": True,
                "names": list(D2_RATIO_BANDS),
                "formula": "VH_dB - VV_dB = 10*log10(VH/VV)",
                "orbit_order": ["ASCENDING", "DESCENDING"],
                "quarter_order_within_orbit": ["Q1", "Q2", "Q3", "Q4"],
            },
            "local_spatial": {"enabled": False},
        },
    })
    feature_config["normalization"].update({
        "scope": "frozen_parent_constants_plus_training_tiles_only_ratios",
        "validation_and_test_used": False,
    })
    feature_config["normalization"]["bands"].update(ratio_bands)
    feature_config["experimental_control"] = {
        "intervention": "eight orbit-specific quarterly log ratio channels",
        "source_tiles_labels_splits_unchanged": True,
        "paired_parent": "unet_v3_mt_d2_e0_cross_orbit_50ep",
        "sampling_strategy": "standard",
        "test_split_locked": True,
    }

    normalization = {
        "schema_version": "dataset-v3-mt-d2-f1-ratio-normalization-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "dataset_variant": "v3-mt-d2-f1",
        "training_tile_count": len(training),
        "validation_and_test_used": False,
        "band_order": list(D2_RATIO_BANDS),
        "bands": ratio_bands,
    }
    provenance = {
        "schema_version": "dataset-v3-mt-d2-f1-provenance-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "parent_manifest_path": str(args.manifest),
        "parent_manifest_sha256": sha256_file(args.manifest),
        "parent_config_path": str(args.parent_config),
        "parent_config_sha256": sha256_file(args.parent_config),
        "training_tile_count": len(training),
        "training_city_ids": sorted(training["city_id"].astype(str).unique()),
        "validation_and_test_used": False,
        "sample_per_tile_per_ratio": args.sample_per_tile_per_ratio,
        "seed": args.seed,
        "epsilon": epsilon,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        json.dumps(feature_config, indent=2) + "\n", encoding="utf-8"
    )
    outputs[1].write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )
    outputs[2].write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "dataset_variant": "v3-mt-d2-f1",
        "source_channels": 16,
        "derived_ratio_channels": 8,
        "model_channels": 24,
        "training_tiles": len(training),
        "validation_and_test_used": False,
        "dataset_config": str(outputs[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
