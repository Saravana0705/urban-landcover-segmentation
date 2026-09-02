"""Build training-only normalization for V3-MT quarterly cross-ratios.

The source Dataset V3-MT remains immutable. This script reads its frozen
training tiles, derives four log-domain VH/VV ratios, and creates a separate
12-channel model-facing configuration for the MT-F1 ablation.
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


SOURCE_BANDS = (
    "VV_Q1", "VH_Q1", "VV_Q2", "VH_Q2",
    "VV_Q3", "VH_Q3", "VV_Q4", "VH_Q4",
)
RATIO_BANDS = ("CR_Q1", "CR_Q2", "CR_Q3", "CR_Q4")
OUTPUT_BANDS = SOURCE_BANDS + RATIO_BANDS


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
        "--manifest", type=Path,
        default=Path("metadata/dataset_v3_mt/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--parent-config", type=Path,
        default=Path("metadata/dataset_v3_mt/freeze/dataset_config.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("metadata/dataset_v3_mt_features/cross_ratio"),
    )
    parser.add_argument("--sample-per-tile-per-ratio", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def resolve_path(value: Any) -> Path:
    return Path(str(value).replace("\\", "/"))


def ratio_arrays(image: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    if image.shape[0] != len(SOURCE_BANDS):
        raise ValueError(f"Expected eight source bands, got {image.shape}")
    finite_positive = np.isfinite(image) & (image > epsilon)
    db = np.full_like(image, np.nan, dtype=np.float32)
    db[finite_positive] = 10.0 * np.log10(
        np.maximum(image[finite_positive], epsilon)
    )
    output: dict[str, np.ndarray] = {}
    for quarter in range(4):
        vv = db[quarter * 2]
        vh = db[quarter * 2 + 1]
        ratio = vh - vv
        ratio[~(np.isfinite(vv) & np.isfinite(vh))] = np.nan
        output[RATIO_BANDS[quarter]] = ratio
    return output


def deterministic_sample(
    values: np.ndarray, size: int, generator: np.random.Generator
) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= size:
        return values.copy()
    indices = generator.choice(values.size, size=size, replace=False)
    return values[indices]


def main() -> None:
    args = parse_args()
    require_file(args.manifest, "V3-MT manifest")
    require_file(args.parent_config, "frozen V3-MT dataset config")
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
        raise RuntimeError("Parent Dataset V3-MT config is not frozen")
    if tuple(parent["input"]["channels"]) != SOURCE_BANDS:
        raise RuntimeError("Unexpected parent V3-MT source band order")
    if parent["normalization"].get("validation_and_test_used") is not False:
        raise RuntimeError("Parent normalization provenance is not leakage-free")
    epsilon = float(parent["normalization"].get("epsilon", 1e-10))

    manifest = pd.read_csv(args.manifest)
    training = manifest.loc[
        manifest["split"].astype(str).str.lower() == "train"
    ].sort_values(["city_id", "tile_id"])
    if len(training) != int(parent["splits"]["train_tile_count"]):
        raise RuntimeError("Training tile count does not match frozen config")

    samples = {name: [] for name in RATIO_BANDS}
    generator = np.random.default_rng(args.seed)
    print("Pass 1/2: deterministic training-only ratio quantile samples")
    for position, row in enumerate(training.itertuples(index=False), start=1):
        image_path = resolve_path(getattr(row, "image_path"))
        validity_path = resolve_path(getattr(row, "validity_mask_path"))
        require_file(image_path, "training image tile")
        require_file(validity_path, "training validity tile")
        with rasterio.open(image_path) as dataset:
            if dataset.count != 8 or tuple(dataset.descriptions) != SOURCE_BANDS:
                raise RuntimeError(f"Unexpected source schema: {image_path}")
            ratios = ratio_arrays(dataset.read().astype(np.float32), epsilon)
        with rasterio.open(validity_path) as dataset:
            valid = dataset.read(1) == 1
        for name, values in ratios.items():
            samples[name].append(deterministic_sample(
                values[valid], args.sample_per_tile_per_ratio, generator
            ))
        if position % 100 == 0:
            print(f"  sampled {position}/{len(training)} tiles")

    limits: dict[str, tuple[float, float]] = {}
    for name in RATIO_BANDS:
        combined = np.concatenate(samples[name]).astype(np.float64, copy=False)
        if combined.size == 0:
            raise RuntimeError(f"No sampled values for {name}")
        lower, upper = np.quantile(combined, [0.01, 0.99])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise RuntimeError(f"Invalid quantiles for {name}")
        limits[name] = (float(lower), float(upper))

    stats = {name: RunningStatistics() for name in RATIO_BANDS}
    print("Pass 2/2: exact clipped ratio mean and population standard deviation")
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
        name: stats[name].model_input(*limits[name]) for name in RATIO_BANDS
    }
    if any(entry["std_db"] <= 0 for entry in ratio_bands.values()):
        raise RuntimeError("Ratio normalization contains a non-positive std")

    feature_config = json.loads(json.dumps(parent))
    feature_config.update({
        "dataset_version": "v3-mt-f1",
        "parent_dataset_version": "v3-mt",
        "freeze_status": "DERIVED_FROM_FROZEN_PARENT",
        "feature_config_generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    feature_config["input"].update({
        "source_channels": list(SOURCE_BANDS),
        "source_channel_count": len(SOURCE_BANDS),
        "channels": list(OUTPUT_BANDS),
        "channel_count": len(OUTPUT_BANDS),
        "derived_features": {
            "cross_ratio": {
                "enabled": True,
                "names": list(RATIO_BANDS),
                "formula": "VH_dB - VV_dB = 10*log10(VH/VV)",
                "quarter_order": ["Q1", "Q2", "Q3", "Q4"],
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
        "intervention": "four quarterly log cross-polarisation ratio channels",
        "source_tiles_labels_splits_unchanged": True,
        "paired_parent": "unet_v3_mt_e0_paired_50ep",
        "sampling_strategy": "standard",
        "test_split_locked": True,
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    normalization = {
        "schema_version": "dataset-v3-mt-f1-ratio-normalization-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "dataset_variant": "v3-mt-f1",
        "training_tile_count": len(training),
        "validation_and_test_used": False,
        "band_order": list(RATIO_BANDS),
        "bands": ratio_bands,
    }
    provenance = {
        "schema_version": "dataset-v3-mt-f1-provenance-0.1",
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
    outputs[0].write_text(json.dumps(feature_config, indent=2) + "\n", encoding="utf-8")
    outputs[1].write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "dataset_variant": "v3-mt-f1",
        "source_channels": 8,
        "model_channels": 12,
        "training_tiles": len(training),
        "validation_and_test_used": False,
        "dataset_config": str(outputs[0]),
    }, indent=2))


if __name__ == "__main__":
    main()

