"""Build training-only normalization for V3-MT-F2 local VH features."""

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
    F1_BANDS,
    F2_BANDS,
    LOCAL_BANDS,
    RATIO_BANDS,
    SOURCE_BANDS,
    quarterly_vh_local_features,
    source_linear_to_db,
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

    def model_input(self, lower: float, upper: float) -> dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty spatial statistics")
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
        default=Path(
            "metadata/dataset_v3_mt_features/cross_ratio/dataset_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("metadata/dataset_v3_mt_features/local_spatial"),
    )
    parser.add_argument("--sample-per-tile-per-feature", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def resolve_path(value: Any) -> Path:
    return Path(str(value).replace("\\", "/"))


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


def local_features_for_tile(
    image: np.ndarray,
    *,
    epsilon: float,
    fill_values: dict[str, float],
) -> np.ndarray:
    source_db = source_linear_to_db(image, epsilon)
    return quarterly_vh_local_features(
        source_db,
        fill_values_db=fill_values,
        window_size=5,
    )


def main() -> None:
    args = parse_args()
    require_file(args.manifest, "V3-MT manifest")
    require_file(args.parent_config, "V3-MT-F1 dataset config")
    if args.sample_per_tile_per_feature <= 0:
        raise ValueError("sample-per-tile-per-feature must be positive")

    outputs = (
        args.output_dir / "dataset_config.json",
        args.output_dir / "training_spatial_normalization.json",
        args.output_dir / "normalization_provenance.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    parent = json.loads(args.parent_config.read_text(encoding="utf-8"))
    if parent.get("dataset_version") != "v3-mt-f1":
        raise RuntimeError("Expected Dataset V3-MT-F1 as the feature parent")
    if tuple(parent["input"]["source_channels"]) != SOURCE_BANDS:
        raise RuntimeError("Unexpected V3-MT-F1 source band order")
    if tuple(parent["input"]["channels"]) != F1_BANDS:
        raise RuntimeError("Unexpected V3-MT-F1 model channel order")
    if parent["input"]["derived_features"]["cross_ratio"].get("enabled") is not True:
        raise RuntimeError("V3-MT-F1 cross-ratio feature is not enabled")
    if parent["normalization"].get("validation_and_test_used") is not False:
        raise RuntimeError("Parent normalization provenance is not leakage-free")

    normalization_bands = parent["normalization"]["bands"]
    missing_parent = [name for name in F1_BANDS if name not in normalization_bands]
    if missing_parent:
        raise RuntimeError(f"Parent normalization missing: {missing_parent}")
    fill_values = {
        name: float(normalization_bands[name]["mean_db"])
        for name in ("VH_Q1", "VH_Q2", "VH_Q3", "VH_Q4")
    }
    if not np.isfinite(list(fill_values.values())).all():
        raise RuntimeError("Parent VH normalization means are non-finite")
    epsilon = float(parent["normalization"].get("epsilon", 1e-10))

    manifest = pd.read_csv(args.manifest)
    training = manifest.loc[
        manifest["split"].astype(str).str.lower() == "train"
    ].sort_values(["city_id", "tile_id"])
    if len(training) != int(parent["splits"]["train_tile_count"]):
        raise RuntimeError("Training tile count does not match frozen parent")
    training_city_ids = sorted(training["city_id"].astype(str).unique())
    if training_city_ids != [f"DE{index:02d}" for index in range(1, 15)]:
        raise RuntimeError("Expected training cities DE01-DE14 only")

    samples = {name: [] for name in LOCAL_BANDS}
    generator = np.random.default_rng(args.seed)
    print("Pass 1/2: deterministic training-only spatial quantile samples")
    for position, row in enumerate(training.itertuples(index=False), start=1):
        image_path = resolve_path(getattr(row, "image_path"))
        validity_path = resolve_path(getattr(row, "validity_mask_path"))
        require_file(image_path, "training image tile")
        require_file(validity_path, "training validity tile")
        with rasterio.open(image_path) as dataset:
            if dataset.count != 8 or tuple(dataset.descriptions) != SOURCE_BANDS:
                raise RuntimeError(f"Unexpected source schema: {image_path}")
            features = local_features_for_tile(
                dataset.read().astype(np.float32),
                epsilon=epsilon,
                fill_values=fill_values,
            )
        with rasterio.open(validity_path) as dataset:
            valid = dataset.read(1) == 1
        for index, name in enumerate(LOCAL_BANDS):
            samples[name].append(deterministic_sample(
                features[index][valid],
                args.sample_per_tile_per_feature,
                generator,
            ))
        if position % 100 == 0:
            print(f"  sampled {position}/{len(training)} tiles")

    limits: dict[str, tuple[float, float]] = {}
    for name in LOCAL_BANDS:
        combined = np.concatenate(samples[name]).astype(np.float64, copy=False)
        if combined.size == 0:
            raise RuntimeError(f"No sampled values for {name}")
        lower, upper = np.quantile(combined, [0.01, 0.99])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise RuntimeError(f"Invalid spatial quantiles for {name}")
        limits[name] = (float(lower), float(upper))

    stats = {name: RunningStatistics() for name in LOCAL_BANDS}
    print("Pass 2/2: exact clipped spatial mean and population standard deviation")
    for position, row in enumerate(training.itertuples(index=False), start=1):
        image_path = resolve_path(getattr(row, "image_path"))
        validity_path = resolve_path(getattr(row, "validity_mask_path"))
        with rasterio.open(image_path) as dataset:
            features = local_features_for_tile(
                dataset.read().astype(np.float32),
                epsilon=epsilon,
                fill_values=fill_values,
            )
        with rasterio.open(validity_path) as dataset:
            valid = dataset.read(1) == 1
        for index, name in enumerate(LOCAL_BANDS):
            lower, upper = limits[name]
            selected = features[index][valid & np.isfinite(features[index])]
            stats[name].update(np.clip(selected, lower, upper))
        if position % 100 == 0:
            print(f"  processed {position}/{len(training)} tiles")

    spatial_bands = {
        name: stats[name].model_input(*limits[name]) for name in LOCAL_BANDS
    }
    if any(entry["std_db"] <= 0 for entry in spatial_bands.values()):
        raise RuntimeError("Spatial normalization contains a non-positive std")

    generated_at = datetime.now(timezone.utc).isoformat()
    feature_config = json.loads(json.dumps(parent))
    feature_config.update({
        "dataset_version": "v3-mt-f2",
        "parent_dataset_version": "v3-mt-f1",
        "freeze_status": "DERIVED_FROM_FROZEN_PARENT",
        "feature_config_generated_at_utc": generated_at,
    })
    feature_config["input"].update({
        "channels": list(F2_BANDS),
        "channel_count": len(F2_BANDS),
    })
    feature_config["input"]["derived_features"]["local_spatial"] = {
        "enabled": True,
        "names": list(LOCAL_BANDS),
        "source_bands": ["VH_Q1", "VH_Q2", "VH_Q3", "VH_Q4"],
        "source_domain": "dB",
        "statistics": ["median", "minimum", "maximum", "range"],
        "window_size": 5,
        "boundary_mode": "reflect",
        "invalid_neighbour_fill": "frozen_training_mean_per_VH_band",
        "invalid_centre_output": "NaN_then_zero_after_normalization",
    }
    feature_config["normalization"].update({
        "scope": (
            "frozen_F1_constants_plus_training_tiles_only_spatial_features"
        ),
        "validation_and_test_used": False,
    })
    feature_config["normalization"]["bands"].update(spatial_bands)
    feature_config["experimental_control"] = {
        "intervention": "sixteen quarterly 5x5 VH local spatial channels",
        "source_tiles_labels_splits_unchanged": True,
        "paired_parent": "unet_v3_mt_f1_cross_ratio_50ep",
        "sampling_strategy": "standard",
        "test_split_locked": True,
    }

    normalization = {
        "schema_version": "dataset-v3-mt-f2-spatial-normalization-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "dataset_variant": "v3-mt-f2",
        "training_tile_count": len(training),
        "validation_and_test_used": False,
        "band_order": list(LOCAL_BANDS),
        "bands": spatial_bands,
    }
    provenance = {
        "schema_version": "dataset-v3-mt-f2-provenance-0.1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "parent_manifest_path": str(args.manifest),
        "parent_manifest_sha256": sha256_file(args.manifest),
        "parent_config_path": str(args.parent_config),
        "parent_config_sha256": sha256_file(args.parent_config),
        "training_tile_count": len(training),
        "training_city_ids": training_city_ids,
        "validation_and_test_used": False,
        "sample_per_tile_per_feature": args.sample_per_tile_per_feature,
        "seed": args.seed,
        "epsilon": epsilon,
        "feature_order": list(LOCAL_BANDS),
        "feature_window_size": 5,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(feature_config, indent=2) + "\n", encoding="utf-8")
    outputs[1].write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "dataset_variant": "v3-mt-f2",
        "source_channels": len(SOURCE_BANDS),
        "cross_ratio_channels": len(RATIO_BANDS),
        "local_spatial_channels": len(LOCAL_BANDS),
        "model_channels": len(F2_BANDS),
        "training_tiles": len(training),
        "validation_and_test_used": False,
        "dataset_config": str(outputs[0]),
    }, indent=2))


if __name__ == "__main__":
    main()

