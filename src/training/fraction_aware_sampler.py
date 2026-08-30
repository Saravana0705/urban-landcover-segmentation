from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


@dataclass(frozen=True)
class FractionAwareSamplingConfig:
    manifest_path: str
    split: str = "train"

    # Bare-land thresholds
    bare_land_low_threshold: float = 0.01
    bare_land_high_threshold: float = 0.05

    # Road / water thresholds
    road_threshold: float = 0.10
    water_threshold: float = 0.05

    # Multiplicative/additive sampling boosts
    bare_land_low_boost: float = 1.0
    bare_land_high_boost: float = 1.5
    road_boost: float = 0.0
    water_boost: float = 0.5

    max_weight: float = 4.0

    replacement: bool = True
    seed: int = 42


def load_training_manifest(
    config: FractionAwareSamplingConfig,
) -> pd.DataFrame:
    path = Path(config.manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "tile_id",
        "city_id",
        "split",
        "valid_pixel_count",
        "bare_land_pixel_count",
        "roads_pixel_count",
        "water_pixel_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    split_df = df[
        df["split"].astype(str).str.lower()
        == config.split.lower()
    ].copy()

    if split_df.empty:
        raise ValueError(
            f"No samples found for split '{config.split}'."
        )

    # Match Sentinel1UrbanDataset behaviour:
    # exclude tiles with zero usable/valid pixels.
    split_df["valid_pixel_count"] = pd.to_numeric(
        split_df["valid_pixel_count"],
        errors="coerce",
    ).fillna(0)

    split_df = split_df.loc[
        split_df["valid_pixel_count"] > 0
    ].copy()

    if split_df.empty:
        raise ValueError(
            f"No usable samples remain for split '{config.split}' "
            "after excluding zero-valid-pixel tiles."
        )

    numeric_columns = [
        "valid_pixel_count",
        "bare_land_pixel_count",
        "roads_pixel_count",
        "water_pixel_count",
    ]

    for column in numeric_columns:
        split_df[column] = pd.to_numeric(
            split_df[column],
            errors="coerce",
        ).fillna(0)

    denominator = split_df["valid_pixel_count"].astype("float64")

    # Avoid division by zero without introducing pd.NA/object dtype.
    denominator = denominator.where(denominator > 0, 1.0)

    split_df["bare_land_fraction"] = (
        split_df["bare_land_pixel_count"].astype("float64")
        / denominator
    )

    split_df["road_fraction"] = (
        split_df["roads_pixel_count"].astype("float64")
        / denominator
    )

    split_df["water_fraction"] = (
        split_df["water_pixel_count"].astype("float64")
        / denominator
    )

    # Explicitly guarantee numeric dtype for PyTorch conversion.
    split_df["bare_land_fraction"] = pd.to_numeric(
        split_df["bare_land_fraction"],
        errors="coerce",
    ).fillna(0.0).astype("float64")

    split_df["road_fraction"] = pd.to_numeric(
        split_df["road_fraction"],
        errors="coerce",
    ).fillna(0.0).astype("float64")

    split_df["water_fraction"] = pd.to_numeric(
        split_df["water_fraction"],
        errors="coerce",
    ).fillna(0.0).astype("float64")

    # IMPORTANT:
    # Match the exact sample ordering used by SegmentationDataset. 
    # WeightedRandomSampler works by positional dataset index, so sampler row i must correspond to dataset sample i.
    
    split_df = split_df.sort_values(
        ["city_id", "tile_id"]
    ).reset_index(drop=True)

    return split_df


def calculate_sampling_weights(
    manifest: pd.DataFrame,
    config: FractionAwareSamplingConfig,
) -> torch.Tensor:
    weights = torch.ones(
        len(manifest),
        dtype=torch.double,
    )

    bare_fraction = torch.from_numpy(
        manifest["bare_land_fraction"]
        .to_numpy(dtype="float64", na_value=0.0)
        .copy()
    )

    road_fraction = torch.from_numpy(
        manifest["road_fraction"]
        .to_numpy(dtype="float64", na_value=0.0)
        .copy()
    )

    water_fraction = torch.from_numpy(
        manifest["water_fraction"]
        .to_numpy(dtype="float64", na_value=0.0)
        .copy()
    )

    # Mild boost for meaningful bare-land presence.
    weights += (
        bare_fraction
        >= config.bare_land_low_threshold
    ).double() * config.bare_land_low_boost

    # Additional boost for genuinely bare-land-rich tiles.
    weights += (
        bare_fraction
        >= config.bare_land_high_threshold
    ).double() * config.bare_land_high_boost

    # Moderate road boost when enabled.
    weights += (
        road_fraction
        >= config.road_threshold
    ).double() * config.road_boost

    # Smaller water boost.
    weights += (
        water_fraction
        >= config.water_threshold
    ).double() * config.water_boost

    weights = torch.clamp(
        weights,
        max=config.max_weight,
    )

    return weights


def build_fraction_aware_sampler(
    config: FractionAwareSamplingConfig,
) -> tuple[WeightedRandomSampler, pd.DataFrame, torch.Tensor]:
    manifest = load_training_manifest(config)

    weights = calculate_sampling_weights(
        manifest,
        config,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=config.replacement,
        generator=generator,
    )

    return sampler, manifest, weights


def sampling_summary(
    manifest: pd.DataFrame,
    weights: torch.Tensor,
) -> dict:
    weighted = manifest.copy()
    weighted["sampling_weight"] = weights.numpy()

    return {
        "tiles": len(weighted),
        "mean_weight": float(
            weighted["sampling_weight"].mean()
        ),
        "max_weight": float(
            weighted["sampling_weight"].max()
        ),
        "tiles_weight_gt_1": int(
            (weighted["sampling_weight"] > 1).sum()
        ),
        "pct_tiles_weight_gt_1": float(
            100
            * (weighted["sampling_weight"] > 1).mean()
        ),
        "bare_land_ge_1pct": int(
            (weighted["bare_land_fraction"] >= 0.01).sum()
        ),
        "bare_land_ge_5pct": int(
            (weighted["bare_land_fraction"] >= 0.05).sum()
        ),
        "water_ge_5pct": int(
            (weighted["water_fraction"] >= 0.05).sum()
        ),
    }