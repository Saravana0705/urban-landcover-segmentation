"""Canonical deterministic features for quarterly Sentinel-1 V3-MT inputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np


SOURCE_BANDS = (
    "VV_Q1", "VH_Q1", "VV_Q2", "VH_Q2",
    "VV_Q3", "VH_Q3", "VV_Q4", "VH_Q4",
)
RATIO_BANDS = ("CR_Q1", "CR_Q2", "CR_Q3", "CR_Q4")
LOCAL_STATISTICS = ("MEDIAN", "MIN", "MAX", "RANGE")
LOCAL_BANDS = tuple(
    f"VH_Q{quarter}_{statistic}_5X5"
    for quarter in range(1, 5)
    for statistic in LOCAL_STATISTICS
)
F1_BANDS = SOURCE_BANDS + RATIO_BANDS
F2_BANDS = F1_BANDS + LOCAL_BANDS


def source_linear_to_db(image: np.ndarray, epsilon: float) -> np.ndarray:
    """Convert eight linear-sigma0 source bands to dB, preserving invalids."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != len(SOURCE_BANDS):
        raise ValueError(
            f"Expected source image shaped (8, H, W), received {image.shape}."
        )
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    usable = np.isfinite(image) & (image > epsilon)
    db = np.full_like(image, np.nan, dtype=np.float32)
    db[usable] = (
        10.0 * np.log10(np.maximum(image[usable], epsilon))
    ).astype(np.float32)
    return db


def quarterly_cross_ratios(source_db: np.ndarray) -> np.ndarray:
    """Return CR_Q1--CR_Q4 as VH_dB minus VV_dB."""
    source_db = np.asarray(source_db, dtype=np.float32)
    if source_db.ndim != 3 or source_db.shape[0] != len(SOURCE_BANDS):
        raise ValueError("Cross-ratios require canonical eight-band source dB.")

    ratios: list[np.ndarray] = []
    for quarter in range(4):
        vv = source_db[quarter * 2]
        vh = source_db[quarter * 2 + 1]
        ratio = (vh - vv).astype(np.float32, copy=False)
        ratio[~(np.isfinite(vv) & np.isfinite(vh))] = np.nan
        ratios.append(ratio)
    return np.stack(ratios, axis=0).astype(np.float32, copy=False)


def quarterly_vh_local_features(
    source_db: np.ndarray,
    *,
    fill_values_db: Mapping[str, float],
    window_size: int = 5,
) -> np.ndarray:
    """Return median/min/max/range features for each quarterly VH band.

    Invalid source pixels are filled with the corresponding frozen training
    mean only for neighbourhood calculation. Outputs at invalid centre pixels
    are reset to NaN and therefore become zero after model normalization.
    """
    if window_size != 5:
        raise ValueError("Dataset V3-MT-F2 requires a fixed 5x5 window.")
    source_db = np.asarray(source_db, dtype=np.float32)
    if source_db.ndim != 3 or source_db.shape[0] != len(SOURCE_BANDS):
        raise ValueError("Local features require canonical eight-band source dB.")

    try:
        from scipy.ndimage import maximum_filter, median_filter, minimum_filter
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError(
            "Dataset V3-MT-F2 local features require scipy."
        ) from exc

    features: list[np.ndarray] = []
    for quarter in range(1, 5):
        band_name = f"VH_Q{quarter}"
        if band_name not in fill_values_db:
            raise ValueError(f"Missing local-feature fill value: {band_name}")
        fill_value = float(fill_values_db[band_name])
        if not np.isfinite(fill_value):
            raise ValueError(f"Non-finite local-feature fill value: {band_name}")

        vh = source_db[(quarter - 1) * 2 + 1]
        centre_valid = np.isfinite(vh)
        filled = np.where(centre_valid, vh, fill_value).astype(np.float32)
        median = median_filter(filled, size=window_size, mode="reflect")
        minimum = minimum_filter(filled, size=window_size, mode="reflect")
        maximum = maximum_filter(filled, size=window_size, mode="reflect")
        value_range = maximum - minimum

        for value in (median, minimum, maximum, value_range):
            value = value.astype(np.float32, copy=False)
            value[~centre_valid] = np.nan
            features.append(value)

    return np.stack(features, axis=0).astype(np.float32, copy=False)


def build_model_input_db(
    image: np.ndarray,
    *,
    epsilon: float,
    include_cross_ratio: bool,
    include_local_spatial: bool,
    local_fill_values_db: Mapping[str, float] | None = None,
    local_window_size: int = 5,
) -> np.ndarray:
    """Build ordered source, ratio and optional local-spatial model channels."""
    source_db = source_linear_to_db(image, epsilon)
    parts = [source_db]
    if include_cross_ratio:
        parts.append(quarterly_cross_ratios(source_db))
    if include_local_spatial:
        if not include_cross_ratio:
            raise ValueError(
                "F2 local spatial features require the F1 cross-ratio parent."
            )
        if local_fill_values_db is None:
            raise ValueError("Local spatial features require frozen VH fill values.")
        parts.append(quarterly_vh_local_features(
            source_db,
            fill_values_db=local_fill_values_db,
            window_size=local_window_size,
        ))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

