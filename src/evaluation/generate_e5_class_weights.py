from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "metadata/dataset_v1/statistics/class_weights_candidates.csv"
)

OUTPUT_DIR = Path(
    "metadata/model_development/e5_class_weight_tuning"
)

OUTPUT_CSV = OUTPUT_DIR / "e5_class_weight_candidates.csv"
OUTPUT_JSON = OUTPUT_DIR / "e5_class_weight_candidates.json"


CLASS_NAMES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]


# Current median-frequency weights used by E2/E3/E4.
#
# We keep these explicitly because this is the known control condition.
CURRENT_MEDIAN_FREQUENCY = np.array(
    [
        0.368622,
        0.511828,
        0.043741,
        3.075984,
        0.999825,
    ],
    dtype=np.float64,
)


def normalize_mean_one(weights: np.ndarray) -> np.ndarray:
    """Normalize weights so the mean class weight equals 1."""
    weights = np.asarray(weights, dtype=np.float64)

    mean_weight = weights.mean()

    if mean_weight <= 0:
        raise ValueError("Mean class weight must be positive.")

    return weights / mean_weight


def sqrt_softened(weights: np.ndarray) -> np.ndarray:
    """
    Reduce extreme class-weight ratios using square-root compression.

    Example:
        very large bare-land weights become less aggressive,
        while very small vegetation weights become less suppressed.
    """
    softened = np.sqrt(weights)
    return normalize_mean_one(softened)


def clipped_weights(
    weights: np.ndarray,
    minimum: float = 0.30,
    maximum: float = 2.00,
) -> np.ndarray:
    """
    Clip extreme weights and normalize afterwards.
    """
    clipped = np.clip(
        weights,
        a_min=minimum,
        a_max=maximum,
    )

    return normalize_mean_one(clipped)


def log_softened(weights: np.ndarray) -> np.ndarray:
    """
    Logarithmic compression of the class-weight range.
    """
    softened = np.log1p(weights)

    return normalize_mean_one(softened)


def build_candidates() -> dict[str, np.ndarray]:
    current = CURRENT_MEDIAN_FREQUENCY.copy()

    return {
        # Control: reproduce E2 weighting exactly.
        "median_frequency_original": current,

        # Uniform CE weighting.
        "uniform": np.ones(
            len(CLASS_NAMES),
            dtype=np.float64,
        ),

        # Primary E5 candidate.
        "sqrt_softened": sqrt_softened(current),

        # Conservative capped weighting.
        "clipped_0p30_2p00": clipped_weights(
            current,
            minimum=0.30,
            maximum=2.00,
        ),

        # Additional smooth compression candidate.
        "log_softened": log_softened(current),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if INPUT_CSV.exists():
        print(f"Existing class-weight source found: {INPUT_CSV}")
    else:
        print(
            "Warning: expected source CSV was not found at "
            f"{INPUT_CSV}."
        )
        print(
            "The E5 candidates will still be generated using the "
            "confirmed E2 median-frequency weights."
        )

    candidates = build_candidates()

    rows: list[dict[str, object]] = []

    json_output: dict[str, dict[str, float]] = {}

    for candidate_name, weights in candidates.items():
        row: dict[str, object] = {
            "candidate_name": candidate_name,
        }

        json_output[candidate_name] = {}

        for class_name, weight in zip(
            CLASS_NAMES,
            weights,
            strict=True,
        ):
            value = float(weight)

            row[class_name] = value
            json_output[candidate_name][class_name] = value

        row["minimum_weight"] = float(weights.min())
        row["maximum_weight"] = float(weights.max())
        row["mean_weight"] = float(weights.mean())
        row["max_min_ratio"] = float(
            weights.max() / weights.min()
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_output,
            file,
            indent=2,
        )

    print()
    print("E5 class-weight candidates")
    print("==========================")
    print()

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(f"CSV written to:  {OUTPUT_CSV}")
    print(f"JSON written to: {OUTPUT_JSON}")

    print()
    print(
        "Primary candidates for E5 pilot:"
    )
    print(
        "1. median_frequency_original  -> E2 control"
    )
    print(
        "2. sqrt_softened             -> main candidate"
    )
    print(
        "3. uniform                   -> no class weighting"
    )
    print(
        "4. clipped_0p30_2p00         -> conservative weighting"
    )

    print()
    print(
        "Next step: create short E5 pilot training configurations."
    )


if __name__ == "__main__":
    main()