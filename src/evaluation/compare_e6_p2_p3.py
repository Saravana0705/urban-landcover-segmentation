"""Compare E6 P2 (Dataset V2.1) against E6 P3 (Dataset V2.2).

Purpose
-------
Quantify whether targeted Dataset V2.2 enrichment improved the controlled
U-Net experiment relative to E6 P2.

The script compares:
- overall evaluation metrics
- per-class IoU / Dice / Precision / Recall / F1
- confusion matrices
- minority-class performance
- training/runtime metadata from the experiment registry

No model training or inference is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------

P2_NAME = "unet_e6_p2_v21_ce30_tversky70_fn80"
P3_NAME = "unet_e6_p3_v22_ce30_tversky70_fn80"

P2_ROOT = Path(
    "outputs/model_experiments/"
    "unet_e6_p2_v21_ce30_tversky70_fn80"
)

P3_ROOT = Path(
    "outputs/model_experiments/"
    "unet_e6_p3_v22_ce30_tversky70_fn80"
)

REGISTRY_PATH = Path(
    "metadata/model_development/experiment_registry.csv"
)

OUTPUT_DIR = Path(
    "reports/e6_p2_vs_p3"
)


CLASS_ORDER = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]

MINORITY_CLASSES = [
    "buildings",
    "roads",
    "bare_land",
    "water",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}"
        )


def metric_path(
    experiment_root: Path,
    filename: str,
) -> Path:

    return (
        experiment_root
        / "metrics"
        / filename
    )


def load_overall(
    root: Path,
) -> pd.Series:

    path = metric_path(
        root,
        "overall_metrics.csv",
    )

    require_file(
        path,
        "overall metrics",
    )

    dataframe = pd.read_csv(
        path
    )

    if len(dataframe) != 1:
        raise ValueError(
            f"Expected one overall-metrics row in {path}."
        )

    return dataframe.iloc[0]


def load_per_class(
    root: Path,
) -> pd.DataFrame:

    path = metric_path(
        root,
        "per_class_metrics.csv",
    )

    require_file(
        path,
        "per-class metrics",
    )

    dataframe = pd.read_csv(
        path
    )

    required = {
        "class_name",
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
        "target_pixels",
        "predicted_pixels",
        "true_positive",
        "false_positive",
        "false_negative",
    }

    missing = required.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Missing per-class metric columns: "
            f"{sorted(missing)}"
        )

    observed = set(
        dataframe["class_name"]
        .astype(str)
    )

    if observed != set(CLASS_ORDER):
        raise ValueError(
            "Unexpected class set: "
            f"{sorted(observed)}"
        )

    return dataframe.copy()


def load_confusion(
    root: Path,
) -> pd.DataFrame:

    path = metric_path(
        root,
        "confusion_matrix.csv",
    )

    require_file(
        path,
        "confusion matrix",
    )

    dataframe = pd.read_csv(
        path
    )

    return dataframe


def safe_relative_change(
    old: float,
    new: float,
) -> float:

    if old == 0:
        return np.nan

    return (
        (new - old)
        / abs(old)
    )


# ---------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------

def compare_overall(
    p2: pd.Series,
    p3: pd.Series,
) -> pd.DataFrame:

    metrics = [
        "pixel_accuracy",
        "mean_iou",
        "mean_dice",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "frequency_weighted_iou",
        "frequency_weighted_dice",
    ]

    rows = []

    for metric in metrics:

        p2_value = float(
            p2[metric]
        )

        p3_value = float(
            p3[metric]
        )

        rows.append(
            {
                "metric": metric,
                "e6_p2": p2_value,
                "e6_p3": p3_value,
                "absolute_change": (
                    p3_value - p2_value
                ),
                "relative_change_pct": (
                    100.0
                    * safe_relative_change(
                        p2_value,
                        p3_value,
                    )
                ),
                "improved": (
                    p3_value > p2_value
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_per_class(
    p2: pd.DataFrame,
    p3: pd.DataFrame,
) -> pd.DataFrame:

    metric_columns = [
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
    ]

    p2 = (
        p2.set_index(
            "class_name"
        )
        .loc[CLASS_ORDER]
    )

    p3 = (
        p3.set_index(
            "class_name"
        )
        .loc[CLASS_ORDER]
    )

    rows = []

    for class_name in CLASS_ORDER:

        row: dict[str, Any] = {
            "class_name": class_name,
            "minority_class": (
                class_name
                in MINORITY_CLASSES
            ),
        }

        for metric in metric_columns:

            old = float(
                p2.loc[
                    class_name,
                    metric,
                ]
            )

            new = float(
                p3.loc[
                    class_name,
                    metric,
                ]
            )

            row[
                f"p2_{metric}"
            ] = old

            row[
                f"p3_{metric}"
            ] = new

            row[
                f"{metric}_change"
            ] = new - old

            row[
                f"{metric}_relative_change_pct"
            ] = (
                100.0
                * safe_relative_change(
                    old,
                    new,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_minority_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:

    minority = comparison.loc[
        comparison[
            "minority_class"
        ]
    ].copy()

    rows = []

    for metric in (
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
    ):

        p2_mean = float(
            minority[
                f"p2_{metric}"
            ].mean()
        )

        p3_mean = float(
            minority[
                f"p3_{metric}"
            ].mean()
        )

        rows.append(
            {
                "metric": metric,
                "p2_minority_macro": (
                    p2_mean
                ),
                "p3_minority_macro": (
                    p3_mean
                ),
                "absolute_change": (
                    p3_mean - p2_mean
                ),
                "relative_change_pct": (
                    100.0
                    * safe_relative_change(
                        p2_mean,
                        p3_mean,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_registry() -> pd.DataFrame:

    require_file(
        REGISTRY_PATH,
        "experiment registry",
    )

    registry = pd.read_csv(
        REGISTRY_PATH
    )

    requested = registry.loc[
        registry[
            "experiment_name"
        ].isin(
            [
                P2_NAME,
                P3_NAME,
            ]
        )
    ].copy()

    if len(requested) != 2:
        raise RuntimeError(
            "Could not find both E6 P2 and E6 P3 "
            "in experiment registry."
        )

    columns = [
        "experiment_name",
        "completed_epochs",
        "best_epoch",
        "best_metric",
        "training_time_seconds",
        "training_time_hours",
        "inference_images_per_second",
        "inference_milliseconds_per_image",
        "pixel_accuracy",
        "mean_iou",
        "mean_dice",
        "mean_precision",
        "mean_recall",
    ]

    return requested[
        columns
    ].copy()


# ---------------------------------------------------------------------
# Decision summary
# ---------------------------------------------------------------------

def make_decision(
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    minority: pd.DataFrame,
) -> dict[str, Any]:

    mean_iou_row = (
        overall.loc[
            overall["metric"]
            == "mean_iou"
        ]
        .iloc[0]
    )

    bare_land = (
        per_class.loc[
            per_class[
                "class_name"
            ]
            == "bare_land"
        ]
        .iloc[0]
    )

    roads = (
        per_class.loc[
            per_class[
                "class_name"
            ]
            == "roads"
        ]
        .iloc[0]
    )

    buildings = (
        per_class.loc[
            per_class[
                "class_name"
            ]
            == "buildings"
        ]
        .iloc[0]
    )

    water = (
        per_class.loc[
            per_class[
                "class_name"
            ]
            == "water"
        ]
        .iloc[0]
    )

    vegetation = (
        per_class.loc[
            per_class[
                "class_name"
            ]
            == "vegetation"
        ]
        .iloc[0]
    )

    minority_iou = (
        minority.loc[
            minority["metric"]
            == "iou"
        ]
        .iloc[0]
    )

    improvements = {
        "mean_iou_improved": bool(
            mean_iou_row[
                "absolute_change"
            ]
            > 0
        ),
        "minority_macro_iou_improved": bool(
            minority_iou[
                "absolute_change"
            ]
            > 0
        ),
        "bare_land_iou_improved": bool(
            bare_land[
                "iou_change"
            ]
            > 0
        ),
        "roads_iou_improved": bool(
            roads[
                "iou_change"
            ]
            > 0
        ),
        "buildings_iou_improved": bool(
            buildings[
                "iou_change"
            ]
            > 0
        ),
        "water_iou_improved": bool(
            water[
                "iou_change"
            ]
            > 0
        ),
        "vegetation_iou_change": float(
            vegetation[
                "iou_change"
            ]
        ),
    }

    # Dataset V2.2 is considered validated if:
    # - overall mIoU improves,
    # - minority macro IoU improves,
    # - and bare land OR roads improves.
    enrichment_validated = bool(
        improvements[
            "mean_iou_improved"
        ]
        and improvements[
            "minority_macro_iou_improved"
        ]
        and (
            improvements[
                "bare_land_iou_improved"
            ]
            or improvements[
                "roads_iou_improved"
            ]
        )
    )

    return {
        "experiment_control": (
            "E6 P2 vs E6 P3; model/loss/weights/"
            "validation/normalization held fixed; "
            "training dataset changed from V2.1 to V2.2."
        ),
        "e6_p2": P2_NAME,
        "e6_p3": P3_NAME,
        "improvements": improvements,
        "dataset_v22_enrichment_validated": (
            enrichment_validated
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print()
    print("E6 P2 vs E6 P3 comparison")
    print("==========================")

    p2_overall = load_overall(
        P2_ROOT
    )

    p3_overall = load_overall(
        P3_ROOT
    )

    p2_per_class = load_per_class(
        P2_ROOT
    )

    p3_per_class = load_per_class(
        P3_ROOT
    )

    # Also validate availability of confusion matrices.
    load_confusion(
        P2_ROOT
    )

    load_confusion(
        P3_ROOT
    )

    overall_comparison = (
        compare_overall(
            p2_overall,
            p3_overall,
        )
    )

    per_class_comparison = (
        compare_per_class(
            p2_per_class,
            p3_per_class,
        )
    )

    minority_summary = (
        build_minority_summary(
            per_class_comparison
        )
    )

    registry_comparison = (
        compare_registry()
    )

    decision = make_decision(
        overall_comparison,
        per_class_comparison,
        minority_summary,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        OUTPUT_DIR
        / "overall_comparison.csv"
    )

    per_class_path = (
        OUTPUT_DIR
        / "per_class_comparison.csv"
    )

    minority_path = (
        OUTPUT_DIR
        / "minority_class_summary.csv"
    )

    registry_path = (
        OUTPUT_DIR
        / "experiment_metadata_comparison.csv"
    )

    decision_path = (
        OUTPUT_DIR
        / "comparison_summary.json"
    )

    overall_comparison.to_csv(
        overall_path,
        index=False,
    )

    per_class_comparison.to_csv(
        per_class_path,
        index=False,
    )

    minority_summary.to_csv(
        minority_path,
        index=False,
    )

    registry_comparison.to_csv(
        registry_path,
        index=False,
    )

    with decision_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            decision,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Terminal output
    # ---------------------------------------------------------

    print()
    print("Overall evaluation")
    print("------------------")

    display_overall = (
        overall_comparison.copy()
    )

    for column in (
        "e6_p2",
        "e6_p3",
        "absolute_change",
    ):
        display_overall[
            column
        ] = display_overall[
            column
        ].map(
            lambda x: f"{x:.6f}"
        )

    display_overall[
        "relative_change_pct"
    ] = display_overall[
        "relative_change_pct"
    ].map(
        lambda x: f"{x:.2f}"
    )

    print(
        display_overall[
            [
                "metric",
                "e6_p2",
                "e6_p3",
                "absolute_change",
                "relative_change_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Per-class IoU comparison")
    print("------------------------")

    iou_display = (
        per_class_comparison[
            [
                "class_name",
                "p2_iou",
                "p3_iou",
                "iou_change",
                "iou_relative_change_pct",
            ]
        ]
        .copy()
    )

    print(
        iou_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Minority-class macro comparison")
    print("-------------------------------")

    print(
        minority_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Decision")
    print("--------")

    for key, value in (
        decision[
            "improvements"
        ].items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "Dataset V2.2 enrichment validated: "
        f"{decision['dataset_v22_enrichment_validated']}"
    )

    print()
    print("Reports written")
    print("---------------")
    print(overall_path)
    print(per_class_path)
    print(minority_path)
    print(registry_path)
    print(decision_path)


if __name__ == "__main__":
    main()