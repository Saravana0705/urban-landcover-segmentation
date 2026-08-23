from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CLASS_NAMES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]

EXPERIMENTS = {
    "U-Net": Path(
        "outputs/model_experiments/"
        "unet_baseline_v1_weighted_extended/"
        "metrics/evaluation_metrics.json"
    ),
    "Attention U-Net": Path(
        "outputs/model_experiments/"
        "attention_unet_v1_weighted_extended/"
        "metrics/evaluation_metrics.json"
    ),
    "U-Net++": Path(
        "outputs/model_experiments/"
        "unetpp_v1_weighted_extended/"
        "metrics/evaluation_metrics.json"
    ),
    "Swin Transformer": Path(
        "outputs/model_experiments/"
        "swin_transformer_tiny_v1_weighted_extended/"
        "metrics/evaluation_metrics.json"
    ),
}

OUTPUT_DIR = Path("reports/minority_class_analysis")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_class_rows = []
    confusion_rows = []

    for model_name, path in EXPERIMENTS.items():
        metrics = load_json(path)

        for class_name, values in metrics["per_class"].items():
            per_class_rows.append(
                {
                    "model": model_name,
                    "class": class_name,
                    "target_pixels": values["target_pixels"],
                    "predicted_pixels": values["predicted_pixels"],
                    "iou": values["iou"],
                    "dice": values["dice"],
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "true_positive": values["true_positive"],
                    "false_positive": values["false_positive"],
                    "false_negative": values["false_negative"],
                }
            )

        matrix = metrics["confusion_matrix"]

        for true_id, true_class in enumerate(CLASS_NAMES):
            row_total = sum(matrix[true_id])

            for pred_id, pred_class in enumerate(CLASS_NAMES):
                count = matrix[true_id][pred_id]

                confusion_rows.append(
                    {
                        "model": model_name,
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "count": count,
                        "row_percentage": (
                            count / row_total if row_total else 0.0
                        ),
                    }
                )

    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(confusion_rows)

    per_class.to_csv(
        OUTPUT_DIR / "per_class_model_comparison.csv",
        index=False,
    )

    confusion.to_csv(
        OUTPUT_DIR / "normalized_confusion_long.csv",
        index=False,
    )

    class_distribution = (
        per_class[
            ["class", "target_pixels"]
        ]
        .drop_duplicates("class")
        .copy()
    )

    total_pixels = class_distribution["target_pixels"].sum()

    class_distribution["pixel_fraction"] = (
        class_distribution["target_pixels"] / total_pixels
    )

    class_distribution["imbalance_vs_rarest"] = (
        class_distribution["target_pixels"]
        / class_distribution["target_pixels"].min()
    )

    class_distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False,
    )

    minority = per_class[
        per_class["class"].isin(["roads", "bare_land"])
    ].copy()

    minority.to_csv(
        OUTPUT_DIR / "minority_class_metrics.csv",
        index=False,
    )

    errors = confusion[
        (
            confusion["true_class"].isin(
                ["roads", "bare_land"]
            )
        )
        & (
            confusion["true_class"]
            != confusion["predicted_class"]
        )
    ].copy()

    errors = errors.sort_values(
        ["model", "true_class", "row_percentage"],
        ascending=[True, True, False],
    )

    errors.to_csv(
        OUTPUT_DIR / "minority_confusion_destinations.csv",
        index=False,
    )

    summary = (
        errors.groupby(
            ["model", "true_class"],
            as_index=False,
        )
        .first()
        [
            [
                "model",
                "true_class",
                "predicted_class",
                "row_percentage",
            ]
        ]
        .rename(
            columns={
                "predicted_class": "main_wrong_prediction",
                "row_percentage": "fraction_sent_to_main_wrong_class",
            }
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "minority_failure_summary.csv",
        index=False,
    )

    print("\nClass distribution")
    print("------------------")
    print(
        class_distribution.to_string(
            index=False
        )
    )

    print("\nMinority-class IoU")
    print("------------------")
    print(
        minority.pivot(
            index="model",
            columns="class",
            values="iou",
        ).round(4)
    )

    print("\nMain minority-class confusion")
    print("-----------------------------")
    print(summary.to_string(index=False))

    print(
        f"\nReports written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()