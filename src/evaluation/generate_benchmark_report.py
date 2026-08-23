"""Generate thesis-ready benchmark tables, rankings, and comparison plots.

The script uses the experiment registry as the authoritative source for overall
metrics and discovers per-class metrics from each experiment's metrics folder.
It never invents missing experiment values: incomplete models are reported in
``benchmark_warnings.txt`` and omitted only from plots that require the missing
field.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_REGISTRY = Path("metadata/model_development/experiment_registry.csv")
DEFAULT_OUTPUT = Path("reports/final_benchmark")
MODEL_ORDER = [
    "unet",
    "attention_unet",
    "unetpp",
    "deeplabv3plus",
    "segformer",
    "swin_transformer",
    "mask2former",
]
DISPLAY_ORDER = {
    name: index for index, name in enumerate(MODEL_ORDER)
}
CLASS_ORDER = ["buildings", "roads", "vegetation", "bare_land", "water"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution.",
    )
    parser.add_argument(
        "--balanced-weights",
        type=float,
        nargs=4,
        metavar=("MIOU", "SPEED", "PARAMS", "TIME"),
        default=(0.60, 0.15, 0.15, 0.10),
        help="Weights for balanced ranking; must sum to one.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Registry is missing required columns: {missing}")


def safe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def model_sort_key(name: str) -> tuple[int, str]:
    normalized = str(name).strip().lower()
    return DISPLAY_ORDER.get(normalized, len(DISPLAY_ORDER)), normalized


def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Experiment registry not found: {path}")
    frame = pd.read_csv(path)
    require_columns(
        frame,
        (
            "experiment_name",
            "model_name",
            "model_display_name",
            "status",
            "experiment_root",
        ),
    )
    frame = frame.loc[frame["status"].astype(str).str.lower() == "completed"].copy()
    frame = safe_numeric(
        frame,
        (
            "trainable_parameters",
            "completed_epochs",
            "best_epoch",
            "training_time_seconds",
            "training_time_hours",
            "peak_gpu_memory_mb",
            "inference_images_per_second",
            "inference_milliseconds_per_image",
            "pixel_accuracy",
            "mean_iou",
            "mean_dice",
            "mean_f1",
            "mean_precision",
            "mean_recall",
        ),
    )
    frame["_sort"] = frame["model_name"].map(model_sort_key)
    frame = frame.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return frame


def normalize_benefit(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    minimum, maximum = float(valid.min()), float(valid.max())
    if math.isclose(minimum, maximum):
        result = pd.Series(np.nan, index=series.index)
        result.loc[valid.index] = 1.0
        return result
    return (values - minimum) / (maximum - minimum)


def normalize_cost(series: pd.Series) -> pd.Series:
    benefit = normalize_benefit(series)
    return 1.0 - benefit


def build_ranking(frame: pd.DataFrame, weights: tuple[float, ...]) -> pd.DataFrame:
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Balanced ranking weights must sum to 1.0.")
    ranking = frame.copy()
    ranking["score_miou"] = normalize_benefit(ranking["mean_iou"])
    ranking["score_speed"] = normalize_benefit(ranking["inference_images_per_second"])
    ranking["score_parameters"] = normalize_cost(ranking["trainable_parameters"])
    ranking["score_training_time"] = normalize_cost(ranking["training_time_seconds"])
    components = [
        ranking["score_miou"],
        ranking["score_speed"],
        ranking["score_parameters"],
        ranking["score_training_time"],
    ]
    # Weighted average only across available components. This prevents missing
    # GPU-only fields or an incomplete historical row from becoming zero.
    numerator = sum(component.fillna(0.0) * weight for component, weight in zip(components, weights))
    denominator = sum(component.notna().astype(float) * weight for component, weight in zip(components, weights))
    ranking["balanced_score"] = numerator / denominator.replace(0.0, np.nan)
    ranking = ranking.sort_values(
        ["balanced_score", "mean_iou"], ascending=[False, False]
    ).reset_index(drop=True)
    ranking.insert(0, "balanced_rank", np.arange(1, len(ranking) + 1))
    columns = [
        "balanced_rank",
        "model_display_name",
        "model_name",
        "balanced_score",
        "mean_iou",
        "mean_dice",
        "pixel_accuracy",
        "inference_images_per_second",
        "trainable_parameters",
        "training_time_hours",
    ]
    return ranking[[column for column in columns if column in ranking.columns]]


def load_per_class(frame: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        path = Path(str(record["experiment_root"])) / "metrics" / "per_class_metrics.csv"
        if not path.exists():
            warnings.append(
                f"Missing per-class metrics for {record['model_display_name']}: {path}"
            )
            continue
        metrics = pd.read_csv(path)
        required = {"class_name", "iou", "dice", "precision", "recall", "f1"}
        missing = required.difference(metrics.columns)
        if missing:
            warnings.append(
                f"Incomplete per-class metrics for {record['model_display_name']}: {sorted(missing)}"
            )
            continue
        for metric in metrics.to_dict(orient="records"):
            rows.append(
                {
                    "model_name": record["model_name"],
                    "model_display_name": record["model_display_name"],
                    **metric,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["class_name"] = pd.Categorical(
            result["class_name"], categories=CLASS_ORDER, ordered=True
        )
        result["_sort"] = result["model_name"].map(model_sort_key)
        result = result.sort_values(["_sort", "class_name"]).drop(columns="_sort")
    return result


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def bar_plot(
    frame: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output: Path,
    dpi: int,
    scale: float = 1.0,
) -> None:
    subset = frame[["model_display_name", column]].dropna()
    if subset.empty:
        return
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    values = subset[column].to_numpy(dtype=float) * scale
    bars = axis.bar(subset["model_display_name"], values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig, output, dpi)


def grouped_per_class_plot(per_class: pd.DataFrame, output: Path, dpi: int) -> None:
    if per_class.empty:
        return
    pivot = per_class.pivot(
        index="class_name", columns="model_display_name", values="iou"
    ).reindex(CLASS_ORDER)
    fig, axis = plt.subplots(figsize=(12.0, 6.2))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Per-class Intersection over Union")
    axis.set_xlabel("Land-cover class")
    axis.set_ylabel("IoU")
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output, dpi)


def efficiency_plot(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    subset = frame.dropna(
        subset=["trainable_parameters", "inference_images_per_second", "mean_iou"]
    )
    if subset.empty:
        return
    fig, axis = plt.subplots(figsize=(8.8, 5.8))
    sizes = 160 + 900 * normalize_benefit(subset["mean_iou"]).fillna(0.5)
    axis.scatter(
        subset["trainable_parameters"] / 1_000_000,
        subset["inference_images_per_second"],
        s=sizes,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in subset.iterrows():
        axis.annotate(
            str(row["model_display_name"]),
            (row["trainable_parameters"] / 1_000_000, row["inference_images_per_second"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("Accuracy-efficiency trade-off")
    axis.set_xlabel("Trainable parameters (millions)")
    axis.set_ylabel("Inference throughput (images/s)")
    axis.grid(alpha=0.25)
    save_figure(fig, output, dpi)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    rename = {
        "model_display_name": "Model",
        "mean_iou": "mIoU",
        "mean_dice": "Dice",
        "mean_f1": "F1",
        "mean_precision": "Precision",
        "mean_recall": "Recall",
        "pixel_accuracy": "Pixel Accuracy",
        "inference_images_per_second": "Images/s",
        "inference_milliseconds_per_image": "ms/image",
        "trainable_parameters": "Parameters",
        "training_time_hours": "Training Hours",
        "peak_gpu_memory_mb": "Peak GPU MB",
    }
    columns = [column for column in rename if column in display.columns]
    display = display[columns].rename(columns=rename)
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column == "Parameters":
            display[column] = display[column].map(lambda value: f"{int(value):,}" if pd.notna(value) else "")
        else:
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_markdown(index=False)


def write_summary(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
    warnings: list[str],
    weights: tuple[float, ...],
) -> None:
    lines = [
        "# Final Model Benchmark Summary",
        "",
        "## Registered experiments",
        "",
        markdown_table(frame),
        "",
        "## Balanced ranking",
        "",
        ranking.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The balanced score is a transparent decision aid, not a statistical test. "
        f"Weights: mIoU={weights[0]:.2f}, speed={weights[1]:.2f}, "
        f"parameter efficiency={weights[2]:.2f}, training-time efficiency={weights[3]:.2f}.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    registry = load_registry(args.registry)
    expected_missing = [name for name in MODEL_ORDER if name not in set(registry["model_name"])]
    if expected_missing:
        warnings.append(
            "Planned models missing from the registry: " + ", ".join(expected_missing)
        )

    benchmark_columns = [
        "model_name",
        "model_display_name",
        "model_class",
        "experiment_name",
        "completed_epochs",
        "best_epoch",
        "pixel_accuracy",
        "mean_iou",
        "mean_dice",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "inference_images_per_second",
        "inference_milliseconds_per_image",
        "peak_gpu_memory_mb",
        "trainable_parameters",
        "training_time_seconds",
        "training_time_hours",
        "device",
        "config_path",
        "best_checkpoint",
    ]
    benchmark = registry[[column for column in benchmark_columns if column in registry.columns]].copy()
    benchmark.to_csv(output / "final_benchmark_table.csv", index=False, encoding="utf-8-sig")
    (output / "final_benchmark_table.md").write_text(
        markdown_table(benchmark) + "\n", encoding="utf-8"
    )

    per_class = load_per_class(registry, warnings)
    if not per_class.empty:
        per_class.to_csv(output / "per_class_metrics_all_models.csv", index=False, encoding="utf-8-sig")
        per_class.pivot(
            index="model_display_name", columns="class_name", values="iou"
        ).reindex(columns=CLASS_ORDER).to_csv(
            output / "per_class_iou_table.csv", encoding="utf-8-sig"
        )

    ranking = build_ranking(registry, tuple(args.balanced_weights))
    ranking.to_csv(output / "model_ranking.csv", index=False, encoding="utf-8-sig")

    bar_plot(registry, "mean_iou", "Mean IoU comparison", "Mean IoU", plots / "mean_iou_comparison", args.dpi)
    bar_plot(registry, "mean_dice", "Mean Dice comparison", "Mean Dice", plots / "mean_dice_comparison", args.dpi)
    bar_plot(registry, "pixel_accuracy", "Pixel accuracy comparison", "Pixel accuracy", plots / "pixel_accuracy_comparison", args.dpi)
    bar_plot(registry, "inference_images_per_second", "Inference throughput comparison", "Images per second", plots / "inference_speed_comparison", args.dpi)
    bar_plot(registry, "trainable_parameters", "Model size comparison", "Parameters (millions)", plots / "parameter_comparison", args.dpi, scale=1 / 1_000_000)
    bar_plot(registry, "training_time_hours", "Training-time comparison", "Training time (hours)", plots / "training_time_comparison", args.dpi)
    grouped_per_class_plot(per_class, plots / "per_class_iou_comparison", args.dpi)
    efficiency_plot(registry, plots / "accuracy_efficiency_tradeoff", args.dpi)

    write_summary(
        benchmark,
        ranking,
        output / "benchmark_summary.md",
        warnings,
        tuple(args.balanced_weights),
    )
    (output / "benchmark_warnings.txt").write_text(
        "\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8"
    )
    metadata = {
        "registry": str(args.registry),
        "output": str(output),
        "completed_model_count": int(len(registry)),
        "registered_models": registry["model_name"].tolist(),
        "missing_planned_models": expected_missing,
        "balanced_ranking_weights": {
            "mean_iou": args.balanced_weights[0],
            "inference_speed": args.balanced_weights[1],
            "parameter_efficiency": args.balanced_weights[2],
            "training_time_efficiency": args.balanced_weights[3],
        },
    }
    (output / "benchmark_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Benchmark models: {len(registry)}")
    print(f"Output directory: {output}")
    print(f"Final table: {output / 'final_benchmark_table.csv'}")
    print(f"Ranking: {output / 'model_ranking.csv'}")
    if warnings:
        print(f"Warnings: {output / 'benchmark_warnings.txt'}")


if __name__ == "__main__":
    main()
