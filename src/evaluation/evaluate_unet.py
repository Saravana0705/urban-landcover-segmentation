"""Post-training evaluation for Dataset V1 U-Net experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor
from tqdm.auto import tqdm

from src.training.train_unet import (
    CLASS_NAMES,
    IGNORE_INDEX,
    build_project_dataloaders,
    build_unet_from_config,
    nested_get,
    read_yaml,
    set_seed,
)
from src.training.training_state import load_checkpoint


COLOURS = ["#d9d9d9", "#e41a1c", "#ffd92f", "#1a9850", "#a65628", "#2c7fb8"]


def resolve_device(name: str) -> torch.device:
    name = name.lower().strip()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def unpack_batch(batch: Any) -> tuple[Tensor, Tensor, Tensor | None, dict[str, Any]]:
    if isinstance(batch, Mapping):
        image = batch.get("image", batch.get("images"))
        target = batch.get("target", batch.get("mask"))
        validity = batch.get("validity", batch.get("validity_mask"))
        metadata = {
            str(k): v for k, v in batch.items()
            if k not in {"image", "images", "target", "mask", "validity", "validity_mask"}
        }
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        image, target = batch[0], batch[1]
        validity = batch[2] if len(batch) >= 3 else None
        metadata = {}
    else:
        raise TypeError("Unsupported DataLoader batch format.")

    if not isinstance(image, Tensor) or not isinstance(target, Tensor):
        raise TypeError("Image and target must be tensors.")
    return image, target, validity, metadata


def update_confusion(
    confusion: Tensor,
    prediction: Tensor,
    target: Tensor,
    validity: Tensor | None,
) -> None:
    valid = target != IGNORE_INDEX
    if validity is not None:
        valid &= validity.bool()

    target = target[valid].to(torch.int64)
    prediction = prediction[valid].to(torch.int64)
    if target.numel() == 0:
        return

    encoded = target * len(CLASS_NAMES) + prediction
    confusion += torch.bincount(
        encoded,
        minlength=len(CLASS_NAMES) ** 2,
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES)).cpu()


def divide(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def metrics_from_confusion(confusion: Tensor) -> dict[str, Any]:
    matrix = confusion.to(torch.float64)
    tp = torch.diag(matrix)
    target_support = matrix.sum(dim=1)
    prediction_support = matrix.sum(dim=0)
    union = target_support + prediction_support - tp

    per_class: dict[str, dict[str, Any]] = {}
    ious, dices, precisions, recalls = [], [], [], []

    for idx, name in enumerate(CLASS_NAMES):
        true_positive = float(tp[idx])
        target_count = float(target_support[idx])
        prediction_count = float(prediction_support[idx])
        union_count = float(union[idx])

        iou = divide(true_positive, union_count)
        dice = divide(2 * true_positive, target_count + prediction_count)
        precision = divide(true_positive, prediction_count)
        recall = divide(true_positive, target_count)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)

        per_class[name] = {
            "class_id": idx,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "target_pixels": int(target_count),
            "predicted_pixels": int(prediction_count),
            "true_positive_pixels": int(true_positive),
        }

    total = float(matrix.sum())
    correct = float(tp.sum())

    return {
        "pixel_accuracy": divide(correct, total),
        "mean_iou": sum(ious) / len(ious),
        "mean_dice": sum(dices) / len(dices),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "supervised_pixels": int(total),
        "per_class": per_class,
    }


def save_confusion_csv(path: Path, confusion: Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target\\prediction", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, confusion.tolist()):
            writer.writerow([name, *row])


def plot_confusion(path: Path, confusion: Tensor, normalised: bool) -> None:
    matrix = confusion.numpy().astype(float)
    if normalised:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix)
    ax.set_xticks(range(5), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Target class")
    ax.set_title("Normalised confusion matrix" if normalised else "Confusion matrix")

    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(5):
        for col in range(5):
            value = matrix[row, col]
            label = f"{value:.2f}" if normalised else f"{int(value):,}"
            ax.text(
                col, row, label, ha="center", va="center",
                color="white" if value > threshold else "black", fontsize=8
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def history_value(record: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def plot_curves(experiment_root: Path, output_dir: Path) -> dict[str, str]:
    history_path = experiment_root / "logs" / "training_history.json"
    if not history_path.exists():
        return {}

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [int(row.get("epoch", i)) + 1 for i, row in enumerate(history)]
    train_loss = [history_value(row, "train_loss") for row in history]
    val_loss = [history_value(row, "val_loss") for row in history]
    val_miou = [history_value(row, "val_mean_iou", "mean_iou") for row in history]
    artifacts: dict[str, str] = {}

    if all(v is not None for v in train_loss + val_loss):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, marker="o", label="Training loss")
        ax.plot(epochs, val_loss, marker="o", label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "training_loss_curve.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        artifacts["loss_curve"] = str(path)

    if all(v is not None for v in val_miou):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, val_miou, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean IoU")
        ax.set_title("Validation mean IoU")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / "validation_miou_curve.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        artifacts["miou_curve"] = str(path)

    return artifacts


def stretch(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(array[finite], [2, 98])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    result = np.clip((array - low) / (high - low), 0, 1)
    result[~finite] = 0
    return result.astype(np.float32)


def sample_label(metadata: Mapping[str, Any], item_index: int, fallback: str) -> str:
    for key in ("tile_id", "tile_ids", "id", "ids"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)) and item_index < len(value):
            return str(value[item_index])
        if isinstance(value, str):
            return value
    return fallback


def plot_prediction(
    path: Path,
    image: Tensor,
    target: Tensor,
    prediction: Tensor,
    validity: Tensor | None,
    title: str,
) -> None:
    image_np = image.cpu().numpy()
    target_np = target.cpu().numpy()
    prediction_np = prediction.cpu().numpy()

    valid = target_np != IGNORE_INDEX
    if validity is not None:
        valid &= validity.cpu().numpy().astype(bool)

    target_display = target_np.astype(np.int16) + 1
    prediction_display = prediction_np.astype(np.int16) + 1
    target_display[~valid] = 0
    prediction_display[~valid] = 0

    error = np.zeros_like(target_display, dtype=np.uint8)
    error[valid & (target_np == prediction_np)] = 1
    error[valid & (target_np != prediction_np)] = 2

    false_colour = np.stack(
        [stretch(image_np[0]), stretch(image_np[1]), stretch(image_np[0] - image_np[1])],
        axis=-1,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(stretch(image_np[0]), cmap="gray")
    axes[0, 0].set_title("Normalised VV")
    axes[0, 1].imshow(stretch(image_np[1]), cmap="gray")
    axes[0, 1].set_title("Normalised VH")
    axes[0, 2].imshow(false_colour)
    axes[0, 2].set_title("False colour: VV / VH / VV−VH")

    class_cmap = ListedColormap(COLOURS)
    axes[1, 0].imshow(target_display, cmap=class_cmap, vmin=0, vmax=5)
    axes[1, 0].set_title("Ground truth")
    axes[1, 1].imshow(prediction_display, cmap=class_cmap, vmin=0, vmax=5)
    axes[1, 1].set_title("Prediction")
    axes[1, 2].imshow(error, cmap=ListedColormap(["#d9d9d9", "#ffffff", "#d73027"]), vmin=0, vmax=2)
    axes[1, 2].set_title("Error map: red=incorrect")

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    output_dir: Path,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], Tensor, list[str]]:
    model.eval()
    confusion = torch.zeros((5, 5), dtype=torch.int64)
    reservoir: list[tuple[Tensor, Tensor, Tensor, Tensor | None, str]] = []
    rng = random.Random(seed)
    seen = 0

    for batch_index, batch in enumerate(tqdm(loader, desc="Evaluation")):
        image, target, validity, metadata = unpack_batch(batch)
        image = image.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.long)
        validity_device = validity.to(device, dtype=torch.bool) if validity is not None else None

        prediction = model(image).argmax(dim=1)
        update_confusion(confusion, prediction, target, validity_device)

        for item_index in range(image.shape[0]):
            seen += 1
            item = (
                image[item_index].detach().cpu(),
                target[item_index].detach().cpu(),
                prediction[item_index].detach().cpu(),
                validity_device[item_index].detach().cpu() if validity_device is not None else None,
                sample_label(metadata, item_index, f"batch_{batch_index:04d}_item_{item_index:02d}"),
            )
            if len(reservoir) < samples:
                reservoir.append(item)
            else:
                replacement = rng.randrange(seen)
                if replacement < samples:
                    reservoir[replacement] = item

    qualitative_paths: list[str] = []
    for index, item in enumerate(reservoir, start=1):
        image, target, prediction, validity, label = item
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        path = output_dir / "qualitative_predictions" / f"{index:02d}_{safe}.png"
        plot_prediction(path, image, target, prediction, validity, label)
        qualitative_paths.append(str(path))

    return metrics_from_confusion(confusion), confusion, qualitative_paths


def write_per_class_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    fields = [
        "class_id", "class_name", "iou", "dice", "precision", "recall",
        "target_pixels", "predicted_pixels", "true_positive_pixels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in CLASS_NAMES:
            row = dict(metrics["per_class"][name])
            row["class_name"] = name
            writer.writerow({key: row[key] for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    seed = int(nested_get(config, "seed", "training.seed", default=20260725))
    set_seed(seed)
    device = resolve_device(args.device)

    loaders = build_project_dataloaders(config, integration_check=False)
    if args.split not in loaders:
        raise KeyError(f"Missing DataLoader split: {args.split}")

    model = build_unet_from_config(config).to(device)
    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        map_location=device,
        restore_rng=False,
    )

    configured_root = nested_get(config, "experiment.output_directory", default=None)
    experiment_root = (
        Path(str(configured_root))
        if configured_root is not None
        else args.checkpoint.parent.parent
    )
    output_dir = args.output_dir or experiment_root / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, confusion, qualitative = run_evaluation(
        model, loaders[args.split], device, output_dir, args.samples, seed
    )

    metrics_path = output_dir / "metrics.json"
    class_csv = output_dir / "per_class_metrics.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_per_class_csv(class_csv, metrics)
    save_confusion_csv(confusion_csv, confusion)
    plot_confusion(output_dir / "confusion_matrix.png", confusion, False)
    plot_confusion(output_dir / "confusion_matrix_normalised.png", confusion, True)
    curves = plot_curves(experiment_root, output_dir)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "device": str(device),
        "metrics": metrics,
        "artifacts": {
            "metrics_json": str(metrics_path),
            "per_class_metrics_csv": str(class_csv),
            "confusion_matrix_csv": str(confusion_csv),
            "confusion_matrix_png": str(output_dir / "confusion_matrix.png"),
            "confusion_matrix_normalised_png": str(output_dir / "confusion_matrix_normalised.png"),
            "qualitative_predictions": qualitative,
            **curves,
        },
        "checks": {
            "checkpoint_loaded": True,
            "supervised_pixels_positive": metrics["supervised_pixels"] > 0,
            "mean_iou_finite": math.isfinite(metrics["mean_iou"]),
            "mean_dice_finite": math.isfinite(metrics["mean_dice"]),
            "all_five_classes_reported": len(metrics["per_class"]) == 5,
            "confusion_total_matches_supervised_pixels": int(confusion.sum()) == metrics["supervised_pixels"],
            "qualitative_sample_count_matches_request": len(qualitative) == args.samples,
        },
    }
    summary["checks"]["all_checks_passed"] = all(summary["checks"].values())

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata_path = Path("metadata/model_development") / f"unet_{args.split}_evaluation_summary.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nU-Net evaluation")
    print("----------------")
    print(f"Split: {args.split}")
    print(f"Checkpoint epoch: {summary['checkpoint_epoch']}")
    print(f"Supervised pixels: {metrics['supervised_pixels']:,}")
    print(f"Pixel accuracy: {metrics['pixel_accuracy']:.6f}")
    print(f"Mean IoU: {metrics['mean_iou']:.6f}")
    print(f"Mean Dice: {metrics['mean_dice']:.6f}")
    print(f"Macro precision: {metrics['macro_precision']:.6f}")
    print(f"Macro recall: {metrics['macro_recall']:.6f}")

    print("\nPer-class metrics")
    print("-----------------")
    for name in CLASS_NAMES:
        row = metrics["per_class"][name]
        print(
            f"{name:12s} IoU={row['iou']:.6f} Dice={row['dice']:.6f} "
            f"Precision={row['precision']:.6f} Recall={row['recall']:.6f}"
        )

    print("\nChecks")
    print("------")
    for name, passed in summary["checks"].items():
        print(f"{name}: {passed}")

    print(f"\nEvaluation directory: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Metadata report: {metadata_path}")
    print("\nResult: U-Net evaluation and visual-report generation completed successfully.")


if __name__ == "__main__":
    main()
