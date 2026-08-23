"""Architecture-independent full-dataset semantic-segmentation evaluation.

Evaluates a model with a single dataset-level confusion matrix and exports
thesis-ready overall and per-class CSV files plus a complete JSON report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from src.evaluation.metrics import MetricsConfig, SegmentationMetrics


def _extract_batch(batch: Any) -> tuple[Tensor, Tensor, Tensor | None]:
    if isinstance(batch, Mapping):
        image = next((batch.get(k) for k in ("image", "images", "input", "inputs", "x") if isinstance(batch.get(k), Tensor)), None)
        target = next((batch.get(k) for k in ("mask", "target", "targets", "label", "labels", "y") if isinstance(batch.get(k), Tensor)), None)
        validity = next((batch.get(k) for k in ("validity", "validity_mask", "valid_mask") if isinstance(batch.get(k), Tensor)), None)
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        image, target = batch[0], batch[1]
        validity = batch[2] if len(batch) >= 3 and isinstance(batch[2], Tensor) else None
    else:
        raise TypeError("Unsupported evaluation batch structure.")
    if not isinstance(image, Tensor) or not isinstance(target, Tensor):
        raise TypeError("Evaluation batch must contain image and target tensors.")
    return image, target, validity


def evaluate_segmentation_model(
    *,
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    num_classes: int = 5,
    ignore_index: int = 255,
    class_names: tuple[str, ...] = (
        "buildings", "roads", "vegetation", "bare_land", "water"
    ),
) -> dict[str, Any]:
    """Evaluate all validation/test pixels using one confusion matrix."""
    accumulator = SegmentationMetrics(
        MetricsConfig(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_names=class_names,
            include_absent_classes_in_macro=False,
        ),
        device="cpu",
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image, target, validity = _extract_batch(batch)
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            target = target.to(device, dtype=torch.long, non_blocking=True)
            if validity is not None:
                validity = validity.to(device, dtype=torch.bool, non_blocking=True)
                target = target.clone()
                target[~validity] = ignore_index
            logits = model(image)
            accumulator.update(logits.detach().cpu(), target.detach().cpu())
    return accumulator.compute()


def write_metrics_outputs(report: Mapping[str, Any], output_directory: Path) -> dict[str, Path]:
    """Write overall CSV, per-class CSV, confusion matrix CSV, and JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    overall_path = output_directory / "overall_metrics.csv"
    per_class_path = output_directory / "per_class_metrics.csv"
    confusion_path = output_directory / "confusion_matrix.csv"
    json_path = output_directory / "evaluation_metrics.json"

    overall_fields = (
        "pixel_accuracy", "mean_iou", "mean_dice", "mean_f1",
        "mean_precision", "mean_recall", "frequency_weighted_iou",
        "frequency_weighted_dice", "total_supervised_pixels",
        "correctly_classified_pixels", "batch_count", "macro_class_policy",
    )
    with overall_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall_fields)
        writer.writeheader()
        writer.writerow({field: report.get(field) for field in overall_fields})

    class_fields = (
        "class_id", "class_name", "target_pixels", "predicted_pixels",
        "true_positive", "false_positive", "false_negative", "iou", "dice",
        "precision", "recall", "f1", "present_in_target",
    )
    with per_class_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fields)
        writer.writeheader()
        for class_name, values in report.get("per_class", {}).items():
            row = dict(values)
            row["class_name"] = class_name
            writer.writerow({field: row.get(field) for field in class_fields})

    matrix = report.get("confusion_matrix", [])
    class_names = list(report.get("per_class", {}).keys())
    with confusion_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])

    json_path.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")
    return {
        "overall_metrics": overall_path,
        "per_class_metrics": per_class_path,
        "confusion_matrix": confusion_path,
        "evaluation_json": json_path,
    }
