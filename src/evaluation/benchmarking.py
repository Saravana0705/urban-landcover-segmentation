"""Reproducible experiment registry and computational benchmarking utilities."""
from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

REGISTRY_FIELDS = (
    "experiment_name", "model_name", "model_display_name", "model_class",
    "config_path", "status", "completed_at_utc", "device",
    "trainable_parameters", "completed_epochs", "best_epoch",
    "best_metric_name", "best_metric", "training_time_seconds",
    "training_time_hours", "peak_gpu_memory_mb", "inference_images_per_second",
    "inference_milliseconds_per_image", "benchmark_batches", "benchmark_images",
    "pixel_accuracy", "mean_iou", "mean_dice", "mean_f1",
    "mean_precision", "mean_recall", "experiment_root", "best_checkpoint",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _image_from_batch(batch: Any) -> Tensor:
    if isinstance(batch, Mapping):
        for key in ("image", "images", "input", "inputs", "x"):
            value = batch.get(key)
            if isinstance(value, Tensor):
                return value
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], Tensor):
        return batch[0]
    raise TypeError("Unable to extract image tensor from benchmark batch.")


def load_best_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Mapping[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(f"Checkpoint lacks model_state_dict: {checkpoint_path}")
    model.load_state_dict(state)
    return checkpoint


def benchmark_inference(
    *, model: nn.Module, loader: Iterable[Any], device: torch.device,
    warmup_batches: int = 3, measured_batches: int = 20,
) -> dict[str, Any]:
    """Measure batch-1 compatible throughput on real validation batches."""
    if measured_batches < 1:
        raise ValueError("measured_batches must be >= 1")
    model.eval()
    elapsed = 0.0
    images = 0
    measured = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            image = _image_from_batch(batch).to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            if index >= warmup_batches:
                elapsed += duration
                images += int(image.shape[0])
                measured += 1
                if measured >= measured_batches:
                    break
    if images == 0 or elapsed <= 0:
        raise RuntimeError("No inference batches were measured.")
    return {
        "inference_images_per_second": images / elapsed,
        "inference_milliseconds_per_image": 1000.0 * elapsed / images,
        "benchmark_batches": measured,
        "benchmark_images": images,
        "benchmark_elapsed_seconds": elapsed,
    }


def peak_gpu_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            value = _finite(metrics[name])
            if value is not None:
                return value
    return None


def build_registry_row(
    *, experiment_name: str, model_name: str, model_display_name: str,
    model_class: str, config_path: str, trainable_parameters: int,
    experiment_root: Path, summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any], benchmark: Mapping[str, Any],
    peak_memory_mb: float | None,
) -> dict[str, Any]:
    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    duration = _finite(summary.get("duration_seconds"))
    best_epoch_index = summary.get("best_epoch")
    best_epoch = None if best_epoch_index is None else int(best_epoch_index) + 1
    row = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "model_display_name": model_display_name,
        "model_class": model_class,
        "config_path": config_path,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": summary.get("device"),
        "trainable_parameters": trainable_parameters,
        "completed_epochs": summary.get("completed_epochs"),
        "best_epoch": best_epoch,
        "best_metric_name": summary.get("monitor"),
        "best_metric": _finite(summary.get("best_metric")),
        "training_time_seconds": duration,
        "training_time_hours": None if duration is None else duration / 3600.0,
        "peak_gpu_memory_mb": peak_memory_mb,
        "inference_images_per_second": benchmark.get("inference_images_per_second"),
        "inference_milliseconds_per_image": benchmark.get("inference_milliseconds_per_image"),
        "benchmark_batches": benchmark.get("benchmark_batches"),
        "benchmark_images": benchmark.get("benchmark_images"),
        "pixel_accuracy": _metric(metrics, "pixel_accuracy", "overall_accuracy"),
        "mean_iou": _metric(metrics, "mean_iou", "miou"),
        "mean_dice": _metric(metrics, "mean_dice", "dice"),
        "mean_f1": _metric(metrics, "mean_f1", "macro_f1", "f1", "f1_score"),
        "mean_precision": _metric(metrics, "mean_precision", "macro_precision", "precision"),
        "mean_recall": _metric(metrics, "mean_recall", "macro_recall", "recall"),
        "experiment_root": str(experiment_root),
        "best_checkpoint": str(experiment_root / "checkpoints/best.pt"),
    }
    return {field: row.get(field) for field in REGISTRY_FIELDS}


def upsert_registry(csv_path: Path, row: Mapping[str, Any]) -> None:
    """Insert or replace one experiment row atomically by experiment name."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    key = str(row["experiment_name"])
    rows = [item for item in rows if item.get("experiment_name") != key]
    rows.append({field: row.get(field, "") for field in REGISTRY_FIELDS})
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def write_benchmark_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
