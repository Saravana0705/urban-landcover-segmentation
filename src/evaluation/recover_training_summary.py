"""Recover a missing training-summary JSON from durable experiment artifacts.

This utility is intended for completed or interrupted runs where checkpoints and
training history were saved, but the final metadata summary was not written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from src.models.model_factory import build_model
from src.training.train_segmentation import (
    model_display_name,
    nested_get,
    normalise_model_name,
    read_yaml,
)


def _first_present(row: Mapping[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, "", "None", "nan", "NaN"):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    return rows


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        last_error: PermissionError | None = None
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as error:
                last_error = error
                time.sleep(min(0.10 * (2**attempt), 2.0))
        raise PermissionError(
            f"Could not replace '{path}' after 8 attempts."
        ) from last_error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def recover_training_summary(
    config_path: Path,
    *,
    force: bool = False,
) -> Path:
    config = read_yaml(config_path)
    model_name = normalise_model_name(config)
    display_name = model_display_name(config)
    experiment_name = str(
        nested_get(config, "experiment.name", "experiment_name")
    )
    experiment_root = Path(
        nested_get(
            config,
            "experiment.output_directory",
            "output_directory",
            default=f"outputs/model_experiments/{experiment_name}",
        )
    )
    output_path = (
        Path("metadata/model_development")
        / f"{model_name}_training_summary.json"
    )
    if output_path.exists() and not force:
        return output_path

    history_path = experiment_root / "logs/training_history.csv"
    rows = _read_history(history_path)
    state = _read_optional_json(
        experiment_root / "reports/experiment_state.json"
    )

    monitor = str(
        state.get("monitor")
        or nested_get(config, "training.monitor", default="mean_iou")
    )
    monitor_mode = str(
        state.get("monitor_mode")
        or nested_get(config, "training.monitor_mode", default="max")
    ).lower()

    metric_names = (
        monitor,
        f"val_{monitor}",
        "mean_iou",
        "val_mean_iou",
        "metric",
    )
    candidates: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        metric = _first_present(row, *metric_names)
        if metric is not None:
            candidates.append((index, metric))
    if not candidates:
        raise ValueError(
            "Could not recover the monitored metric from training history. "
            f"Checked columns: {metric_names}; available: {list(rows[0])}"
        )

    best_index, best_metric = (
        min(candidates, key=lambda item: item[1])
        if monitor_mode == "min"
        else max(candidates, key=lambda item: item[1])
    )
    best_row = rows[best_index]
    best_epoch_value = _first_present(best_row, "epoch")
    best_epoch = (
        int(best_epoch_value) if best_epoch_value is not None else best_index
    )

    last_row = rows[-1]
    global_step_value = _first_present(last_row, "global_step", "step")
    global_step = int(global_step_value or 0)
    optimizer_step = int(
        state.get("optimizer_step")
        or global_step
    )
    duration_seconds = sum(
        _first_present(row, "epoch_duration_seconds", "duration_seconds") or 0.0
        for row in rows
    )
    if duration_seconds <= 0:
        duration_seconds = state.get("duration_seconds")

    model = build_model(config)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    model_class = type(model).__name__

    best_checkpoint = experiment_root / "checkpoints/best.pt"
    latest_checkpoint = experiment_root / "checkpoints/latest.pt"
    history_json = experiment_root / "logs/training_history.json"

    summary = {
        "completed_epochs": len(rows),
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "best_metric": float(best_metric),
        "best_epoch": best_epoch,
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "stopped_early": bool(state.get("stopped_early", False)),
        "device": str(
            state.get("device")
            or nested_get(config, "training.device", "device", default="auto")
        ),
        "mixed_precision_enabled": bool(
            state.get("mixed_precision_enabled", False)
        ),
        "duration_seconds": duration_seconds,
        "checkpoint_directory": str(experiment_root / "checkpoints"),
    }

    checks = {
        "training_completed": len(rows) >= 1,
        "global_step_positive": global_step > 0,
        "optimizer_step_positive": optimizer_step > 0,
        "best_metric_recorded": best_metric is not None,
        "best_epoch_recorded": best_epoch is not None,
        "latest_checkpoint_exists": latest_checkpoint.exists(),
        "best_checkpoint_exists": best_checkpoint.exists(),
        "history_exists": history_path.exists() or history_json.exists(),
    }
    checks["all_checks_passed"] = all(checks.values())

    report = {
        "model_name": model_name,
        "model_display_name": display_name,
        "model_class": model_class,
        "experiment_name": experiment_name,
        "config": str(config_path),
        "model_config": dict(config.get("model", {})),
        "integration_check": False,
        "trainable_parameters": trainable_parameters,
        "experiment_root": str(experiment_root),
        "summary": summary,
        "peak_gpu_memory_mb": None,
        "inference_benchmark": {},
        "best_checkpoint_metrics": {},
        "evaluation_outputs": {},
        "checks": checks,
        "recovery": {
            "recovered": True,
            "source_history": str(history_path),
            "source_state": str(
                experiment_root / "reports/experiment_state.json"
            ),
        },
    }
    _atomic_write_json(output_path, report)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = recover_training_summary(args.config, force=args.force)
    report = json.loads(output.read_text(encoding="utf-8"))
    summary = report["summary"]
    print(f"Recovered: {output}")
    print(f"Experiment: {report['experiment_name']}")
    print(f"Completed epochs: {summary['completed_epochs']}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best {summary['monitor']}: {summary['best_metric']:.6f}")
    print(f"All checks passed: {report['checks']['all_checks_passed']}")


if __name__ == "__main__":
    main()
