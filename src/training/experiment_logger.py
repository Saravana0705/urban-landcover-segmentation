"""Experiment logging utilities for segmentation-model training.

The logger stores epoch-level metrics in:
- CSV for quick inspection and plotting;
- JSON for structured reproducibility;
- latest.json for checkpoint/resume workflows;
- TensorBoard when the optional dependency is available.

The implementation is architecture-independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CLASS_NAMES = (
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        return _json_safe(value.item())

    return str(value)


def _atomic_write_text(
    destination: Path,
    text: str,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(text)
        temporary_path = Path(
            temporary_file.name
        )

    try:
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class LoggerConfig:
    experiment_name: str
    output_directory: Path
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    enable_tensorboard: bool = True
    flush_every_epoch: bool = True

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError(
                "experiment_name cannot be empty."
            )

        if not self.class_names:
            raise ValueError(
                "class_names cannot be empty."
            )

        if len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError(
                "class_names must be unique."
            )


@dataclass
class ExperimentState:
    experiment_name: str
    created_at_utc: str
    updated_at_utc: str
    completed: bool = False
    best_epoch: int | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    last_epoch: int | None = None
    history_length: int = 0
    extra: dict[str, Any] = field(
        default_factory=dict
    )


class ExperimentLogger:
    """Resume-safe epoch logger for segmentation experiments."""

    BASE_FIELDS = (
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "epoch_duration_seconds",
        "train_supervised_pixels",
        "val_supervised_pixels",
        "val_pixel_accuracy",
        "val_mean_iou",
        "val_mean_dice",
        "val_macro_precision",
        "val_macro_recall",
        "is_best",
        "timestamp_utc",
    )

    def __init__(
        self,
        config: LoggerConfig,
        *,
        resume: bool = True,
    ) -> None:
        config.validate()

        self.config = config
        self.root = config.output_directory
        self.logs_directory = (
            self.root / "logs"
        )
        self.reports_directory = (
            self.root / "reports"
        )

        self.csv_path = (
            self.logs_directory
            / "training_history.csv"
        )
        self.json_path = (
            self.logs_directory
            / "training_history.json"
        )
        self.latest_path = (
            self.logs_directory
            / "latest.json"
        )
        self.state_path = (
            self.reports_directory
            / "experiment_state.json"
        )

        self.logs_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.reports_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history: list[
            dict[str, Any]
        ] = []

        if resume and self.json_path.exists():
            loaded = json.loads(
                self.json_path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(loaded, list):
                raise TypeError(
                    "training_history.json must "
                    "contain a list."
                )

            self.history = [
                dict(item) for item in loaded
            ]

        if self.state_path.exists() and resume:
            raw_state = json.loads(
                self.state_path.read_text(
                    encoding="utf-8"
                )
            )
            self.state = ExperimentState(
                experiment_name=str(
                    raw_state[
                        "experiment_name"
                    ]
                ),
                created_at_utc=str(
                    raw_state[
                        "created_at_utc"
                    ]
                ),
                updated_at_utc=str(
                    raw_state[
                        "updated_at_utc"
                    ]
                ),
                completed=bool(
                    raw_state.get(
                        "completed",
                        False,
                    )
                ),
                best_epoch=raw_state.get(
                    "best_epoch"
                ),
                best_metric_name=(
                    raw_state.get(
                        "best_metric_name"
                    )
                ),
                best_metric_value=(
                    raw_state.get(
                        "best_metric_value"
                    )
                ),
                last_epoch=raw_state.get(
                    "last_epoch"
                ),
                history_length=int(
                    raw_state.get(
                        "history_length",
                        len(self.history),
                    )
                ),
                extra=dict(
                    raw_state.get(
                        "extra",
                        {},
                    )
                ),
            )
        else:
            now = _utc_now()
            self.state = ExperimentState(
                experiment_name=(
                    config.experiment_name
                ),
                created_at_utc=now,
                updated_at_utc=now,
                history_length=len(
                    self.history
                ),
            )

        if (
            self.state.experiment_name
            != config.experiment_name
        ):
            raise ValueError(
                "Existing logger state belongs to "
                "a different experiment."
            )

        self.tensorboard_writer = None
        self.tensorboard_available = False

        if config.enable_tensorboard:
            try:
                from torch.utils.tensorboard import (
                    SummaryWriter,
                )

                self.tensorboard_writer = (
                    SummaryWriter(
                        log_dir=str(
                            self.logs_directory
                            / "tensorboard"
                        )
                    )
                )
                self.tensorboard_available = True
            except ImportError:
                self.tensorboard_writer = None
                self.tensorboard_available = False

    @property
    def field_names(self) -> list[str]:
        fields = list(self.BASE_FIELDS)

        for class_name in self.config.class_names:
            fields.extend(
                [
                    f"val_iou_{class_name}",
                    f"val_dice_{class_name}",
                    f"val_precision_{class_name}",
                    f"val_recall_{class_name}",
                ]
            )

        return fields

    @property
    def last_epoch(self) -> int | None:
        if not self.history:
            return None

        return int(
            self.history[-1]["epoch"]
        )

    def _normalise_class_metrics(
        self,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}

        per_class = metrics.get(
            "per_class",
            {},
        )

        if not isinstance(per_class, Mapping):
            per_class = {}

        for class_name in self.config.class_names:
            class_metrics = per_class.get(
                class_name,
                {},
            )

            if not isinstance(
                class_metrics,
                Mapping,
            ):
                class_metrics = {}

            output[
                f"val_iou_{class_name}"
            ] = class_metrics.get("iou")

            output[
                f"val_dice_{class_name}"
            ] = class_metrics.get("dice")

            output[
                f"val_precision_{class_name}"
            ] = class_metrics.get(
                "precision"
            )

            output[
                f"val_recall_{class_name}"
            ] = class_metrics.get(
                "recall"
            )

        return output

    def _build_record(
        self,
        *,
        epoch: int,
        global_step: int,
        train_metrics: Mapping[str, Any],
        validation_metrics: Mapping[str, Any],
        learning_rate: float,
        epoch_duration_seconds: float,
        is_best: bool,
    ) -> dict[str, Any]:
        if epoch < 0:
            raise ValueError(
                "epoch cannot be negative."
            )

        if global_step < 0:
            raise ValueError(
                "global_step cannot be negative."
            )

        if not math.isfinite(
            learning_rate
        ):
            raise ValueError(
                "learning_rate must be finite."
            )

        if epoch_duration_seconds < 0:
            raise ValueError(
                "epoch duration cannot be negative."
            )

        record: dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": train_metrics.get(
                "loss"
            ),
            "val_loss": validation_metrics.get(
                "loss"
            ),
            "learning_rate": float(
                learning_rate
            ),
            "epoch_duration_seconds": float(
                epoch_duration_seconds
            ),
            "train_supervised_pixels": (
                train_metrics.get(
                    "supervised_pixels"
                )
            ),
            "val_supervised_pixels": (
                validation_metrics.get(
                    "supervised_pixels"
                )
            ),
            "val_pixel_accuracy": (
                validation_metrics.get(
                    "pixel_accuracy"
                )
            ),
            "val_mean_iou": (
                validation_metrics.get(
                    "mean_iou"
                )
            ),
            "val_mean_dice": (
                validation_metrics.get(
                    "mean_dice"
                )
            ),
            "val_macro_precision": (
                validation_metrics.get(
                    "macro_precision"
                )
            ),
            "val_macro_recall": (
                validation_metrics.get(
                    "macro_recall"
                )
            ),
            "is_best": bool(is_best),
            "timestamp_utc": _utc_now(),
        }

        record.update(
            self._normalise_class_metrics(
                validation_metrics
            )
        )

        return _json_safe(record)

    def log_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        train_metrics: Mapping[str, Any],
        validation_metrics: Mapping[str, Any],
        learning_rate: float,
        epoch_duration_seconds: float,
        is_best: bool = False,
        best_metric_name: str = "val_mean_iou",
    ) -> dict[str, Any]:
        record = self._build_record(
            epoch=epoch,
            global_step=global_step,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            learning_rate=learning_rate,
            epoch_duration_seconds=(
                epoch_duration_seconds
            ),
            is_best=is_best,
        )

        existing_index = next(
            (
                index
                for index, item
                in enumerate(self.history)
                if int(item["epoch"]) == epoch
            ),
            None,
        )

        if existing_index is None:
            if (
                self.last_epoch is not None
                and epoch < self.last_epoch
            ):
                raise ValueError(
                    "Cannot append an epoch older "
                    "than the latest logged epoch."
                )

            self.history.append(record)
        else:
            self.history[
                existing_index
            ] = record

        self.history.sort(
            key=lambda item: int(
                item["epoch"]
            )
        )

        self.state.last_epoch = int(epoch)
        self.state.history_length = len(
            self.history
        )
        self.state.updated_at_utc = (
            _utc_now()
        )

        if is_best:
            metric_value = record.get(
                best_metric_name
            )

            if metric_value is None:
                raise ValueError(
                    f"Best metric "
                    f"{best_metric_name!r} "
                    "is absent from the record."
                )

            self.state.best_epoch = int(
                epoch
            )
            self.state.best_metric_name = (
                best_metric_name
            )
            self.state.best_metric_value = (
                float(metric_value)
            )

        self._log_tensorboard(record)

        if self.config.flush_every_epoch:
            self.flush()

        return record

    def _log_tensorboard(
        self,
        record: Mapping[str, Any],
    ) -> None:
        if self.tensorboard_writer is None:
            return

        epoch = int(record["epoch"])

        for key, value in record.items():
            if key in {
                "epoch",
                "global_step",
                "timestamp_utc",
                "is_best",
            }:
                continue

            if isinstance(
                value,
                (int, float),
            ):
                self.tensorboard_writer.add_scalar(
                    key,
                    value,
                    epoch,
                )

        self.tensorboard_writer.flush()

    def flush(self) -> None:
        csv_lines: list[str] = []

        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            delete=False,
        ) as temporary_csv:
            writer = csv.DictWriter(
                temporary_csv,
                fieldnames=self.field_names,
                extrasaction="ignore",
            )
            writer.writeheader()

            for record in self.history:
                writer.writerow(record)

            temp_csv_path = Path(
                temporary_csv.name
            )

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            shutil.move(
                str(temp_csv_path),
                str(self.csv_path),
            )
        finally:
            if temp_csv_path.exists():
                temp_csv_path.unlink()

        _atomic_write_text(
            self.json_path,
            json.dumps(
                _json_safe(self.history),
                indent=2,
                ensure_ascii=False,
            ),
        )

        latest = (
            {}
            if not self.history
            else self.history[-1]
        )

        _atomic_write_text(
            self.latest_path,
            json.dumps(
                _json_safe(latest),
                indent=2,
                ensure_ascii=False,
            ),
        )

        _atomic_write_text(
            self.state_path,
            json.dumps(
                _json_safe(
                    asdict(self.state)
                ),
                indent=2,
                ensure_ascii=False,
            ),
        )

    def mark_completed(
        self,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.state.completed = True
        self.state.updated_at_utc = (
            _utc_now()
        )

        if extra:
            self.state.extra.update(
                _json_safe(dict(extra))
            )

        self.flush()

    def close(self) -> None:
        self.flush()

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


def smoke_test(
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        shutil.rmtree(output_directory)

    config = LoggerConfig(
        experiment_name=(
            "experiment_logger_smoke_test"
        ),
        output_directory=(
            output_directory
        ),
        enable_tensorboard=True,
    )

    train_0 = {
        "loss": 1.80,
        "supervised_pixels": 100_000,
    }

    val_0 = {
        "loss": 1.65,
        "supervised_pixels": 25_000,
        "pixel_accuracy": 0.50,
        "mean_iou": 0.25,
        "mean_dice": 0.38,
        "macro_precision": 0.42,
        "macro_recall": 0.40,
        "per_class": {
            name: {
                "iou": 0.20 + index * 0.02,
                "dice": 0.33 + index * 0.02,
                "precision": 0.40,
                "recall": 0.39,
            }
            for index, name in enumerate(
                DEFAULT_CLASS_NAMES
            )
        },
    }

    train_1 = {
        "loss": 1.40,
        "supervised_pixels": 100_500,
    }

    val_1 = {
        "loss": 1.30,
        "supervised_pixels": 25_200,
        "pixel_accuracy": 0.61,
        "mean_iou": 0.34,
        "mean_dice": 0.48,
        "macro_precision": 0.52,
        "macro_recall": 0.49,
        "per_class": {
            name: {
                "iou": 0.28 + index * 0.02,
                "dice": 0.44 + index * 0.02,
                "precision": 0.51,
                "recall": 0.48,
            }
            for index, name in enumerate(
                DEFAULT_CLASS_NAMES
            )
        },
    }

    logger = ExperimentLogger(
        config,
        resume=False,
    )

    logger.log_epoch(
        epoch=0,
        global_step=100,
        train_metrics=train_0,
        validation_metrics=val_0,
        learning_rate=1e-3,
        epoch_duration_seconds=12.5,
        is_best=True,
    )

    logger.log_epoch(
        epoch=1,
        global_step=200,
        train_metrics=train_1,
        validation_metrics=val_1,
        learning_rate=5e-4,
        epoch_duration_seconds=11.8,
        is_best=True,
    )

    logger.close()

    resumed = ExperimentLogger(
        config,
        resume=True,
    )

    resumed.log_epoch(
        epoch=1,
        global_step=200,
        train_metrics=train_1,
        validation_metrics=val_1,
        learning_rate=5e-4,
        epoch_duration_seconds=11.8,
        is_best=True,
    )

    resumed.mark_completed(
        extra={
            "status": "PASS",
        }
    )
    resumed.close()

    csv_rows: list[
        dict[str, str]
    ] = []

    with config.output_directory.joinpath(
        "logs/training_history.csv"
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        csv_rows = list(
            csv.DictReader(csv_file)
        )

    json_history = json.loads(
        config.output_directory.joinpath(
            "logs/training_history.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    state = json.loads(
        config.output_directory.joinpath(
            "reports/experiment_state.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    checks = {
        "csv_created": config.output_directory.joinpath(
            "logs/training_history.csv"
        ).exists(),
        "json_created": config.output_directory.joinpath(
            "logs/training_history.json"
        ).exists(),
        "latest_created": config.output_directory.joinpath(
            "logs/latest.json"
        ).exists(),
        "state_created": config.output_directory.joinpath(
            "reports/experiment_state.json"
        ).exists(),
        "two_unique_epochs_logged": (
            len(csv_rows) == 2
            and len(json_history) == 2
        ),
        "resume_did_not_duplicate_epoch": (
            len(json_history) == 2
        ),
        "best_epoch_recorded": (
            state["best_epoch"] == 1
        ),
        "best_metric_recorded": (
            abs(
                state["best_metric_value"]
                - 0.34
            )
            < 1e-12
        ),
        "completion_recorded": bool(
            state["completed"]
        ),
        "per_class_fields_recorded": all(
            f"val_iou_{name}" in csv_rows[-1]
            for name in DEFAULT_CLASS_NAMES
        ),
    }

    checks["all_checks_passed"] = all(
        checks.values()
    )

    if not checks["all_checks_passed"]:
        failed = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "Experiment logger smoke test failed: "
            + ", ".join(failed)
        )

    return {
        "experiment_name": (
            config.experiment_name
        ),
        "output_directory": str(
            config.output_directory
        ),
        "tensorboard_available": (
            resumed.tensorboard_available
        ),
        "history_length": len(
            json_history
        ),
        "best_epoch": state[
            "best_epoch"
        ],
        "best_metric_name": state[
            "best_metric_name"
        ],
        "best_metric_value": state[
            "best_metric_value"
        ],
        "checks": checks,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run experiment-logger validation."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/model_experiments/"
            "experiment_logger_smoke_test"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "metadata/model_development/"
            "experiment_logger_smoke_test.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    report = smoke_test(
        args.output_dir
    )

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _atomic_write_text(
        args.report,
        json.dumps(
            _json_safe(report),
            indent=2,
            ensure_ascii=False,
        ),
    )

    print("\nExperiment logger smoke test")
    print("----------------------------")
    print(
        f"Experiment: "
        f"{report['experiment_name']}"
    )
    print(
        f"History length: "
        f"{report['history_length']}"
    )
    print(
        f"Best epoch: "
        f"{report['best_epoch']}"
    )
    print(
        f"Best metric: "
        f"{report['best_metric_name']}="
        f"{report['best_metric_value']}"
    )
    print(
        "TensorBoard available: "
        f"{report['tensorboard_available']}"
    )

    for name, passed in report[
        "checks"
    ].items():
        print(f"{name}: {passed}")

    print(
        f"Output directory: "
        f"{report['output_directory']}"
    )
    print(
        f"Report: {args.report}"
    )
    print(
        "\nResult: experiment logger passed "
        "CSV, JSON, resume, and "
        "best-epoch validation."
    )


if __name__ == "__main__":
    main()
