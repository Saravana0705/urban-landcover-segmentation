"""Reusable optimizer, scheduler, early-stopping, and checkpoint utilities.

This module is model-independent and is intended to be shared by all semantic
segmentation architectures used in the project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
)

from src.training.training_config import (
    OptimizerConfig,
    SchedulerConfig,
)


@dataclass
class EarlyStoppingState:
    """Serializable state for metric-based early stopping."""

    patience: int
    mode: str = "max"
    minimum_delta: float = 0.0
    best_metric: float | None = None
    bad_epoch_count: int = 0
    stopped: bool = False
    best_epoch: int | None = None

    def validate(self) -> None:
        if self.patience < 0:
            raise ValueError("patience cannot be negative.")

        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")

        if self.minimum_delta < 0:
            raise ValueError(
                "minimum_delta cannot be negative."
            )


class EarlyStopping:
    """Track a validation metric and decide when training should stop."""

    def __init__(
        self,
        patience: int,
        *,
        mode: str = "max",
        minimum_delta: float = 0.0,
    ) -> None:
        self.state = EarlyStoppingState(
            patience=patience,
            mode=mode,
            minimum_delta=minimum_delta,
        )
        self.state.validate()

    def _is_improvement(self, metric: float) -> bool:
        if self.state.best_metric is None:
            return True

        if self.state.mode == "max":
            return (
                metric
                > self.state.best_metric
                + self.state.minimum_delta
            )

        return (
            metric
            < self.state.best_metric
            - self.state.minimum_delta
        )

    def update(
        self,
        metric: float,
        *,
        epoch: int,
    ) -> bool:
        """Update state and return whether training should stop."""
        if not math.isfinite(metric):
            raise ValueError(
                "Early-stopping metric must be finite."
            )

        if epoch < 0:
            raise ValueError("epoch cannot be negative.")

        if self._is_improvement(metric):
            self.state.best_metric = float(metric)
            self.state.bad_epoch_count = 0
            self.state.best_epoch = int(epoch)
            self.state.stopped = False
        else:
            self.state.bad_epoch_count += 1

            if (
                self.state.bad_epoch_count
                >= self.state.patience
            ):
                self.state.stopped = True

        return self.state.stopped

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> None:
        restored = EarlyStoppingState(
            patience=int(state_dict["patience"]),
            mode=str(state_dict["mode"]),
            minimum_delta=float(
                state_dict["minimum_delta"]
            ),
            best_metric=(
                None
                if state_dict.get("best_metric") is None
                else float(state_dict["best_metric"])
            ),
            bad_epoch_count=int(
                state_dict.get("bad_epoch_count", 0)
            ),
            stopped=bool(
                state_dict.get("stopped", False)
            ),
            best_epoch=(
                None
                if state_dict.get("best_epoch") is None
                else int(state_dict["best_epoch"])
            ),
        )
        restored.validate()
        self.state = restored


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> Optimizer:
    """Create an optimizer from validated configuration."""
    config.validate()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError(
            "Model has no trainable parameters."
        )

    if config.name == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.epsilon,
        )

    if config.name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.epsilon,
        )

    if config.name == "sgd":
        return torch.optim.SGD(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.betas[0],
            nesterov=True,
        )

    raise ValueError(
        f"Unsupported optimizer: {config.name}"
    )


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    max_epochs: int,
) -> ReduceLROnPlateau | LRScheduler | None:
    """Create a learning-rate scheduler."""
    config.validate()

    if max_epochs <= 0:
        raise ValueError(
            "max_epochs must be positive."
        )

    if config.name == "none":
        return None

    if config.name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            min_lr=config.minimum_learning_rate,
        )

    if config.name == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=config.minimum_learning_rate,
        )

    raise ValueError(
        f"Unsupported scheduler: {config.name}"
    )


def step_scheduler(
    scheduler: ReduceLROnPlateau | LRScheduler | None,
    *,
    monitored_metric: float | None = None,
) -> None:
    """Step any supported scheduler safely."""
    if scheduler is None:
        return

    if isinstance(
        scheduler,
        ReduceLROnPlateau,
    ):
        if monitored_metric is None:
            raise ValueError(
                "ReduceLROnPlateau requires "
                "a monitored metric."
            )

        if not math.isfinite(monitored_metric):
            raise ValueError(
                "Scheduler metric must be finite."
            )

        scheduler.step(monitored_metric)
        return

    scheduler.step()


def get_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    """Return current learning rates for all parameter groups."""
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def capture_random_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch random states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_random_state(
    state: Mapping[str, Any],
) -> None:
    """Restore Python, NumPy, and PyTorch random states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


def _atomic_torch_save(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write a checkpoint atomically."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temporary_path = Path(temp_file.name)

    try:
        torch.save(
            payload,
            temporary_path,
        )
        os.replace(
            temporary_path,
            destination,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau | LRScheduler | None,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    early_stopping: EarlyStopping | None = None,
    scaler: torch.amp.GradScaler | None = None,
    is_best: bool = False,
    best_path: str | Path | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    """Save all state required to resume training."""
    if epoch < 0:
        raise ValueError("epoch cannot be negative.")

    if global_step < 0:
        raise ValueError(
            "global_step cannot be negative."
        )

    checkpoint_path = Path(path)

    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            None
            if scheduler is None
            else scheduler.state_dict()
        ),
        "scaler_state_dict": (
            None
            if scaler is None
            else scaler.state_dict()
        ),
        "early_stopping_state_dict": (
            None
            if early_stopping is None
            else early_stopping.state_dict()
        ),
        "metrics": {
            str(key): float(value)
            for key, value in metrics.items()
        },
        "config": dict(config),
        "random_state": capture_random_state(),
        "extra_state": (
            {}
            if extra_state is None
            else dict(extra_state)
        ),
    }

    _atomic_torch_save(
        payload,
        checkpoint_path,
    )

    if is_best:
        if best_path is None:
            raise ValueError(
                "best_path is required when is_best=True."
            )

        best_checkpoint_path = Path(best_path)
        best_checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(
            checkpoint_path,
            best_checkpoint_path,
        )

    return checkpoint_path


def load_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: (
        ReduceLROnPlateau
        | LRScheduler
        | None
    ) = None,
    early_stopping: EarlyStopping | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict_model: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load checkpoint and restore requested training state."""
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint root must be a dictionary."
        )

    required = {
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "metrics",
        "config",
    }

    missing = sorted(
        required.difference(checkpoint)
    )

    if missing:
        raise KeyError(
            "Checkpoint is missing fields: "
            + ", ".join(missing)
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict_model,
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get(
            "scheduler_state_dict"
        ) is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if (
        early_stopping is not None
        and checkpoint.get(
            "early_stopping_state_dict"
        ) is not None
    ):
        early_stopping.load_state_dict(
            checkpoint[
                "early_stopping_state_dict"
            ]
        )

    if (
        scaler is not None
        and checkpoint.get(
            "scaler_state_dict"
        ) is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    if (
        restore_rng
        and checkpoint.get("random_state")
        is not None
    ):
        restore_random_state(
            checkpoint["random_state"]
        )

    return checkpoint


def latest_checkpoint(
    checkpoint_directory: str | Path,
    pattern: str = "epoch_*.pt",
) -> Path | None:
    """Return the newest checkpoint by modification time."""
    directory = Path(checkpoint_directory)

    if not directory.exists():
        return None

    candidates = list(
        directory.glob(pattern)
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def smoke_test(
    output_directory: Path,
) -> dict[str, Any]:
    """Validate all training-state utilities."""
    torch.manual_seed(20260725)
    np.random.seed(20260725)
    random.seed(20260725)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )

    optimizer_config = OptimizerConfig(
        name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    scheduler_config = SchedulerConfig(
        name="reduce_on_plateau",
        monitor="val_mean_iou",
        mode="max",
        factor=0.5,
        patience=1,
        minimum_learning_rate=1e-6,
    )

    optimizer = build_optimizer(
        model,
        optimizer_config,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_config,
        max_epochs=10,
    )
    early_stopping = EarlyStopping(
        patience=2,
        mode="max",
        minimum_delta=0.0,
    )

    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    x = torch.randn(6, 4)
    target = torch.randint(0, 3, (6,))

    logits = model(x)
    loss = nn.CrossEntropyLoss()(
        logits,
        target,
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    optimizer_updated_parameters = any(
        not torch.equal(
            initial_parameters[name],
            parameter.detach(),
        )
        for name, parameter
        in model.named_parameters()
    )

    initial_lr = get_learning_rates(
        optimizer
    )[0]

    step_scheduler(
        scheduler,
        monitored_metric=0.50,
    )
    step_scheduler(
        scheduler,
        monitored_metric=0.49,
    )
    step_scheduler(
        scheduler,
        monitored_metric=0.48,
    )

    reduced_lr = get_learning_rates(
        optimizer
    )[0]

    stop_sequence = []
    for epoch, metric in enumerate(
        [0.40, 0.45, 0.44, 0.43]
    ):
        stop_sequence.append(
            early_stopping.update(
                metric,
                epoch=epoch,
            )
        )

    checkpoint_directory = (
        output_directory / "checkpoints"
    )
    checkpoint_path = (
        checkpoint_directory
        / "epoch_0003.pt"
    )
    best_path = (
        checkpoint_directory
        / "best.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=42,
        metrics={
            "val_mean_iou": 0.45,
            "val_loss": 1.2,
        },
        config={
            "experiment_name": (
                "training_state_smoke_test"
            )
        },
        early_stopping=early_stopping,
        is_best=True,
        best_path=best_path,
        extra_state={"note": "smoke test"},
    )

    restored_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    restored_optimizer = build_optimizer(
        restored_model,
        optimizer_config,
    )
    restored_scheduler = build_scheduler(
        restored_optimizer,
        scheduler_config,
        max_epochs=10,
    )
    restored_early_stopping = EarlyStopping(
        patience=2,
        mode="max",
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        early_stopping=restored_early_stopping,
        restore_rng=True,
    )

    model_state_matches = all(
        torch.equal(
            model.state_dict()[key],
            restored_model.state_dict()[key],
        )
        for key in model.state_dict()
    )

    optimizer_lr_matches = (
        get_learning_rates(optimizer)
        == get_learning_rates(
            restored_optimizer
        )
    )

    early_stopping_matches = (
        early_stopping.state_dict()
        == restored_early_stopping.state_dict()
    )

    latest = latest_checkpoint(
        checkpoint_directory
    )

    checks = {
        "optimizer_created": isinstance(
            optimizer,
            torch.optim.AdamW,
        ),
        "optimizer_updated_parameters": (
            optimizer_updated_parameters
        ),
        "scheduler_created": isinstance(
            scheduler,
            ReduceLROnPlateau,
        ),
        "scheduler_reduced_learning_rate": (
            reduced_lr < initial_lr
        ),
        "early_stopping_triggered": bool(
            stop_sequence[-1]
        ),
        "checkpoint_created": (
            checkpoint_path.exists()
        ),
        "best_checkpoint_created": (
            best_path.exists()
        ),
        "checkpoint_epoch_restored": (
            checkpoint["epoch"] == 3
        ),
        "checkpoint_global_step_restored": (
            checkpoint["global_step"] == 42
        ),
        "model_state_restored": (
            model_state_matches
        ),
        "optimizer_state_restored": (
            optimizer_lr_matches
        ),
        "scheduler_state_restored": (
            restored_scheduler is not None
        ),
        "early_stopping_state_restored": (
            early_stopping_matches
        ),
        "latest_checkpoint_detected": (
            latest == checkpoint_path
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
            "Training-state smoke test failed: "
            + ", ".join(failed)
        )

    report = {
        "optimizer": optimizer_config.name,
        "scheduler": scheduler_config.name,
        "initial_learning_rate": initial_lr,
        "reduced_learning_rate": reduced_lr,
        "early_stopping_state": (
            early_stopping.state_dict()
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "best_checkpoint_path": str(
            best_path
        ),
        "latest_checkpoint": (
            None
            if latest is None
            else str(latest)
        ),
        "checks": checks,
    }

    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate optimizer, scheduler, "
            "early stopping, and checkpoints."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/model_experiments/"
            "training_state_smoke_test"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "metadata/model_development/"
            "training_state_smoke_test.json"
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
    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nTraining-state utilities smoke test")
    print("-----------------------------------")
    print(
        f"Optimizer: {report['optimizer']}"
    )
    print(
        f"Scheduler: {report['scheduler']}"
    )
    print(
        "Initial learning rate: "
        f"{report['initial_learning_rate']:.8f}"
    )
    print(
        "Reduced learning rate: "
        f"{report['reduced_learning_rate']:.8f}"
    )
    print(
        "Early stopping best metric: "
        f"{report['early_stopping_state']['best_metric']}"
    )
    print(
        "Early stopping best epoch: "
        f"{report['early_stopping_state']['best_epoch']}"
    )
    print(
        "Early stopping triggered: "
        f"{report['early_stopping_state']['stopped']}"
    )

    for name, passed in report[
        "checks"
    ].items():
        print(f"{name}: {passed}")

    print(
        f"Checkpoint: "
        f"{report['checkpoint_path']}"
    )
    print(
        f"Best checkpoint: "
        f"{report['best_checkpoint_path']}"
    )
    print(
        f"Report: {args.report}"
    )
    print(
        "\nResult: optimizer, scheduler, "
        "early stopping, and checkpoint "
        "utilities passed validation."
    )


if __name__ == "__main__":
    main()
