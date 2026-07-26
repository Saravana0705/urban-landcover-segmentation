"""Architecture-independent semantic-segmentation Trainer.

The Trainer accepts already-built project components:
- model;
- train and validation DataLoaders;
- criterion;
- validation-metrics callable;
- optimizer and scheduler;
- ExperimentLogger;
- EarlyStopping and checkpoint utilities.

This separation keeps the engine reusable for U-Net, Attention U-Net, U-Net++,
DeepLabV3+, SegFormer, Swin-based models, and Mask2Former.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.training.experiment_logger import ExperimentLogger, LoggerConfig
from src.training.training_state import (
    EarlyStopping,
    get_learning_rates,
    load_checkpoint,
    save_checkpoint,
    step_scheduler,
)


class MetricCallable(Protocol):
    """Validation metric callback protocol."""

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        validity: Tensor | None = None,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class TrainerConfig:
    """Runtime controls for the generic Trainer."""

    max_epochs: int = 2
    device: str = "auto"
    use_mixed_precision: bool = True
    gradient_clip_norm: float | None = 1.0
    accumulation_steps: int = 1
    monitor: str = "mean_iou"
    monitor_mode: str = "max"
    save_every_epochs: int = 1
    progress_bar: bool = True
    non_blocking_transfer: bool = True
    restore_rng_on_resume: bool = True

    def validate(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive.")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'.")
        if self.save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be positive.")
        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive or None."
            )


@dataclass
class TrainerState:
    """Mutable training progress."""

    epoch: int = -1
    global_step: int = 0
    optimizer_step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    stopped_early: bool = False


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps device selection."""
    requested = requested.lower().strip()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if device.type == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable.")

    return device


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    if seed < 0:
        raise ValueError("seed cannot be negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_batch(
    batch: Any,
) -> tuple[Tensor, Tensor, Tensor | None, dict[str, Any]]:
    """Support mapping and tuple/list DataLoader batches."""
    metadata: dict[str, Any] = {}

    if isinstance(batch, Mapping):
        image = (
            batch.get("image")
            if "image" in batch
            else batch.get("images")
        )
        target = (
            batch.get("target")
            if "target" in batch
            else batch.get("mask")
        )
        validity = (
            batch.get("validity")
            if "validity" in batch
            else batch.get("validity_mask")
        )

        metadata = {
            str(key): value
            for key, value in batch.items()
            if key
            not in {
                "image",
                "images",
                "target",
                "mask",
                "validity",
                "validity_mask",
            }
        }
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                "Tuple/list batches require image and target."
            )

        image = batch[0]
        target = batch[1]
        validity = batch[2] if len(batch) >= 3 else None
    else:
        raise TypeError(
            "Batch must be a mapping, tuple, or list."
        )

    if not isinstance(image, Tensor):
        raise TypeError("Batch image must be a Tensor.")

    if not isinstance(target, Tensor):
        raise TypeError("Batch target must be a Tensor.")

    if validity is not None and not isinstance(validity, Tensor):
        raise TypeError("Batch validity must be a Tensor or None.")

    return image, target, validity, metadata


def _extract_loss(
    loss_output: Any,
) -> tuple[Tensor, dict[str, float]]:
    """Accept Tensor, mapping, or (Tensor, mapping) criterion outputs."""
    if isinstance(loss_output, Tensor):
        return loss_output, {"loss": float(loss_output.detach().item())}

    if isinstance(loss_output, Mapping):
        candidate = (
            loss_output.get("loss")
            if "loss" in loss_output
            else loss_output.get("total_loss")
        )

        if not isinstance(candidate, Tensor):
            raise TypeError(
                "Loss mapping must contain Tensor 'loss' "
                "or 'total_loss'."
            )

        components: dict[str, float] = {}

        for key, value in loss_output.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                components[str(key)] = float(value.detach().item())
            elif isinstance(value, (int, float)):
                components[str(key)] = float(value)

        components.setdefault(
            "loss",
            float(candidate.detach().item()),
        )
        return candidate, components

    if (
        isinstance(loss_output, (tuple, list))
        and len(loss_output) == 2
        and isinstance(loss_output[0], Tensor)
        and isinstance(loss_output[1], Mapping)
    ):
        total = loss_output[0]
        components = {
            str(key): float(
                value.detach().item()
                if isinstance(value, Tensor)
                else value
            )
            for key, value in loss_output[1].items()
            if (
                isinstance(value, (int, float))
                or (
                    isinstance(value, Tensor)
                    and value.numel() == 1
                )
            )
        }
        components.setdefault("loss", float(total.detach().item()))
        return total, components

    raise TypeError(
        "Criterion must return a Tensor, loss mapping, "
        "or (Tensor, mapping)."
    )


def _call_criterion(
    criterion: Callable[..., Any],
    logits: Tensor,
    target: Tensor,
    validity: Tensor | None,
) -> tuple[Tensor, dict[str, float]]:
    """Call criterion with optional validity-mask support."""
    if validity is not None:
        try:
            return _extract_loss(
                criterion(logits, target, validity)
            )
        except TypeError:
            pass

    return _extract_loss(
        criterion(logits, target)
    )


def _metric_is_better(
    value: float,
    best: float | None,
    mode: str,
) -> bool:
    if best is None:
        return True
    if mode == "max":
        return value > best
    return value < best


def _weighted_merge(
    records: Iterable[tuple[Mapping[str, Any], int]],
) -> dict[str, Any]:
    """Merge scalar and per-class metric dictionaries by weight."""
    records = list(records)

    if not records:
        return {}

    total_weight = sum(max(0, int(weight)) for _, weight in records)

    if total_weight <= 0:
        total_weight = len(records)
        records = [(record, 1) for record, _ in records]

    scalar_sums: dict[str, float] = {}
    scalar_weights: dict[str, int] = {}
    per_class_sums: dict[str, dict[str, float]] = {}
    per_class_weights: dict[str, dict[str, int]] = {}

    for record, weight in records:
        weight = max(1, int(weight))

        for key, value in record.items():
            if key == "per_class" and isinstance(value, Mapping):
                for class_name, class_metrics in value.items():
                    if not isinstance(class_metrics, Mapping):
                        continue

                    class_key = str(class_name)
                    per_class_sums.setdefault(class_key, {})
                    per_class_weights.setdefault(class_key, {})

                    for metric_name, metric_value in class_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            metric_value = float(metric_value)
                            if math.isfinite(metric_value):
                                metric_key = str(metric_name)
                                per_class_sums[class_key][metric_key] = (
                                    per_class_sums[class_key].get(
                                        metric_key, 0.0
                                    )
                                    + metric_value * weight
                                )
                                per_class_weights[class_key][metric_key] = (
                                    per_class_weights[class_key].get(
                                        metric_key, 0
                                    )
                                    + weight
                                )
                continue

            if isinstance(value, (int, float)):
                value = float(value)
                if math.isfinite(value):
                    scalar_sums[key] = scalar_sums.get(key, 0.0) + value * weight
                    scalar_weights[key] = scalar_weights.get(key, 0) + weight

    merged: dict[str, Any] = {
        key: scalar_sums[key] / scalar_weights[key]
        for key in scalar_sums
    }

    if per_class_sums:
        merged["per_class"] = {
            class_name: {
                metric_name: (
                    metric_sum
                    / per_class_weights[class_name][metric_name]
                )
                for metric_name, metric_sum in class_metrics.items()
            }
            for class_name, class_metrics in per_class_sums.items()
        }

    return merged


class GenericTrainer:
    """Reusable training and validation engine."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        criterion: Callable[..., Any],
        metric_function: MetricCallable,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | LRScheduler | None,
        logger: ExperimentLogger,
        early_stopping: EarlyStopping | None,
        checkpoint_directory: str | Path,
        config: TrainerConfig,
        resolved_config: Mapping[str, Any],
        seed: int = 20260725,
    ) -> None:
        config.validate()
        set_reproducible_seed(seed)

        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.early_stopping = early_stopping
        self.checkpoint_directory = Path(checkpoint_directory)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.resolved_config = dict(resolved_config)
        self.seed = seed
        self.device = resolve_device(config.device)
        self.model.to(self.device)

        self.amp_enabled = (
            config.use_mixed_precision
            and self.device.type == "cuda"
        )

        try:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=self.amp_enabled,
            )
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.amp_enabled
            )

        self.state = TrainerState()

    def _autocast_context(self):
        return torch.autocast(
            device_type=self.device.type,
            enabled=self.amp_enabled,
        )

    def _move_batch(
        self,
        batch: Any,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        image, target, validity, _ = _extract_batch(batch)

        image = image.to(
            self.device,
            non_blocking=self.config.non_blocking_transfer,
            dtype=torch.float32,
        )
        target = target.to(
            self.device,
            non_blocking=self.config.non_blocking_transfer,
            dtype=torch.long,
        )

        if validity is not None:
            validity = validity.to(
                self.device,
                non_blocking=self.config.non_blocking_transfer,
                dtype=torch.bool,
            )

        return image, target, validity

    def train_one_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        supervised_pixels = 0
        batch_count = 0
        component_sums: dict[str, float] = {}

        iterator = tqdm(
            self.train_loader,
            desc=f"Train {epoch + 1}/{self.config.max_epochs}",
            leave=False,
            disable=not self.config.progress_bar,
        )

        for batch_index, batch in enumerate(iterator):
            image, target, validity = self._move_batch(batch)

            with self._autocast_context():
                logits = self.model(image)
                loss, components = _call_criterion(
                    self.criterion,
                    logits,
                    target,
                    validity,
                )
                scaled_loss = loss / self.config.accumulation_steps

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, "
                    f"batch {batch_index}."
                )

            self.scaler.scale(scaled_loss).backward()

            should_step = (
                (batch_index + 1) % self.config.accumulation_steps == 0
                or batch_index + 1 == len(self.train_loader)
            )

            if should_step:
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.optimizer_step += 1

            batch_size = int(image.shape[0])
            loss_sum += float(loss.detach().item()) * batch_size
            batch_count += batch_size

            valid_count = (
                int(validity.sum().item())
                if validity is not None
                else int((target != 255).sum().item())
            )
            supervised_pixels += valid_count

            for key, value in components.items():
                component_sums[key] = (
                    component_sums.get(key, 0.0)
                    + float(value) * batch_size
                )

            self.state.global_step += 1

            iterator.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}"
            )

        if batch_count == 0:
            raise RuntimeError("Training DataLoader produced no samples.")

        metrics: dict[str, Any] = {
            "loss": loss_sum / batch_count,
            "supervised_pixels": supervised_pixels,
        }

        for key, value in component_sums.items():
            if key != "loss":
                metrics[key] = value / batch_count

        return metrics

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.eval()

        loss_sum = 0.0
        sample_count = 0
        supervised_pixels = 0
        metric_records: list[
            tuple[Mapping[str, Any], int]
        ] = []

        iterator = tqdm(
            self.validation_loader,
            desc=f"Val   {epoch + 1}/{self.config.max_epochs}",
            leave=False,
            disable=not self.config.progress_bar,
        )

        for batch_index, batch in enumerate(iterator):
            image, target, validity = self._move_batch(batch)

            with self._autocast_context():
                logits = self.model(image)
                loss, _ = _call_criterion(
                    self.criterion,
                    logits,
                    target,
                    validity,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite validation loss at epoch {epoch}, "
                    f"batch {batch_index}."
                )

            batch_size = int(image.shape[0])
            sample_count += batch_size
            loss_sum += float(loss.detach().item()) * batch_size

            valid_count = (
                int(validity.sum().item())
                if validity is not None
                else int((target != 255).sum().item())
            )
            supervised_pixels += valid_count

            metric_output = self.metric_function(
                logits.detach(),
                target.detach(),
                validity.detach() if validity is not None else None,
            )

            if not isinstance(metric_output, Mapping):
                raise TypeError(
                    "metric_function must return a mapping."
                )

            metric_records.append(
                (dict(metric_output), max(1, valid_count))
            )

            iterator.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}"
            )

        if sample_count == 0:
            raise RuntimeError(
                "Validation DataLoader produced no samples."
            )

        metrics = _weighted_merge(metric_records)
        metrics["loss"] = loss_sum / sample_count
        metrics["supervised_pixels"] = supervised_pixels

        return metrics

    def _save_training_checkpoint(
        self,
        *,
        epoch: int,
        validation_metrics: Mapping[str, Any],
        is_best: bool,
    ) -> Path:
        latest_path = self.checkpoint_directory / "latest.pt"
        best_path = self.checkpoint_directory / "best.pt"

        epoch_path = (
            self.checkpoint_directory
            / f"epoch_{epoch:04d}.pt"
        )

        checkpoint_path = (
            epoch_path
            if (
                (epoch + 1) % self.config.save_every_epochs == 0
                or is_best
            )
            else latest_path
        )

        saved = save_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            global_step=self.state.global_step,
            metrics={
                key: float(value)
                for key, value in validation_metrics.items()
                if isinstance(value, (int, float))
                and math.isfinite(float(value))
            },
            config=self.resolved_config,
            early_stopping=self.early_stopping,
            scaler=self.scaler,
            is_best=is_best,
            best_path=best_path if is_best else None,
            extra_state={
                "trainer_state": asdict(self.state),
                "trainer_config": asdict(self.config),
                "seed": self.seed,
            },
        )

        if saved != latest_path:
            shutil.copy2(saved, latest_path)

        return saved

    def resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            early_stopping=self.early_stopping,
            scaler=self.scaler,
            map_location=self.device,
            restore_rng=self.config.restore_rng_on_resume,
        )

        self.state.epoch = int(checkpoint["epoch"])
        self.state.global_step = int(checkpoint["global_step"])

        extra = checkpoint.get("extra_state", {})
        trainer_state = (
            extra.get("trainer_state", {})
            if isinstance(extra, Mapping)
            else {}
        )

        self.state.optimizer_step = int(
            trainer_state.get("optimizer_step", 0)
        )
        self.state.best_metric = trainer_state.get("best_metric")
        self.state.best_epoch = trainer_state.get("best_epoch")
        self.state.stopped_early = bool(
            trainer_state.get("stopped_early", False)
        )

        return checkpoint

    def fit(
        self,
        *,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        start_epoch = 0

        if resume_from is not None:
            self.resume(resume_from)
            start_epoch = self.state.epoch + 1

        if start_epoch >= self.config.max_epochs:
            raise ValueError(
                "Checkpoint epoch is already at or beyond max_epochs."
            )

        training_started = time.perf_counter()

        for epoch in range(start_epoch, self.config.max_epochs):
            epoch_started = time.perf_counter()

            train_metrics = self.train_one_epoch(epoch)
            validation_metrics = self.validate_one_epoch(epoch)

            monitor_value = validation_metrics.get(self.config.monitor)

            if not isinstance(monitor_value, (int, float)):
                raise KeyError(
                    f"Validation metrics do not contain scalar "
                    f"{self.config.monitor!r}."
                )

            monitor_value = float(monitor_value)

            if not math.isfinite(monitor_value):
                raise RuntimeError(
                    f"Monitored metric {self.config.monitor!r} "
                    "is non-finite."
                )

            is_best = _metric_is_better(
                monitor_value,
                self.state.best_metric,
                self.config.monitor_mode,
            )

            if is_best:
                self.state.best_metric = monitor_value
                self.state.best_epoch = epoch

            step_scheduler(
                self.scheduler,
                monitored_metric=monitor_value,
            )

            learning_rates = get_learning_rates(self.optimizer)
            learning_rate = learning_rates[0]
            epoch_duration = time.perf_counter() - epoch_started

            self.state.epoch = epoch

            self.logger.log_epoch(
                epoch=epoch,
                global_step=self.state.global_step,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                learning_rate=learning_rate,
                epoch_duration_seconds=epoch_duration,
                is_best=is_best,
                best_metric_name=f"val_{self.config.monitor}",
            )

            self._save_training_checkpoint(
                epoch=epoch,
                validation_metrics=validation_metrics,
                is_best=is_best,
            )

            should_stop = False

            if self.early_stopping is not None:
                should_stop = self.early_stopping.update(
                    monitor_value,
                    epoch=epoch,
                )

            print(
                f"Epoch {epoch + 1}/{self.config.max_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={validation_metrics['loss']:.4f} | "
                f"{self.config.monitor}={monitor_value:.4f} | "
                f"lr={learning_rate:.8f} | "
                f"best={is_best}"
            )

            if should_stop:
                self.state.stopped_early = True
                break

        total_duration = time.perf_counter() - training_started

        summary = {
            "completed_epochs": self.state.epoch + 1,
            "global_step": self.state.global_step,
            "optimizer_step": self.state.optimizer_step,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "monitor": self.config.monitor,
            "monitor_mode": self.config.monitor_mode,
            "stopped_early": self.state.stopped_early,
            "device": str(self.device),
            "mixed_precision_enabled": self.amp_enabled,
            "duration_seconds": total_duration,
            "checkpoint_directory": str(self.checkpoint_directory),
        }

        self.logger.mark_completed(extra=summary)
        return summary


class _SyntheticSegmentationDataset(Dataset):
    """Small deterministic dataset used only by the smoke test."""

    def __init__(self, length: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(
            length,
            2,
            32,
            32,
            generator=generator,
        )
        self.targets = torch.randint(
            0,
            5,
            (length, 32, 32),
            generator=generator,
        )
        self.validity = torch.rand(
            length,
            32,
            32,
            generator=generator,
        ) > 0.1
        self.targets = self.targets.clone()
        self.targets[~self.validity] = 255

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "image": self.images[index],
            "target": self.targets[index],
            "validity": self.validity[index],
        }


class _TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 5, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def _synthetic_criterion(
    logits: Tensor,
    target: Tensor,
    validity: Tensor | None = None,
) -> dict[str, Tensor]:
    loss = nn.functional.cross_entropy(
        logits,
        target,
        ignore_index=255,
    )
    return {
        "loss": loss,
        "cross_entropy": loss,
    }


def _synthetic_metrics(
    logits: Tensor,
    target: Tensor,
    validity: Tensor | None = None,
) -> dict[str, Any]:
    prediction = logits.argmax(dim=1)
    valid = target != 255

    if validity is not None:
        valid = valid & validity

    class_names = (
        "buildings",
        "roads",
        "vegetation",
        "bare_land",
        "water",
    )

    total_valid = int(valid.sum().item())

    if total_valid == 0:
        raise RuntimeError("Synthetic validation batch has no valid pixels.")

    correct = int(((prediction == target) & valid).sum().item())
    per_class: dict[str, dict[str, float]] = {}

    ious: list[float] = []
    dices: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for class_id, class_name in enumerate(class_names):
        predicted = (prediction == class_id) & valid
        actual = (target == class_id) & valid

        intersection = int((predicted & actual).sum().item())
        union = int((predicted | actual).sum().item())
        predicted_count = int(predicted.sum().item())
        actual_count = int(actual.sum().item())

        iou = intersection / union if union > 0 else 0.0
        dice_denominator = predicted_count + actual_count
        dice = (
            2 * intersection / dice_denominator
            if dice_denominator > 0
            else 0.0
        )
        precision = (
            intersection / predicted_count
            if predicted_count > 0
            else 0.0
        )
        recall = (
            intersection / actual_count
            if actual_count > 0
            else 0.0
        )

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)

        per_class[class_name] = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

    return {
        "pixel_accuracy": correct / total_valid,
        "mean_iou": sum(ious) / len(ious),
        "mean_dice": sum(dices) / len(dices),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "supervised_pixels": total_valid,
        "per_class": per_class,
    }


def smoke_test(output_directory: Path) -> dict[str, Any]:
    if output_directory.exists():
        shutil.rmtree(output_directory)

    train_loader = DataLoader(
        _SyntheticSegmentationDataset(8, seed=1),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _SyntheticSegmentationDataset(4, seed=2),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = _TinySegmentationModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    experiment_root = output_directory / "experiment"

    logger = ExperimentLogger(
        LoggerConfig(
            experiment_name="generic_trainer_smoke_test",
            output_directory=experiment_root,
            enable_tensorboard=False,
        ),
        resume=False,
    )

    trainer = GenericTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=_synthetic_criterion,
        metric_function=_synthetic_metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        early_stopping=EarlyStopping(
            patience=5,
            mode="max",
        ),
        checkpoint_directory=(
            experiment_root / "checkpoints"
        ),
        config=TrainerConfig(
            max_epochs=2,
            device="cpu",
            use_mixed_precision=False,
            progress_bar=False,
        ),
        resolved_config={
            "experiment_name": "generic_trainer_smoke_test",
            "model": "tiny_segmentation_model",
        },
        seed=20260725,
    )

    summary = trainer.fit()
    logger.close()

    history_path = (
        experiment_root / "logs/training_history.json"
    )
    history = json.loads(
        history_path.read_text(encoding="utf-8")
    )

    latest_checkpoint = (
        experiment_root / "checkpoints/latest.pt"
    )
    best_checkpoint = (
        experiment_root / "checkpoints/best.pt"
    )

    restored_model = _TinySegmentationModel()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )
    restored_scheduler = ReduceLROnPlateau(
        restored_optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    restored_early_stopping = EarlyStopping(
        patience=5,
        mode="max",
    )

    restored = load_checkpoint(
        path=latest_checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        early_stopping=restored_early_stopping,
        map_location="cpu",
        restore_rng=False,
    )

    checks = {
        "two_epochs_completed": summary["completed_epochs"] == 2,
        "global_step_positive": summary["global_step"] > 0,
        "optimizer_step_positive": summary["optimizer_step"] > 0,
        "best_metric_recorded": summary["best_metric"] is not None,
        "best_epoch_recorded": summary["best_epoch"] is not None,
        "history_has_two_epochs": len(history) == 2,
        "latest_checkpoint_created": latest_checkpoint.exists(),
        "best_checkpoint_created": best_checkpoint.exists(),
        "checkpoint_epoch_restored": int(restored["epoch"]) == 1,
        "checkpoint_global_step_restored": (
            int(restored["global_step"]) == summary["global_step"]
        ),
        "training_losses_finite": all(
            math.isfinite(float(item["train_loss"]))
            for item in history
        ),
        "validation_losses_finite": all(
            math.isfinite(float(item["val_loss"]))
            for item in history
        ),
        "validation_miou_finite": all(
            math.isfinite(float(item["val_mean_iou"]))
            for item in history
        ),
    }
    checks["all_checks_passed"] = all(checks.values())

    if not checks["all_checks_passed"]:
        failed = [
            name for name, passed in checks.items() if not passed
        ]
        raise RuntimeError(
            "Generic Trainer smoke test failed: " + ", ".join(failed)
        )

    return {
        "summary": summary,
        "history_path": str(history_path),
        "latest_checkpoint": str(latest_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "checks": checks,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the generic Trainer smoke test."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/model_experiments/generic_trainer_smoke_test"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "metadata/model_development/generic_trainer_smoke_test.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    report = smoke_test(args.output_dir)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    summary = report["summary"]

    print("\nGeneric Trainer smoke test")
    print("--------------------------")
    print(f"Completed epochs: {summary['completed_epochs']}")
    print(f"Global steps: {summary['global_step']}")
    print(f"Optimizer steps: {summary['optimizer_step']}")
    print(f"Best metric: {summary['best_metric']}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Device: {summary['device']}")
    print(
        "Mixed precision enabled: "
        f"{summary['mixed_precision_enabled']}"
    )

    for name, passed in report["checks"].items():
        print(f"{name}: {passed}")

    print(f"History: {report['history_path']}")
    print(f"Latest checkpoint: {report['latest_checkpoint']}")
    print(f"Best checkpoint: {report['best_checkpoint']}")
    print(f"Report: {args.report}")
    print(
        "\nResult: generic Trainer passed end-to-end "
        "training, validation, logging, and checkpoint validation."
    )


if __name__ == "__main__":
    main()
