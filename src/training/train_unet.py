"""Train the baseline U-Net on frozen Dataset V1.

Recommended sequence
--------------------
1. Integration check on one real training and validation batch:

   python -m src.training.train_unet --config config/training_unet_baseline.yaml --integration-check

2. Full configured training:

   python -m src.training.train_unet --config config/training_unet_baseline.yaml

3. Resume:

   python -m src.training.train_unet --config config/training_unet_baseline.yaml \
       --resume outputs/model_experiments/unet_baseline_v1/checkpoints/latest.pt

The entry point intentionally contains small compatibility adapters because the
project's Dataset/DataLoader and U-Net modules were developed independently.
It does not alter Dataset V1.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data.dataloader import create_dataloaders
from src.data.transforms import (
    build_evaluation_transform,
    build_train_transform,
)
from src.models.unet import UNet
from src.training.experiment_logger import (
    ExperimentLogger,
    LoggerConfig,
)
from src.training.trainer import (
    GenericTrainer,
    TrainerConfig,
)
from src.training.training_state import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
)
from src.training.training_config import (
    OptimizerConfig,
    SchedulerConfig,
)


CLASS_NAMES = (
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
)

IGNORE_INDEX = 255


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if payload is None:
        return {}

    if not isinstance(payload, dict):
        raise TypeError("Training YAML root must be a mapping.")

    return payload


def nested_get(
    payload: Mapping[str, Any],
    *paths: str,
    default: Any = None,
) -> Any:
    """Return the first matching dotted configuration path."""
    for dotted_path in paths:
        current: Any = payload
        found = True

        for part in dotted_path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]

        if found:
            return current

    return default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_unet_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Instantiate the project U-Net while tolerating common parameter names."""
    signature = inspect.signature(UNet)
    parameters = signature.parameters

    candidate_values: dict[str, Any] = {
        "in_channels": int(
            nested_get(
                config,
                "model.in_channels",
                "model.input_channels",
                default=2,
            )
        ),
        "input_channels": int(
            nested_get(
                config,
                "model.input_channels",
                "model.in_channels",
                default=2,
            )
        ),
        "n_channels": int(
            nested_get(
                config,
                "model.in_channels",
                "model.input_channels",
                default=2,
            )
        ),
        "num_classes": int(
            nested_get(
                config,
                "model.num_classes",
                "model.classes",
                default=5,
            )
        ),
        "out_channels": int(
            nested_get(
                config,
                "model.num_classes",
                "model.classes",
                default=5,
            )
        ),
        "n_classes": int(
            nested_get(
                config,
                "model.num_classes",
                "model.classes",
                default=5,
            )
        ),
        "base_channels": int(
            nested_get(
                config,
                "model.base_channels",
                "model.base_features",
                default=32,
            )
        ),
        "base_features": int(
            nested_get(
                config,
                "model.base_features",
                "model.base_channels",
                default=32,
            )
        ),
        "features": int(
            nested_get(
                config,
                "model.base_channels",
                "model.base_features",
                default=32,
            )
        ),
        "dropout": float(
            nested_get(
                config,
                "model.dropout",
                default=0.1,
            )
        ),
        "bilinear": bool(
            nested_get(
                config,
                "model.bilinear",
                default=False,
            )
        ),
    }

    kwargs = {
        name: candidate_values[name]
        for name in parameters
        if name in candidate_values
    }

    model = UNet(**kwargs)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters <= 0:
        raise RuntimeError("Constructed U-Net has no trainable parameters.")

    return model


def _call_supported(
    function: Callable[..., Any],
    candidate_kwargs: Mapping[str, Any],
) -> Any:
    """Call a function with only the keyword arguments it supports."""
    signature = inspect.signature(function)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return function(**dict(candidate_kwargs))

    supported = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }

    return function(**supported)


def _normalise_loader_bundle(bundle: Any) -> dict[str, DataLoader]:
    """Convert common DataLoader bundle return formats to a dictionary."""
    if isinstance(bundle, Mapping):
        aliases = {
            "train": ("train", "train_loader"),
            "val": ("val", "validation", "val_loader", "validation_loader"),
            "test": ("test", "test_loader"),
        }

        output: dict[str, DataLoader] = {}

        for split, names in aliases.items():
            for name in names:
                candidate = bundle.get(name)
                if isinstance(candidate, DataLoader):
                    output[split] = candidate
                    break

        if "train" in output and "val" in output:
            return output

    if isinstance(bundle, (tuple, list)):
        if len(bundle) >= 2:
            if isinstance(bundle[0], DataLoader) and isinstance(
                bundle[1], DataLoader
            ):
                output = {
                    "train": bundle[0],
                    "val": bundle[1],
                }

                if len(bundle) >= 3 and isinstance(bundle[2], DataLoader):
                    output["test"] = bundle[2]

                return output

    attributes = {
        "train": ("train", "train_loader"),
        "val": ("val", "validation", "val_loader", "validation_loader"),
        "test": ("test", "test_loader"),
    }

    output = {}

    for split, names in attributes.items():
        for name in names:
            candidate = getattr(bundle, name, None)
            if isinstance(candidate, DataLoader):
                output[split] = candidate
                break

    if "train" in output and "val" in output:
        return output

    raise TypeError(
        "create_dataloaders() returned an unsupported structure. "
        "Expected mapping, tuple/list, or object containing train and val loaders."
    )


def build_project_dataloaders(
    config: Mapping[str, Any],
    *,
    integration_check: bool,
) -> dict[str, DataLoader]:
    batch_size = int(
        nested_get(
            config,
            "data.batch_size",
            "training.batch_size",
            default=4,
        )
    )
    num_workers = int(
        nested_get(
            config,
            "data.num_workers",
            "training.num_workers",
            default=0,
        )
    )
    pin_memory = bool(
        nested_get(
            config,
            "data.pin_memory",
            "training.pin_memory",
            default=torch.cuda.is_available(),
        )
    )
    seed = int(
        nested_get(
            config,
            "seed",
            "training.seed",
            default=20260725,
        )
    )

    train_transform = build_train_transform()
    evaluation_transform = build_evaluation_transform()

    candidate_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "seed": seed,
        "train_transform": train_transform,
        "val_transform": evaluation_transform,
        "validation_transform": evaluation_transform,
        "test_transform": evaluation_transform,
        "drop_last": False,
        "persistent_workers": num_workers > 0,
    }

    bundle = _call_supported(
        create_dataloaders,
        candidate_kwargs,
    )
    loaders = _normalise_loader_bundle(bundle)

    if integration_check:
        # The Trainer receives temporary one-batch loader wrappers in main().
        # The underlying project DataLoaders remain unchanged.
        print(
            "Integration-check mode: only the first real train and "
            "validation batches will be used."
        )

    return loaders


class OneBatchLoader:
    """Expose exactly one existing DataLoader batch without mutating it."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def __iter__(self):
        iterator = iter(self.loader)
        yield next(iterator)

    def __len__(self) -> int:
        return 1


class WeightedCrossEntropyDiceLoss(nn.Module):
    """Ignore-aware weighted CE + multiclass soft Dice loss."""

    def __init__(
        self,
        *,
        class_weights: Tensor | None,
        ce_weight: float,
        dice_weight: float,
        smoothing: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()

        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights cannot be negative.")

        if ce_weight + dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive.")

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smoothing = float(smoothing)
        self.ignore_index = int(ignore_index)

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights",
                class_weights.to(dtype=torch.float32),
            )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        validity: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cross_entropy = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )

        valid = target != self.ignore_index

        if validity is not None:
            valid = valid & validity.bool()

        safe_target = target.clone()
        safe_target[~valid] = 0

        probabilities = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(
            safe_target,
            num_classes=logits.shape[1],
        ).permute(0, 3, 1, 2).to(probabilities.dtype)

        valid_float = valid.unsqueeze(1).to(probabilities.dtype)
        probabilities = probabilities * valid_float
        one_hot = one_hot * valid_float

        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + one_hot.sum(
            dim=dimensions
        )

        dice_score = (
            2.0 * intersection + self.smoothing
        ) / (denominator + self.smoothing)

        present_classes = one_hot.sum(dim=dimensions) > 0

        if bool(present_classes.any()):
            dice_loss = 1.0 - dice_score[present_classes].mean()
        else:
            dice_loss = logits.sum() * 0.0

        total = (
            self.ce_weight * cross_entropy
            + self.dice_weight * dice_loss
        )

        return {
            "loss": total,
            "cross_entropy": cross_entropy,
            "dice": dice_loss,
        }


@torch.no_grad()
def segmentation_metrics(
    logits: Tensor,
    target: Tensor,
    validity: Tensor | None = None,
) -> dict[str, Any]:
    prediction = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX

    if validity is not None:
        valid = valid & validity.bool()

    supervised_pixels = int(valid.sum().item())

    if supervised_pixels <= 0:
        raise RuntimeError("Validation batch contains no supervised pixels.")

    correct = int(((prediction == target) & valid).sum().item())

    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []
    dices: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        predicted = (prediction == class_id) & valid
        actual = (target == class_id) & valid

        true_positive = int((predicted & actual).sum().item())
        predicted_count = int(predicted.sum().item())
        target_count = int(actual.sum().item())
        union = predicted_count + target_count - true_positive

        iou = true_positive / union if union > 0 else 0.0
        denominator = predicted_count + target_count
        dice = (
            2.0 * true_positive / denominator
            if denominator > 0
            else 0.0
        )
        precision = (
            true_positive / predicted_count
            if predicted_count > 0
            else 0.0
        )
        recall = (
            true_positive / target_count
            if target_count > 0
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
            "present_in_target": target_count > 0,
        }

    return {
        "pixel_accuracy": correct / supervised_pixels,
        "mean_iou": sum(ious) / len(ious),
        "mean_dice": sum(dices) / len(dices),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "supervised_pixels": supervised_pixels,
        "per_class": per_class,
    }


def load_class_weights(
    config: Mapping[str, Any],
) -> Tensor | None:
    """Load five training-only class weights from YAML, CSV, or JSON.

    Supported strategy names:
    - inverse_frequency
    - median_frequency
    - logarithmic_inverse
    - effective_number

    The Dataset V1 statistics CSV uses columns ending in ``_mean_one``.
    """
    configured = nested_get(
        config,
        "loss.class_weights",
        "training.loss.class_weights",
        default=None,
    )

    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) != 5:
            raise ValueError("Configured class_weights must contain 5 values.")

        weights = torch.tensor(configured, dtype=torch.float32)

        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("Configured class weights must be finite and positive.")

        return weights

    candidate_path = Path(
        nested_get(
            config,
            "loss.class_weights_file",
            "training.loss.class_weights_file",
            default=(
                "metadata/dataset_v1/statistics/"
                "class_weights_candidates.csv"
            ),
        )
    )

    if not candidate_path.exists():
        print(
            f"Class-weight candidate file was not found: {candidate_path}; "
            "using unweighted cross-entropy."
        )
        return None

    strategy = str(
        nested_get(
            config,
            "loss.class_weight_strategy",
            "training.loss.class_weight_strategy",
            default="median_frequency",
        )
    ).strip().lower()

    strategy_aliases = {
        "inverse_frequency": "inverse_frequency_mean_one",
        "inverse_frequency_mean_one": "inverse_frequency_mean_one",
        "median_frequency": "median_frequency_balancing_mean_one",
        "median_frequency_balancing": "median_frequency_balancing_mean_one",
        "median_frequency_balancing_mean_one": (
            "median_frequency_balancing_mean_one"
        ),
        "logarithmic_inverse": "logarithmic_inverse_c1_02_mean_one",
        "log_inverse": "logarithmic_inverse_c1_02_mean_one",
        "logarithmic_inverse_c1_02_mean_one": (
            "logarithmic_inverse_c1_02_mean_one"
        ),
        "effective_number": "effective_number_mean_one",
        "effective_number_mean_one": "effective_number_mean_one",
    }

    strategy_key = strategy_aliases.get(strategy)

    if strategy_key is None:
        raise ValueError(
            f"Unsupported class-weight strategy {strategy!r}. "
            f"Supported values: {sorted(strategy_aliases)}"
        )

    values: list[float] | None = None

    if candidate_path.suffix.lower() == ".csv":
        import csv

        with candidate_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))

        if not rows:
            raise ValueError(
                f"Class-weight CSV is empty: {candidate_path}"
            )

        required_columns = {
            "class_name",
            strategy_key,
        }
        missing_columns = required_columns.difference(rows[0].keys())

        if missing_columns:
            raise ValueError(
                "Class-weight CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        by_class = {
            str(row["class_name"]).strip(): float(row[strategy_key])
            for row in rows
        }

        missing_classes = [
            class_name
            for class_name in CLASS_NAMES
            if class_name not in by_class
        ]

        if missing_classes:
            raise ValueError(
                "Class-weight CSV is missing classes: "
                + ", ".join(missing_classes)
            )

        values = [
            by_class[class_name]
            for class_name in CLASS_NAMES
        ]

    elif candidate_path.suffix.lower() == ".json":
        payload = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )

        candidate: Any = None

        if isinstance(payload, Mapping):
            candidate = payload.get(strategy)
            if candidate is None:
                candidate = payload.get(strategy_key)

            if candidate is None:
                for key in (
                    "weights",
                    "class_weights",
                    "candidate_weights",
                ):
                    possible = payload.get(key)
                    if isinstance(possible, Mapping):
                        candidate = possible.get(strategy)
                        if candidate is None:
                            candidate = possible.get(strategy_key)
                        if candidate is not None:
                            break

        if isinstance(candidate, Mapping):
            values = [
                float(candidate[class_name])
                for class_name in CLASS_NAMES
            ]
        elif isinstance(candidate, (list, tuple)):
            values = [float(value) for value in candidate]

    else:
        raise ValueError(
            "Class-weight file must be CSV or JSON: "
            f"{candidate_path}"
        )

    if values is None or len(values) != 5:
        raise ValueError(
            f"Could not resolve five class weights for strategy "
            f"{strategy!r} from {candidate_path}."
        )

    weights = torch.tensor(values, dtype=torch.float32)

    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("Class weights must be finite and positive.")

    print(
        f"Loaded class weights ({strategy}) from {candidate_path}: "
        f"{[round(float(value), 6) for value in weights]}"
    )

    return weights

def make_optimizer_config(config: Mapping[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(
        name=str(
            nested_get(
                config,
                "optimizer.name",
                "training.optimizer.name",
                default="adamw",
            )
        ).lower(),
        learning_rate=float(
            nested_get(
                config,
                "optimizer.learning_rate",
                "optimizer.lr",
                "training.optimizer.learning_rate",
                default=1e-3,
            )
        ),
        weight_decay=float(
            nested_get(
                config,
                "optimizer.weight_decay",
                "training.optimizer.weight_decay",
                default=1e-4,
            )
        ),
        betas=tuple(
            nested_get(
                config,
                "optimizer.betas",
                "training.optimizer.betas",
                default=(0.9, 0.999),
            )
        ),
        epsilon=float(
            nested_get(
                config,
                "optimizer.epsilon",
                "optimizer.eps",
                "training.optimizer.epsilon",
                default=1e-8,
            )
        ),
    )


def make_scheduler_config(config: Mapping[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(
        name=str(
            nested_get(
                config,
                "scheduler.name",
                "training.scheduler.name",
                default="reduce_on_plateau",
            )
        ).lower(),
        monitor=str(
            nested_get(
                config,
                "scheduler.monitor",
                "training.scheduler.monitor",
                default="val_mean_iou",
            )
        ),
        mode=str(
            nested_get(
                config,
                "scheduler.mode",
                "training.scheduler.mode",
                default="max",
            )
        ).lower(),
        factor=float(
            nested_get(
                config,
                "scheduler.factor",
                "training.scheduler.factor",
                default=0.5,
            )
        ),
        patience=int(
            nested_get(
                config,
                "scheduler.patience",
                "training.scheduler.patience",
                default=3,
            )
        ),
        minimum_learning_rate=float(
            nested_get(
                config,
                "scheduler.minimum_learning_rate",
                "scheduler.min_lr",
                "training.scheduler.minimum_learning_rate",
                default=1e-6,
            )
        ),
    )


def build_components(
    config: Mapping[str, Any],
    *,
    integration_check: bool,
) -> tuple[
    nn.Module,
    Any,
    Any,
    WeightedCrossEntropyDiceLoss,
    Any,
    Any,
]:
    loaders = build_project_dataloaders(
        config,
        integration_check=integration_check,
    )

    train_loader: Any = loaders["train"]
    validation_loader: Any = loaders["val"]

    if integration_check:
        train_loader = OneBatchLoader(train_loader)
        validation_loader = OneBatchLoader(validation_loader)

    model = build_unet_from_config(config)
    class_weights = load_class_weights(config)

    criterion = WeightedCrossEntropyDiceLoss(
        class_weights=class_weights,
        ce_weight=float(
            nested_get(
                config,
                "loss.cross_entropy_weight",
                "loss.ce_weight",
                "training.loss.cross_entropy_weight",
                default=1.0,
            )
        ),
        dice_weight=float(
            nested_get(
                config,
                "loss.dice_weight",
                "training.loss.dice_weight",
                default=1.0,
            )
        ),
        smoothing=float(
            nested_get(
                config,
                "loss.dice_smoothing",
                "training.loss.dice_smoothing",
                default=1.0,
            )
        ),
    )

    optimizer = build_optimizer(
        model,
        make_optimizer_config(config),
    )

    max_epochs = (
        1
        if integration_check
        else int(
            nested_get(
                config,
                "training.max_epochs",
                "max_epochs",
                "epochs",
                default=20,
            )
        )
    )

    scheduler = build_scheduler(
        optimizer,
        make_scheduler_config(config),
        max_epochs=max_epochs,
    )

    return (
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline U-Net on frozen Dataset V1."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training_unet_baseline.yaml"),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--integration-check",
        action="store_true",
        help=(
            "Run one real train batch and one real validation batch "
            "for one epoch."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override: auto, cpu, cuda, cuda:0, or mps.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    raw_config = read_yaml(args.config)

    seed = int(
        nested_get(
            raw_config,
            "seed",
            "training.seed",
            default=20260725,
        )
    )
    set_seed(seed)

    experiment_name = str(
        nested_get(
            raw_config,
            "experiment.name",
            "experiment_name",
            default="unet_baseline_v1",
        )
    )

    if args.integration_check:
        experiment_name = f"{experiment_name}_integration_check"

    experiment_root = Path(
        nested_get(
            raw_config,
            "experiment.output_directory",
            "output_directory",
            default=f"outputs/model_experiments/{experiment_name}",
        )
    )

    if args.integration_check:
        experiment_root = (
            Path("outputs/model_experiments")
            / experiment_name
        )

        if experiment_root.exists():
            shutil.rmtree(experiment_root)

    (
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
    ) = build_components(
        raw_config,
        integration_check=args.integration_check,
    )

    configured_device = str(
        nested_get(
            raw_config,
            "training.device",
            "device",
            default="auto",
        )
    )

    device = args.device or configured_device

    max_epochs = (
        1
        if args.integration_check
        else int(
            nested_get(
                raw_config,
                "training.max_epochs",
                "max_epochs",
                "epochs",
                default=20,
            )
        )
    )

    monitor = str(
        nested_get(
            raw_config,
            "training.monitor",
            "scheduler.monitor",
            default="mean_iou",
        )
    )

    if monitor.startswith("val_"):
        monitor = monitor[4:]

    monitor_mode = str(
        nested_get(
            raw_config,
            "training.monitor_mode",
            "scheduler.mode",
            default="max",
        )
    ).lower()

    logger = ExperimentLogger(
        LoggerConfig(
            experiment_name=experiment_name,
            output_directory=experiment_root,
            class_names=CLASS_NAMES,
            enable_tensorboard=bool(
                nested_get(
                    raw_config,
                    "logging.tensorboard",
                    "training.logging.tensorboard",
                    default=True,
                )
            ),
        ),
        resume=args.resume is not None,
    )

    early_stopping = EarlyStopping(
        patience=int(
            nested_get(
                raw_config,
                "early_stopping.patience",
                "training.early_stopping.patience",
                default=7,
            )
        ),
        mode=monitor_mode,
        minimum_delta=float(
            nested_get(
                raw_config,
                "early_stopping.minimum_delta",
                "training.early_stopping.minimum_delta",
                default=0.0,
            )
        ),
    )

    trainer = GenericTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        metric_function=segmentation_metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        early_stopping=early_stopping,
        checkpoint_directory=experiment_root / "checkpoints",
        config=TrainerConfig(
            max_epochs=max_epochs,
            device=device,
            use_mixed_precision=bool(
                nested_get(
                    raw_config,
                    "training.mixed_precision",
                    "mixed_precision",
                    default=True,
                )
            ),
            gradient_clip_norm=nested_get(
                raw_config,
                "training.gradient_clip_norm",
                "gradient_clip_norm",
                default=1.0,
            ),
            accumulation_steps=int(
                nested_get(
                    raw_config,
                    "training.accumulation_steps",
                    "accumulation_steps",
                    default=1,
                )
            ),
            monitor=monitor,
            monitor_mode=monitor_mode,
            save_every_epochs=int(
                nested_get(
                    raw_config,
                    "checkpoint.save_every_epochs",
                    "training.checkpoint.save_every_epochs",
                    default=1,
                )
            ),
            progress_bar=bool(
                nested_get(
                    raw_config,
                    "logging.progress_bar",
                    "training.logging.progress_bar",
                    default=True,
                )
            ),
        ),
        resolved_config={
            "source_config": str(args.config),
            "experiment_name": experiment_name,
            "raw_config": raw_config,
            "integration_check": args.integration_check,
        },
        seed=seed,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nBaseline U-Net training")
    print("-----------------------")
    print(f"Experiment: {experiment_name}")
    print(f"Configuration: {args.config}")
    print(f"Integration check: {args.integration_check}")
    print(f"Epochs: {max_epochs}")
    print(f"Requested device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Output directory: {experiment_root}")

    try:
        summary = trainer.fit(
            resume_from=args.resume,
        )
    finally:
        logger.close()

    report_directory = Path(
        "metadata/model_development"
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_name = (
        "unet_real_data_integration_check.json"
        if args.integration_check
        else "unet_baseline_training_summary.json"
    )
    report_path = report_directory / report_name

    report = {
        "experiment_name": experiment_name,
        "config": str(args.config),
        "integration_check": args.integration_check,
        "trainable_parameters": trainable_parameters,
        "experiment_root": str(experiment_root),
        "summary": summary,
        "checks": {
            "training_completed": summary["completed_epochs"] >= 1,
            "global_step_positive": summary["global_step"] > 0,
            "optimizer_step_positive": summary["optimizer_step"] > 0,
            "best_metric_recorded": summary["best_metric"] is not None,
            "best_epoch_recorded": summary["best_epoch"] is not None,
            "latest_checkpoint_exists": (
                experiment_root / "checkpoints/latest.pt"
            ).exists(),
            "best_checkpoint_exists": (
                experiment_root / "checkpoints/best.pt"
            ).exists(),
            "history_exists": (
                experiment_root / "logs/training_history.json"
            ).exists(),
        },
    }
    report["checks"]["all_checks_passed"] = all(
        report["checks"].values()
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nTraining result")
    print("---------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    for key, value in report["checks"].items():
        print(f"{key}: {value}")

    print(f"Report: {report_path}")

    if args.integration_check:
        print(
            "\nResult: real Dataset V1 → U-Net integration "
            "check completed successfully."
        )
        print(
            "Next command: python -m src.training.train_unet "
            "--config config/training_unet_baseline.yaml"
        )
    else:
        print(
            "\nResult: configured baseline U-Net training completed."
        )


if __name__ == "__main__":
    main()
