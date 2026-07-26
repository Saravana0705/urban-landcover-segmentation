"""Validated training configuration and reproducibility utilities.

This module is model-independent. Every segmentation architecture should use
the same experiment configuration structure so that model comparisons remain
fair and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


DEFAULT_CLASS_NAMES = (
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-related experiment settings."""

    dataset_version: str = "v1.0"
    manifest_path: str = (
        "metadata/dataset_v1/dataset_manifest.csv"
    )
    dataset_config_path: str = (
        "metadata/dataset_v1/freeze/dataset_config.json"
    )
    normalization_path: str = (
        "metadata/dataset_v1/normalization/"
        "training_normalization.json"
    )
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    tile_size: int = 256
    input_channels: int = 2
    num_classes: int = 5
    ignore_index: int = 255
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    exclude_zero_valid: bool = True

    def validate(self) -> None:
        if not self.dataset_version:
            raise ValueError("dataset_version cannot be empty.")

        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive.")

        if self.input_channels != 2:
            raise ValueError(
                "Dataset V1 expects two Sentinel-1 channels."
            )

        if self.num_classes != 5:
            raise ValueError(
                "Dataset V1 expects five model classes."
            )

        if self.ignore_index != 255:
            raise ValueError(
                "Dataset V1 expects ignore_index=255."
            )

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names length must equal num_classes."
            )

        splits = {
            self.train_split,
            self.val_split,
            self.test_split,
        }

        if len(splits) != 3:
            raise ValueError(
                "Train, validation, and test splits must differ."
            )


@dataclass(frozen=True)
class ModelConfig:
    """Model settings shared by all architectures."""

    name: str = "unet"
    input_channels: int = 2
    num_classes: int = 5
    base_channels: int = 32
    dropout: float = 0.1
    use_batch_norm: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Model name cannot be empty.")

        if self.input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if self.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than one."
            )

        if self.base_channels <= 0:
            raise ValueError(
                "base_channels must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )


@dataclass(frozen=True)
class LossConfig:
    """Training-loss settings."""

    name: str = "ce_dice"
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    use_class_weights: bool = False
    class_weights_path: str = (
        "metadata/dataset_v1/statistics/"
        "class_weights_candidates.json"
    )
    class_weight_strategy: str = "none"
    include_absent_classes: bool = False

    def validate(self) -> None:
        if self.name not in {
            "cross_entropy",
            "dice",
            "ce_dice",
        }:
            raise ValueError(
                "Unsupported loss name."
            )

        if self.ce_weight < 0 or self.dice_weight < 0:
            raise ValueError(
                "Loss weights cannot be negative."
            )

        if (
            self.name == "ce_dice"
            and self.ce_weight + self.dice_weight <= 0
        ):
            raise ValueError(
                "Combined loss needs a positive total weight."
            )

        if (
            not self.use_class_weights
            and self.class_weight_strategy != "none"
        ):
            raise ValueError(
                "class_weight_strategy must be 'none' when "
                "class weighting is disabled."
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.name not in {"adam", "adamw", "sgd"}:
            raise ValueError(
                "Unsupported optimizer name."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not (
            0.0 <= self.betas[0] < 1.0
            and 0.0 <= self.betas[1] < 1.0
        ):
            raise ValueError(
                "Adam beta values must be in [0, 1)."
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    name: str = "reduce_on_plateau"
    monitor: str = "val_mean_iou"
    mode: str = "max"
    factor: float = 0.5
    patience: int = 3
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        if self.name not in {
            "none",
            "reduce_on_plateau",
            "cosine_annealing",
        }:
            raise ValueError(
                "Unsupported scheduler name."
            )

        if self.mode not in {"min", "max"}:
            raise ValueError(
                "Scheduler mode must be 'min' or 'max'."
            )

        if not 0.0 < self.factor < 1.0:
            raise ValueError(
                "Scheduler factor must be in (0, 1)."
            )

        if self.patience < 0:
            raise ValueError(
                "Scheduler patience cannot be negative."
            )

        if self.minimum_learning_rate < 0:
            raise ValueError(
                "minimum_learning_rate cannot be negative."
            )


@dataclass(frozen=True)
class DataLoaderConfig:
    """DataLoader settings."""

    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last_train: bool = False
    shuffle_train: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if (
            self.persistent_workers
            and self.num_workers == 0
        ):
            raise ValueError(
                "persistent_workers requires num_workers > 0."
            )


@dataclass(frozen=True)
class AugmentationConfig:
    """Training augmentation settings."""

    enabled: bool = True
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_90_probability: float = 0.5
    enable_sar_intensity: bool = False
    enable_speckle: bool = False

    def validate(self) -> None:
        for name, value in (
            (
                "horizontal_flip_probability",
                self.horizontal_flip_probability,
            ),
            (
                "vertical_flip_probability",
                self.vertical_flip_probability,
            ),
            (
                "rotate_90_probability",
                self.rotate_90_probability,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1]."
                )


@dataclass(frozen=True)
class RuntimeConfig:
    """Training runtime and reproducibility settings."""

    seed: int = 20260725
    max_epochs: int = 50
    early_stopping_patience: int = 10
    gradient_clip_norm: float | None = 1.0
    use_mixed_precision: bool = True
    deterministic_algorithms: bool = False
    cudnn_benchmark: bool = False
    device: str = "auto"
    monitor_metric: str = "val_mean_iou"
    monitor_mode: str = "max"

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative."
            )

        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive."
            )

        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience cannot be negative."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive."
            )

        if self.device not in {
            "auto",
            "cpu",
            "cuda",
            "mps",
        }:
            raise ValueError(
                "device must be auto, cpu, cuda, or mps."
            )

        if self.monitor_mode not in {"min", "max"}:
            raise ValueError(
                "monitor_mode must be min or max."
            )


@dataclass(frozen=True)
class OutputConfig:
    """Experiment output paths."""

    experiment_name: str = "unet_baseline_v1"
    output_root: str = "outputs/model_experiments"
    checkpoint_subdirectory: str = "checkpoints"
    log_subdirectory: str = "logs"
    prediction_subdirectory: str = "predictions"
    report_subdirectory: str = "reports"

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError(
                "experiment_name cannot be empty."
            )

        if not self.output_root:
            raise ValueError(
                "output_root cannot be empty."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Complete experiment configuration."""

    dataset: DatasetConfig = field(
        default_factory=DatasetConfig
    )
    model: ModelConfig = field(
        default_factory=ModelConfig
    )
    loss: LossConfig = field(
        default_factory=LossConfig
    )
    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    dataloader: DataLoaderConfig = field(
        default_factory=DataLoaderConfig
    )
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig
    )
    output: OutputConfig = field(
        default_factory=OutputConfig
    )

    def validate(self) -> None:
        """Validate the complete experiment configuration."""
        self.dataset.validate()
        self.model.validate()
        self.loss.validate()
        self.optimizer.validate()
        self.scheduler.validate()
        self.dataloader.validate()
        self.augmentation.validate()
        self.runtime.validate()
        self.output.validate()

        if (
            self.dataset.input_channels
            != self.model.input_channels
        ):
            raise ValueError(
                "Dataset and model input channels differ."
            )

        if (
            self.dataset.num_classes
            != self.model.num_classes
        ):
            raise ValueError(
                "Dataset and model class counts differ."
            )

        if (
            self.scheduler.monitor
            != self.runtime.monitor_metric
        ):
            raise ValueError(
                "Scheduler and runtime monitor metrics differ."
            )

        if (
            self.scheduler.mode
            != self.runtime.monitor_mode
        ):
            raise ValueError(
                "Scheduler and runtime monitor modes differ."
            )

    @property
    def experiment_directory(self) -> Path:
        return (
            Path(self.output.output_root)
            / self.output.experiment_name
        )

    @property
    def checkpoint_directory(self) -> Path:
        return (
            self.experiment_directory
            / self.output.checkpoint_subdirectory
        )

    @property
    def log_directory(self) -> Path:
        return (
            self.experiment_directory
            / self.output.log_subdirectory
        )

    @property
    def prediction_directory(self) -> Path:
        return (
            self.experiment_directory
            / self.output.prediction_subdirectory
        )

    @property
    def report_directory(self) -> Path:
        return (
            self.experiment_directory
            / self.output.report_subdirectory
        )

    def create_output_directories(self) -> None:
        for path in (
            self.experiment_directory,
            self.checkpoint_directory,
            self.log_directory,
            self.prediction_directory,
            self.report_directory,
        ):
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_from_value(
    value: Any,
    *,
    field_name: str,
) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{field_name} must be a list or tuple."
        )

    return tuple(value)


def training_config_from_dict(
    raw: dict[str, Any],
) -> TrainingConfig:
    """Construct a TrainingConfig from nested dictionaries."""
    dataset_raw = dict(
        raw.get("dataset", {})
    )
    model_raw = dict(
        raw.get("model", {})
    )
    loss_raw = dict(
        raw.get("loss", {})
    )
    optimizer_raw = dict(
        raw.get("optimizer", {})
    )
    scheduler_raw = dict(
        raw.get("scheduler", {})
    )
    dataloader_raw = dict(
        raw.get("dataloader", {})
    )
    augmentation_raw = dict(
        raw.get("augmentation", {})
    )
    runtime_raw = dict(
        raw.get("runtime", {})
    )
    output_raw = dict(
        raw.get("output", {})
    )

    if "class_names" in dataset_raw:
        dataset_raw["class_names"] = _tuple_from_value(
            dataset_raw["class_names"],
            field_name="dataset.class_names",
        )

    if "betas" in optimizer_raw:
        optimizer_raw["betas"] = _tuple_from_value(
            optimizer_raw["betas"],
            field_name="optimizer.betas",
        )

    config = TrainingConfig(
        dataset=DatasetConfig(**dataset_raw),
        model=ModelConfig(**model_raw),
        loss=LossConfig(**loss_raw),
        optimizer=OptimizerConfig(**optimizer_raw),
        scheduler=SchedulerConfig(**scheduler_raw),
        dataloader=DataLoaderConfig(**dataloader_raw),
        augmentation=AugmentationConfig(
            **augmentation_raw
        ),
        runtime=RuntimeConfig(**runtime_raw),
        output=OutputConfig(**output_raw),
    )

    config.validate()
    return config


def load_training_config(
    path: str | Path,
) -> TrainingConfig:
    """Load and validate YAML or JSON configuration."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {config_path}"
        )

    suffix = config_path.suffix.lower()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        if suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(file)
        elif suffix == ".json":
            raw = json.load(file)
        else:
            raise ValueError(
                "Training config must be YAML or JSON."
            )

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise TypeError(
            "Training configuration root must be a mapping."
        )

    return training_config_from_dict(raw)


def save_training_config(
    config: TrainingConfig,
    path: str | Path,
) -> None:
    """Save a validated configuration."""
    config.validate()
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_path.suffix.lower()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        if suffix in {".yaml", ".yml"}:
            yaml.safe_dump(
                config.to_dict(),
                file,
                sort_keys=False,
                allow_unicode=True,
            )
        elif suffix == ".json":
            json.dump(
                config.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )
        else:
            raise ValueError(
                "Output config must be YAML or JSON."
            )


def resolve_device(
    requested: str = "auto",
) -> torch.device:
    """Resolve requested training device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    if requested == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not mps_available:
            raise RuntimeError(
                "MPS was requested but is unavailable."
            )

    return torch.device(requested)


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = False,
    cudnn_benchmark: bool = False,
) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch."""
    if seed < 0:
        raise ValueError("seed cannot be negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = (
            cudnn_benchmark
        )
        torch.backends.cudnn.deterministic = (
            deterministic_algorithms
        )

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=True,
    )

    return {
        "seed": seed,
        "python_hash_seed": os.environ[
            "PYTHONHASHSEED"
        ],
        "deterministic_algorithms": (
            deterministic_algorithms
        ),
        "cudnn_benchmark": cudnn_benchmark,
        "cuda_seeded": torch.cuda.is_available(),
    }


def collect_environment_metadata(
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Collect runtime metadata for experiment provenance."""
    cuda_device_name = None

    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(
            device
        )

    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_name": (
            config.output.experiment_name
        ),
        "dataset_version": (
            config.dataset.dataset_version
        ),
        "model_name": config.model.name,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "resolved_device": str(device),
        "cuda_device_name": cuda_device_name,
        "cpu_count": os.cpu_count(),
    }


def initialize_experiment(
    config: TrainingConfig,
) -> dict[str, Any]:
    """Validate and initialize one experiment."""
    config.validate()
    config.create_output_directories()

    device = resolve_device(
        config.runtime.device
    )

    reproducibility = seed_everything(
        config.runtime.seed,
        deterministic_algorithms=(
            config.runtime.deterministic_algorithms
        ),
        cudnn_benchmark=(
            config.runtime.cudnn_benchmark
        ),
    )

    environment = collect_environment_metadata(
        config,
        device,
    )

    resolved_mixed_precision = bool(
        config.runtime.use_mixed_precision
        and device.type == "cuda"
    )

    resolved = {
        "config": config.to_dict(),
        "paths": {
            "experiment_directory": str(
                config.experiment_directory
            ),
            "checkpoint_directory": str(
                config.checkpoint_directory
            ),
            "log_directory": str(
                config.log_directory
            ),
            "prediction_directory": str(
                config.prediction_directory
            ),
            "report_directory": str(
                config.report_directory
            ),
        },
        "runtime": {
            "device": str(device),
            "mixed_precision_enabled": (
                resolved_mixed_precision
            ),
        },
        "reproducibility": reproducibility,
        "environment": environment,
    }

    resolved_path = (
        config.experiment_directory
        / "resolved_experiment.json"
    )

    with resolved_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            resolved,
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_training_config(
        config,
        config.experiment_directory
        / "resolved_training_config.yaml",
    )

    return resolved


def smoke_test(
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate configuration loading and reproducibility."""
    config = load_training_config(
        config_path
    )

    first = initialize_experiment(config)

    torch_first = torch.rand(4)
    numpy_first = np.random.rand(4)
    python_first = [
        random.random()
        for _ in range(4)
    ]

    seed_everything(
        config.runtime.seed,
        deterministic_algorithms=(
            config.runtime.deterministic_algorithms
        ),
        cudnn_benchmark=(
            config.runtime.cudnn_benchmark
        ),
    )

    torch_second = torch.rand(4)
    numpy_second = np.random.rand(4)
    python_second = [
        random.random()
        for _ in range(4)
    ]

    checks = {
        "config_valid": True,
        "output_directories_exist": all(
            path.exists()
            for path in (
                config.experiment_directory,
                config.checkpoint_directory,
                config.log_directory,
                config.prediction_directory,
                config.report_directory,
            )
        ),
        "torch_seed_reproducible": bool(
            torch.equal(
                torch_first,
                torch_second,
            )
        ),
        "numpy_seed_reproducible": bool(
            np.array_equal(
                numpy_first,
                numpy_second,
            )
        ),
        "python_seed_reproducible": (
            python_first == python_second
        ),
        "dataset_model_channels_match": (
            config.dataset.input_channels
            == config.model.input_channels
        ),
        "dataset_model_classes_match": (
            config.dataset.num_classes
            == config.model.num_classes
        ),
        "monitor_policy_consistent": (
            config.scheduler.monitor
            == config.runtime.monitor_metric
            and config.scheduler.mode
            == config.runtime.monitor_mode
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
            "Training configuration smoke test failed: "
            + ", ".join(failed)
        )

    report = {
        "config_path": str(
            Path(config_path)
        ),
        "experiment_name": (
            config.output.experiment_name
        ),
        "dataset_version": (
            config.dataset.dataset_version
        ),
        "model_name": config.model.name,
        "resolved_device": first[
            "runtime"
        ]["device"],
        "mixed_precision_enabled": first[
            "runtime"
        ]["mixed_precision_enabled"],
        "checks": checks,
        "resolved_experiment_path": str(
            config.experiment_directory
            / "resolved_experiment.json"
        ),
        "resolved_config_path": str(
            config.experiment_directory
            / "resolved_training_config.yaml"
        ),
    }

    report_path = Path(
        "metadata/model_development/"
        "training_config_smoke_test.json"
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    report["report_path"] = str(report_path)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate training configuration and reproducibility."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/training_unet_baseline.yaml"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    report = smoke_test(
        args.config
    )

    print("\nTraining configuration smoke test")
    print("---------------------------------")
    print(
        f"Experiment: "
        f"{report['experiment_name']}"
    )
    print(
        f"Dataset version: "
        f"{report['dataset_version']}"
    )
    print(
        f"Model: {report['model_name']}"
    )
    print(
        f"Resolved device: "
        f"{report['resolved_device']}"
    )
    print(
        "Mixed precision enabled: "
        f"{report['mixed_precision_enabled']}"
    )

    for name, passed in report[
        "checks"
    ].items():
        print(f"{name}: {passed}")

    print(
        "Resolved experiment: "
        f"{report['resolved_experiment_path']}"
    )
    print(
        "Resolved config: "
        f"{report['resolved_config_path']}"
    )
    print(
        f"Report: {report['report_path']}"
    )
    print(
        "\nResult: training configuration and "
        "reproducibility checks passed."
    )


if __name__ == "__main__":
    main()
