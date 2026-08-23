"""Reproducible DataLoader factory for frozen Dataset Version 1.

This module builds model-independent PyTorch DataLoaders from
Sentinel1UrbanDataset and validates one batch from each split.

Recommended initial Windows settings:
    batch_size=4
    num_workers=0

After the pipeline is stable, num_workers may be benchmarked at 2 or 4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import Sentinel1UrbanDataset


@dataclass(frozen=True)
class LoaderSettings:
    """Serializable DataLoader settings."""

    batch_size: int = 4
    num_workers: int = 0
    seed: int = 20260725
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last_train: bool = False
    exclude_zero_valid: bool = True
    verify_raster_metadata: bool = True


@dataclass(frozen=True)
class LoaderBundle:
    """Datasets and DataLoaders for all frozen splits."""

    train_dataset: Sentinel1UrbanDataset
    val_dataset: Sentinel1UrbanDataset
    test_dataset: Sentinel1UrbanDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    settings: LoaderSettings


def seed_everything(
    seed: int,
    deterministic_algorithms: bool = False,
) -> None:
    """Seed Python, NumPy and PyTorch.

    Full bitwise determinism can reduce performance and may not be available
    for every CUDA operation. It is therefore optional.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = (
            deterministic_algorithms
        )


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's worker seed."""
    del worker_id

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _validate_loader_arguments(
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
) -> None:
    """Validate DataLoader options."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if prefetch_factor is not None:
        if num_workers == 0:
            raise ValueError(
                "prefetch_factor requires num_workers greater than zero."
            )
        if prefetch_factor <= 0:
            raise ValueError(
                "prefetch_factor must be greater than zero."
            )


def _build_loader(
    dataset: Sentinel1UrbanDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    seed: int,
) -> DataLoader:
    """Build one deterministic DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (
            persistent_workers and num_workers > 0
        ),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


def create_dataloaders(
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 20260725,
    manifest_path: str | Path = (
        "metadata/dataset_v1/dataset_manifest.csv"
    ),
    dataset_config_path: str | Path = (
        "metadata/dataset_v1/freeze/dataset_config.json"
    ),
    evaluation_manifest_path: str | Path | None = None,
    evaluation_dataset_config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    train_transform: Callable[
        [dict[str, np.ndarray]],
        Mapping[str, np.ndarray],
    ]
    | None = None,
    val_transform: Callable[
        [dict[str, np.ndarray]],
        Mapping[str, np.ndarray],
    ]
    | None = None,
    test_transform: Callable[
        [dict[str, np.ndarray]],
        Mapping[str, np.ndarray],
    ]
    | None = None,
    exclude_zero_valid: bool = True,
    verify_raster_metadata: bool = True,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    drop_last_train: bool = False,
    deterministic_algorithms: bool = False,
) -> LoaderBundle:
    """Create frozen train, validation and test DataLoaders."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    effective_prefetch_factor = (
        None if num_workers == 0 else prefetch_factor
    )

    _validate_loader_arguments(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=effective_prefetch_factor,
    )

    seed_everything(
        seed,
        deterministic_algorithms=deterministic_algorithms,
    )

    if evaluation_manifest_path is None:
        evaluation_manifest_path = manifest_path

    if evaluation_dataset_config_path is None:
        evaluation_dataset_config_path = dataset_config_path

    train_dataset = Sentinel1UrbanDataset(
        split="train",
        manifest_path=manifest_path,
        dataset_config_path=dataset_config_path,
        project_root=project_root,
        joint_transform=train_transform,
        exclude_zero_valid=exclude_zero_valid,
        verify_raster_metadata=verify_raster_metadata,
    )

    val_dataset = Sentinel1UrbanDataset(
        split="val",
        manifest_path=evaluation_manifest_path,
        dataset_config_path=evaluation_dataset_config_path,
        project_root=project_root,
        joint_transform=val_transform,
        exclude_zero_valid=exclude_zero_valid,
        verify_raster_metadata=verify_raster_metadata,
    )

    test_dataset = Sentinel1UrbanDataset(
        split="test",
        manifest_path=evaluation_manifest_path,
        dataset_config_path=evaluation_dataset_config_path,
        project_root=project_root,
        joint_transform=test_transform,
        exclude_zero_valid=exclude_zero_valid,
        verify_raster_metadata=verify_raster_metadata,
    )

    train_loader = _build_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=effective_prefetch_factor,
        seed=seed,
    )

    val_loader = _build_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=effective_prefetch_factor,
        seed=seed + 1,
    )

    test_loader = _build_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=effective_prefetch_factor,
        seed=seed + 2,
    )

    settings = LoaderSettings(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=(
            persistent_workers and num_workers > 0
        ),
        prefetch_factor=effective_prefetch_factor,
        drop_last_train=drop_last_train,
        exclude_zero_valid=exclude_zero_valid,
        verify_raster_metadata=verify_raster_metadata,
    )

    return LoaderBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        settings=settings,
    )


def validate_batch(
    batch: Mapping[str, Any],
    *,
    num_classes: int = 5,
    ignore_index: int = 255,
    expected_channels: int = 2,
    expected_tile_size: int = 256,
) -> dict[str, Any]:
    """Validate one collated segmentation batch."""
    required_keys = {
        "image",
        "target",
        "validity",
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "ignore_index",
        "paths",
    }

    missing = required_keys.difference(batch)

    if missing:
        raise ValueError(
            f"Batch is missing keys: {sorted(missing)}"
        )

    image = batch["image"]
    target = batch["target"]
    validity = batch["validity"]

    if not isinstance(image, Tensor):
        raise TypeError("batch['image'] must be a Tensor.")

    if not isinstance(target, Tensor):
        raise TypeError("batch['target'] must be a Tensor.")

    if not isinstance(validity, Tensor):
        raise TypeError("batch['validity'] must be a Tensor.")

    if image.ndim != 4:
        raise ValueError(
            f"Expected image rank 4, found {image.ndim}."
        )

    if target.ndim != 3:
        raise ValueError(
            f"Expected target rank 3, found {target.ndim}."
        )

    if validity.ndim != 3:
        raise ValueError(
            f"Expected validity rank 3, found {validity.ndim}."
        )

    batch_size = image.shape[0]
    expected_image_shape = (
        batch_size,
        expected_channels,
        expected_tile_size,
        expected_tile_size,
    )
    expected_mask_shape = (
        batch_size,
        expected_tile_size,
        expected_tile_size,
    )

    if tuple(image.shape) != expected_image_shape:
        raise ValueError(
            f"Unexpected image shape: {tuple(image.shape)}."
        )

    if tuple(target.shape) != expected_mask_shape:
        raise ValueError(
            f"Unexpected target shape: {tuple(target.shape)}."
        )

    if tuple(validity.shape) != expected_mask_shape:
        raise ValueError(
            f"Unexpected validity shape: {tuple(validity.shape)}."
        )

    if image.dtype != torch.float32:
        raise TypeError(
            f"Expected float32 image, found {image.dtype}."
        )

    if target.dtype != torch.int64:
        raise TypeError(
            f"Expected int64 target, found {target.dtype}."
        )

    if validity.dtype != torch.bool:
        raise TypeError(
            f"Expected bool validity, found {validity.dtype}."
        )

    image_all_finite = bool(
        torch.isfinite(image).all().item()
    )

    if not image_all_finite:
        raise ValueError(
            "Batch contains non-finite normalized image values."
        )

    unique_targets = sorted(
        int(value)
        for value in torch.unique(target).tolist()
    )
    allowed_targets = set(range(num_classes)) | {
        ignore_index
    }

    if not set(unique_targets).issubset(
        allowed_targets
    ):
        raise ValueError(
            f"Unexpected target values: {unique_targets}."
        )

    supervised = target != ignore_index
    invalid_marked_ignore = bool(
        torch.all(target[~validity] == ignore_index).item()
    )
    supervised_marked_valid = bool(
        torch.all(validity[supervised]).item()
    )

    if not invalid_marked_ignore:
        raise ValueError(
            "At least one validity=False pixel is not ignore_index."
        )

    if not supervised_marked_valid:
        raise ValueError(
            "At least one supervised pixel has validity=False."
        )

    supervised_pixel_count = int(
        supervised.sum().item()
    )
    ignored_pixel_count = int(
        (target == ignore_index).sum().item()
    )
    total_pixels = int(target.numel())

    if supervised_pixel_count == 0:
        raise ValueError(
            "The batch contains no supervised pixels."
        )

    per_class_counts = {
        class_id: int(
            (target == class_id).sum().item()
        )
        for class_id in range(num_classes)
    }

    return {
        "batch_size": int(batch_size),
        "image_shape": tuple(image.shape),
        "target_shape": tuple(target.shape),
        "validity_shape": tuple(validity.shape),
        "image_dtype": str(image.dtype),
        "target_dtype": str(target.dtype),
        "validity_dtype": str(validity.dtype),
        "image_all_finite": image_all_finite,
        "target_values": unique_targets,
        "supervised_pixel_count": supervised_pixel_count,
        "ignored_pixel_count": ignored_pixel_count,
        "total_pixel_count": total_pixels,
        "supervised_fraction": (
            supervised_pixel_count / total_pixels
        ),
        "per_class_pixel_counts": per_class_counts,
        "invalid_pixels_are_ignored":
            invalid_marked_ignore,
        "supervised_pixels_are_valid":
            supervised_marked_valid,
        "tile_ids": list(batch["tile_id"]),
        "city_ids": list(batch["city_id"]),
    }


def inspect_dataloaders(
    bundle: LoaderBundle,
) -> dict[str, Any]:
    """Inspect and validate one batch from each split."""
    result: dict[str, Any] = {
        "settings": asdict(bundle.settings),
        "splits": {},
    }

    split_objects = {
        "train": (
            bundle.train_dataset,
            bundle.train_loader,
        ),
        "val": (
            bundle.val_dataset,
            bundle.val_loader,
        ),
        "test": (
            bundle.test_dataset,
            bundle.test_loader,
        ),
    }

    for split_name, (
        dataset,
        loader,
    ) in split_objects.items():
        first_batch = next(iter(loader))
        batch_report = validate_batch(
            first_batch,
            num_classes=dataset.num_classes,
            ignore_index=dataset.ignore_index,
            expected_channels=dataset.channel_count,
            expected_tile_size=dataset.tile_size,
        )

        result["splits"][split_name] = {
            "dataset": dataset.describe(),
            "batch_count": len(loader),
            "first_batch": batch_report,
        }

    return result


def parse_arguments() -> argparse.Namespace:
    """Parse smoke-test command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate Dataset V1 DataLoaders."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v1/"
            "dataloader_smoke_test.json"
        ),
    )

    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run DataLoader smoke tests."""
    args = parse_arguments()

    bundle = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic_algorithms=(
            args.deterministic_algorithms
        ),
    )

    report = inspect_dataloaders(bundle)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDataset V1 DataLoader smoke test")
    print("--------------------------------")
    print(
        f"Batch size: {bundle.settings.batch_size}"
    )
    print(
        f"Workers: {bundle.settings.num_workers}"
    )
    print(
        f"Pin memory: {bundle.settings.pin_memory}"
    )
    print(f"Seed: {bundle.settings.seed}")

    for split_name in ("train", "val", "test"):
        split_report = report["splits"][
            split_name
        ]
        dataset_report = split_report["dataset"]
        batch_report = split_report[
            "first_batch"
        ]

        print(f"\n{split_name.upper()}")
        print(
            f"  usable tiles: "
            f"{dataset_report['tile_count']}"
        )
        print(
            f"  batches: {split_report['batch_count']}"
        )
        print(
            f"  first batch image: "
            f"{batch_report['image_shape']}"
        )
        print(
            f"  target values: "
            f"{batch_report['target_values']}"
        )
        print(
            f"  supervised pixels: "
            f"{batch_report['supervised_pixel_count']:,}"
        )
        print(
            f"  image finite: "
            f"{batch_report['image_all_finite']}"
        )
        print(
            f"  invalid pixels ignored: "
            f"{batch_report['invalid_pixels_are_ignored']}"
        )
        print(
            f"  supervised pixels valid: "
            f"{batch_report['supervised_pixels_are_valid']}"
        )

    print(f"\nReport: {args.output}")
    print(
        "\nResult: all Dataset V1 DataLoaders passed "
        "batch-level validation."
    )


if __name__ == "__main__":
    main()
