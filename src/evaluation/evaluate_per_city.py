"""Validation-only per-city evaluation for frozen Dataset V3.

This diagnostic evaluator:
- does not train or modify a model;
- evaluates only the configured validation split;
- preserves the official aggregate metric implementation;
- reports metrics separately by validation city;
- performs exact confusion-matrix and valid-pixel reconciliation;
- records reproducibility hashes and evaluation provenance.

It is reusable across the frozen baseline and subsequent experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from src.evaluation.metrics import (
    MetricsConfig,
    SegmentationMetrics,
)
from src.training.train_segmentation import (
    CLASS_NAMES,
    IGNORE_INDEX,
    build_project_dataloaders,
    build_segmentation_model,
    nested_get,
    read_yaml,
)


REPORT_SCHEMA_VERSION = "1.0"


def file_sha256(path: Path) -> str:
    """Return SHA-256 for one non-empty file."""
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit_hash() -> str | None:
    """Return current Git commit when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)

    if path.is_absolute():
        return path.resolve()

    return (Path.cwd() / path).resolve()


def build_metric_accumulator() -> SegmentationMetrics:
    """Create an accumulator using the official metric policy."""
    return SegmentationMetrics(
        MetricsConfig(
            num_classes=len(CLASS_NAMES),
            ignore_index=IGNORE_INDEX,
            class_names=tuple(CLASS_NAMES),
            include_absent_classes_in_macro=False,
        ),
        device="cpu",
    )


def extract_batch(
    batch: Mapping[str, Any],
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Extract tensors and validation identity metadata."""
    image = batch.get("image")
    target = batch.get("target")
    validity = batch.get("validity")

    if not isinstance(image, Tensor):
        raise TypeError("Batch image is missing or not a tensor.")

    if not isinstance(target, Tensor):
        raise TypeError("Batch target is missing or not a tensor.")

    if validity is not None and not isinstance(validity, Tensor):
        raise TypeError("Batch validity must be a tensor.")

    required_metadata = (
        "tile_id",
        "city_id",
        "city_name",
        "split",
    )

    for key in required_metadata:
        if key not in batch:
            raise RuntimeError(
                f"Validation batch is missing required metadata: {key}"
            )

    tile_ids = [str(value) for value in batch["tile_id"]]
    city_ids = [str(value) for value in batch["city_id"]]
    city_names = [str(value) for value in batch["city_name"]]

    split_value = batch["split"]

    if isinstance(split_value, str):
        splits = [split_value] * int(image.shape[0])
    else:
        splits = [str(value) for value in split_value]

    batch_size = int(image.shape[0])

    lengths = {
        len(tile_ids),
        len(city_ids),
        len(city_names),
        len(splits),
        batch_size,
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "Validation batch metadata is not aligned with "
            "the tensor batch."
        )

    return (
        image,
        target,
        validity,
        tile_ids,
        city_ids,
        city_names,
        splits,
    )


def validate_dataset(
    loader: Any,
    *,
    expected_city_ids: set[str],
    expected_tile_count: int | None,
) -> None:
    """Validate dataset records before any model inference."""
    dataset = loader.dataset

    if getattr(dataset, "split", None) != "val":
        raise RuntimeError(
            "Per-city diagnostic is validation-only. "
            f"Received dataset split: {getattr(dataset, 'split', None)!r}"
        )

    records = getattr(dataset, "records", None)

    if not isinstance(records, list) or not records:
        raise RuntimeError(
            "Validation dataset does not expose usable manifest records."
        )

    if (
        expected_tile_count is not None
        and len(records) != expected_tile_count
    ):
        raise RuntimeError(
            "Validation tile count does not match frozen configuration: "
            f"dataset={len(records)}, expected={expected_tile_count}."
        )

    observed_cities: set[str] = set()

    for record in records:
        tile_id = str(record.get("tile_id", "")).strip()
        city_id = str(record.get("city_id", "")).strip()
        split = str(record.get("split", "")).strip().lower()

        if not tile_id:
            raise RuntimeError(
                "Validation manifest contains a missing tile ID."
            )

        if not city_id:
            raise RuntimeError(
                f"Validation tile {tile_id} has a missing city ID."
            )

        if split != "val":
            if split == "test":
                raise RuntimeError(
                    "TEST TILE DETECTED — per-city validation "
                    f"evaluation aborted: {tile_id}."
                )

            raise RuntimeError(
                "Non-validation tile detected — per-city validation "
                f"evaluation aborted: tile={tile_id}, split={split!r}."
            )

        if city_id not in expected_city_ids:
            raise RuntimeError(
                "Unknown/non-validation city detected: "
                f"tile={tile_id}, city_id={city_id!r}."
            )

        observed_cities.add(city_id)

    if observed_cities != expected_city_ids:
        raise RuntimeError(
            "Observed validation cities do not exactly match "
            "the frozen validation city set. "
            f"Observed={sorted(observed_cities)}, "
            f"expected={sorted(expected_city_ids)}."
        )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    """Load one project model checkpoint."""
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint root must be a mapping.")

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return checkpoint


def evaluate(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    expected_city_ids: set[str],
) -> dict[str, Any]:
    """Run one validation pass and accumulate aggregate/per-city metrics."""
    aggregate = build_metric_accumulator()

    city_accumulators = {
        city_id: build_metric_accumulator()
        for city_id in sorted(expected_city_ids)
    }

    city_names: dict[str, str] = {}
    tile_counts = {
        city_id: 0
        for city_id in sorted(expected_city_ids)
    }
    ignore_counts = {
        city_id: 0
        for city_id in sorted(expected_city_ids)
    }
    valid_counts = {
        city_id: 0
        for city_id in sorted(expected_city_ids)
    }

    total_ignore = 0
    total_valid = 0
    evaluated_tile_ids: set[str] = set()

    model.eval()

    with torch.inference_mode():
        for batch in loader:
            (
                image,
                target,
                validity,
                tile_ids,
                city_ids,
                batch_city_names,
                splits,
            ) = extract_batch(batch)

            for index, split in enumerate(splits):
                normalized_split = split.strip().lower()

                if normalized_split != "val":
                    if normalized_split == "test":
                        raise RuntimeError(
                            "TEST TILE DETECTED — per-city validation "
                            f"evaluation aborted: {tile_ids[index]}."
                        )

                    raise RuntimeError(
                        "Non-validation tile detected during inference: "
                        f"{tile_ids[index]}, split={split!r}."
                    )

            image = image.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            target = target.to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )

            if validity is not None:
                validity = validity.to(
                    device,
                    dtype=torch.bool,
                    non_blocking=True,
                )

                target = target.clone()
                target[~validity] = IGNORE_INDEX

            logits = model(image)

            if logits.ndim != 4:
                raise RuntimeError(
                    f"Expected BCHW logits, got {tuple(logits.shape)}."
                )

            if logits.shape[1] != len(CLASS_NAMES):
                raise RuntimeError(
                    "Model output class count does not match "
                    "the frozen five-class ontology."
                )

            aggregate.update(
                logits.detach().cpu(),
                target.detach().cpu(),
            )

            for index, city_id in enumerate(city_ids):
                tile_id = tile_ids[index]
                city_name = batch_city_names[index]

                if not city_id:
                    raise RuntimeError(
                        f"Tile {tile_id} has a missing city ID."
                    )

                if city_id not in expected_city_ids:
                    raise RuntimeError(
                        "Unknown/non-validation city encountered "
                        f"during inference: {city_id!r}."
                    )

                if tile_id in evaluated_tile_ids:
                    raise RuntimeError(
                        f"Duplicate validation tile evaluated: {tile_id}"
                    )

                evaluated_tile_ids.add(tile_id)

                previous_name = city_names.get(city_id)

                if (
                    previous_name is not None
                    and previous_name != city_name
                ):
                    raise RuntimeError(
                        f"Inconsistent city name for {city_id}: "
                        f"{previous_name!r} vs {city_name!r}."
                    )

                city_names[city_id] = city_name

                sample_target = target[index:index + 1]

                sample_valid = sample_target != IGNORE_INDEX

                valid_pixels = int(
                    sample_valid.sum().item()
                )
                total_pixels = int(
                    sample_target.numel()
                )
                ignore_pixels = total_pixels - valid_pixels

                valid_counts[city_id] += valid_pixels
                ignore_counts[city_id] += ignore_pixels
                total_valid += valid_pixels
                total_ignore += ignore_pixels

                city_accumulators[city_id].update(
                    logits[index:index + 1]
                    .detach()
                    .cpu(),
                    sample_target.detach().cpu(),
                )

                tile_counts[city_id] += 1

    aggregate_report = aggregate.compute()

    city_reports = {
        city_id: city_accumulators[city_id].compute()
        for city_id in sorted(expected_city_ids)
    }

    combined_matrix = torch.zeros_like(
        aggregate.confusion_matrix
    )

    for city_id in sorted(expected_city_ids):
        combined_matrix += (
            city_accumulators[city_id].confusion_matrix
        )

    confusion_match = torch.equal(
        combined_matrix,
        aggregate.confusion_matrix,
    )

    aggregate_valid = int(
        aggregate_report["total_supervised_pixels"]
    )

    city_valid_sum = sum(valid_counts.values())

    valid_match = (
        city_valid_sum
        == aggregate_valid
        == total_valid
    )

    if not valid_match:
        raise RuntimeError(
            "City-level valid-pixel counts do not reconcile "
            "with aggregate validation count."
        )

    if not confusion_match:
        raise RuntimeError(
            "Combined city confusion matrices do not exactly equal "
            "the aggregate validation confusion matrix."
        )

    return {
        "aggregate": aggregate_report,
        "aggregate_confusion_matrix": (
            aggregate.confusion_matrix.clone()
        ),
        "city_reports": city_reports,
        "city_accumulators": city_accumulators,
        "city_names": city_names,
        "tile_counts": tile_counts,
        "valid_counts": valid_counts,
        "ignore_counts": ignore_counts,
        "total_valid": total_valid,
        "total_ignore": total_ignore,
        "evaluated_tile_count": len(evaluated_tile_ids),
        "reconciliation": {
            "valid_pixel_counts_match": valid_match,
            "confusion_matrices_match": confusion_match,
        },
    }


def prepare_output_directory(
    output_directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Refuse accidental replacement of an existing report."""
    if output_directory.exists():
        existing = list(output_directory.iterdir())

        if existing and not overwrite:
            raise FileExistsError(
                "Per-city report already exists and will not be "
                f"overwritten: {output_directory}. "
                "Use --overwrite explicitly if replacement is intended."
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_confusion_matrix(
    path: Path,
    matrix: Tensor,
) -> None:
    """Write a labeled confusion matrix."""
    values = matrix.detach().cpu().numpy()

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["target\\predicted", *CLASS_NAMES]
        )

        for class_id, class_name in enumerate(CLASS_NAMES):
            writer.writerow(
                [
                    class_name,
                    *[
                        int(value)
                        for value in values[class_id]
                    ],
                ]
            )


def write_overall_csv(
    path: Path,
    result: Mapping[str, Any],
    experiment_id: str,
) -> None:
    """Write one row per validation city."""
    fields = (
        "experiment_id",
        "city_id",
        "city_name",
        "split",
        "tile_count",
        "valid_pixel_count",
        "ignore_pixel_count",
        "mean_iou",
        "mean_dice",
        "pixel_accuracy",
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for city_id in sorted(result["city_reports"]):
            report = result["city_reports"][city_id]

            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "city_id": city_id,
                    "city_name": result["city_names"][city_id],
                    "split": "val",
                    "tile_count": result["tile_counts"][city_id],
                    "valid_pixel_count": (
                        result["valid_counts"][city_id]
                    ),
                    "ignore_pixel_count": (
                        result["ignore_counts"][city_id]
                    ),
                    "mean_iou": report["mean_iou"],
                    "mean_dice": report["mean_dice"],
                    "pixel_accuracy": report["pixel_accuracy"],
                }
            )


def write_class_csv(
    path: Path,
    result: Mapping[str, Any],
    experiment_id: str,
) -> None:
    """Write one row per validation city and semantic class."""
    fields = (
        "experiment_id",
        "city_id",
        "city_name",
        "class_id",
        "class_name",
        "iou",
        "dice",
        "f1",
        "precision",
        "recall",
        "target_pixel_count",
        "predicted_pixel_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "present_in_target",
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for city_id in sorted(result["city_reports"]):
            report = result["city_reports"][city_id]

            for class_name, values in report["per_class"].items():
                writer.writerow(
                    {
                        "experiment_id": experiment_id,
                        "city_id": city_id,
                        "city_name": result[
                            "city_names"
                        ][city_id],
                        "class_id": values["class_id"],
                        "class_name": class_name,
                        "iou": values["iou"],
                        "dice": values["dice"],
                        "f1": values["f1"],
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "target_pixel_count": values[
                            "target_pixels"
                        ],
                        "predicted_pixel_count": values[
                            "predicted_pixels"
                        ],
                        "true_positive": values[
                            "true_positive"
                        ],
                        "false_positive": values[
                            "false_positive"
                        ],
                        "false_negative": values[
                            "false_negative"
                        ],
                        "present_in_target": values[
                            "present_in_target"
                        ],
                    }
                )

def read_existing_confusion_matrix(
    path: Path,
) -> Tensor:
    """Read the existing official aggregate confusion-matrix CSV."""
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Existing official aggregate confusion matrix was not found: "
            f"{path}"
        )

    rows: list[list[int]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

        if header is None:
            raise ValueError(
                f"Aggregate confusion matrix is empty: {path}"
            )

        for row in reader:
            if not row:
                continue

            values = [int(value) for value in row[1:]]

            if len(values) != len(CLASS_NAMES):
                raise ValueError(
                    "Aggregate confusion matrix has an unexpected "
                    f"column count: {path}"
                )

            rows.append(values)

    if len(rows) != len(CLASS_NAMES):
        raise ValueError(
            "Aggregate confusion matrix has an unexpected "
            f"row count: {path}"
        )

    return torch.tensor(
        rows,
        dtype=torch.int64,
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-only per-city evaluation "
            "for frozen Dataset V3."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Default: "
            "<experiment output>/metrics/per_city"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--aggregate-confusion",
        type=Path,
        default=None,
        help=(
            "Existing official aggregate validation confusion matrix. "
            "Default: <experiment output>/metrics/confusion_matrix.csv"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of an existing report.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = read_yaml(args.config)

    experiment_id = str(
        nested_get(
            config,
            "experiment.name",
            "experiment_name",
            default="unknown_experiment",
        )
    )

    experiment_root = Path(
        nested_get(
            config,
            "experiment.output_directory",
            "output_directory",
            default=(
                f"outputs/model_experiments/{experiment_id}"
            ),
        )
    )

    output_directory = (
        args.output
        if args.output is not None
        else experiment_root / "metrics" / "per_city"
    )

    output_directory = output_directory.resolve()

    prepare_output_directory(
        output_directory,
        overwrite=args.overwrite,
    )

    aggregate_confusion_path = (
        args.aggregate_confusion
        if args.aggregate_confusion is not None
        else experiment_root / "metrics" / "confusion_matrix.csv"
    )

    aggregate_confusion_path = (
        aggregate_confusion_path.resolve()
    )

    manifest_path = resolve_project_path(
        nested_get(
            config,
            "dataset.evaluation_manifest_path",
            "dataset.manifest_path",
            default="metadata/dataset_v3/dataset_manifest.csv",
        )
    )

    dataset_config_path = resolve_project_path(
        nested_get(
            config,
            "dataset.evaluation_dataset_config_path",
            "dataset.dataset_config_path",
            default=(
                "metadata/dataset_v3/freeze/"
                "dataset_config.json"
            ),
        )
    )

    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {dataset_config_path}"
        )

    dataset_config = json.loads(
        dataset_config_path.read_text(
            encoding="utf-8"
        )
    )

    dataset_version = str(
        dataset_config.get("dataset_version", "")
    )

    if dataset_version.lower() != "v3":
        raise RuntimeError(
            "This diagnostic is restricted to frozen Dataset V3. "
            f"Received dataset_version={dataset_version!r}."
        )

    splits = dataset_config.get("splits", {})

    expected_city_ids = {
        str(value)
        for value in splits.get(
            "validation_city_ids",
            [],
        )
    }

    if expected_city_ids != {
        "DE15",
        "DE16",
        "DE17",
    }:
        raise RuntimeError(
            "Frozen validation city configuration is unexpected: "
            f"{sorted(expected_city_ids)}."
        )

    train_city_ids = {
        str(value)
        for value in splits.get("train_city_ids", [])
    }

    test_city_ids = {
        str(value)
        for value in splits.get("test_city_ids", [])
    }

    if expected_city_ids & train_city_ids:
        raise RuntimeError(
            "Validation and training city sets overlap."
        )

    if expected_city_ids & test_city_ids:
        raise RuntimeError(
            "Validation and test city sets overlap."
        )

    expected_tile_count_raw = splits.get(
        "validation_tile_count"
    )

    expected_tile_count = (
        int(expected_tile_count_raw)
        if expected_tile_count_raw is not None
        else None
    )

    manifest_hash = file_sha256(manifest_path)
    dataset_config_hash = file_sha256(
        dataset_config_path
    )
    checkpoint_hash = file_sha256(
        args.checkpoint
    )

    frozen_manifest_hash = dataset_config.get(
        "manifest_sha256"
    )

    if (
        frozen_manifest_hash is not None
        and str(frozen_manifest_hash).lower()
        != manifest_hash.lower()
    ):
        raise RuntimeError(
            "Dataset manifest SHA-256 does not match the "
            "hash recorded in the frozen Dataset V3 config."
        )

    loaders = build_project_dataloaders(
        config,
        integration_check=False,
    )

    validation_loader = loaders["val"]

    validate_dataset(
        validation_loader,
        expected_city_ids=expected_city_ids,
        expected_tile_count=expected_tile_count,
    )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    model = build_segmentation_model(config)

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    print("\nDataset V3 per-city validation")
    print("------------------------------")
    print(f"Experiment: {experiment_id}")
    print(f"Configuration: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(
        f"Validation cities: "
        f"{sorted(expected_city_ids)}"
    )
    print(
        "Validation tiles: "
        f"{len(validation_loader.dataset)}"
    )
    print("Test evaluation: DISABLED")

    result = evaluate(
        model=model,
        loader=validation_loader,
        device=device,
        expected_city_ids=expected_city_ids,
    )

    existing_aggregate_confusion = (
        read_existing_confusion_matrix(
            aggregate_confusion_path
        )
    )

    recomputed_aggregate_confusion = result[
        "aggregate_confusion_matrix"
    ].cpu()

    existing_confusion_match = torch.equal(
        existing_aggregate_confusion,
        recomputed_aggregate_confusion,
    )

    if not existing_confusion_match:
        difference = (
            recomputed_aggregate_confusion
            - existing_aggregate_confusion
        )

        raise RuntimeError(
            "Recomputed validation confusion matrix does not exactly "
            "match the existing official aggregate validation confusion "
            "matrix.\n"
            f"Existing:\n{existing_aggregate_confusion}\n"
            f"Recomputed:\n{recomputed_aggregate_confusion}\n"
            f"Difference:\n{difference}"
        )

    if (
        expected_tile_count is not None
        and result["evaluated_tile_count"]
        != expected_tile_count
    ):
        raise RuntimeError(
            "Evaluated validation tile count does not match "
            "the frozen configuration."
        )

    overall_path = (
        output_directory
        / "per_city_overall_metrics.csv"
    )
    class_path = (
        output_directory
        / "per_city_class_metrics.csv"
    )

    write_overall_csv(
        overall_path,
        result,
        experiment_id,
    )

    write_class_csv(
        class_path,
        result,
        experiment_id,
    )

    confusion_paths: dict[str, str] = {}

    for city_id in sorted(expected_city_ids):
        path = (
            output_directory
            / f"{city_id}_confusion_matrix.csv"
        )

        write_confusion_matrix(
            path,
            result["city_accumulators"][
                city_id
            ].confusion_matrix,
        )

        confusion_paths[city_id] = str(path)

    evaluation_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    summary = {
        "report_schema_version": (
            REPORT_SCHEMA_VERSION
        ),
        "experiment_identifier": experiment_id,
        "evaluation_timestamp_utc": (
            evaluation_timestamp
        ),
        "git_commit": git_commit_hash(),
        "checkpoint": {
            "path": str(
                args.checkpoint.resolve()
            ),
            "sha256": checkpoint_hash,
            "epoch": checkpoint.get("epoch"),
        },
        "dataset": {
            "dataset_version": dataset_version,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "manifest_sha256_recorded_in_freeze": (
                frozen_manifest_hash
            ),
            "dataset_config_path": str(
                dataset_config_path
            ),
            "dataset_freeze_config_sha256": (
                dataset_config_hash
            ),
        },
        "evaluation": {
            "split": "val",
            "validation_split_confirmed": True,
            "test_data_evaluated": False,
            "ignore_index": IGNORE_INDEX,
            "class_id_to_name": {
                str(index): name
                for index, name in enumerate(
                    CLASS_NAMES
                )
            },
            "number_of_evaluated_cities": (
                len(expected_city_ids)
            ),
            "number_of_evaluated_tiles": (
                result["evaluated_tile_count"]
            ),
            "valid_pixel_count": (
                result["total_valid"]
            ),
            "ignore_pixel_count": (
                result["total_ignore"]
            ),
            "existing_aggregate_confusion_matrix_path": (
                str(aggregate_confusion_path)
            ),
            "existing_aggregate_confusion_matrix_sha256": (
                file_sha256(aggregate_confusion_path)
            ),
        },
        "validation_cities": {
            city_id: {
                "city_name": result[
                    "city_names"
                ][city_id],
                "tile_count": result[
                    "tile_counts"
                ][city_id],
                "valid_pixel_count": result[
                    "valid_counts"
                ][city_id],
                "ignore_pixel_count": result[
                    "ignore_counts"
                ][city_id],
                "mean_iou": result[
                    "city_reports"
                ][city_id]["mean_iou"],
                "mean_dice": result[
                    "city_reports"
                ][city_id]["mean_dice"],
                "pixel_accuracy": result[
                    "city_reports"
                ][city_id]["pixel_accuracy"],
            }
            for city_id in sorted(
                expected_city_ids
            )
        },
        "aggregate_validation_metrics_recomputed": {
            "mean_iou": result[
                "aggregate"
            ]["mean_iou"],
            "mean_dice": result[
                "aggregate"
            ]["mean_dice"],
            "pixel_accuracy": result[
                "aggregate"
            ]["pixel_accuracy"],
            "valid_pixel_count": result[
                "aggregate"
            ]["total_supervised_pixels"],
        },
        "safety_checks": {
            "dataset_version_is_v3": True,
            "validation_city_set_exact": True,
            "train_validation_city_overlap": False,
            "test_validation_city_overlap": False,
            "validation_tile_count_confirmed": (
                expected_tile_count is None
                or result["evaluated_tile_count"]
                == expected_tile_count
            ),
            "manifest_hash_matches_freeze": (
                frozen_manifest_hash is None
                or str(
                    frozen_manifest_hash
                ).lower()
                == manifest_hash.lower()
            ),
            "city_valid_pixels_reconcile": result[
                "reconciliation"
            ]["valid_pixel_counts_match"],
            "city_confusion_matrices_reconcile": (
                result["reconciliation"][
                    "confusion_matrices_match"
                ]
            ),
            "test_data_not_evaluated": True,
            "recomputed_aggregate_matches_existing_official": (
                existing_confusion_match
            ),
        },
        "outputs": {
            "per_city_overall_metrics": str(
                overall_path
            ),
            "per_city_class_metrics": str(
                class_path
            ),
            "confusion_matrices": (
                confusion_paths
            ),
        },
    }

    summary_path = (
        output_directory
        / "per_city_evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nAggregate validation")
    print("--------------------")
    print(
        "mIoU: "
        f"{result['aggregate']['mean_iou']:.6f}"
    )
    print(
        "Mean Dice: "
        f"{result['aggregate']['mean_dice']:.6f}"
    )
    print(
        "Pixel accuracy: "
        f"{result['aggregate']['pixel_accuracy']:.6f}"
    )

    print("\nPer-city validation")
    print("-------------------")

    for city_id in sorted(expected_city_ids):
        city_report = result[
            "city_reports"
        ][city_id]

        print(
            f"{city_id} "
            f"({result['city_names'][city_id]}): "
            f"tiles={result['tile_counts'][city_id]}, "
            f"valid={result['valid_counts'][city_id]:,}, "
            f"ignore={result['ignore_counts'][city_id]:,}, "
            f"mIoU={city_report['mean_iou']:.6f}, "
            f"Dice={city_report['mean_dice']:.6f}, "
            f"Acc={city_report['pixel_accuracy']:.6f}"
        )

    print("\nSafety/reconciliation")
    print("---------------------")
    print(
        "Recomputed aggregate == existing official: "
        f"{existing_confusion_match}"
    )
    print(
        "City valid pixels == aggregate: "
        f"{result['reconciliation']['valid_pixel_counts_match']}"
    )
    print(
        "City confusion matrices == aggregate: "
        f"{result['reconciliation']['confusion_matrices_match']}"
    )
    print("Test data evaluated: False")

    print(f"\nOutput directory: {output_directory}")
    print(f"Summary: {summary_path}")
    print(
        "\nResult: Dataset V3 per-city validation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()