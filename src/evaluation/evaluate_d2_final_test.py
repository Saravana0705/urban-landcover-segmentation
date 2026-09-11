"""One-time test evaluation of the validation-locked final D2 method.

The selected method is the E9B U-Net++ checkpoint with the exact four-way TTA,
float16-CPU probability replay, and five class multipliers locked by E10C/E11.
The program has a metadata-only preflight mode and requires explicit
``--authorize-test`` for the sole test-raster pass. It never trains or updates
a checkpoint and refuses to reuse an existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataloader import seed_everything, seed_worker
from src.data.segmentation_dataset import Sentinel1UrbanDataset
from src.evaluation.evaluate_segmentation import _extract_batch, write_metrics_outputs
from src.evaluation.metrics import metrics_from_confusion_matrix
from src.models.model_factory import build_model


DEFAULT_CONFIG = Path("config/evaluation_d2_final_locked_test.yaml")
CLASS_NAMES = ("buildings", "roads", "vegetation", "bare_land", "water")
IGNORE_INDEX = 255
LOCKED_TRANSFORMS = (
    "identity",
    "horizontal_flip",
    "vertical_flip",
    "rotate_180",
)
LOCKED_MULTIPLIERS = (
    1.149999976158142,
    1.149999976158142,
    1.024999976158142,
    0.699999988079071,
    1.0750000476837158,
)
LOCKED_CHECKPOINT_SHA256 = (
    "9ea0b5220b96acc26cce52f289e2f5b35a085dab0b0927d0794490907f70c799"
)
LOCKED_TRAINING_CONFIG_SHA256 = (
    "e62269e45eda7356d87876a28a743dfc358fd582df61f1b812f110a7ac52fe38"
)
LOCKED_VALIDATION_SHA256 = (
    "06c5d6cba9774bc0c0ad70ddddcaba8e9d90f3ea5e2cf2f068c3f3ea0946233c"
)
LOCKED_VALIDATION_MIOU = 0.6982309104696373
EXPECTED_TEST_CITIES = ("DE18", "DE19", "DE20")
EXPECTED_TEST_CITY_NAMES = {
    "DE18": "Freiburg",
    "DE19": "Kiel",
    "DE20": "Rostock",
}
EXPECTED_TEST_CITY_TILE_COUNTS = {"DE18": 120, "DE19": 117, "DE20": 102}
EXPECTED_TEST_TILE_COUNT = 339
EXPECTED_PARAMETERS = 9_049_701


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    """Hash text after normalizing platform line endings to LF."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_nested_file(root: Path, relative_path: str) -> Path:
    """Resolve a checkpoint even when Kaggle adds one directory level."""
    root = root.resolve()
    direct = (root / relative_path).resolve()
    if direct.is_file():
        return direct
    suffix = Path(relative_path).parts
    matches = [
        candidate.resolve()
        for candidate in root.rglob(Path(relative_path).name)
        if candidate.is_file()
        and tuple(candidate.parts[-len(suffix) :]) == suffix
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"'{relative_path}' was not found under {root}.")
    raise RuntimeError(
        f"'{relative_path}' is ambiguous under {root}: "
        + ", ".join(str(path) for path in matches)
    )


def validate_lock(config: Mapping[str, Any]) -> dict[str, bool]:
    selection = config.get("selection_lock", {})
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    inference = config.get("inference", {})
    safety = config.get("safety", {})
    checks = {
        "architecture_is_unetpp": selection.get("architecture_benchmark_winner") == "U-Net++",
        "final_candidate_is_e9b": (
            selection.get("final_selection_experiment") == "d2_e11_checkpoint_model_soup"
            and selection.get("selected_candidate") == "s0_e9b_reference"
            and selection.get("selected_weights")
            == {"e9b": 1.0, "e10b": 0.0, "e10d": 0.0}
        ),
        "validation_result_locked": (
            abs(float(selection.get("validation_mean_iou", 0.0)) - LOCKED_VALIDATION_MIOU)
            < 1e-12
            and selection.get("validation_lock_sha256") == LOCKED_VALIDATION_SHA256
            and selection.get("validation_tuning_closed") is True
        ),
        "dataset_is_frozen_d2_test": (
            dataset.get("dataset_version") == "v3-mt-d2"
            and dataset.get("split") == "test"
            and tuple(dataset.get("expected_city_ids", ())) == EXPECTED_TEST_CITIES
            and dict(zip(
                dataset.get("expected_city_ids", ()),
                dataset.get("expected_city_names", ()),
            )) == EXPECTED_TEST_CITY_NAMES
            and dict(dataset.get("expected_city_tile_counts", {}))
            == EXPECTED_TEST_CITY_TILE_COUNTS
            and int(dataset.get("expected_tile_count", -1)) == EXPECTED_TEST_TILE_COUNT
            and int(dataset.get("input_channels", -1)) == 16
            and int(dataset.get("num_classes", -1)) == 5
            and tuple(dataset.get("class_names", ())) == CLASS_NAMES
            and int(dataset.get("ignore_index", -1)) == IGNORE_INDEX
        ),
        "checkpoint_exact": (
            model.get("checkpoint_relative_path") == "e9b/best.pt"
            and model.get("expected_checkpoint_sha256") == LOCKED_CHECKPOINT_SHA256
            and model.get("expected_training_config_sha256")
            == LOCKED_TRAINING_CONFIG_SHA256
            and model.get("expected_model_name") == "unetpp"
            and int(model.get("expected_trainable_parameters", -1)) == EXPECTED_PARAMETERS
        ),
        "inference_policy_exact": (
            tuple(inference.get("tta_transforms", ())) == LOCKED_TRANSFORMS
            and tuple(float(value) for value in inference.get("class_probability_multipliers", ()))
            == LOCKED_MULTIPLIERS
            and inference.get("probability_quantization") == "float16_cpu"
            and inference.get("mixed_precision") is False
            and int(inference.get("batch_size", -1)) == 2
            and int(inference.get("num_workers", -1)) == 0
        ),
        "one_time_safety_exact": (
            safety.get("selection_locked_before_test") is True
            and safety.get("explicit_test_authorization_required") is True
            and safety.get("additional_validation_search_permitted") is False
            and safety.get("additional_training_permitted") is False
            and safety.get("checkpoint_updates_permitted") is False
            and safety.get("test_result_may_trigger_reselection") is False
            and safety.get("refuse_existing_output_directory") is True
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise PermissionError("Final test lock failed: " + ", ".join(failed))
    return checks


def inspect_dataset_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the test manifest contract without opening any raster."""
    dataset = config["dataset"]
    manifest_path = Path(str(dataset["manifest_path"]))
    dataset_config_path = Path(str(dataset["dataset_config_path"]))
    if not manifest_path.is_file() or not dataset_config_path.is_file():
        raise FileNotFoundError("Frozen D2 manifest/config is not available.")
    frozen = json.loads(dataset_config_path.read_text(encoding="utf-8"))
    if frozen.get("dataset_version") != "v3-mt-d2":
        raise ValueError("Frozen dataset configuration is not v3-mt-d2.")
    if tuple(frozen["splits"]["test_city_ids"]) != EXPECTED_TEST_CITIES:
        raise ValueError("Frozen test cities differ from the final lock.")
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "")).strip().lower() == "test":
                rows.append(row)
    cities = tuple(sorted({str(row["city_id"]) for row in rows}))
    usable = [row for row in rows if int(float(row.get("valid_pixel_count", 0))) > 0]
    city_names = {
        str(row["city_id"]): str(row["city_name"])
        for row in usable
    }
    city_tile_counts = {
        city_id: sum(str(row["city_id"]) == city_id for row in usable)
        for city_id in EXPECTED_TEST_CITIES
    }
    if (
        cities != EXPECTED_TEST_CITIES
        or city_names != EXPECTED_TEST_CITY_NAMES
        or city_tile_counts != EXPECTED_TEST_CITY_TILE_COUNTS
        or len(usable) != EXPECTED_TEST_TILE_COUNT
    ):
        raise ValueError(
            "Frozen test metadata mismatch: "
            f"cities={cities}, names={city_names}, "
            f"city_tile_counts={city_tile_counts}, usable_tiles={len(usable)}"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_config_path": str(dataset_config_path),
        "dataset_config_sha256": file_sha256(dataset_config_path),
        "test_city_ids": list(cities),
        "test_city_names": city_names,
        "test_city_tile_counts": city_tile_counts,
        "test_tile_count": len(usable),
        "test_rasters_opened": False,
    }


def load_locked_model(
    config: Mapping[str, Any], checkpoint_root: Path, device: torch.device
) -> tuple[nn.Module, Path, dict[str, Any], dict[str, Any]]:
    model_lock = config["model"]
    training_config_path = Path(str(model_lock["training_config"]))
    actual_training_hash = file_sha256(training_config_path)
    canonical_training_hash = canonical_text_sha256(training_config_path)
    if LOCKED_TRAINING_CONFIG_SHA256 not in {
        actual_training_hash,
        canonical_training_hash,
    }:
        raise RuntimeError(
            "Training config SHA-256 mismatch: "
            f"expected={LOCKED_TRAINING_CONFIG_SHA256}, "
            f"raw={actual_training_hash}, canonical={canonical_training_hash}"
        )
    training_config = read_yaml(training_config_path)
    if training_config.get("model", {}).get("name") != "unetpp":
        raise ValueError("Locked training configuration is not U-Net++.")
    checkpoint_path = resolve_nested_file(
        checkpoint_root, str(model_lock["checkpoint_relative_path"])
    )
    actual_checkpoint_hash = file_sha256(checkpoint_path)
    if actual_checkpoint_hash != LOCKED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: "
            f"expected={LOCKED_CHECKPOINT_SHA256}, actual={actual_checkpoint_hash}"
        )
    model = build_model(training_config).to(device)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETERS:
        raise RuntimeError("Locked U-Net++ parameter count mismatch.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint root must be a mapping.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint lacks model_state_dict.")
    model.load_state_dict(state, strict=True)
    model.eval()
    provenance = {
        "training_config": str(training_config_path),
        "training_config_sha256": LOCKED_TRAINING_CONFIG_SHA256,
        "training_config_raw_sha256": actual_training_hash,
        "training_config_line_endings_normalized": (
            actual_training_hash != canonical_training_hash
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_hash,
        "model_name": "unetpp",
        "model_class": model.__class__.__name__,
        "trainable_parameters": EXPECTED_PARAMETERS,
    }
    return model, checkpoint_path, training_config, provenance


def transform_tensor(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return value
    if name == "horizontal_flip":
        return torch.flip(value, dims=(-1,))
    if name == "vertical_flip":
        return torch.flip(value, dims=(-2,))
    if name == "rotate_180":
        return torch.flip(value, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA transform: {name}")


def tta_probability(model: nn.Module, image: Tensor, transforms: Sequence[str]) -> Tensor:
    total: Tensor | None = None
    for name in transforms:
        logits = model(transform_tensor(image, name))
        if not isinstance(logits, Tensor):
            raise TypeError("Final model must return one logits tensor.")
        probability = transform_tensor(torch.softmax(logits, dim=1), name)
        total = probability if total is None else total + probability
    if total is None:
        raise ValueError("At least one TTA transform is required.")
    return total / len(transforms)


def add_confusion(
    confusion: Tensor, prediction: Tensor, target: Tensor, validity: Tensor | None
) -> int:
    valid = target != IGNORE_INDEX
    if validity is not None:
        valid &= validity.bool()
    encoded = target[valid] * len(CLASS_NAMES) + prediction[valid]
    confusion += torch.bincount(
        encoded, minlength=len(CLASS_NAMES) ** 2
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    return int(valid.sum().item())


def metadata_values(batch: Mapping[str, Any], name: str, batch_size: int) -> list[str]:
    raw = batch.get(name)
    if raw is None:
        raise RuntimeError(f"Evaluation batch lacks '{name}' metadata.")
    if isinstance(raw, str):
        return [raw] * batch_size
    values = [str(value) for value in raw]
    if len(values) != batch_size:
        raise RuntimeError(f"Batch metadata '{name}' is not tensor-aligned.")
    return values


def create_test_loader(config: Mapping[str, Any]) -> DataLoader[Any]:
    dataset_lock = config["dataset"]
    inference = config["inference"]
    dataset = Sentinel1UrbanDataset(
        split="test",
        manifest_path=str(dataset_lock["manifest_path"]),
        dataset_config_path=str(dataset_lock["dataset_config_path"]),
        project_root=Path.cwd(),
        joint_transform=None,
        exclude_zero_valid=True,
        verify_raster_metadata=True,
    )
    description = dataset.describe()
    if (
        description.get("split") != "test"
        or tuple(description.get("city_ids", ())) != EXPECTED_TEST_CITIES
        or int(description.get("tile_count", -1)) != EXPECTED_TEST_TILE_COUNT
        or int(description.get("channel_count", -1)) != 16
        or int(description.get("num_classes", -1)) != 5
    ):
        raise RuntimeError(f"Instantiated test dataset violates lock: {description}")
    generator = torch.Generator().manual_seed(int(inference["seed"]) + 2)
    return DataLoader(
        dataset,
        batch_size=int(inference["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(inference["num_workers"]),
        pin_memory=bool(inference["pin_memory"]),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def evaluate_test(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, config: Mapping[str, Any]
) -> dict[str, Any]:
    inference = config["inference"]
    transforms = tuple(inference["tta_transforms"])
    multipliers = torch.tensor(
        inference["class_probability_multipliers"], dtype=torch.float32
    ).view(1, 5, 1, 1)
    aggregate = torch.zeros(5, 5, dtype=torch.int64)
    by_city: dict[str, Tensor] = defaultdict(lambda: torch.zeros(5, 5, dtype=torch.int64))
    city_names: dict[str, str] = {}
    city_tiles: dict[str, int] = defaultdict(int)
    tile_ids_seen: set[str] = set()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batch_count = 0
    image_count = 0
    model.eval()
    with torch.inference_mode():
        progress = tqdm(loader, desc="Final test E9B", dynamic_ncols=True, leave=True)
        for batch in progress:
            if not isinstance(batch, Mapping):
                raise TypeError("Final test requires mapping batches with provenance metadata.")
            image, target, validity = _extract_batch(batch)
            batch_size = int(image.shape[0])
            tile_ids = metadata_values(batch, "tile_id", batch_size)
            city_ids = metadata_values(batch, "city_id", batch_size)
            names = metadata_values(batch, "city_name", batch_size)
            splits = metadata_values(batch, "split", batch_size)
            if any(value != "test" for value in splits):
                raise PermissionError("A non-test sample entered final test evaluation.")
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            probability = tta_probability(model, image, transforms)
            # Exact E10C/E11 replay contract.
            probability = probability.detach().cpu().to(torch.float16).to(torch.float32)
            prediction = (probability * multipliers).argmax(dim=1)
            target_cpu = target.detach().cpu().to(torch.long)
            validity_cpu = validity.detach().cpu().bool() if validity is not None else None
            add_confusion(aggregate, prediction, target_cpu, validity_cpu)
            for index, (tile_id, city_id, city_name) in enumerate(
                zip(tile_ids, city_ids, names)
            ):
                if tile_id in tile_ids_seen:
                    raise RuntimeError(f"Duplicate test tile encountered: {tile_id}")
                tile_ids_seen.add(tile_id)
                previous_name = city_names.get(city_id)
                if previous_name is not None and previous_name != city_name:
                    raise RuntimeError(f"Inconsistent name for city {city_id}.")
                city_names[city_id] = city_name
                sample_validity = (
                    validity_cpu[index : index + 1] if validity_cpu is not None else None
                )
                add_confusion(
                    by_city[city_id],
                    prediction[index : index + 1],
                    target_cpu[index : index + 1],
                    sample_validity,
                )
                city_tiles[city_id] += 1
            batch_count += 1
            image_count += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    if len(tile_ids_seen) != EXPECTED_TEST_TILE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_TILE_COUNT} test tiles, evaluated {len(tile_ids_seen)}."
        )
    if tuple(sorted(by_city)) != EXPECTED_TEST_CITIES:
        raise RuntimeError(f"Unexpected evaluated test cities: {sorted(by_city)}")
    if city_names != EXPECTED_TEST_CITY_NAMES:
        raise RuntimeError(f"Unexpected evaluated test city names: {city_names}")
    if dict(city_tiles) != EXPECTED_TEST_CITY_TILE_COUNTS:
        raise RuntimeError(f"Unexpected per-city test tile counts: {dict(city_tiles)}")
    summed = sum(by_city.values(), torch.zeros_like(aggregate))
    if not torch.equal(summed, aggregate):
        raise RuntimeError("Per-city confusion matrices do not reconcile with aggregate.")
    aggregate_metrics = metrics_from_confusion_matrix(
        aggregate, class_names=CLASS_NAMES, include_absent_classes_in_macro=False
    )
    aggregate_metrics["batch_count"] = batch_count
    cities = {
        city_id: {
            "city_id": city_id,
            "city_name": city_names[city_id],
            "tile_count": city_tiles[city_id],
            **metrics_from_confusion_matrix(
                matrix, class_names=CLASS_NAMES, include_absent_classes_in_macro=False
            ),
        }
        for city_id, matrix in sorted(by_city.items())
    }
    return {
        "aggregate": aggregate_metrics,
        "cities": cities,
        "evaluated_tile_count": len(tile_ids_seen),
        "reconciliation_passed": True,
        "runtime": {
            "duration_seconds": duration,
            "image_count": image_count,
            "images_per_second": image_count / duration if duration > 0 else None,
            "milliseconds_per_image": 1000.0 * duration / image_count,
            "peak_gpu_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024.0**2)
                if device.type == "cuda" else None
            ),
            "peak_gpu_memory_reserved_mb": (
                torch.cuda.max_memory_reserved(device) / (1024.0**2)
                if device.type == "cuda" else None
            ),
        },
    }


def write_city_outputs(output: Path, cities: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    city_path = reports / "per_city_metrics.csv"
    class_path = reports / "per_city_class_metrics.csv"
    city_fields = (
        "city_id", "city_name", "tile_count", "mean_iou", "mean_dice",
        "mean_f1", "mean_precision", "mean_recall", "pixel_accuracy",
        "total_supervised_pixels",
    )
    with city_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=city_fields)
        writer.writeheader()
        for city in cities.values():
            writer.writerow({field: city.get(field) for field in city_fields})
    class_fields = (
        "city_id", "city_name", "class_id", "class_name", "target_pixels",
        "predicted_pixels", "true_positive", "false_positive", "false_negative",
        "iou", "dice", "precision", "recall", "f1", "present_in_target",
    )
    with class_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fields)
        writer.writeheader()
        for city in cities.values():
            for class_name, values in city["per_class"].items():
                row = {"city_id": city["city_id"], "city_name": city["city_name"]}
                row.update(values)
                row["class_name"] = class_name
                writer.writerow({field: row.get(field) for field in class_fields})
    return [city_path, class_path]


def write_inventory(output: Path, paths: Sequence[Path]) -> Path:
    inventory_path = output / "reports" / "artifact_inventory.json"
    rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(set(paths))
        if path.is_file() and path != inventory_path
    ]
    inventory_path.write_text(
        json.dumps({"status": "PASS", "files": rows}, indent=2), encoding="utf-8"
    )
    return inventory_path


def preflight(config_path: Path, checkpoint_root: Path, device: torch.device) -> dict[str, Any]:
    config = read_yaml(config_path)
    checks = validate_lock(config)
    metadata = inspect_dataset_metadata(config)
    model, checkpoint_path, _, provenance = load_locked_model(config, checkpoint_root, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "mode": "metadata_and_checkpoint_preflight",
        "configuration": str(config_path),
        "configuration_sha256": file_sha256(config_path),
        "selection_locked": True,
        "selected_method": "E9B U-Net++ + locked four-way TTA + class multipliers",
        "validation_reference_mean_iou": LOCKED_VALIDATION_MIOU,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "training_config_sha256": provenance["training_config_sha256"],
        "model_class": provenance["model_class"],
        "trainable_parameters": provenance["trainable_parameters"],
        "dataset_metadata": metadata,
        "test_split_loaded": False,
        "test_rasters_opened": False,
        "training_started": False,
        "checks": checks,
        "all_checks_passed": True,
    }


def run_once(config_path: Path, checkpoint_root: Path, device: torch.device) -> dict[str, Any]:
    config = read_yaml(config_path)
    lock_checks = validate_lock(config)
    metadata = inspect_dataset_metadata(config)
    model, checkpoint_path, _, provenance = load_locked_model(config, checkpoint_root, device)
    output = Path(str(config["experiment"]["output_directory"]))
    if output.exists():
        raise FileExistsError(
            f"Final test output already exists: {output}. Refusing another test run."
        )
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / "test_access_ledger.json"
    ledger_payload: dict[str, Any] = {
        "status": "STARTED",
        "authorized_at_utc": utc_now(),
        "test_authorized": True,
        "test_evaluation_count_this_output": 1,
        "configuration_sha256": file_sha256(config_path),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "git_commit": git_commit(),
    }
    ledger.write_text(json.dumps(ledger_payload, indent=2), encoding="utf-8")
    try:
        seed_everything(int(config["inference"]["seed"]), deterministic_algorithms=False)
        loader = create_test_loader(config)
        result = evaluate_test(model, loader, device, config)
        report: dict[str, Any] = {
            "status": "PASS",
            "created_at_utc": utc_now(),
            "experiment": config["experiment"]["name"],
            "method": "E9B U-Net++ + locked four-way TTA + locked class multipliers",
            "dataset_version": "v3-mt-d2",
            "split": "test",
            "test_authorized": True,
            "test_used": True,
            "test_evaluation_count_this_output": 1,
            "test_split_loaded": True,
            "test_rasters_opened": True,
            "training_started": False,
            "selection_locked_before_test": True,
            "validation_reference_mean_iou": LOCKED_VALIDATION_MIOU,
            "validation_lock_sha256": LOCKED_VALIDATION_SHA256,
            "git_commit": git_commit(),
            "configuration": str(config_path),
            "configuration_sha256": file_sha256(config_path),
            "dataset_metadata": metadata,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "training_config": provenance["training_config"],
            "training_config_sha256": provenance["training_config_sha256"],
            "model_name": provenance["model_name"],
            "model_class": provenance["model_class"],
            "trainable_parameters": provenance["trainable_parameters"],
            "tta_transforms": list(LOCKED_TRANSFORMS),
            "probability_quantization": "float16_cpu",
            "class_probability_multipliers": list(LOCKED_MULTIPLIERS),
            "device": str(device),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "result": result,
            "checks": {
                **lock_checks,
                "test_tile_count_exact": result["evaluated_tile_count"] == EXPECTED_TEST_TILE_COUNT,
                "per_city_reconciles": result["reconciliation_passed"] is True,
            },
            "interpretation_policy": config["safety"]["interpretation_policy"],
        }
        report["all_checks_passed"] = all(report["checks"].values())
        metric_paths = list(
            write_metrics_outputs(result["aggregate"], output / "metrics").values()
        )
        city_paths = write_city_outputs(output, result["cities"])
        summary = output / "reports" / "final_evaluation_summary.json"
        summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        ledger_payload.update(
            {
                "status": "COMPLETED",
                "completed_at_utc": utc_now(),
                "evaluated_test_tiles": result["evaluated_tile_count"],
                "test_rasters_opened": True,
                "training_performed": False,
                "checkpoint_updated": False,
            }
        )
        ledger.write_text(json.dumps(ledger_payload, indent=2), encoding="utf-8")
        inventory = write_inventory(
            output, [*metric_paths, *city_paths, summary, ledger]
        )
        report["artifact_inventory"] = str(inventory)
        return report
    except Exception as error:
        ledger_payload.update(
            {
                "status": "FAILED_AFTER_AUTHORIZATION",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "rerun_requires_manual_review": True,
            }
        )
        ledger.write_text(json.dumps(ledger_payload, indent=2), encoding="utf-8")
        raise


def smoke_test() -> dict[str, Any]:
    class Tiny(nn.Module):
        def forward(self, value: Tensor) -> Tensor:
            base = value[:, :1]
            return torch.cat((base, base * 0.5, -base, base * 0.25, -base * 0.5), dim=1)

    torch.manual_seed(17)
    image = torch.randn(2, 1, 9, 11)
    probability = tta_probability(Tiny(), image, LOCKED_TRANSFORMS)
    replay = probability.to(torch.float16).to(torch.float32)
    prediction = (replay * torch.tensor(LOCKED_MULTIPLIERS).view(1, 5, 1, 1)).argmax(1)
    round_trips = all(
        torch.equal(transform_tensor(transform_tensor(image, name), name), image)
        for name in LOCKED_TRANSFORMS
    )
    checks = {
        "probability_shape_correct": tuple(probability.shape) == (2, 5, 9, 11),
        "probability_finite": bool(torch.isfinite(probability).all()),
        "probability_sums_to_one": bool(
            torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]), atol=1e-6)
        ),
        "transform_round_trips": round_trips,
        "float16_cpu_replay_finite": bool(torch.isfinite(replay).all()),
        "prediction_shape_correct": tuple(prediction.shape) == (2, 9, 11),
        "test_path_not_implemented": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "test_split_loaded": False,
        "test_rasters_opened": False,
        "training_started": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--authorize-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        report = smoke_test()
    else:
        if args.checkpoint_root is None:
            parser.error("--checkpoint-root is required outside smoke-test mode.")
        device = torch.device(args.device)
        if args.preflight_only:
            if args.authorize_test:
                parser.error("--preflight-only and --authorize-test are mutually exclusive.")
            report = preflight(args.config, args.checkpoint_root, device)
        else:
            if not args.authorize_test:
                raise PermissionError(
                    "The final test split is sealed. Supply --authorize-test only after "
                    "reviewing a successful preflight and confirming the validation lock."
                )
            report = run_once(args.config, args.checkpoint_root, device)
    print(json.dumps(report, indent=2))
    if not report.get("all_checks_passed", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
