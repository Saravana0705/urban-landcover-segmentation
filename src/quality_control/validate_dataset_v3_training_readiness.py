"""Validate frozen Dataset V3 and one real batch from every split."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.data.dataloader import create_dataloaders, inspect_dataloaders


EXPECTED_COUNTS = {"train": 974, "val": 351, "test": 339}
EXPECTED_CONFIG = Path("config/training_unet_v3_sanity_e6p3_2ep.yaml")
EXPECTED_MANIFEST = Path("metadata/dataset_v3/dataset_manifest.csv")
EXPECTED_DATASET_CONFIG = Path("metadata/dataset_v3/freeze/dataset_config.json")
EXPECTED_NORMALIZATION = Path("metadata/dataset_v3/normalization/training_normalization.json")
EXPECTED_FREEZE_RECORD = Path("metadata/dataset_v3/freeze/dataset_v3_freeze.json")
EXPECTED_FREEZE_MARKER = Path("metadata/dataset_v3/freeze/FROZEN")
OUTPUT = Path("metadata/dataset_v3/training_readiness/unet_v3_sanity_readiness.json")


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON input")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def normalized(value: Any) -> str:
    return str(value).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPECTED_CONFIG)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.config, "sanity configuration"),
        (EXPECTED_MANIFEST, "V3 manifest"),
        (EXPECTED_DATASET_CONFIG, "frozen V3 dataset config"),
        (EXPECTED_NORMALIZATION, "V3 normalization"),
        (EXPECTED_FREEZE_RECORD, "V3 freeze record"),
        (EXPECTED_FREEZE_MARKER, "V3 freeze marker"),
    ):
        require_file(path, label)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite readiness report: {OUTPUT}")

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping.")

    freeze = load_json(EXPECTED_FREEZE_RECORD)
    marker = load_json(EXPECTED_FREEZE_MARKER)
    dataset_config = load_json(EXPECTED_DATASET_CONFIG)
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    loss = config.get("loss", {})
    training = config.get("training", {})
    sampling = config.get("sampling", {})
    experiment = config.get("experiment", {})
    benchmark = config.get("benchmark", {})

    checks = {
        "freeze_status_is_frozen": freeze.get("freeze_status") == "FROZEN" and marker.get("status") == "FROZEN",
        "freeze_all_checks_passed": freeze.get("all_checks_passed") is True,
        "freeze_record_hash_matches_marker": marker.get("freeze_record_sha256") == sha256_file(EXPECTED_FREEZE_RECORD),
        "manifest_hash_matches_freeze": freeze.get("manifest_sha256") == sha256_file(EXPECTED_MANIFEST),
        "dataset_config_hash_matches_freeze": freeze.get("dataset_config_sha256") == sha256_file(EXPECTED_DATASET_CONFIG),
        "dataset_config_is_v3_frozen": dataset_config.get("dataset_version") == "v3" and dataset_config.get("freeze_status") == "FROZEN",
        "training_manifest_is_v3": normalized(dataset.get("manifest_path")) == normalized(EXPECTED_MANIFEST),
        "evaluation_manifest_is_v3": normalized(dataset.get("evaluation_manifest_path")) == normalized(EXPECTED_MANIFEST),
        "training_dataset_config_is_frozen_v3": normalized(dataset.get("dataset_config_path")) == normalized(EXPECTED_DATASET_CONFIG),
        "evaluation_dataset_config_is_frozen_v3": normalized(dataset.get("evaluation_dataset_config_path")) == normalized(EXPECTED_DATASET_CONFIG),
        "normalization_path_is_v3": normalized(dataset.get("normalization_path")) == normalized(EXPECTED_NORMALIZATION),
        "model_matches_e6_p3_unet": (
            str(model.get("name", "")).lower() == "unet"
            and int(model.get("base_channels", -1)) == 32
            and int(model.get("input_channels", -1)) == 2
            and int(model.get("num_classes", -1)) == 5
        ),
        "loss_matches_e6_p3_ce_tversky": (
            str(loss.get("name", "")).lower() == "ce_tversky"
            and float(loss.get("ce_weight", -1)) == 0.30
            and float(loss.get("tversky_weight", -1)) == 0.70
            and float(loss.get("tversky_alpha", -1)) == 0.20
            and float(loss.get("tversky_beta", -1)) == 0.80
            and list(loss.get("class_weights", []))
            == [0.708374, 0.834706, 0.244015, 2.046274, 1.166632]
        ),
        "sanity_epoch_count_is_2": int(training.get("max_epochs", -1)) == 2,
        "standard_shuffle_without_weighted_resampling": str(sampling.get("strategy", "")).lower() == "standard",
        "unique_sanity_output_directory": normalized(experiment.get("output_directory")) == "outputs/model_experiments/unet_v3_sanity_e6p3_2ep",
        "inference_benchmark_disabled": benchmark.get("enabled") is False,
        "frozen_split_counts_match": freeze.get("split_tile_counts") == EXPECTED_COUNTS,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "V3 training-readiness metadata checks failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    bundle = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(config.get("seed", 20260725)),
        manifest_path=EXPECTED_MANIFEST,
        dataset_config_path=EXPECTED_DATASET_CONFIG,
        evaluation_manifest_path=EXPECTED_MANIFEST,
        evaluation_dataset_config_path=EXPECTED_DATASET_CONFIG,
        exclude_zero_valid=True,
        verify_raster_metadata=True,
        pin_memory=False,
        persistent_workers=False,
    )
    loader_report = inspect_dataloaders(bundle)
    observed_counts = {
        split: int(loader_report["splits"][split]["dataset"]["tile_count"])
        for split in EXPECTED_COUNTS
    }
    batch_checks = {
        "loader_split_counts_match": observed_counts == EXPECTED_COUNTS,
        "all_first_batches_finite": all(
            loader_report["splits"][split]["first_batch"]["image_all_finite"]
            for split in EXPECTED_COUNTS
        ),
        "all_invalid_pixels_ignored": all(
            loader_report["splits"][split]["first_batch"]["invalid_pixels_are_ignored"]
            for split in EXPECTED_COUNTS
        ),
        "all_supervised_pixels_valid": all(
            loader_report["splits"][split]["first_batch"]["supervised_pixels_are_valid"]
            for split in EXPECTED_COUNTS
        ),
    }
    if not all(batch_checks.values()):
        raise RuntimeError(
            "V3 DataLoader readiness checks failed: "
            + ", ".join(name for name, passed in batch_checks.items() if not passed)
        )

    report = {
        "schema_version": "dataset-v3-unet-sanity-readiness-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_version": "v3",
        "training_config": str(args.config),
        "training_config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(EXPECTED_MANIFEST),
        "dataset_config_sha256": sha256_file(EXPECTED_DATASET_CONFIG),
        "freeze_record_sha256": sha256_file(EXPECTED_FREEZE_RECORD),
        "split_tile_counts": observed_counts,
        "metadata_checks": checks,
        "batch_checks": batch_checks,
        "dataloader_report": loader_report,
        "next": (
            "python -m src.training.train_segmentation --config "
            "config/training_unet_v3_sanity_e6p3_2ep.yaml --integration-check"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PASS",
        "split_tile_counts": observed_counts,
        "first_batch_validation": "PASS",
        "next": report["next"],
    }, indent=2))


if __name__ == "__main__":
    main()
