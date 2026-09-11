"""Evaluate cross-city consistency, speckle sensitivity and inference cost.

Only the frozen V3-MT-D2 validation split is instantiated. No training occurs,
checkpoints are never modified, and test rasters are never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.segmentation_dataset import Sentinel1UrbanDataset
from src.data.transforms import AugmentationConfig, IdentityTransform, Sentinel1TrainTransform
from src.evaluation.benchmarking import benchmark_inference, load_best_checkpoint, runtime_environment
from src.evaluation.evaluate_per_city import evaluate, validate_dataset
from src.models.model_factory import build_model
from src.training.train_segmentation import read_yaml


DEFAULT_CONFIG = Path("config/evaluation_d2_final_robustness.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="CODE=PATH")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_overrides(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CODE=PATH, received {value!r}")
        code, raw_path = value.split("=", 1)
        result[code.strip().upper()] = Path(raw_path.strip()).expanduser().resolve()
    return result


def find_checkpoint(root: Path, item: Mapping[str, Any], overrides: Mapping[str, Path]) -> Path:
    code = str(item["code"]).upper()
    if code in overrides:
        path = overrides[code]
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint override not found: {path}")
        return path
    experiment = str(item["experiment_name"]).lower()
    matches = []
    for path in root.rglob("best.pt"):
        normalized = path.as_posix().lower()
        if "integration_check" not in normalized and experiment in normalized:
            matches.append(path.resolve())
    unique = {sha256(path): path for path in matches}
    if len(unique) != 1:
        raise RuntimeError(f"{code}: expected one distinct checkpoint beneath {root}, found {len(unique)}. Use --checkpoint {code}=PATH.")
    return next(iter(unique.values()))


def noise_transform(looks: float | None):
    if looks is None:
        return IdentityTransform()
    return Sentinel1TrainTransform(AugmentationConfig(
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotation_90_probability=0.0,
        enable_sar_intensity=False,
        sar_intensity_probability=0.0,
        enable_speckle=True,
        speckle_probability=1.0,
        speckle_looks=float(looks),
        ignore_index=255,
    ))


def build_validation_loader(config: Mapping[str, Any], transform: Any) -> DataLoader:
    dataset_config = config["dataset"]
    runtime = config["runtime"]
    dataset = Sentinel1UrbanDataset(
        split="val",
        manifest_path=dataset_config["manifest_path"],
        dataset_config_path=dataset_config["dataset_config_path"],
        project_root=Path.cwd(),
        joint_transform=transform,
        exclude_zero_valid=True,
        verify_raster_metadata=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(runtime["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=bool(runtime.get("pin_memory", False)),
    )


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite explicitly.")
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def city_rows(item: Mapping[str, Any], condition: str, result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows = []
    class_rows = []
    for city_id in sorted(result["city_reports"]):
        report = result["city_reports"][city_id]
        overall_rows.append({
            "experiment_code": item["code"], "model": item["display_name"],
            "condition": condition, "city_id": city_id,
            "city_name": result["city_names"][city_id],
            "tile_count": result["tile_counts"][city_id],
            "valid_pixel_count": result["valid_counts"][city_id],
            "mean_iou": report["mean_iou"], "mean_dice": report["mean_dice"],
            "pixel_accuracy": report["pixel_accuracy"],
        })
        for class_name, values in report["per_class"].items():
            class_rows.append({
                "experiment_code": item["code"], "model": item["display_name"],
                "condition": condition, "city_id": city_id,
                "city_name": result["city_names"][city_id],
                "class_name": class_name, "iou": values["iou"],
                "dice": values["dice"], "precision": values["precision"],
                "recall": values["recall"], "target_pixels": values["target_pixels"],
            })
    return overall_rows, class_rows


def save_figures(overall: pd.DataFrame, aggregate: pd.DataFrame, inference: pd.DataFrame, output: Path) -> None:
    clean = overall.loc[overall["condition"] == "clean"]
    pivot = clean.pivot(index="model", columns="city_id", values="mean_iou")
    order = list(dict.fromkeys(overall["model"]))
    pivot = pivot.reindex(order)
    fig, axis = plt.subplots(figsize=(7.5, 5.3))
    values = pivot.to_numpy(float)
    image = axis.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", color="white" if values[row, column] > 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=axis, label="Validation mIoU")
    fig.tight_layout()
    fig.savefig(output / "cross_city_miou_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "cross_city_miou_heatmap.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.8, 5.3))
    condition_order = list(dict.fromkeys(aggregate["condition"]))
    for model, group in aggregate.groupby("model", sort=False):
        group = group.set_index("condition").reindex(condition_order)
        axis.plot(condition_order, group["mean_iou"], marker="o", label=model)
    axis.set_ylabel("Validation mIoU")
    axis.set_xlabel("Additional synthetic Gamma speckle")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "speckle_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "speckle_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    if inference["inference_images_per_second"].notna().any():
        fig, axis = plt.subplots(figsize=(8.8, 5.1))
        axis.bar(inference["model"], inference["inference_images_per_second"], color="#2f6f9f")
        axis.set_ylabel("Inference throughput (images/s)")
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "inference_throughput.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / "inference_throughput.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    if config.get("dataset", {}).get("split") != "val" or config.get("safety", {}).get("test_evaluation_enabled") is not False:
        raise RuntimeError("Safety lock failed: this command is validation-only")
    checkpoint_root = args.checkpoint_root.resolve()
    overrides = parse_overrides(args.checkpoint)
    models = config["models"]
    checkpoints = {item["code"]: find_checkpoint(checkpoint_root, item, overrides) for item in models}

    audit = []
    for item in models:
        path = checkpoints[item["code"]]
        audit.append({"code": item["code"], "model": item["display_name"], "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    preflight = {
        "status": "PASS", "model_count": len(models), "checkpoint_count": len(checkpoints),
        "selection_split": "validation", "test_split_loaded": False,
        "training_started": False, "checkpoints": audit,
    }
    print(json.dumps(preflight, indent=2))
    if args.preflight_only:
        return

    output = (args.output_dir or Path(config["experiment"]["output_directory"])).resolve()
    prepare_output(output, args.overwrite)
    requested_device = args.device or str(config["runtime"].get("device", "auto"))
    device = resolve_device(requested_device)
    seed = int(config["runtime"]["seed"])
    expected_cities = set(config["dataset"]["expected_city_ids"])
    expected_tiles = int(config["dataset"]["expected_tile_count"])
    aggregate_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []

    clean_loader = build_validation_loader(config, IdentityTransform())
    validate_dataset(clean_loader, expected_city_ids=expected_cities, expected_tile_count=expected_tiles)

    for model_index, item in enumerate(models, start=1):
        model_config = read_yaml(Path(item["config_path"]))
        model = build_model(model_config)
        parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
        if parameter_count != int(item["expected_trainable_parameters"]):
            raise RuntimeError(f"{item['code']}: parameter count mismatch")
        load_best_checkpoint(model, checkpoints[item["code"]], device, strict=True)
        model.to(device).eval()
        print(f"\n[{model_index}/{len(models)}] {item['display_name']}")

        profile = benchmark_inference(
            model=model, loader=clean_loader, device=device,
            warmup_batches=int(config["runtime"]["warmup_batches"]),
            measured_batches=int(config["runtime"]["measured_batches"]),
            use_mixed_precision=bool(config["runtime"]["mixed_precision_inference_benchmark"]),
        )
        inference_rows.append({"experiment_code": item["code"], "model": item["display_name"], "trainable_parameters": parameter_count, **profile})

        clean_miou = None
        for condition_index, condition in enumerate(config["noise_conditions"]):
            condition_name = str(condition["name"])
            looks = finite(condition.get("speckle_looks"))
            np.random.seed(seed + condition_index * 1000)
            torch.manual_seed(seed + condition_index * 1000)
            loader = clean_loader if looks is None else build_validation_loader(config, noise_transform(looks))
            progress = tqdm(loader, desc=f"{item['code']} {condition_name}", dynamic_ncols=True, leave=False, mininterval=0.5)
            result = evaluate(model=model, loader=progress, device=device, expected_city_ids=expected_cities)
            report = result["aggregate"]
            current_miou = float(report["mean_iou"])
            if condition_name == "clean":
                clean_miou = current_miou
                expected = float(item["expected_mean_iou"])
                tolerance = float(item["clean_miou_tolerance"])
                if abs(current_miou - expected) > tolerance:
                    raise RuntimeError(f"{item['code']}: clean mIoU {current_miou:.9f} differs from locked {expected:.9f} by more than {tolerance}")
            aggregate_rows.append({
                "experiment_code": item["code"], "model": item["display_name"],
                "condition": condition_name, "speckle_looks": looks,
                "mean_iou": current_miou, "mean_dice": report["mean_dice"],
                "pixel_accuracy": report["pixel_accuracy"],
                "delta_miou_from_clean": current_miou - float(clean_miou),
                "relative_miou_change_percent": 100.0 * (current_miou - float(clean_miou)) / float(clean_miou),
                "evaluated_tiles": result["evaluated_tile_count"],
            })
            cities, classes = city_rows(item, condition_name, result)
            overall_rows.extend(cities)
            class_rows.extend(classes)
            print(f"  {condition_name}: mIoU={current_miou:.6f}, delta={current_miou - float(clean_miou):+.6f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = pd.DataFrame(aggregate_rows)
    overall = pd.DataFrame(overall_rows)
    classes = pd.DataFrame(class_rows)
    inference = pd.DataFrame(inference_rows)
    clean_city = overall.loc[overall["condition"] == "clean"]
    city_summary = clean_city.groupby(["experiment_code", "model"], sort=False)["mean_iou"].agg(
        city_mean_miou="mean", city_std_miou="std", city_min_miou="min", city_max_miou="max"
    ).reset_index()
    city_summary["city_miou_range"] = city_summary["city_max_miou"] - city_summary["city_min_miou"]

    aggregate.to_csv(output / "noise_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "cross_city_condition_metrics.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(output / "cross_city_class_metrics.csv", index=False, encoding="utf-8-sig")
    city_summary.to_csv(output / "cross_city_consistency_summary.csv", index=False, encoding="utf-8-sig")
    inference.to_csv(output / "inference_gpu_profile.csv", index=False, encoding="utf-8-sig")
    save_figures(overall, aggregate, inference, output)

    environment = runtime_environment(requested_device=requested_device, resolved_device=device, mixed_precision_enabled=bool(config["runtime"]["mixed_precision_inference_benchmark"]))
    summary = {
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "configuration": str(args.config.resolve()),
        "configuration_sha256": sha256(args.config.resolve()),
        "scope": "v3-mt-d2 validation only", "test_rasters_opened": False,
        "training_performed": False, "models_evaluated": len(models),
        "validation_cities": sorted(expected_cities), "validation_tiles": expected_tiles,
        "noise_method": "multiplicative Gamma noise on raw linear Sigma0 before frozen clipping and normalization",
        "runtime_environment": environment, "checkpoints": audit,
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "robustness_evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "models_evaluated": len(models),
        "validation_cities": sorted(expected_cities), "test_rasters_opened": False,
        "output_directory": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()

