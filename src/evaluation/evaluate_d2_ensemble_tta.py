"""Validation-first weighted ensemble and flip TTA for D2 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor, nn

from src.data.dataloader import create_dataloaders
from src.evaluation.benchmarking import load_best_checkpoint
from src.evaluation.evaluate_segmentation import _extract_batch, write_metrics_outputs
from src.evaluation.metrics import MetricsConfig, SegmentationMetrics
from src.models.model_factory import build_model


CLASS_NAMES = ("buildings", "roads", "vegetation", "bare_land", "water")
TRANSFORMS = ("identity", "horizontal_flip", "vertical_flip")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _flip(x: Tensor, name: str) -> Tensor:
    if name == "identity":
        return x
    if name == "horizontal_flip":
        return torch.flip(x, dims=(-1,))
    if name == "vertical_flip":
        return torch.flip(x, dims=(-2,))
    raise ValueError(f"Unsupported TTA transform: {name}")


class EnsembleTTA(nn.Module):
    def __init__(self, members: list[tuple[str, nn.Module, float]], transforms: list[str]) -> None:
        super().__init__()
        if not members or not transforms:
            raise ValueError("At least one member and TTA transform are required.")
        if any(name not in TRANSFORMS for name in transforms):
            raise ValueError(f"TTA transforms must be selected from {TRANSFORMS}.")
        weights = torch.tensor([item[2] for item in members], dtype=torch.float32)
        if bool((weights < 0).any()) or float(weights.sum()) <= 0:
            raise ValueError("Ensemble weights must be non-negative with a positive sum.")
        self.names = [item[0] for item in members]
        self.models = nn.ModuleList([item[1] for item in members])
        self.register_buffer("weights", weights / weights.sum())
        self.transforms = transforms

    def forward(self, images: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        if len(images) != len(self.models):
            raise ValueError("One aligned input tensor is required per ensemble member.")
        probability_sum: Tensor | None = None
        for image, weight, model in zip(images, self.weights, self.models):
            member_probability = None
            for transform in self.transforms:
                logits = model(_flip(image, transform))
                probability = _flip(torch.softmax(logits, dim=1), transform)
                member_probability = probability if member_probability is None else member_probability + probability
            member_probability = member_probability / len(self.transforms)
            probability_sum = weight * member_probability if probability_sum is None else probability_sum + weight * member_probability
        return torch.log(probability_sum.clamp_min(1e-8))


def build_members(config: Mapping[str, Any], device: torch.device, *, require_checkpoints: bool) -> list[tuple[str, nn.Module, float]]:
    members: list[tuple[str, nn.Module, float]] = []
    for item in config.get("members", []):
        training_config_path = Path(str(item["training_config"]))
        training_config = read_yaml(training_config_path)
        model = build_model(training_config).to(device)
        checkpoint_path = Path(str(item["checkpoint"]))
        if require_checkpoints:
            load_best_checkpoint(model, checkpoint_path, device)
        model.eval()
        members.append((str(item["name"]), model, float(item.get("weight", 1.0))))
    return members


def evaluate(config: Mapping[str, Any], *, device: torch.device, max_batches: int | None = None) -> dict[str, Any]:
    split = str(config.get("split", "val")).lower()
    if split != "val":
        raise ValueError("E6A is validation-only. Test access is intentionally blocked.")
    loaders = []
    for item in config["members"]:
        training_config = read_yaml(Path(str(item["training_config"])))
        dataset = training_config["dataset"]
        bundle = create_dataloaders(
            batch_size=int(config.get("batch_size", 2)),
            num_workers=int(config.get("num_workers", 0)),
            manifest_path=dataset["manifest_path"],
            dataset_config_path=dataset["dataset_config_path"],
            evaluation_manifest_path=dataset.get("evaluation_manifest_path", dataset["manifest_path"]),
            evaluation_dataset_config_path=dataset.get("evaluation_dataset_config_path", dataset["dataset_config_path"]),
        )
        loaders.append(bundle.val_loader)
    model = EnsembleTTA(
        build_members(config, device, require_checkpoints=True),
        list(config.get("tta_transforms", TRANSFORMS)),
    ).to(device).eval()
    metrics = SegmentationMetrics(
        MetricsConfig(num_classes=5, ignore_index=255, class_names=CLASS_NAMES), device="cpu"
    )
    with torch.inference_mode():
        for index, batches in enumerate(zip(*loaders)):
            extracted = [_extract_batch(batch) for batch in batches]
            tile_ids = [list(batch["tile_id"]) for batch in batches]
            if any(value != tile_ids[0] for value in tile_ids[1:]):
                raise RuntimeError("Ensemble member datasets are not tile-aligned.")
            images = [item[0].to(device) for item in extracted]
            target, validity = extracted[0][1], extracted[0][2]
            target = target.clone()
            if validity is not None:
                target[~validity.bool()] = 255
            metrics.update(model(images).cpu(), target.cpu())
            if max_batches is not None and index + 1 >= max_batches:
                break
    return metrics.compute()


def smoke_test(config: Mapping[str, Any]) -> dict[str, Any]:
    members = build_members(config, torch.device("cpu"), require_checkpoints=False)
    model = EnsembleTTA(members, list(config.get("tta_transforms", TRANSFORMS)))
    channels = [
        int(read_yaml(Path(item["training_config"]))["model"]["input_channels"])
        for item in config["members"]
    ]
    output = model([torch.randn(1, value, 65, 67) for value in channels])
    report = {
        "member_count": len(members),
        "tta_transforms": model.transforms,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "validation_only": str(config.get("split", "val")).lower() == "val",
    }
    report["all_checks_passed"] = all((report["member_count"] >= 2, report["output_shape"] == [1, 5, 65, 67], report["output_finite"], report["validation_only"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    config = read_yaml(args.config)
    if args.smoke_test:
        report = smoke_test(config)
    else:
        report = evaluate(config, device=torch.device(args.device), max_batches=args.max_batches)
        output = Path(str(config.get("output_directory", "outputs/model_experiments/d2_e6a_ensemble_tta/metrics")))
        write_metrics_outputs(report, output)
        (output / "ensemble_report.json").write_text(json.dumps({"config": config, "metrics": report}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.smoke_test and not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
