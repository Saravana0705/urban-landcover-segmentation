"""Cache D2 ensemble probabilities and calibrate them on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor

from src.data.dataloader import create_dataloaders
from src.evaluation.benchmarking import load_best_checkpoint
from src.evaluation.evaluate_segmentation import _extract_batch
from src.models.model_factory import build_model


CLASS_NAMES = ("buildings", "roads", "vegetation", "bare_land", "water")
TRANSFORMS = ("identity", "horizontal_flip", "vertical_flip", "rotate_180")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _transform(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return value
    if name == "horizontal_flip":
        return torch.flip(value, dims=(-1,))
    if name == "vertical_flip":
        return torch.flip(value, dims=(-2,))
    if name == "rotate_180":
        return torch.flip(value, dims=(-2, -1))
    raise ValueError(f"Unsupported transform: {name}")


def _loader(training_config: Mapping[str, Any], config: Mapping[str, Any], split: str):
    dataset = training_config["dataset"]
    bundle = create_dataloaders(
        batch_size=int(config.get("batch_size", 2)),
        num_workers=int(config.get("num_workers", 0)),
        manifest_path=dataset["manifest_path"],
        dataset_config_path=dataset["dataset_config_path"],
        evaluation_manifest_path=dataset.get("evaluation_manifest_path", dataset["manifest_path"]),
        evaluation_dataset_config_path=dataset.get("evaluation_dataset_config_path", dataset["dataset_config_path"]),
    )
    return bundle.val_loader if split == "val" else bundle.test_loader


def _member_probability(model, image: Tensor, transforms: list[str]) -> Tensor:
    total = None
    for name in transforms:
        logits = model(_transform(image, name))
        probability = _transform(torch.softmax(logits, dim=1), name)
        total = probability if total is None else total + probability
    if total is None:
        raise ValueError("At least one TTA transform is required.")
    return total / len(transforms)


def cache_probabilities(
    config: Mapping[str, Any], *, device: torch.device, split: str, authorize_test: bool
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test.")
    if split == "test" and not authorize_test:
        raise PermissionError("Test caching is blocked until --authorize-test is supplied after locking.")
    transforms = list(config.get("tta_transforms", TRANSFORMS))
    if any(name not in TRANSFORMS for name in transforms):
        raise ValueError(f"TTA transforms must be selected from {TRANSFORMS}.")
    members = list(config.get("members", []))
    if len(members) < 2:
        raise ValueError("At least two ensemble members are required.")

    models = []
    loaders = []
    for member in members:
        training_config = read_yaml(Path(str(member["training_config"])))
        model = build_model(training_config).to(device)
        load_best_checkpoint(model, Path(str(member["checkpoint"])), device)
        models.append(model.eval())
        loaders.append(_loader(training_config, config, split))

    cache_root = Path(str(config.get("cache_directory", "outputs/cache/d2_e9a"))) / split
    cache_root.mkdir(parents=True, exist_ok=True)
    for old in cache_root.glob("batch_*.pt"):
        old.unlink()

    batch_count = 0
    tile_count = 0
    with torch.inference_mode():
        for batch_index, batches in enumerate(zip(*loaders)):
            extracted = [_extract_batch(batch) for batch in batches]
            tile_ids = [list(batch["tile_id"]) for batch in batches]
            if any(ids != tile_ids[0] for ids in tile_ids[1:]):
                raise RuntimeError("Ensemble member datasets are not tile-aligned.")
            target, validity = extracted[0][1].clone(), extracted[0][2]
            if validity is not None:
                target[~validity.bool()] = 255
            probabilities = []
            for model, item in zip(models, extracted):
                image = item[0].to(device, dtype=torch.float32, non_blocking=True)
                probabilities.append(_member_probability(model, image, transforms).cpu().half())
            torch.save(
                {
                    "probabilities": torch.stack(probabilities),
                    "target": target.to(torch.uint8),
                    "tile_ids": tile_ids[0],
                },
                cache_root / f"batch_{batch_index:04d}.pt",
            )
            batch_count += 1
            tile_count += len(tile_ids[0])

    manifest = {
        "status": "PASS",
        "split": split,
        "validation_only_search": True,
        "member_names": [str(item["name"]) for item in members],
        "tta_transforms": transforms,
        "batch_count": batch_count,
        "tile_count": tile_count,
        "dtype": "float16",
    }
    (cache_root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _confusion_metrics(confusion: Tensor) -> dict[str, Any]:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    return {
        "mean_iou": float(torch.nanmean(iou)),
        "per_class_iou": {name: float(iou[index]) for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def _score_candidates(
    cache_files: list[Path], weights: Tensor, multipliers: Tensor
) -> list[dict[str, Any]]:
    if weights.ndim != 2 or multipliers.ndim != 2 or weights.shape[0] != multipliers.shape[0]:
        raise ValueError("Candidate weights and multipliers must be aligned matrices.")
    candidate_count = weights.shape[0]
    confusions = torch.zeros(candidate_count, len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    normalized_weights = weights / weights.sum(1, keepdim=True)
    for cache_file in cache_files:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        probabilities = payload["probabilities"].float()
        target = payload["target"].long()
        combined = torch.einsum("km,mbchw->kbchw", normalized_weights, probabilities)
        combined = combined * multipliers[:, None, :, None, None]
        predictions = combined.argmax(2)
        valid = target != 255
        for index in range(candidate_count):
            encoded = target[valid] * len(CLASS_NAMES) + predictions[index][valid]
            confusions[index] += torch.bincount(encoded, minlength=25).reshape(5, 5)
    return [_confusion_metrics(value) for value in confusions]


def search_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    cache_root = Path(str(config.get("cache_directory", "outputs/cache/d2_e9a"))) / "val"
    cache_files = sorted(cache_root.glob("batch_*.pt"))
    if not cache_files:
        raise FileNotFoundError(f"No validation cache shards found in {cache_root}.")
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = [str(item["name"]) for item in config["members"]]
    if manifest.get("split") != "val" or manifest.get("member_names") != expected_names:
        raise ValueError("Validation cache manifest does not match the configured member order.")
    if manifest.get("tta_transforms") != list(config.get("tta_transforms", TRANSFORMS)):
        raise ValueError("Validation cache TTA transforms do not match the runtime configuration.")
    member_count = len(config["members"])
    weight_candidates = torch.tensor(config["search"]["weight_candidates"], dtype=torch.float32)
    if weight_candidates.ndim != 2 or weight_candidates.shape[1] != member_count:
        raise ValueError("Every weight candidate must provide one value per member.")
    ones = torch.ones(weight_candidates.shape[0], len(CLASS_NAMES))
    weight_results = _score_candidates(cache_files, weight_candidates, ones)
    best_index = max(range(len(weight_results)), key=lambda index: weight_results[index]["mean_iou"])
    best_weights = weight_candidates[best_index].tolist()
    best_multipliers = torch.ones(len(CLASS_NAMES), dtype=torch.float32)
    best_metrics = weight_results[best_index]

    class_grid = config["search"].get("class_multiplier_candidates", {})
    trace: list[dict[str, Any]] = [{"stage": "weights", "candidate": best_weights, **best_metrics}]
    for pass_index in range(int(config["search"].get("coordinate_passes", 2))):
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = [float(value) for value in class_grid.get(class_name, [1.0])]
            candidate_multipliers = best_multipliers.repeat(len(values), 1)
            candidate_multipliers[:, class_index] = torch.tensor(values)
            repeated_weights = torch.tensor(best_weights).repeat(len(values), 1)
            results = _score_candidates(cache_files, repeated_weights, candidate_multipliers)
            selected = max(range(len(results)), key=lambda index: results[index]["mean_iou"])
            if results[selected]["mean_iou"] >= best_metrics["mean_iou"]:
                best_multipliers = candidate_multipliers[selected]
                best_metrics = results[selected]
            trace.append({
                "stage": f"class_pass_{pass_index + 1}_{class_name}",
                "candidate": best_multipliers.tolist(),
                **best_metrics,
            })

    locked = {
        "status": "PASS",
        "selection_split": "val",
        "test_used": False,
        "member_names": [str(item["name"]) for item in config["members"]],
        "weights": best_weights,
        "class_names": list(CLASS_NAMES),
        "class_probability_multipliers": best_multipliers.tolist(),
        "validation_metrics": best_metrics,
        "search_trace": trace,
    }
    digest_payload = json.dumps(
        {"weights": best_weights, "multipliers": best_multipliers.tolist()}, sort_keys=True
    ).encode("utf-8")
    locked["lock_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    output = Path(str(config.get("output_directory", "outputs/model_experiments/d2_e9a_calibrated_ensemble")))
    output.mkdir(parents=True, exist_ok=True)
    (output / "locked_calibration.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
    return locked


def evaluate_locked(config: Mapping[str, Any], locked_path: Path, split: str) -> dict[str, Any]:
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    expected_names = [str(item["name"]) for item in config["members"]]
    if locked.get("member_names") != expected_names:
        raise ValueError("Locked member order does not match the runtime configuration.")
    digest_payload = json.dumps(
        {
            "weights": locked["weights"],
            "multipliers": locked["class_probability_multipliers"],
        },
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(digest_payload).hexdigest() != locked.get("lock_sha256"):
        raise ValueError("Locked calibration checksum is invalid.")
    cache_root = Path(str(config.get("cache_directory", "outputs/cache/d2_e9a"))) / split
    cache_files = sorted(cache_root.glob("batch_*.pt"))
    if not cache_files:
        raise FileNotFoundError(f"No {split} cache shards found in {cache_root}.")
    weights = torch.tensor([locked["weights"]], dtype=torch.float32)
    multipliers = torch.tensor(
        [locked["class_probability_multipliers"]], dtype=torch.float32
    )
    metrics = _score_candidates(cache_files, weights, multipliers)[0]
    report = {
        "status": "PASS",
        "split": split,
        "locked_calibration": str(locked_path),
        "lock_sha256": locked["lock_sha256"],
        "metrics": metrics,
    }
    output = Path(str(config.get("output_directory", "outputs/model_experiments/d2_e9a_calibrated_ensemble")))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"locked_{split}_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def smoke_test() -> dict[str, Any]:
    root = Path("outputs/smoke_tests/d2_e9a_cache")
    root.mkdir(parents=True, exist_ok=True)
    target = torch.tensor([[[0, 1], [3, 4]]], dtype=torch.uint8)
    first = torch.full((1, 1, 5, 2, 2), 0.01, dtype=torch.float16)
    second = torch.full_like(first, 0.01)
    for row, cls in enumerate((0, 1, 3, 4)):
        y, x = divmod(row, 2)
        first[0, 0, cls, y, x] = 0.90
        second[0, 0, (cls + 1) % 5, y, x] = 0.90
    cache_file = root / "batch_0000.pt"
    torch.save({"probabilities": torch.cat((first, second), 0), "target": target}, cache_file)
    results = _score_candidates(
        [cache_file],
        torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
        torch.ones(2, 5),
    )
    report = {
        "candidate_count": len(results),
        "best_candidate_index": max(range(2), key=lambda index: results[index]["mean_iou"]),
        "best_mean_iou": max(item["mean_iou"] for item in results),
        "test_requires_explicit_authorization": True,
    }
    report["all_checks_passed"] = report["best_candidate_index"] == 0 and report["best_mean_iou"] == 1.0
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--evaluate-locked", action="store_true")
    parser.add_argument("--locked-config", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--authorize-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        report = smoke_test()
    else:
        if args.config is None:
            parser.error("--config is required unless --smoke-test is used.")
        config = read_yaml(args.config)
        if args.search and args.split != "val":
            raise PermissionError("Calibration search is validation-only; test labels cannot tune parameters.")
        if args.evaluate_locked and args.split == "test" and not args.authorize_test:
            raise PermissionError("Locked test evaluation requires --authorize-test.")
        reports: dict[str, Any] = {}
        if args.cache:
            reports["cache"] = cache_probabilities(
                config, device=torch.device(args.device), split=args.split,
                authorize_test=args.authorize_test,
            )
        if args.search:
            reports["search"] = search_calibration(config)
        if args.evaluate_locked:
            if args.locked_config is None:
                parser.error("--locked-config is required with --evaluate-locked.")
            reports["locked_evaluation"] = evaluate_locked(config, args.locked_config, args.split)
        if not args.cache and not args.search and not args.evaluate_locked:
            parser.error("Choose --cache, --search, --evaluate-locked, or a combination.")
        report = reports
    print(json.dumps(report, indent=2))
    if args.smoke_test and not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
