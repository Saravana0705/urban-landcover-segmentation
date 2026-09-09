"""Evaluate a fixed, validation-only U-Net++ checkpoint model soup.

E11 averages compatible E9B, E10B and E10D state dictionaries using five
predeclared candidates. It deliberately performs no training, multiplier
search or test evaluation. The winning validation candidate is locked and
written as one ordinary project checkpoint for later, one-time test use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch import Tensor, nn

from src.data.dataloader import create_dataloaders, seed_everything
from src.evaluation.evaluate_segmentation import _extract_batch, write_metrics_outputs
from src.evaluation.metrics import metrics_from_confusion_matrix
from src.models.model_factory import build_model


ALLOWED_TRANSFORMS = (
    "identity",
    "horizontal_flip",
    "vertical_flip",
    "rotate_180",
)


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_contract(training_config: Mapping[str, Any]) -> dict[str, Any]:
    model = training_config.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Training configuration lacks a model mapping.")
    keys = (
        "name",
        "input_channels",
        "num_classes",
        "base_channels",
        "dropout",
        "use_batch_norm",
        "bilinear_upsampling",
        "deep_supervision",
    )
    contract = {key: model.get(key) for key in keys}
    contract["deep_supervision"] = bool(model.get("deep_supervision", False))
    return contract


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("selection_split") != "val":
        raise PermissionError("E11 selection_split must remain 'val'.")
    if config.get("test_access") != "blocked":
        raise PermissionError("E11 test_access must remain blocked.")

    members = config.get("model_members")
    if not isinstance(members, list) or len(members) != 3:
        raise ValueError("E11 requires exactly three checkpoint members.")
    names = [str(item.get("name")) for item in members]
    if names != ["e9b", "e10b", "e10d"]:
        raise ValueError("E11 member order must be e9b, e10b, e10d.")

    candidates = config.get("soup_candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("E11 requires exactly five predeclared candidates.")
    for candidate in candidates:
        weights = candidate.get("weights")
        if not isinstance(weights, list) or len(weights) != len(members):
            raise ValueError("Each soup candidate needs three weights.")
        values = [float(value) for value in weights]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Soup weights must be finite and non-negative.")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Every soup candidate must sum exactly to one.")

    transforms = list(config.get("tta_transforms", []))
    if transforms != list(ALLOWED_TRANSFORMS):
        raise ValueError(
            "E11 must use identity, horizontal flip, vertical flip and "
            "180-degree rotation in that order."
        )
    class_names = list(config.get("class_names", []))
    multipliers = list(config.get("class_probability_multipliers", []))
    if len(class_names) != 5 or len(multipliers) != 5:
        raise ValueError("Exactly five classes and five multipliers are required.")
    if config.get("probability_quantization") != "float16_cpu":
        raise ValueError("E11 must preserve E10C float16 CPU probability scoring.")

    policy = config.get("selection_policy", {})
    forbidden = (
        policy.get("additional_search_permitted"),
        policy.get("multiplier_search_permitted"),
        policy.get("training_permitted"),
    )
    if any(bool(value) for value in forbidden):
        raise PermissionError("E11 cannot enable further search or training.")


def _resolve_checkpoint(root: Path, relative_path: str) -> Path:
    direct = (root / relative_path).resolve()
    if direct.is_file():
        return direct

    parts = Path(relative_path).parts
    matches = []
    for candidate in root.rglob(Path(relative_path).name):
        if candidate.is_file() and tuple(candidate.parts[-len(parts) :]) == parts:
            matches.append(candidate.resolve())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Checkpoint '{relative_path}' was not found under {root.resolve()}."
        )
    raise RuntimeError(
        f"Checkpoint '{relative_path}' is ambiguous under {root.resolve()}: "
        + ", ".join(str(path) for path in matches)
    )


def _load_checkpoint_state(path: Path) -> tuple[dict[str, Tensor], Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping: {path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(f"Checkpoint lacks model_state_dict: {path}")
    if not state or any(not isinstance(value, Tensor) for value in state.values()):
        raise TypeError(f"model_state_dict must contain tensors only: {path}")
    return {str(key): value.detach().cpu() for key, value in state.items()}, checkpoint


def state_signature(state: Mapping[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "floating_point": bool(value.is_floating_point()),
        }
        for key, value in state.items()
    ]


def validate_state_compatibility(states: Sequence[Mapping[str, Tensor]]) -> None:
    if not states:
        raise ValueError("At least one state dictionary is required.")
    reference_keys = list(states[0].keys())
    for index, state in enumerate(states[1:], start=1):
        if list(state.keys()) != reference_keys:
            raise ValueError(f"State-dictionary keys/order differ for member {index}.")
        for key in reference_keys:
            left, right = states[0][key], state[key]
            if left.shape != right.shape:
                raise ValueError(f"Tensor shape mismatch for '{key}'.")
            if left.dtype != right.dtype:
                raise ValueError(f"Tensor dtype mismatch for '{key}'.")


def average_state_dicts(
    states: Sequence[Mapping[str, Tensor]],
    weights: Sequence[float],
    *,
    base_index: int = 0,
) -> dict[str, Tensor]:
    validate_state_compatibility(states)
    if len(states) != len(weights):
        raise ValueError("State dictionaries and weights must have equal lengths.")
    values = [float(value) for value in weights]
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("Weights must be finite and non-negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("Weights must have a positive sum.")
    values = [value / total for value in values]

    nonzero = [index for index, value in enumerate(values) if value > 0]
    if len(nonzero) == 1 and math.isclose(values[nonzero[0]], 1.0):
        return {key: tensor.clone() for key, tensor in states[nonzero[0]].items()}

    result: dict[str, Tensor] = {}
    for key, reference in states[0].items():
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for weight, state in zip(values, states):
                if weight:
                    accumulator.add_(state[key].to(torch.float32), alpha=weight)
            result[key] = accumulator.to(reference.dtype)
        else:
            result[key] = states[base_index][key].clone()
    return result


def _transform(value: Tensor, name: str) -> Tensor:
    if name == "identity":
        return value
    if name == "horizontal_flip":
        return torch.flip(value, dims=(-1,))
    if name == "vertical_flip":
        return torch.flip(value, dims=(-2,))
    if name == "rotate_180":
        return torch.flip(value, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA transform: {name}")


def _tta_probability(model: nn.Module, image: Tensor, transforms: Sequence[str]) -> Tensor:
    total: Tensor | None = None
    for transform in transforms:
        logits = model(_transform(image, transform))
        if not isinstance(logits, Tensor):
            raise TypeError("E11 model evaluation must return one logits tensor.")
        probability = _transform(torch.softmax(logits, dim=1), transform)
        total = probability if total is None else total + probability
    if total is None:
        raise ValueError("At least one TTA transform is required.")
    return total / len(transforms)


def evaluate_candidate(
    *,
    model: nn.Module,
    loader: Any,
    device: torch.device,
    transforms: Sequence[str],
    multipliers: Sequence[float],
    class_names: Sequence[str],
    ignore_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.int64)
    batch_count = 0
    image_count = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    multiplier_tensor = torch.tensor(multipliers, dtype=torch.float32).view(1, -1, 1, 1)

    with torch.inference_mode():
        for batch in loader:
            image, target, validity = _extract_batch(batch)
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            probability = _tta_probability(model, image, transforms)

            # Match E10C: probability shards were stored as float16 on CPU,
            # then restored to float32 for calibration and argmax.
            probability = probability.detach().cpu().to(torch.float16).to(torch.float32)
            prediction = (probability * multiplier_tensor).argmax(dim=1)
            target = target.detach().cpu().to(torch.long)
            valid = target != ignore_index
            if validity is not None:
                valid &= validity.detach().cpu().bool()
            encoded = target[valid] * len(class_names) + prediction[valid]
            confusion += torch.bincount(
                encoded,
                minlength=len(class_names) ** 2,
            ).reshape(len(class_names), len(class_names))
            batch_count += 1
            image_count += int(image.shape[0])

    duration = time.perf_counter() - started
    metrics = metrics_from_confusion_matrix(
        confusion,
        class_names=tuple(class_names),
        include_absent_classes_in_macro=False,
    )
    metrics["batch_count"] = batch_count
    runtime = {
        "duration_seconds": duration,
        "images": image_count,
        "images_per_second": image_count / duration if duration > 0 else None,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else None
        ),
    }
    return metrics, runtime


def _create_validation_loader(
    training_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    dataset = training_config["dataset"]
    bundle = create_dataloaders(
        batch_size=int(config.get("batch_size", 2)),
        num_workers=int(config.get("num_workers", 0)),
        seed=int(config.get("seed", 20260725)),
        manifest_path=dataset["manifest_path"],
        dataset_config_path=dataset["dataset_config_path"],
        evaluation_manifest_path=dataset.get(
            "evaluation_manifest_path", dataset["manifest_path"]
        ),
        evaluation_dataset_config_path=dataset.get(
            "evaluation_dataset_config_path", dataset["dataset_config_path"]
        ),
        pin_memory=torch.cuda.is_available(),
    )
    return bundle.val_loader


def _write_candidate_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "candidate_index",
        "candidate_name",
        "weight_e9b",
        "weight_e10b",
        "weight_e10d",
        "mean_iou",
        "delta_from_e10c",
        "strict_target_crossed",
        "duration_seconds",
        "images_per_second",
        "peak_gpu_memory_mb",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_checksums(root: Path, paths: Sequence[Path]) -> Path:
    inventory = {
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "files": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(paths)
            if path.is_file()
        ],
    }
    output = root / "checksums.json"
    output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return output


def run_selection(
    config_path: Path,
    *,
    checkpoint_root: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    _validate_config(config)
    seed_everything(int(config.get("seed", 20260725)), deterministic_algorithms=False)

    members = list(config["model_members"])
    training_configs = [
        read_yaml(Path(str(member["training_config"]))) for member in members
    ]
    contracts = [_model_contract(item) for item in training_configs]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("E9B, E10B and E10D model contracts are not identical.")

    expected_parent = training_configs[0]["experiment"]["name"]
    for index in (1, 2):
        actual_parent = training_configs[index].get("initialization", {}).get(
            "source_experiment"
        )
        if actual_parent != expected_parent:
            raise ValueError(
                f"{members[index]['name']} was not initialized from {expected_parent}."
            )

    root = Path(str(checkpoint_root or config["checkpoint_root"]))
    paths = [
        _resolve_checkpoint(root, str(member["checkpoint_relative_path"]))
        for member in members
    ]
    loaded = [_load_checkpoint_state(path) for path in paths]
    states = [item[0] for item in loaded]
    validate_state_compatibility(states)

    expected_model = build_model(training_configs[0])
    incompatibility = expected_model.load_state_dict(states[0], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError("E9B checkpoint does not strictly match the configured model.")
    del expected_model

    source_audit = []
    for member, path, (_, checkpoint) in zip(members, paths, loaded):
        source_audit.append(
            {
                "name": member["name"],
                "checkpoint": str(path),
                "checkpoint_sha256": file_sha256(path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "training_config": str(member["training_config"]),
                "training_config_sha256": file_sha256(
                    Path(str(member["training_config"]))
                ),
            }
        )

    output_root = Path(str(config["experiment"]["output_directory"]))
    report_root = output_root / "reports"
    metrics_root = output_root / "metrics"
    checkpoint_output_root = output_root / "checkpoints"
    for directory in (report_root, metrics_root, checkpoint_output_root):
        directory.mkdir(parents=True, exist_ok=True)

    audit = {
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "selection_split": "val",
        "test_used": False,
        "model_contract": contracts[0],
        "state_tensor_count": len(states[0]),
        "state_signature_sha256": _json_sha256(state_signature(states[0])),
        "members": source_audit,
        "checks": {
            "common_model_contract": True,
            "common_state_keys_shapes_dtypes": True,
            "e10b_and_e10d_initialized_from_e9b": True,
            "test_access_blocked": True,
        },
    }
    audit_path = report_root / "checkpoint_compatibility_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    loader = _create_validation_loader(training_configs[0], config)
    candidates = list(config["soup_candidates"])
    class_names = list(config["class_names"])
    transforms = list(config["tta_transforms"])
    multipliers = [float(value) for value in config["class_probability_multipliers"]]
    reference = float(config["reference"]["validation_mean_iou"])
    target = float(config["reference"]["strict_target_mean_iou"])
    candidate_reports: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_states: list[dict[str, Tensor]] = []

    for index, candidate in enumerate(candidates):
        weights = [float(value) for value in candidate["weights"]]
        state = average_state_dicts(states, weights, base_index=0)
        model = build_model(training_configs[0]).to(device)
        model.load_state_dict(state, strict=True)
        metrics, runtime = evaluate_candidate(
            model=model,
            loader=loader,
            device=device,
            transforms=transforms,
            multipliers=multipliers,
            class_names=class_names,
            ignore_index=int(training_configs[0]["dataset"].get("ignore_index", 255)),
        )
        mean_iou = float(metrics["mean_iou"])
        report = {
            "candidate_index": index,
            "candidate_name": str(candidate["name"]),
            "weights": dict(zip([item["name"] for item in members], weights)),
            "selection_split": "val",
            "test_used": False,
            "tta_transforms": transforms,
            "class_probability_multipliers": dict(zip(class_names, multipliers)),
            "metrics": metrics,
            "runtime": runtime,
            "delta_from_e10c": mean_iou - reference,
            "strict_target_crossed": mean_iou > target,
        }
        candidate_reports.append(report)
        candidate_rows.append(
            {
                "candidate_index": index,
                "candidate_name": candidate["name"],
                "weight_e9b": weights[0],
                "weight_e10b": weights[1],
                "weight_e10d": weights[2],
                "mean_iou": mean_iou,
                "delta_from_e10c": mean_iou - reference,
                "strict_target_crossed": mean_iou > target,
                **runtime,
            }
        )
        candidate_states.append(state)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_index = max(
        range(len(candidate_reports)),
        key=lambda index: (candidate_reports[index]["metrics"]["mean_iou"], -index),
    )
    baseline_iou = float(candidate_reports[0]["metrics"]["mean_iou"])
    replay_delta = baseline_iou - reference
    replay_tolerance = float(config["reference"]["replay_absolute_tolerance"])
    replay_passed = abs(replay_delta) <= replay_tolerance
    selected = candidate_reports[best_index]

    candidates_path = report_root / "candidate_results.json"
    candidates_path.write_text(
        json.dumps(candidate_reports, indent=2), encoding="utf-8"
    )
    table_path = report_root / "candidate_metrics.csv"
    _write_candidate_table(candidate_rows, table_path)

    selected_metrics_paths = write_metrics_outputs(
        selected["metrics"], metrics_root
    )
    selected_checkpoint_path = checkpoint_output_root / "selected_soup.pt"
    checkpoint_payload = {
        "model_state_dict": candidate_states[best_index],
        "epoch": None,
        "metrics": selected["metrics"],
        "e11": {
            "candidate_name": selected["candidate_name"],
            "weights": selected["weights"],
            "source_checkpoint_sha256": {
                item["name"]: item["checkpoint_sha256"] for item in source_audit
            },
            "selection_split": "val",
            "test_used": False,
            "tta_transforms": transforms,
            "class_probability_multipliers": dict(zip(class_names, multipliers)),
        },
    }
    torch.save(checkpoint_payload, selected_checkpoint_path)

    lock_payload = {
        "status": "PASS" if replay_passed else "FAIL",
        "created_at_utc": _utc_now(),
        "experiment": config["experiment"]["name"],
        "selection_split": "val",
        "test_used": False,
        "validation_tuning_closed": True,
        "selected_candidate_index": best_index,
        "selected_candidate_name": selected["candidate_name"],
        "selected_weights": selected["weights"],
        "tta_transforms": transforms,
        "class_probability_multipliers": dict(zip(class_names, multipliers)),
        "validation_metrics": selected["metrics"],
        "reference_e10c_mean_iou": reference,
        "baseline_s0_mean_iou": baseline_iou,
        "baseline_replay_delta": replay_delta,
        "baseline_replay_tolerance": replay_tolerance,
        "baseline_replay_passed": replay_passed,
        "delta_from_e10c": float(selected["metrics"]["mean_iou"]) - reference,
        "strict_target_mean_iou": target,
        "strict_target_crossed": float(selected["metrics"]["mean_iou"]) > target,
        "additional_validation_search_permitted": False,
        "additional_training_permitted": False,
        "source_checkpoints": source_audit,
        "selected_checkpoint": str(selected_checkpoint_path),
    }
    lock_payload["lock_sha256"] = _json_sha256(
        {
            "selected_candidate_name": lock_payload["selected_candidate_name"],
            "selected_weights": lock_payload["selected_weights"],
            "tta_transforms": transforms,
            "class_probability_multipliers": lock_payload[
                "class_probability_multipliers"
            ],
            "source_checkpoint_sha256": {
                item["name"]: item["checkpoint_sha256"] for item in source_audit
            },
        }
    )
    lock_path = report_root / "locked_soup.json"
    lock_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    summary = {
        "status": lock_payload["status"],
        "experiment": config["experiment"]["name"],
        "selection_split": "val",
        "test_used": False,
        "candidate_count": len(candidates),
        "selected_candidate": selected["candidate_name"],
        "selected_weights": selected["weights"],
        "validation_mean_iou": selected["metrics"]["mean_iou"],
        "delta_from_e10c": lock_payload["delta_from_e10c"],
        "strict_target_crossed": lock_payload["strict_target_crossed"],
        "baseline_replay_passed": replay_passed,
        "output_directory": str(output_root),
    }
    summary_path = report_root / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    artifact_paths = [
        audit_path,
        candidates_path,
        table_path,
        lock_path,
        summary_path,
        selected_checkpoint_path,
        *selected_metrics_paths.values(),
    ]
    _write_checksums(output_root, artifact_paths)
    if not replay_passed:
        raise RuntimeError(
            "S0 did not reproduce E10C within the configured tolerance; "
            "E11 selection is not comparable."
        )
    return summary


def smoke_test() -> dict[str, Any]:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor(9, dtype=torch.int64),
    }
    third = {
        "weight": torch.tensor([9.0, 11.0]),
        "counter": torch.tensor(11, dtype=torch.int64),
    }
    states = [first, second, third]
    baseline = average_state_dicts(states, [1.0, 0.0, 0.0])
    soup = average_state_dicts(states, [0.5, 0.25, 0.25])

    image = torch.arange(1 * 2 * 5 * 7, dtype=torch.float32).reshape(1, 2, 5, 7)
    transform_round_trips = all(
        torch.equal(_transform(_transform(image, name), name), image)
        for name in ALLOWED_TRANSFORMS
    )
    checks = {
        "baseline_is_bitwise_e9b": torch.equal(baseline["weight"], first["weight"]),
        "floating_average_correct": torch.allclose(
            soup["weight"], torch.tensor([4.0, 6.0])
        ),
        "integer_buffer_copied_from_e9b": int(soup["counter"]) == 7,
        "transform_round_trips": transform_round_trips,
        "test_path_not_implemented": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate_count": 5,
        "selection_split": "val",
        "test_used": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed validation-only E11 checkpoint model soup."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/evaluation_d2_e11_checkpoint_soup.yaml"),
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        report = smoke_test()
    elif args.select:
        report = run_selection(
            args.config,
            checkpoint_root=args.checkpoint_root,
            device=torch.device(args.device),
        )
    else:
        parser.error("Choose --smoke-test or --select.")

    print(json.dumps(report, indent=2))
    if report.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

