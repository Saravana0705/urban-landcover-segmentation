"""Static and numerical validation for the bounded D2 E10 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.model_factory import build_model
from src.training.losses import LossConfig, build_loss


TRAINING_CONFIGS = (
    Path("config/training_unetpp_v3_mt_d2_e10a_deep_supervision_15ep.yaml"),
    Path("config/training_unetpp_v3_mt_d2_e10b_road_cldice_12ep.yaml"),
    Path("config/training_unetpp_v3_mt_d2_e10d_boundary_12ep.yaml"),
)
ENSEMBLE_CONFIG = Path("config/evaluation_d2_e10c_final_calibrated_ensemble.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _loss_config(payload: dict[str, Any]) -> LossConfig:
    loss = payload["loss"]
    return LossConfig(
        name=str(loss["name"]),
        num_classes=5,
        ignore_index=255,
        ce_weight=float(loss.get("ce_weight", 0.5)),
        lovasz_weight=float(loss.get("lovasz_weight", 0.7)),
        cldice_weight=float(loss.get("cldice_weight", 0.0)),
        cldice_class_id=int(loss.get("cldice_class_id", 1)),
        cldice_iterations=int(loss.get("cldice_iterations", 10)),
        boundary_weight=float(loss.get("boundary_weight", 0.0)),
        boundary_class_ids=tuple(int(value) for value in loss.get("boundary_class_ids", [0, 1])),
        boundary_tolerance=int(loss.get("boundary_tolerance", 2)),
        deep_supervision_weights=tuple(
            float(value) for value in loss.get("deep_supervision_weights", [])
        ),
        include_absent_classes=bool(loss.get("include_absent_classes", False)),
    )


def validate() -> dict[str, Any]:
    torch.manual_seed(20260725)
    reports: list[dict[str, Any]] = []
    image = torch.randn(1, 16, 65, 67)
    target = torch.randint(0, 5, (1, 65, 67), dtype=torch.int64)
    target[:, :2, :] = 255

    for path in TRAINING_CONFIGS:
        payload = _read_yaml(path)
        model = build_model(payload)
        model.train()
        output = model(image)
        criterion = build_loss(
            _loss_config(payload),
            class_weights=payload["loss"]["class_weights"],
        )
        loss = criterion(output, target)
        loss.backward()
        gradients_exist = any(
            parameter.grad is not None and torch.any(parameter.grad != 0).item()
            for parameter in model.parameters()
        )
        model.eval()
        with torch.inference_mode():
            evaluation_output = model(image)
        reports.append(
            {
                "config": str(path),
                "experiment": payload["experiment"]["name"],
                "training_head_count": len(output) if isinstance(output, tuple) else 1,
                "evaluation_shape": list(evaluation_output.shape),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "gradients_exist": gradients_exist,
            }
        )

    ensemble = _read_yaml(ENSEMBLE_CONFIG)
    member_count = len(ensemble["members"])
    candidates = ensemble["search"]["weight_candidates"]
    ensemble_valid = (
        member_count == 8
        and all(len(candidate) == member_count for candidate in candidates)
        and all(abs(sum(candidate) - 1.0) < 1e-6 for candidate in candidates)
        and ensemble.get("selection_split") == "val"
    )
    all_checks_passed = all(
        item["evaluation_shape"] == [1, 5, 65, 67]
        and item["loss_finite"]
        and item["gradients_exist"]
        for item in reports
    ) and ensemble_valid
    return {
        "status": "PASS" if all_checks_passed else "FAIL",
        "training_experiments": reports,
        "ensemble": {
            "config": str(ENSEMBLE_CONFIG),
            "member_count": member_count,
            "weight_candidate_count": len(candidates),
            "validation_only": ensemble.get("selection_split") == "val",
            "checks_passed": ensemble_valid,
        },
        "all_checks_passed": all_checks_passed,
    }


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)
