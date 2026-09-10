"""Validate the three remaining controlled D2 architecture configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.model_factory import build_model


REFERENCE = Path("config/training_unet_v3_mt_d2_e0_cross_orbit_50ep.yaml")
EXPERIMENTS = (
    Path("config/training_deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep.yaml"),
    Path("config/training_segformer_v3_mt_d2_s0_cross_orbit_50ep.yaml"),
    Path("config/training_mask2former_v3_mt_d2_s0_cross_orbit_50ep.yaml"),
)
EXPECTED_MODELS = ("deeplabv3plus", "segformer", "mask2former")
CONTROLLED_SECTIONS = (
    "seed",
    "dataset",
    "loss",
    "optimizer",
    "scheduler",
    "data",
    "augmentation",
    "training",
    "sampling",
    "benchmark",
)


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def main() -> None:
    reference = read_yaml(REFERENCE)
    results: list[dict[str, Any]] = []
    names: set[str] = set()

    for path, expected_model in zip(EXPERIMENTS, EXPECTED_MODELS):
        config = read_yaml(path)
        controlled_match = {
            section: config.get(section) == reference.get(section)
            for section in CONTROLLED_SECTIONS
        }
        model_config = config.get("model", {})
        experiment = config.get("experiment", {})
        model = build_model(config)
        model.eval()
        image = torch.randn(1, 16, 65, 67)
        output = model(image)
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{expected_model} must return one logits tensor.")
        output.mean().backward()
        gradients_exist = any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        experiment_name = str(experiment.get("name", ""))
        names.add(experiment_name)
        checks = {
            "expected_model": model_config.get("name") == expected_model,
            "input_channels_16": model_config.get("input_channels") == 16,
            "semantic_classes_5": model_config.get("num_classes") == 5,
            "controlled_sections_match_reference": all(controlled_match.values()),
            "comparison_parent_is_d2_e0": (
                experiment.get("comparison_parent")
                == "unet_v3_mt_d2_e0_cross_orbit_50ep"
            ),
            "output_shape": list(output.shape) == [1, 5, 65, 67],
            "output_finite": bool(torch.isfinite(output).all()),
            "gradients_exist": gradients_exist,
        }
        results.append(
            {
                "config": str(path),
                "experiment": experiment_name,
                "model": expected_model,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "controlled_section_match": controlled_match,
                "checks": checks,
                "all_checks_passed": all(checks.values()),
            }
        )

    unique_names = len(names) == len(EXPERIMENTS) and "" not in names
    report = {
        "status": (
            "PASS"
            if unique_names and all(item["all_checks_passed"] for item in results)
            else "FAIL"
        ),
        "reference_config": str(REFERENCE),
        "comparison_scope": "same frozen V3-MT-D2 data and training protocol",
        "test_split_loaded": False,
        "unique_experiment_names": unique_names,
        "experiments": results,
    }
    report["all_checks_passed"] = report["status"] == "PASS"
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

