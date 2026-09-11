"""Validate the locked D2 robustness benchmark without loading any dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.model_factory import build_model
from src.training.train_segmentation import read_yaml


DEFAULT_CONFIG = Path("config/evaluation_d2_final_robustness.yaml")
EXPECTED_CODES = [f"E{index}" for index in range(7)]
EXPECTED_CITIES = ["DE15", "DE16", "DE17"]
EXPECTED_NOISE = [("clean", None), ("light_speckle_l25", 25.0), ("moderate_speckle_l9", 9.0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    dataset = config.get("dataset", {})
    runtime = config.get("runtime", {})
    safety = config.get("safety", {})
    models = config.get("models", [])
    noise = config.get("noise_conditions", [])

    observed_noise = [(item.get("name"), item.get("speckle_looks")) for item in noise]
    checks = {
        "seven_models_exact": [item.get("code") for item in models] == EXPECTED_CODES,
        "validation_split_only": dataset.get("split") == "val" and safety.get("allowed_split") == "val",
        "validation_cities_exact": dataset.get("expected_city_ids") == EXPECTED_CITIES,
        "validation_tile_count_locked": dataset.get("expected_tile_count") == 351,
        "dataset_version_locked": dataset.get("dataset_version") == "v3-mt-d2",
        "model_contract_locked": dataset.get("input_channels") == 16 and dataset.get("num_classes") == 5,
        "noise_conditions_locked": observed_noise == EXPECTED_NOISE,
        "deterministic_single_worker": runtime.get("seed") == 20260725 and runtime.get("num_workers") == 0,
        "training_disabled": safety.get("training_enabled") is False,
        "test_evaluation_disabled": safety.get("test_evaluation_enabled") is False,
        "checkpoint_updates_disabled": safety.get("checkpoint_updates_enabled") is False,
    }

    model_audit = []
    for item in models:
        path = Path(str(item["config_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Model config not found: {path}")
        model_config = read_yaml(path)
        if model_config.get("experiment", {}).get("name") != item["experiment_name"]:
            raise ValueError(f"Experiment identity mismatch for {item['code']}")
        if model_config.get("dataset", {}).get("dataset_version") != "v3-mt-d2":
            raise ValueError(f"Dataset version mismatch for {item['code']}")
        model = build_model(model_config)
        parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        expected = int(item["expected_trainable_parameters"])
        model_audit.append({
            "code": item["code"],
            "experiment_name": item["experiment_name"],
            "model_name": model_config.get("model", {}).get("name"),
            "trainable_parameters": parameters,
            "expected_trainable_parameters": expected,
            "parameter_match": parameters == expected,
        })
        del model

    checks["all_model_configs_and_parameters_match"] = all(item["parameter_match"] for item in model_audit)
    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "config": str(args.config.resolve()),
        "experiment": config.get("experiment", {}).get("name"),
        "selection_split": "validation",
        "test_split_loaded": False,
        "training_started": False,
        "checks": checks,
        "models": model_audit,
        "all_checks_passed": passed,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

