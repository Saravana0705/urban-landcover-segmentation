"""Static and numerical checks for the validation-locked final D2 test stage."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.evaluate_d2_final_test import (
    CLASS_NAMES,
    EXPECTED_PARAMETERS,
    EXPECTED_TEST_CITIES,
    EXPECTED_TEST_CITY_NAMES,
    EXPECTED_TEST_CITY_TILE_COUNTS,
    EXPECTED_TEST_TILE_COUNT,
    LOCKED_CHECKPOINT_SHA256,
    LOCKED_MULTIPLIERS,
    LOCKED_TRAINING_CONFIG_SHA256,
    LOCKED_TRANSFORMS,
    LOCKED_VALIDATION_MIOU,
    LOCKED_VALIDATION_SHA256,
    canonical_text_sha256,
    file_sha256,
    read_yaml,
    smoke_test,
    validate_lock,
)
from src.models.model_factory import build_model


CONFIG_PATH = Path("config/evaluation_d2_final_locked_test.yaml")


def main() -> None:
    config = read_yaml(CONFIG_PATH)
    lock_checks = validate_lock(config)
    model_lock = config["model"]
    training_path = Path(str(model_lock["training_config"]))
    training = read_yaml(training_path)
    model = build_model(training)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    smoke = smoke_test()
    checks = {
        **lock_checks,
        "training_config_exists": training_path.is_file(),
        "training_config_hash_exact_or_line_ending_normalized": (
            LOCKED_TRAINING_CONFIG_SHA256
            in {file_sha256(training_path), canonical_text_sha256(training_path)}
        ),
        "training_config_is_frozen_d2": training.get("dataset", {}).get("dataset_version") == "v3-mt-d2",
        "training_config_model_is_unetpp": training.get("model", {}).get("name") == "unetpp",
        "training_config_class_order_exact": tuple(training.get("dataset", {}).get("class_names", ())) == CLASS_NAMES,
        "model_parameter_count_exact": parameters == EXPECTED_PARAMETERS,
        "test_city_contract_exact": tuple(config["dataset"]["expected_city_ids"]) == EXPECTED_TEST_CITIES,
        "test_city_names_locked": dict(zip(
            config["dataset"]["expected_city_ids"],
            config["dataset"]["expected_city_names"],
        )) == EXPECTED_TEST_CITY_NAMES,
        "test_city_tile_counts_locked": (
            config["dataset"]["expected_city_tile_counts"]
            == EXPECTED_TEST_CITY_TILE_COUNTS
        ),
        "test_tile_count_locked": int(config["dataset"]["expected_tile_count"]) == EXPECTED_TEST_TILE_COUNT,
        "checkpoint_hash_locked": config["model"]["expected_checkpoint_sha256"] == LOCKED_CHECKPOINT_SHA256,
        "validation_lock_exact": (
            config["selection_lock"]["validation_lock_sha256"] == LOCKED_VALIDATION_SHA256
            and abs(float(config["selection_lock"]["validation_mean_iou"]) - LOCKED_VALIDATION_MIOU) < 1e-12
        ),
        "tta_and_multipliers_exact": (
            tuple(config["inference"]["tta_transforms"]) == LOCKED_TRANSFORMS
            and tuple(float(value) for value in config["inference"]["class_probability_multipliers"]) == LOCKED_MULTIPLIERS
        ),
        "numerical_smoke_test": smoke["all_checks_passed"],
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config": str(CONFIG_PATH),
        "experiment": config["experiment"]["name"],
        "selected_architecture": "U-Net++",
        "selected_checkpoint": "E9B",
        "validation_reference_mean_iou": LOCKED_VALIDATION_MIOU,
        "expected_test_cities": list(EXPECTED_TEST_CITIES),
        "expected_test_tiles": EXPECTED_TEST_TILE_COUNT,
        "trainable_parameters": parameters,
        "test_split_loaded": False,
        "test_rasters_opened": False,
        "training_started": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
