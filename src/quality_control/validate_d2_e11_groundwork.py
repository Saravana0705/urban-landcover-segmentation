"""Validate E11 configuration and architecture contracts without checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_d2_checkpoint_soup import (
    _model_contract,
    _validate_config,
    read_yaml,
    smoke_test,
    state_signature,
)
from src.models.model_factory import build_model


def validate(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    _validate_config(config)
    members = list(config["model_members"])
    contracts = []
    signatures = []
    configs = []

    for member in members:
        training_config = read_yaml(Path(str(member["training_config"])))
        configs.append(training_config)
        contracts.append(_model_contract(training_config))
        model = build_model(training_config)
        signatures.append(state_signature(model.state_dict()))
        del model

    expected_parent = configs[0]["experiment"]["name"]
    lineage = {
        member["name"]: training_config.get("initialization", {}).get(
            "source_experiment"
        )
        for member, training_config in zip(members[1:], configs[1:])
    }
    checks = {
        "configuration_valid": True,
        "validation_only": config["selection_split"] == "val",
        "test_access_blocked": config["test_access"] == "blocked",
        "three_members_exact": [item["name"] for item in members]
        == ["e9b", "e10b", "e10d"],
        "five_candidates_exact": len(config["soup_candidates"]) == 5,
        "common_model_contract": all(item == contracts[0] for item in contracts[1:]),
        "common_state_signature": all(
            item == signatures[0] for item in signatures[1:]
        ),
        "e10b_initialized_from_e9b": lineage.get("e10b") == expected_parent,
        "e10d_initialized_from_e9b": lineage.get("e10d") == expected_parent,
        "fixed_tta": len(config["tta_transforms"]) == 4,
        "fixed_class_multipliers": len(
            config["class_probability_multipliers"]
        )
        == 5,
        "no_training_or_extra_search": not any(
            bool(config["selection_policy"].get(key))
            for key in (
                "training_permitted",
                "additional_search_permitted",
                "multiplier_search_permitted",
            )
        ),
        "numerical_smoke_test": smoke_test()["all_checks_passed"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config": str(config_path),
        "experiment": config["experiment"]["name"],
        "member_names": [item["name"] for item in members],
        "candidate_names": [item["name"] for item in config["soup_candidates"]],
        "model_contract": contracts[0],
        "state_tensor_count": len(signatures[0]),
        "selection_split": config["selection_split"],
        "test_used": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/evaluation_d2_e11_checkpoint_soup.yaml"),
    )
    args = parser.parse_args()
    report = validate(args.config)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

