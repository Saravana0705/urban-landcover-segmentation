"""Create extended-training configs from the completed pilot configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "config"

EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "source": "training_unet_weighted_pilot.yaml",
        "destination": "training_unet_extended.yaml",
        "name": "unet_baseline_v1_weighted_extended",
        "max_epochs": 50,
    },
    {
        "source": "training_attention_unet_pilot.yaml",
        "destination": "training_attention_unet_extended.yaml",
        "name": "attention_unet_v1_weighted_extended",
        "max_epochs": 50,
    },
    {
        "source": "training_unetpp_pilot.yaml",
        "destination": "training_unetpp_extended.yaml",
        "name": "unetpp_v1_weighted_extended",
        "max_epochs": 40,
    },
    {
        "source": "training_swin_transformer_pilot.yaml",
        "destination": "training_swin_transformer_extended.yaml",
        "name": "swin_transformer_tiny_v1_weighted_extended",
        "max_epochs": 30,
    },
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pilot configuration not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return payload


def create_extended_config(specification: dict[str, Any]) -> Path:
    source_path = CONFIG_DIRECTORY / specification["source"]
    destination_path = CONFIG_DIRECTORY / specification["destination"]

    configuration = deepcopy(load_yaml(source_path))

    training = configuration.setdefault("training", {})
    training["max_epochs"] = int(specification["max_epochs"])

    early_stopping = training.setdefault("early_stopping", {})
    early_stopping["patience"] = 8
    early_stopping["minimum_delta"] = 0.001

    checkpoint = training.setdefault("checkpoint", {})
    checkpoint["save_every_epochs"] = 1

    scheduler = configuration.setdefault("scheduler", {})
    scheduler["monitor"] = "val_mean_iou"
    scheduler["mode"] = "max"
    scheduler["factor"] = 0.5
    scheduler["patience"] = 3
    scheduler["minimum_learning_rate"] = 1.0e-6

    experiment_name = str(specification["name"])
    experiment = configuration.setdefault("experiment", {})
    experiment["name"] = experiment_name
    experiment["output_directory"] = (
        f"outputs/model_experiments/{experiment_name}"
    )

    header = (
        "# Extended convergence experiment generated from "
        f"{specification['source']}.\n"
        "# Start from random initialization; do not resume from the pilot run.\n"
        "# Model selection uses validation mean IoU and early stopping.\n\n"
    )

    destination_path.write_text(
        header
        + yaml.safe_dump(
            configuration,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create extended-training YAML configurations."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing extended configurations.",
    )
    arguments = parser.parse_args()

    generated: list[Path] = []

    for specification in EXPERIMENTS:
        destination = CONFIG_DIRECTORY / specification["destination"]

        if destination.exists() and not arguments.overwrite:
            print(f"Skipped existing: {destination.relative_to(PROJECT_ROOT)}")
            continue

        generated.append(create_extended_config(specification))

    for path in generated:
        print(f"Created: {path.relative_to(PROJECT_ROOT)}")

    print("\nExtended-study protocol:")
    print("- U-Net: max 50 epochs")
    print("- Attention U-Net: max 50 epochs")
    print("- U-Net++: max 40 epochs")
    print("- Swin Transformer: max 30 epochs")
    print("- Early stopping: patience 8, minimum delta 0.001")
    print("- Scheduler: ReduceLROnPlateau patience 3")
    print("- Start every experiment from random initialization")


if __name__ == "__main__":
    main()
