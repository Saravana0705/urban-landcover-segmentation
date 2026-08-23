"""Freeze the selected U-Net baseline for subsequent model benchmarking.

Current baseline:
    E6 P3
    Dataset V2.2
    CE + Tversky loss
    Tversky alpha=0.2, beta=0.8
    Frozen V1 validation/test set

This script does NOT train or evaluate a model.

It validates the completed E6 P3 experiment and creates canonical
benchmark metadata that subsequent architectures can use for comparison.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# Selected experiment
# ---------------------------------------------------------------------

BASELINE_EXPERIMENT = (
    "unet_e6_p3_v22_ce30_tversky70_fn80"
)

EXPERIMENT_ROOT = Path(
    "outputs/model_experiments"
) / BASELINE_EXPERIMENT

METRICS_DIR = (
    EXPERIMENT_ROOT / "metrics"
)

CHECKPOINT_DIR = (
    EXPERIMENT_ROOT / "checkpoints"
)

OVERALL_METRICS = (
    METRICS_DIR
    / "overall_metrics.csv"
)

PER_CLASS_METRICS = (
    METRICS_DIR
    / "per_class_metrics.csv"
)

CONFUSION_MATRIX = (
    METRICS_DIR
    / "confusion_matrix.csv"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best.pt"
)

COMPARISON_SUMMARY = Path(
    "reports/e6_p2_vs_p3/"
    "comparison_summary.json"
)

EXPERIMENT_REGISTRY = Path(
    "metadata/model_development/"
    "experiment_registry.csv"
)

V22_MANIFEST = Path(
    "metadata/dataset_v22/freeze/"
    "dataset_manifest.csv"
)

V22_CONFIG = Path(
    "metadata/dataset_v22/freeze/"
    "dataset_config.json"
)

V22_NORMALIZATION = Path(
    "metadata/dataset_v22/normalization/"
    "training_normalization.json"
)


# ---------------------------------------------------------------------
# Canonical baseline outputs
# ---------------------------------------------------------------------

OUTPUT_DIR = Path(
    "metadata/model_development/"
    "baselines/unet"
)

BASELINE_JSON = (
    OUTPUT_DIR
    / "unet_baseline.json"
)

BASELINE_METRICS = (
    OUTPUT_DIR
    / "unet_baseline_metrics.csv"
)

BASELINE_PER_CLASS = (
    OUTPUT_DIR
    / "unet_baseline_per_class_metrics.csv"
)

BASELINE_BENCHMARK = Path(
    "metadata/model_development/"
    "model_benchmark.csv"
)


CLASS_ORDER = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_file(
    path: Path,
    label: str,
) -> None:
    """Require a non-empty file."""

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}"
        )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return SHA-256 checksum for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_single_row_csv(
    path: Path,
) -> pd.Series:
    """Read a CSV expected to contain exactly one row."""

    dataframe = pd.read_csv(
        path
    )

    if len(dataframe) != 1:
        raise RuntimeError(
            f"Expected exactly one row in {path}; "
            f"found {len(dataframe)}."
        )

    return dataframe.iloc[0]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Validate and freeze E6 P3 as the U-Net baseline."""

    required_files = (
        (
            OVERALL_METRICS,
            "E6 P3 overall metrics",
        ),
        (
            PER_CLASS_METRICS,
            "E6 P3 per-class metrics",
        ),
        (
            CONFUSION_MATRIX,
            "E6 P3 confusion matrix",
        ),
        (
            BEST_CHECKPOINT,
            "E6 P3 best checkpoint",
        ),
        (
            COMPARISON_SUMMARY,
            "E6 P2 vs E6 P3 comparison summary",
        ),
        (
            EXPERIMENT_REGISTRY,
            "experiment registry",
        ),
        (
            V22_MANIFEST,
            "frozen Dataset V2.2 manifest",
        ),
        (
            V22_CONFIG,
            "frozen Dataset V2.2 config",
        ),
        (
            V22_NORMALIZATION,
            "Dataset V2.2 normalization",
        ),
    )

    for path, label in required_files:
        require_file(
            path,
            label,
        )

    print()
    print("Freeze U-Net baseline")
    print("=====================")
    print(
        f"Selected experiment: "
        f"{BASELINE_EXPERIMENT}"
    )

    # ---------------------------------------------------------
    # Validate P2 vs P3 decision
    # ---------------------------------------------------------

    with COMPARISON_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as file:
        comparison = json.load(
            file
        )

    selected_name = str(
        comparison["e6_p3"]
    )

    if (
        selected_name
        != BASELINE_EXPERIMENT
    ):
        raise RuntimeError(
            "Comparison-selected E6 P3 experiment "
            "does not match requested baseline."
        )

    enrichment_validated = bool(
        comparison[
            "dataset_v22_enrichment_validated"
        ]
    )

    if not enrichment_validated:
        raise RuntimeError(
            "Dataset V2.2 enrichment was not validated. "
            "Baseline freeze aborted."
        )

    required_improvements = {
        "mean_iou_improved",
        "minority_macro_iou_improved",
        "bare_land_iou_improved",
        "roads_iou_improved",
        "buildings_iou_improved",
        "water_iou_improved",
    }

    improvements = comparison[
        "improvements"
    ]

    missing_improvements = (
        required_improvements
        - set(improvements)
    )

    if missing_improvements:
        raise RuntimeError(
            "Comparison summary missing decision fields: "
            f"{sorted(missing_improvements)}"
        )

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    overall = read_single_row_csv(
        OVERALL_METRICS
    )

    required_overall = {
        "pixel_accuracy",
        "mean_iou",
        "mean_dice",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "frequency_weighted_iou",
    }

    missing = (
        required_overall
        - set(overall.index)
    )

    if missing:
        raise RuntimeError(
            "Overall metrics missing fields: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------

    per_class = pd.read_csv(
        PER_CLASS_METRICS
    )

    required_per_class = {
        "class_name",
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
    }

    missing = (
        required_per_class
        - set(per_class.columns)
    )

    if missing:
        raise RuntimeError(
            "Per-class metrics missing fields: "
            f"{sorted(missing)}"
        )

    observed_classes = set(
        per_class[
            "class_name"
        ].astype(str)
    )

    if (
        observed_classes
        != set(CLASS_ORDER)
    ):
        raise RuntimeError(
            "Unexpected per-class metric set: "
            f"{sorted(observed_classes)}"
        )

    per_class = (
        per_class
        .set_index("class_name")
        .loc[CLASS_ORDER]
        .reset_index()
    )

    # ---------------------------------------------------------
    # Registry metadata
    # ---------------------------------------------------------

    registry = pd.read_csv(
        EXPERIMENT_REGISTRY
    )

    selected_registry = registry.loc[
        registry[
            "experiment_name"
        ].astype(str)
        == BASELINE_EXPERIMENT
    ].copy()

    if len(selected_registry) != 1:
        raise RuntimeError(
            "Expected exactly one E6 P3 experiment "
            "registry row."
        )

    registry_row = (
        selected_registry.iloc[0]
    )

    completed_epochs = int(
        registry_row[
            "completed_epochs"
        ]
    )

    best_epoch = int(
        registry_row[
            "best_epoch"
        ]
    )

    if completed_epochs != 10:
        raise RuntimeError(
            "Expected completed_epochs=10; "
            f"found {completed_epochs}."
        )

    # ---------------------------------------------------------
    # Dataset membership validation
    # ---------------------------------------------------------

    manifest = pd.read_csv(
        V22_MANIFEST
    )

    if len(manifest) != 2124:
        raise RuntimeError(
            "Expected frozen Dataset V2.2 "
            f"training size 2124; found {len(manifest)}."
        )

    if (
        manifest[
            "tile_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Frozen V2.2 manifest contains "
            "duplicate tile IDs."
        )

    if (
        "tile_qa_status"
        in manifest.columns
    ):

        failed = (
            manifest[
                "tile_qa_status"
            ]
            .astype(str)
            .str.upper()
            .str.strip()
            != "PASS"
        )

        if failed.any():
            raise RuntimeError(
                "Frozen V2.2 contains non-PASS "
                "training tiles."
            )

    # ---------------------------------------------------------
    # Canonical metrics
    # ---------------------------------------------------------

    mean_iou = float(
        overall["mean_iou"]
    )

    pixel_accuracy = float(
        overall["pixel_accuracy"]
    )

    mean_dice = float(
        overall["mean_dice"]
    )

    mean_precision = float(
        overall["mean_precision"]
    )

    mean_recall = float(
        overall["mean_recall"]
    )

    class_iou = {
        row["class_name"]:
            float(row["iou"])
        for _, row in (
            per_class.iterrows()
        )
    }

    # ---------------------------------------------------------
    # Freeze outputs
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        OVERALL_METRICS,
        BASELINE_METRICS,
    )

    shutil.copy2(
        PER_CLASS_METRICS,
        BASELINE_PER_CLASS,
    )

    baseline_record: dict[
        str,
        Any,
    ] = {
        "baseline_type": (
            "unet_reference_baseline"
        ),
        "status": "frozen",
        "frozen_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "experiment": {
            "experiment_name": (
                BASELINE_EXPERIMENT
            ),
            "model": "unet",
            "dataset_version": "v2.2",
            "training_tiles": 2124,
            "training_cities": 14,
            "validation_tiles": 432,
            "completed_epochs": (
                completed_epochs
            ),
            "best_epoch": best_epoch,
        },

        "model_configuration": {
            "input_channels": 2,
            "num_classes": 5,
            "base_channels": 32,
            "loss": "ce_tversky",
            "ce_weight": 0.30,
            "tversky_weight": 0.70,
            "tversky_alpha": 0.20,
            "tversky_beta": 0.80,
            "class_weights": [
                0.708374,
                0.834706,
                0.244015,
                2.046274,
                1.166632,
            ],
        },

        "overall_metrics": {
            "mean_iou": mean_iou,
            "pixel_accuracy": (
                pixel_accuracy
            ),
            "mean_dice": mean_dice,
            "mean_precision": (
                mean_precision
            ),
            "mean_recall": mean_recall,
            "frequency_weighted_iou": (
                float(
                    overall[
                        "frequency_weighted_iou"
                    ]
                )
            ),
        },

        "per_class_iou": class_iou,

        "selection_basis": {
            "previous_experiment": (
                comparison["e6_p2"]
            ),
            "dataset_v22_enrichment_validated": (
                enrichment_validated
            ),
            "mean_iou_improved": bool(
                improvements[
                    "mean_iou_improved"
                ]
            ),
            "minority_macro_iou_improved": bool(
                improvements[
                    "minority_macro_iou_improved"
                ]
            ),
            "bare_land_iou_improved": bool(
                improvements[
                    "bare_land_iou_improved"
                ]
            ),
            "roads_iou_improved": bool(
                improvements[
                    "roads_iou_improved"
                ]
            ),
        },

        "evaluation_policy": {
            "validation_source": (
                "frozen_dataset_v1"
            ),
            "test_source": (
                "frozen_dataset_v1"
            ),
            "normalization": (
                "inherited_training_city_"
                "normalization"
            ),
        },

        "paths": {
            "experiment_root": str(
                EXPERIMENT_ROOT
            ),
            "best_checkpoint": str(
                BEST_CHECKPOINT
            ),
            "dataset_manifest": str(
                V22_MANIFEST
            ),
            "dataset_config": str(
                V22_CONFIG
            ),
            "normalization": str(
                V22_NORMALIZATION
            ),
            "overall_metrics": str(
                OVERALL_METRICS
            ),
            "per_class_metrics": str(
                PER_CLASS_METRICS
            ),
            "confusion_matrix": str(
                CONFUSION_MATRIX
            ),
            "comparison_summary": str(
                COMPARISON_SUMMARY
            ),
        },

        "checksums": {
            "best_checkpoint_sha256": (
                sha256_file(
                    BEST_CHECKPOINT
                )
            ),
            "dataset_manifest_sha256": (
                sha256_file(
                    V22_MANIFEST
                )
            ),
            "dataset_config_sha256": (
                sha256_file(
                    V22_CONFIG
                )
            ),
            "normalization_sha256": (
                sha256_file(
                    V22_NORMALIZATION
                )
            ),
            "overall_metrics_sha256": (
                sha256_file(
                    OVERALL_METRICS
                )
            ),
            "per_class_metrics_sha256": (
                sha256_file(
                    PER_CLASS_METRICS
                )
            ),
        },
    }

    with BASELINE_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            baseline_record,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Canonical benchmark row
    # ---------------------------------------------------------

    benchmark_row = {
        "model_family": "CNN",
        "model_name": "U-Net",
        "model_identifier": "unet",
        "baseline_status": "frozen_baseline",
        "experiment_name": (
            BASELINE_EXPERIMENT
        ),
        "dataset_version": "v2.2",
        "training_tiles": 2124,
        "validation_tiles": 432,
        "best_epoch": best_epoch,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "mean_dice": mean_dice,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "buildings_iou": (
            class_iou["buildings"]
        ),
        "roads_iou": (
            class_iou["roads"]
        ),
        "vegetation_iou": (
            class_iou["vegetation"]
        ),
        "bare_land_iou": (
            class_iou["bare_land"]
        ),
        "water_iou": (
            class_iou["water"]
        ),
        "checkpoint_path": str(
            BEST_CHECKPOINT
        ),
        "frozen_at_utc": (
            baseline_record[
                "frozen_at_utc"
            ]
        ),
    }

    benchmark_dataframe = (
        pd.DataFrame(
            [benchmark_row]
        )
    )

    if BASELINE_BENCHMARK.exists():

        existing = pd.read_csv(
            BASELINE_BENCHMARK
        )

        # Remove any earlier U-Net baseline row so the
        # benchmark has one canonical baseline entry.
        if (
            "model_identifier"
            in existing.columns
        ):

            existing = existing.loc[
                ~(
                    (
                        existing[
                            "model_identifier"
                        ]
                        .astype(str)
                        == "unet"
                    )
                    & (
                        existing[
                            "baseline_status"
                        ]
                        .astype(str)
                        == "frozen_baseline"
                    )
                )
            ].copy()

        benchmark_dataframe = (
            pd.concat(
                [
                    existing,
                    benchmark_dataframe,
                ],
                ignore_index=True,
                sort=False,
            )
        )

    BASELINE_BENCHMARK.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_dataframe.to_csv(
        BASELINE_BENCHMARK,
        index=False,
    )

    # ---------------------------------------------------------
    # Terminal report
    # ---------------------------------------------------------

    print()
    print("Frozen U-Net baseline")
    print("---------------------")
    print(
        f"Experiment:       "
        f"{BASELINE_EXPERIMENT}"
    )
    print(
        "Dataset:          V2.2"
    )
    print(
        "Training tiles:   2124"
    )
    print(
        f"Best epoch:       {best_epoch}"
    )
    print(
        f"Mean IoU:         "
        f"{mean_iou:.6f}"
    )
    print(
        f"Pixel accuracy:   "
        f"{pixel_accuracy:.6f}"
    )
    print(
        f"Mean Dice:        "
        f"{mean_dice:.6f}"
    )

    print()
    print("Per-class IoU")
    print("-------------")

    for class_name in CLASS_ORDER:

        print(
            f"{class_name:12s}: "
            f"{class_iou[class_name]:.6f}"
        )

    print()
    print("Selection validation")
    print("--------------------")
    print(
        "Dataset V2.2 enrichment validated: "
        f"{enrichment_validated}"
    )
    print(
        "Mean IoU improved: "
        f"{improvements['mean_iou_improved']}"
    )
    print(
        "Minority macro IoU improved: "
        f"{improvements['minority_macro_iou_improved']}"
    )
    print(
        "Bare-land IoU improved: "
        f"{improvements['bare_land_iou_improved']}"
    )

    print()
    print("Artifacts")
    print("---------")
    print(BASELINE_JSON)
    print(BASELINE_METRICS)
    print(BASELINE_PER_CLASS)
    print(BASELINE_BENCHMARK)

    print()
    print(
        "Result: E6 P3 frozen successfully "
        "as the canonical U-Net baseline."
    )


if __name__ == "__main__":
    main()