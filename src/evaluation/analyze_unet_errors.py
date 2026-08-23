"""Targeted validation error analysis for the frozen E6 P3 U-Net baseline.

The script evaluates every tile in the chosen split and records:
- supervised pixels
- tile pixel accuracy
- per-class target fractions
- per-class IoU
- total error fraction
- bare-land -> vegetation confusion
- road -> vegetation confusion
- building -> vegetation confusion

It then automatically selects representative examples for:
- strongest overall predictions
- weakest overall predictions
- bare-land-rich tiles
- worst bare-land -> vegetation failures
- road-rich tiles
- worst road -> vegetation failures
- representative mixed-urban tiles

Selected examples are rendered using the exact plotting function already
used by evaluate_unet.py.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.evaluation.evaluate_unet import (
    CLASS_NAMES,
    IGNORE_INDEX,
    build_project_dataloaders,
    build_unet_from_config,
    nested_get,
    plot_prediction,
    read_yaml,
    resolve_device,
    sample_label,
    set_seed,
    unpack_batch,
)
from src.training.training_state import load_checkpoint


# ---------------------------------------------------------------------
# Class configuration
# ---------------------------------------------------------------------

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

BUILDINGS_ID = CLASS_TO_ID["buildings"]
ROADS_ID = CLASS_TO_ID["roads"]
VEGETATION_ID = CLASS_TO_ID["vegetation"]
BARE_LAND_ID = CLASS_TO_ID["bare_land"]
WATER_ID = CLASS_TO_ID["water"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def divide(
    numerator: float,
    denominator: float,
) -> float:
    """Safe division."""

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def safe_name(value: str) -> str:
    """Create filesystem-safe sample label."""

    return "".join(
        character
        if character.isalnum()
        or character in "-_"
        else "_"
        for character in value
    )


def extract_metadata_value(
    metadata: dict[str, Any],
    key: str,
    item_index: int,
    default: str = "",
) -> str:
    """Extract one metadata value from a collated batch."""

    value = metadata.get(key)

    if value is None:
        return default

    if isinstance(
        value,
        (list, tuple),
    ):
        if item_index < len(value):
            return str(
                value[item_index]
            )

        return default

    if isinstance(value, Tensor):

        if value.ndim == 0:
            return str(
                value.item()
            )

        if item_index < len(value):
            item = value[item_index]

            if item.ndim == 0:
                return str(
                    item.item()
                )

            return str(
                item.tolist()
            )

    return str(value)


def tile_confusion(
    target: Tensor,
    prediction: Tensor,
    validity: Tensor | None,
) -> Tensor:
    """Build one 5x5 tile-level confusion matrix."""

    valid = (
        target
        != IGNORE_INDEX
    )

    if validity is not None:
        valid &= validity.bool()

    target_valid = (
        target[valid]
        .to(torch.int64)
    )

    prediction_valid = (
        prediction[valid]
        .to(torch.int64)
    )

    confusion = torch.zeros(
        (
            len(CLASS_NAMES),
            len(CLASS_NAMES),
        ),
        dtype=torch.int64,
    )

    if target_valid.numel() == 0:
        return confusion

    encoded = (
        target_valid
        * len(CLASS_NAMES)
        + prediction_valid
    )

    confusion += torch.bincount(
        encoded,
        minlength=(
            len(CLASS_NAMES) ** 2
        ),
    ).reshape(
        len(CLASS_NAMES),
        len(CLASS_NAMES),
    ).cpu()

    return confusion


def calculate_tile_metrics(
    confusion: Tensor,
) -> dict[str, float | int]:
    """Convert one tile confusion matrix into analysis metrics."""

    matrix = (
        confusion
        .to(torch.float64)
    )

    total = float(
        matrix.sum()
    )

    correct = float(
        torch.diag(
            matrix
        ).sum()
    )

    result: dict[
        str,
        float | int
    ] = {
        "supervised_pixels": int(
            total
        ),
        "correct_pixels": int(
            correct
        ),
        "error_pixels": int(
            total - correct
        ),
        "pixel_accuracy": divide(
            correct,
            total,
        ),
        "error_fraction": divide(
            total - correct,
            total,
        ),
    }

    target_support = (
        matrix.sum(dim=1)
    )

    prediction_support = (
        matrix.sum(dim=0)
    )

    tp = torch.diag(
        matrix
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        target_pixels = float(
            target_support[class_id]
        )

        predicted_pixels = float(
            prediction_support[
                class_id
            ]
        )

        true_positive = float(
            tp[class_id]
        )

        union = (
            target_pixels
            + predicted_pixels
            - true_positive
        )

        result[
            f"{class_name}_target_pixels"
        ] = int(target_pixels)

        result[
            f"{class_name}_target_fraction"
        ] = divide(
            target_pixels,
            total,
        )

        result[
            f"{class_name}_predicted_pixels"
        ] = int(
            predicted_pixels
        )

        result[
            f"{class_name}_iou"
        ] = divide(
            true_positive,
            union,
        )

        result[
            f"{class_name}_recall"
        ] = divide(
            true_positive,
            target_pixels,
        )

    # ---------------------------------------------------------
    # Key failure modes
    # ---------------------------------------------------------

    bare_to_vegetation = float(
        matrix[
            BARE_LAND_ID,
            VEGETATION_ID,
        ]
    )

    roads_to_vegetation = float(
        matrix[
            ROADS_ID,
            VEGETATION_ID,
        ]
    )

    buildings_to_vegetation = float(
        matrix[
            BUILDINGS_ID,
            VEGETATION_ID,
        ]
    )

    bare_target = float(
        target_support[
            BARE_LAND_ID
        ]
    )

    road_target = float(
        target_support[
            ROADS_ID
        ]
    )

    building_target = float(
        target_support[
            BUILDINGS_ID
        ]
    )

    result[
        "bare_land_to_vegetation_pixels"
    ] = int(
        bare_to_vegetation
    )

    result[
        "bare_land_to_vegetation_rate"
    ] = divide(
        bare_to_vegetation,
        bare_target,
    )

    result[
        "roads_to_vegetation_pixels"
    ] = int(
        roads_to_vegetation
    )

    result[
        "roads_to_vegetation_rate"
    ] = divide(
        roads_to_vegetation,
        road_target,
    )

    result[
        "buildings_to_vegetation_pixels"
    ] = int(
        buildings_to_vegetation
    )

    result[
        "buildings_to_vegetation_rate"
    ] = divide(
        buildings_to_vegetation,
        building_target,
    )

    # Mixed urban = meaningful representation of multiple
    # non-vegetation classes.
    minority_fractions = [
        float(
            result[
                "buildings_target_fraction"
            ]
        ),
        float(
            result[
                "roads_target_fraction"
            ]
        ),
        float(
            result[
                "bare_land_target_fraction"
            ]
        ),
        float(
            result[
                "water_target_fraction"
            ]
        ),
    ]

    result[
        "minority_fraction_total"
    ] = sum(
        minority_fractions
    )

    result[
        "minority_classes_present"
    ] = sum(
        fraction >= 0.01
        for fraction
        in minority_fractions
    )

    return result


def rank_unique(
    dataframe: pd.DataFrame,
    column: str,
    count: int,
    ascending: bool,
    minimum_column: str | None = None,
    minimum_value: float = 0.0,
) -> pd.DataFrame:
    """Return top-N ranked rows with optional eligibility threshold."""

    candidates = (
        dataframe.copy()
    )

    if minimum_column is not None:

        candidates = (
            candidates.loc[
                candidates[
                    minimum_column
                ]
                >= minimum_value
            ]
            .copy()
        )

    candidates = (
        candidates.sort_values(
            column,
            ascending=ascending,
        )
        .head(count)
        .copy()
    )

    return candidates


# ---------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------

def build_selection(
    dataframe: pd.DataFrame,
    per_category: int,
) -> pd.DataFrame:
    """Select interpretable qualitative examples."""

    categories: list[
        tuple[str, pd.DataFrame]
    ] = []

    categories.append(
        (
            "strong_prediction",
            rank_unique(
                dataframe,
                column="pixel_accuracy",
                count=per_category,
                ascending=False,
            ),
        )
    )

    categories.append(
        (
            "weak_prediction",
            rank_unique(
                dataframe,
                column="pixel_accuracy",
                count=per_category,
                ascending=True,
            ),
        )
    )

    categories.append(
        (
            "bare_land_rich",
            rank_unique(
                dataframe,
                column=(
                    "bare_land_target_fraction"
                ),
                count=per_category,
                ascending=False,
                minimum_column=(
                    "bare_land_target_pixels"
                ),
                minimum_value=1,
            ),
        )
    )

    categories.append(
        (
            "bare_land_to_vegetation_failure",
            rank_unique(
                dataframe,
                column=(
                    "bare_land_to_vegetation_pixels"
                ),
                count=per_category,
                ascending=False,
                minimum_column=(
                    "bare_land_target_pixels"
                ),
                minimum_value=100,
            ),
        )
    )

    categories.append(
        (
            "road_rich",
            rank_unique(
                dataframe,
                column=(
                    "roads_target_fraction"
                ),
                count=per_category,
                ascending=False,
                minimum_column=(
                    "roads_target_pixels"
                ),
                minimum_value=1,
            ),
        )
    )

    categories.append(
        (
            "road_to_vegetation_failure",
            rank_unique(
                dataframe,
                column=(
                    "roads_to_vegetation_pixels"
                ),
                count=per_category,
                ascending=False,
                minimum_column=(
                    "roads_target_pixels"
                ),
                minimum_value=100,
            ),
        )
    )

    mixed = (
        dataframe.loc[
            dataframe[
                "minority_classes_present"
            ]
            >= 3
        ]
        .sort_values(
            [
                "minority_classes_present",
                "minority_fraction_total",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            per_category
        )
        .copy()
    )

    categories.append(
        (
            "mixed_urban",
            mixed,
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for category, selected in categories:

        for rank, (
            _,
            row,
        ) in enumerate(
            selected.iterrows(),
            start=1,
        ):

            record = row.to_dict()

            record[
                "selection_category"
            ] = category

            record[
                "category_rank"
            ] = rank

            rows.append(
                record
            )

    selection = pd.DataFrame(
        rows
    )

    if selection.empty:
        raise RuntimeError(
            "No error-analysis examples were selected."
        )

    return selection


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

@torch.no_grad()
def analyze_split(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    dict[str, tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
    ]],
]:
    """Evaluate every tile and retain tensors for later plotting."""

    model.eval()

    records: list[
        dict[str, Any]
    ] = []

    tensor_cache: dict[
        str,
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
        ],
    ] = {}

    for batch_index, batch in enumerate(
        tqdm(
            loader,
            desc="Tile error analysis",
        )
    ):

        (
            image,
            target,
            validity,
            metadata,
        ) = unpack_batch(
            batch
        )

        image_device = image.to(
            device,
            dtype=torch.float32,
        )

        target_device = target.to(
            device,
            dtype=torch.long,
        )

        validity_device = (
            validity.to(
                device,
                dtype=torch.bool,
            )
            if validity is not None
            else None
        )

        prediction_device = (
            model(
                image_device
            )
            .argmax(dim=1)
        )

        for item_index in range(
            image.shape[0]
        ):

            fallback = (
                f"batch_{batch_index:04d}"
                f"_item_{item_index:02d}"
            )

            tile_id = sample_label(
                metadata,
                item_index,
                fallback,
            )

            target_item = (
                target_device[
                    item_index
                ]
            )

            prediction_item = (
                prediction_device[
                    item_index
                ]
            )

            validity_item = (
                validity_device[
                    item_index
                ]
                if validity_device
                is not None
                else None
            )

            confusion = (
                tile_confusion(
                    target_item,
                    prediction_item,
                    validity_item,
                )
            )

            metrics = (
                calculate_tile_metrics(
                    confusion
                )
            )

            metrics.update(
                {
                    "tile_id": tile_id,
                    "city_id": (
                        extract_metadata_value(
                            metadata,
                            "city_id",
                            item_index,
                        )
                    ),
                    "city_name": (
                        extract_metadata_value(
                            metadata,
                            "city_name",
                            item_index,
                        )
                    ),
                    "batch_index": (
                        batch_index
                    ),
                    "item_index": (
                        item_index
                    ),
                }
            )

            records.append(
                metrics
            )

            tensor_cache[
                tile_id
            ] = (
                image[
                    item_index
                ].detach().cpu(),
                target[
                    item_index
                ].detach().cpu(),
                prediction_device[
                    item_index
                ].detach().cpu(),
                (
                    validity_device[
                        item_index
                    ]
                    .detach()
                    .cpu()
                    if validity_device
                    is not None
                    else None
                ),
            )

    dataframe = pd.DataFrame(
        records
    )

    if dataframe.empty:
        raise RuntimeError(
            "No validation tiles were analyzed."
        )

    return (
        dataframe,
        tensor_cache,
    )


def render_selection(
    selection: pd.DataFrame,
    tensor_cache: dict[
        str,
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
        ],
    ],
    output_dir: Path,
) -> list[
    dict[str, Any]
]:
    """Render selected error-analysis figures."""

    rendered: list[
        dict[str, Any]
    ] = []

    for _, row in (
        selection.iterrows()
    ):

        tile_id = str(
            row["tile_id"]
        )

        category = str(
            row[
                "selection_category"
            ]
        )

        rank = int(
            row["category_rank"]
        )

        tensors = tensor_cache.get(
            tile_id
        )

        if tensors is None:
            raise KeyError(
                f"No cached tensors for "
                f"{tile_id}."
            )

        (
            image,
            target,
            prediction,
            validity,
        ) = tensors

        category_dir = (
            output_dir
            / "selected_examples"
            / category
        )

        filename = (
            f"{rank:02d}_"
            f"{safe_name(tile_id)}.png"
        )

        path = (
            category_dir
            / filename
        )

        title = (
            f"{tile_id} | {category}\n"
            f"accuracy="
            f"{float(row['pixel_accuracy']):.3f}, "
            f"bare IoU="
            f"{float(row['bare_land_iou']):.3f}, "
            f"road IoU="
            f"{float(row['roads_iou']):.3f}"
        )

        plot_prediction(
            path,
            image,
            target,
            prediction,
            validity,
            title,
        )

        rendered.append(
            {
                "tile_id": tile_id,
                "category": category,
                "rank": rank,
                "path": str(path),
            }
        )

    return rendered


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Targeted tile-level error analysis "
            "for a trained U-Net."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=(
            "val",
            "test",
        ),
        default="val",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--per-category",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/"
            "unet_e6_p3_error_analysis"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    if args.per_category <= 0:
        raise ValueError(
            "--per-category must be positive."
        )

    config = read_yaml(
        args.config
    )

    seed = int(
        nested_get(
            config,
            "seed",
            "training.seed",
            default=20260725,
        )
    )

    set_seed(
        seed
    )

    device = resolve_device(
        args.device
    )

    loaders = (
        build_project_dataloaders(
            config,
            integration_check=False,
        )
    )

    if args.split not in loaders:
        raise KeyError(
            "Missing DataLoader split: "
            f"{args.split}"
        )

    model = (
        build_unet_from_config(
            config
        )
        .to(device)
    )

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        map_location=device,
        restore_rng=False,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("U-Net targeted error analysis")
    print("=============================")
    print(
        f"Split: {args.split}"
    )
    print(
        f"Checkpoint: "
        f"{args.checkpoint}"
    )
    print(
        f"Checkpoint epoch: "
        f"{int(checkpoint.get('epoch', -1))}"
    )
    print(
        f"Device: {device}"
    )

    (
        tile_metrics,
        tensor_cache,
    ) = analyze_split(
        model,
        loaders[
            args.split
        ],
        device,
    )

    metrics_path = (
        args.output_dir
        / "validation_tile_metrics.csv"
    )

    tile_metrics.to_csv(
        metrics_path,
        index=False,
    )

    selection = (
        build_selection(
            tile_metrics,
            args.per_category,
        )
    )

    selection_path = (
        args.output_dir
        / "category_selection.csv"
    )

    selection.to_csv(
        selection_path,
        index=False,
    )

    rendered = (
        render_selection(
            selection,
            tensor_cache,
            args.output_dir,
        )
    )

    # ---------------------------------------------------------
    # Aggregate diagnostics
    # ---------------------------------------------------------

    summary = {
        "config": str(
            args.config
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_epoch": int(
            checkpoint.get(
                "epoch",
                -1,
            )
        ),
        "split": args.split,
        "device": str(
            device
        ),
        "tiles_analyzed": int(
            len(tile_metrics)
        ),
        "selected_examples": int(
            len(selection)
        ),
        "selection_categories": (
            selection[
                "selection_category"
            ]
            .value_counts()
            .to_dict()
        ),
        "tile_level_summary": {
            "mean_pixel_accuracy": float(
                tile_metrics[
                    "pixel_accuracy"
                ].mean()
            ),
            "median_pixel_accuracy": float(
                tile_metrics[
                    "pixel_accuracy"
                ].median()
            ),
            "mean_error_fraction": float(
                tile_metrics[
                    "error_fraction"
                ].mean()
            ),
            "tiles_with_bare_land": int(
                (
                    tile_metrics[
                        "bare_land_target_pixels"
                    ]
                    > 0
                ).sum()
            ),
            "tiles_with_roads": int(
                (
                    tile_metrics[
                        "roads_target_pixels"
                    ]
                    > 0
                ).sum()
            ),
            "tiles_with_water": int(
                (
                    tile_metrics[
                        "water_target_pixels"
                    ]
                    > 0
                ).sum()
            ),
            "bare_land_to_vegetation_pixels": int(
                tile_metrics[
                    "bare_land_to_vegetation_pixels"
                ].sum()
            ),
            "roads_to_vegetation_pixels": int(
                tile_metrics[
                    "roads_to_vegetation_pixels"
                ].sum()
            ),
            "buildings_to_vegetation_pixels": int(
                tile_metrics[
                    "buildings_to_vegetation_pixels"
                ].sum()
            ),
        },
        "artifacts": {
            "validation_tile_metrics": str(
                metrics_path
            ),
            "category_selection": str(
                selection_path
            ),
            "selected_examples": (
                rendered
            ),
        },
        "checks": {
            "checkpoint_loaded": True,
            "tiles_analyzed_positive": bool(
                len(tile_metrics) > 0
            ),
            "all_tile_metrics_finite": bool(
                np.isfinite(
                    tile_metrics[
                        [
                            "pixel_accuracy",
                            "error_fraction",
                            "buildings_iou",
                            "roads_iou",
                            "vegetation_iou",
                            "bare_land_iou",
                            "water_iou",
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                ).all()
            ),
            "selected_examples_positive": bool(
                len(selection) > 0
            ),
        },
    }

    summary[
        "checks"
    ][
        "all_checks_passed"
    ] = all(
        summary[
            "checks"
        ].values()
    )

    summary_path = (
        args.output_dir
        / "error_analysis_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Analysis result")
    print("---------------")
    print(
        f"Tiles analyzed: "
        f"{len(tile_metrics)}"
    )
    print(
        f"Examples selected: "
        f"{len(selection)}"
    )
    print(
        "Mean tile accuracy: "
        f"{tile_metrics['pixel_accuracy'].mean():.4f}"
    )
    print(
        "Median tile accuracy: "
        f"{tile_metrics['pixel_accuracy'].median():.4f}"
    )

    print()
    print("Key confusion totals")
    print("--------------------")
    print(
        "Bare land -> vegetation: "
        f"{int(tile_metrics['bare_land_to_vegetation_pixels'].sum()):,}"
    )
    print(
        "Roads -> vegetation:     "
        f"{int(tile_metrics['roads_to_vegetation_pixels'].sum()):,}"
    )
    print(
        "Buildings -> vegetation: "
        f"{int(tile_metrics['buildings_to_vegetation_pixels'].sum()):,}"
    )

    print()
    print("Selected categories")
    print("-------------------")

    print(
        selection[
            "selection_category"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Reports written")
    print("---------------")
    print(
        metrics_path
    )
    print(
        selection_path
    )
    print(
        summary_path
    )
    print(
        args.output_dir
        / "selected_examples"
    )

    print()
    print(
        "Result: targeted U-Net error "
        "analysis completed successfully."
    )


if __name__ == "__main__":
    main()