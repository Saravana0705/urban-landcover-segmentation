"""Materialize the selected Dataset V2.2 offset enrichment tiles.

This script reads the frozen class-aware V2.2 candidate selection and writes
the corresponding SAR image, semantic-mask, and validity-mask GeoTIFF tiles.

Important:
- Existing Dataset V1/V2.1 tiles are never modified.
- Only training-city offset enrichment tiles are generated.
- Tile windows are taken directly from the selected-candidate CSV.
- Output raster conventions match the existing project tiling pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from src.tiling.tile_city_dataset import (
    TRAINABLE_CLASSES,
    class_counts,
    load_yaml,
    sha256_file,
    validate_alignment,
    validate_config,
)


DEFAULT_SELECTION = Path(
    "reports/dataset_v22_class_aware_selection/"
    "dataset_v22_selected_candidates.csv"
)

DEFAULT_OUTPUT_ROOT = Path(
    "data/processed/tiles_v22_enrichment"
)

DEFAULT_MANIFEST_DIR = Path(
    "metadata/dataset_v22/enrichment"
)

DEFAULT_TILING_CONFIG = Path(
    "config/tiling.yaml"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Materialize Dataset V2.2 selected offset-enrichment tiles."
        )
    )

    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="CSV containing the selected V2.2 offset candidates.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TILING_CONFIG,
        help="Existing project tiling configuration.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already existing V2.2 enrichment tile files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and print the materialization plan "
            "without creating raster tiles."
        ),
    )

    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    """Require a non-empty file."""

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}"
        )


def normalise_path(value: Any) -> Path:
    """Convert CSV path text to a Path."""

    if pd.isna(value):
        raise ValueError("Encountered an empty source path.")

    return Path(str(value))


def resolve_source_paths(
    row: pd.Series,
) -> tuple[Path, Path, Path]:
    """Resolve the full-city SAR, semantic, and validity rasters."""

    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    folder_name = f"{city_id}_{city_name}"

    image_path = (
        Path("data/processed/sar")
        / folder_name
        / f"{folder_name}_S1_VV_VH.tif"
    )

    if "semantic_source" in row.index:
        semantic_path = normalise_path(
            row["semantic_source"]
        )
    else:
        semantic_path = (
            Path("data/processed/masks")
            / folder_name
            / f"{city_id}_label_mask.tif"
        )

    if "validity_source" in row.index:
        validity_path = normalise_path(
            row["validity_source"]
        )
    else:
        validity_path = (
            Path("data/processed/masks")
            / folder_name
            / f"{city_id}_validity_mask.tif"
        )

    return (
        image_path,
        semantic_path,
        validity_path,
    )


def validate_selection(
    selection: pd.DataFrame,
) -> None:
    """Validate the selected-candidate CSV."""

    required = {
        "candidate_id",
        "city_id",
        "city_name",
        "split",
        "row_offset",
        "col_offset",
        "tile_size",
    }

    missing = required.difference(
        selection.columns
    )

    if missing:
        raise ValueError(
            "V2.2 selection CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    if selection.empty:
        raise ValueError(
            "V2.2 selection CSV contains no rows."
        )

    if selection["candidate_id"].duplicated().any():
        duplicates = (
            selection.loc[
                selection["candidate_id"].duplicated(
                    keep=False
                ),
                "candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate V2.2 candidate IDs detected: "
            f"{duplicates[:10]}"
        )

    splits = set(
        selection["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if splits != {"train"}:
        raise ValueError(
            "Dataset V2.2 enrichment must contain training "
            f"candidates only. Found splits: {sorted(splits)}"
        )


def compare_selection_statistics(
    source_row: pd.Series,
    materialized_row: dict[str, Any],
    tolerance: float = 1e-9,
) -> list[str]:
    """Compare recomputed counts against simulation metadata.

    Returns a list of differences. Empty means exact agreement.
    """

    differences: list[str] = []

    candidate_columns = {
        "valid_pixel_count": "valid_pixel_count",
        "buildings_pixel_count": "buildings_pixel_count",
        "roads_pixel_count": "roads_pixel_count",
        "vegetation_pixel_count": "vegetation_pixel_count",
        "bare_land_pixel_count": "bare_land_pixel_count",
        "water_pixel_count": "water_pixel_count",
    }

    for source_column, generated_column in candidate_columns.items():

        if source_column not in source_row.index:
            continue

        expected = int(source_row[source_column])
        actual = int(materialized_row[generated_column])

        if expected != actual:
            differences.append(
                f"{source_column}: expected={expected}, "
                f"generated={actual}"
            )

    if "valid_fraction" in source_row.index:

        expected_fraction = float(
            source_row["valid_fraction"]
        )

        actual_fraction = float(
            materialized_row["valid_fraction"]
        )

        if abs(
            expected_fraction - actual_fraction
        ) > tolerance:
            differences.append(
                "valid_fraction: "
                f"expected={expected_fraction}, "
                f"generated={actual_fraction}"
            )

    return differences


def main() -> None:
    """Materialize Dataset V2.2 enrichment tiles."""

    args = parse_arguments()

    require_file(
        args.selection,
        "V2.2 selected-candidate CSV",
    )

    require_file(
        args.config,
        "tiling configuration",
    )

    config = load_yaml(
        args.config
    )

    validate_config(
        config
    )

    selection = pd.read_csv(
        args.selection
    )

    validate_selection(
        selection
    )

    selection = selection.sort_values(
        [
            "city_id",
            "candidate_id",
        ]
    ).reset_index(drop=True)

    tile_size = int(
        config["tile_size"]
    )

    image_fill = float(
        config["image_fill_value"]
    )

    semantic_fill = int(
        config["semantic_fill_value"]
    )

    validity_fill = int(
        config["validity_fill_value"]
    )

    output_config = config["output"]

    if not (
        selection["tile_size"].astype(int)
        == tile_size
    ).all():
        raise ValueError(
            "Selected V2.2 tile size does not match "
            f"config/tiling.yaml tile_size={tile_size}."
        )

    args.manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.manifest_dir
        / "dataset_v22_enrichment_manifest.csv"
    )

    summary_path = (
        args.manifest_dir
        / "dataset_v22_enrichment_summary.json"
    )

    print()
    print("Dataset V2.2 enrichment materialization")
    print("=======================================")
    print(
        f"Selection: {args.selection}"
    )
    print(
        f"Selected candidates: {len(selection)}"
    )
    print(
        f"Training cities: "
        f"{selection['city_id'].nunique()}"
    )
    print(
        f"Tile size: {tile_size}"
    )
    print(
        f"Output root: {args.output_root}"
    )
    print(
        f"Dry run: {args.dry_run}"
    )
    print(
        f"Force overwrite: {args.force}"
    )

    source_groups = (
        selection
        .groupby(
            [
                "city_id",
                "city_name",
            ],
            sort=True,
        )
    )

    rows: list[dict[str, Any]] = []

    checked_source_files: dict[
        str,
        dict[str, str],
    ] = {}

    validation_mismatch_count = 0

    for (
        city_id,
        city_name,
    ), city_candidates in source_groups:

        city_id = str(city_id)
        city_name = str(city_name)

        first_row = city_candidates.iloc[0]

        (
            image_path,
            semantic_path,
            validity_path,
        ) = resolve_source_paths(
            first_row
        )

        for path, label in (
            (image_path, "SAR image"),
            (semantic_path, "semantic mask"),
            (validity_path, "validity mask"),
        ):
            require_file(
                path,
                f"{city_id} {label}",
            )

        print()
        print(
            f"{city_id} {city_name}: "
            f"{len(city_candidates)} selected tiles"
        )

        city_folder = (
            f"{city_id}_{city_name}"
        )

        city_root = (
            args.output_root
            / "train"
            / city_folder
        )

        image_dir = (
            city_root / "images"
        )

        semantic_dir = (
            city_root / "semantic_masks"
        )

        validity_dir = (
            city_root / "validity_masks"
        )

        if not args.dry_run:
            for directory in (
                image_dir,
                semantic_dir,
                validity_dir,
            ):
                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

        with rasterio.open(
            image_path
        ) as image_ds, rasterio.open(
            semantic_path
        ) as semantic_ds, rasterio.open(
            validity_path
        ) as validity_ds:

            validate_alignment(
                image_ds,
                semantic_ds,
                validity_ds,
            )

            image_profile = (
                image_ds.profile.copy()
            )

            semantic_profile = (
                semantic_ds.profile.copy()
            )

            validity_profile = (
                validity_ds.profile.copy()
            )

            checked_source_files[
                city_id
            ] = {
                "image": str(image_path),
                "semantic": str(semantic_path),
                "validity": str(validity_path),
            }

            for _, candidate in (
                city_candidates.iterrows()
            ):

                candidate_id = str(
                    candidate["candidate_id"]
                )

                row_off = int(
                    candidate["row_offset"]
                )

                col_off = int(
                    candidate["col_offset"]
                )

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=tile_size,
                    height=tile_size,
                )

                tile_transform = (
                    window_transform(
                        window,
                        image_ds.transform,
                    )
                )

                image_array = image_ds.read(
                    window=window,
                    out_shape=(
                        image_ds.count,
                        tile_size,
                        tile_size,
                    ),
                    boundless=True,
                    fill_value=image_fill,
                )

                semantic_array = (
                    semantic_ds.read(
                        1,
                        window=window,
                        out_shape=(
                            tile_size,
                            tile_size,
                        ),
                        boundless=True,
                        fill_value=semantic_fill,
                    )
                )

                validity_array = (
                    validity_ds.read(
                        1,
                        window=window,
                        out_shape=(
                            tile_size,
                            tile_size,
                        ),
                        boundless=True,
                        fill_value=validity_fill,
                    )
                )

                source_width = max(
                    0,
                    min(
                        tile_size,
                        image_ds.width - col_off,
                    ),
                )

                source_height = max(
                    0,
                    min(
                        tile_size,
                        image_ds.height - row_off,
                    ),
                )

                source_pixel_count = (
                    source_width
                    * source_height
                )

                tile_pixel_count = (
                    tile_size
                    * tile_size
                )

                padding_pixel_count = (
                    tile_pixel_count
                    - source_pixel_count
                )

                counts = class_counts(
                    semantic_array
                )

                valid_pixel_count = int(
                    np.count_nonzero(
                        validity_array == 1
                    )
                )

                unlabeled_pixel_count = int(
                    np.count_nonzero(
                        semantic_array == 0
                    )
                )

                nodata_pixel_count = int(
                    np.count_nonzero(
                        semantic_array
                        == semantic_fill
                    )
                )

                image_output = (
                    image_dir
                    / f"{candidate_id}_image.tif"
                )

                semantic_output = (
                    semantic_dir
                    / f"{candidate_id}_semantic.tif"
                )

                validity_output = (
                    validity_dir
                    / f"{candidate_id}_validity.tif"
                )

                generated_row: dict[
                    str,
                    Any,
                ] = {
                    "tile_id": candidate_id,
                    "candidate_id": candidate_id,
                    "dataset_version": "v2.2",
                    "dataset_source": "v22_offset_enrichment",
                    "city_id": city_id,
                    "city_name": city_name,
                    "split": "train",
                    "row_offset": row_off,
                    "col_offset": col_off,
                    "tile_size": tile_size,
                    "grid_offset": int(
                        candidate.get(
                            "grid_offset",
                            tile_size // 2,
                        )
                    ),
                    "source_width_pixels": (
                        source_width
                    ),
                    "source_height_pixels": (
                        source_height
                    ),
                    "padding_pixel_count": (
                        padding_pixel_count
                    ),
                    "padding_fraction": (
                        padding_pixel_count
                        / tile_pixel_count
                    ),
                    "valid_pixel_count": (
                        valid_pixel_count
                    ),
                    "valid_fraction": (
                        valid_pixel_count
                        / tile_pixel_count
                    ),
                    "unlabeled_pixel_count": (
                        unlabeled_pixel_count
                    ),
                    "semantic_nodata_pixel_count": (
                        nodata_pixel_count
                    ),
                    "buildings_pixel_count": (
                        counts.get(1, 0)
                    ),
                    "roads_pixel_count": (
                        counts.get(2, 0)
                    ),
                    "vegetation_pixel_count": (
                        counts.get(3, 0)
                    ),
                    "bare_land_pixel_count": (
                        counts.get(4, 0)
                    ),
                    "water_pixel_count": (
                        counts.get(5, 0)
                    ),
                    "contains_buildings": (
                        counts.get(1, 0) > 0
                    ),
                    "contains_roads": (
                        counts.get(2, 0) > 0
                    ),
                    "contains_vegetation": (
                        counts.get(3, 0) > 0
                    ),
                    "contains_bare_land": (
                        counts.get(4, 0) > 0
                    ),
                    "contains_water": (
                        counts.get(5, 0) > 0
                    ),
                    "selection_rank": (
                        int(
                            candidate[
                                "selection_rank"
                            ]
                        )
                        if (
                            "selection_rank"
                            in candidate.index
                        )
                        else None
                    ),
                    "priority_tier": (
                        int(
                            candidate[
                                "priority_tier"
                            ]
                        )
                        if (
                            "priority_tier"
                            in candidate.index
                        )
                        else None
                    ),
                    "selection_reason": (
                        str(
                            candidate[
                                "selection_reason"
                            ]
                        )
                        if (
                            "selection_reason"
                            in candidate.index
                        )
                        else ""
                    ),
                    "enrichment_score": (
                        float(
                            candidate[
                                "enrichment_score"
                            ]
                        )
                        if (
                            "enrichment_score"
                            in candidate.index
                        )
                        else None
                    ),
                }

                differences = (
                    compare_selection_statistics(
                        candidate,
                        generated_row,
                    )
                )

                generated_row[
                    "simulation_match"
                ] = (
                    len(differences) == 0
                )

                generated_row[
                    "simulation_difference"
                ] = "; ".join(
                    differences
                )

                if differences:
                    validation_mismatch_count += 1

                left, bottom, right, top = (
                    window_bounds(
                        window,
                        image_ds.transform,
                    )
                )

                generated_row.update(
                    {
                        "bounds_left": left,
                        "bounds_bottom": bottom,
                        "bounds_right": right,
                        "bounds_top": top,
                        "crs": str(
                            image_ds.crs
                        ),
                        "source_image": str(
                            image_path
                        ),
                        "source_semantic_mask": (
                            str(semantic_path)
                        ),
                        "source_validity_mask": (
                            str(validity_path)
                        ),
                        "image_path": str(
                            image_output
                        ),
                        "semantic_mask_path": str(
                            semantic_output
                        ),
                        "validity_mask_path": str(
                            validity_output
                        ),
                    }
                )

                rows.append(
                    generated_row
                )

                if args.dry_run:
                    continue

                if not args.force:

                    existing = [
                        path
                        for path in (
                            image_output,
                            semantic_output,
                            validity_output,
                        )
                        if path.exists()
                    ]

                    if existing:
                        raise FileExistsError(
                            "V2.2 enrichment tile "
                            "already exists. "
                            "Use --force only if you "
                            "intend to regenerate it: "
                            f"{existing[0]}"
                        )

                image_tile_profile = (
                    image_profile.copy()
                )

                image_tile_profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=2,
                    transform=tile_transform,
                    compress=output_config[
                        "compress"
                    ],
                    tiled=bool(
                        output_config["tiled"]
                    ),
                    blockxsize=int(
                        output_config[
                            "block_size"
                        ]
                    ),
                    blockysize=int(
                        output_config[
                            "block_size"
                        ]
                    ),
                    predictor=int(
                        output_config[
                            "predictor_image"
                        ]
                    ),
                    BIGTIFF=str(
                        output_config[
                            "bigtiff"
                        ]
                    ).upper(),
                )

                with rasterio.open(
                    image_output,
                    "w",
                    **image_tile_profile,
                ) as output:

                    output.write(
                        image_array
                    )

                    output.set_band_description(
                        1,
                        "Sigma0_VV",
                    )

                    output.set_band_description(
                        2,
                        "Sigma0_VH",
                    )

                    output.update_tags(
                        city_id=city_id,
                        city_name=city_name,
                        split="train",
                        tile_id=candidate_id,
                        dataset_version="v2.2",
                        tile_source=(
                            "offset_enrichment"
                        ),
                        row_offset=str(
                            row_off
                        ),
                        col_offset=str(
                            col_off
                        ),
                        source_image=str(
                            image_path
                        ),
                    )

                semantic_tile_profile = (
                    semantic_profile.copy()
                )

                semantic_tile_profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=1,
                    dtype="uint8",
                    nodata=semantic_fill,
                    transform=tile_transform,
                    compress=output_config[
                        "compress"
                    ],
                    tiled=bool(
                        output_config["tiled"]
                    ),
                    blockxsize=int(
                        output_config[
                            "block_size"
                        ]
                    ),
                    blockysize=int(
                        output_config[
                            "block_size"
                        ]
                    ),
                    predictor=int(
                        output_config[
                            "predictor_mask"
                        ]
                    ),
                    BIGTIFF=str(
                        output_config[
                            "bigtiff"
                        ]
                    ).upper(),
                )

                with rasterio.open(
                    semantic_output,
                    "w",
                    **semantic_tile_profile,
                ) as output:

                    output.write(
                        semantic_array.astype(
                            np.uint8
                        ),
                        1,
                    )

                    output.set_band_description(
                        1,
                        "five_class_semantic_label",
                    )

                    output.update_tags(
                        city_id=city_id,
                        city_name=city_name,
                        split="train",
                        tile_id=candidate_id,
                        dataset_version="v2.2",
                        tile_source=(
                            "offset_enrichment"
                        ),
                        class_0="unlabeled",
                        class_1="buildings",
                        class_2="roads",
                        class_3="vegetation",
                        class_4="bare_land",
                        class_5="water",
                        semantic_nodata=str(
                            semantic_fill
                        ),
                    )

                validity_tile_profile = (
                    validity_profile.copy()
                )

                validity_tile_profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=1,
                    dtype="uint8",
                    nodata=validity_fill,
                    transform=tile_transform,
                    compress=output_config[
                        "compress"
                    ],
                    tiled=bool(
                        output_config["tiled"]
                    ),
                    blockxsize=int(
                        output_config[
                            "block_size"
                        ]
                    ),
                    blockysize=int(
                        output_config[
                            "block_size"
                        ]
                    ),
                    predictor=int(
                        output_config[
                            "predictor_mask"
                        ]
                    ),
                    BIGTIFF=str(
                        output_config[
                            "bigtiff"
                        ]
                    ).upper(),
                )

                with rasterio.open(
                    validity_output,
                    "w",
                    **validity_tile_profile,
                ) as output:

                    output.write(
                        validity_array.astype(
                            np.uint8
                        ),
                        1,
                    )

                    output.set_band_description(
                        1,
                        "training_validity_mask",
                    )

                    output.update_tags(
                        city_id=city_id,
                        city_name=city_name,
                        split="train",
                        tile_id=candidate_id,
                        dataset_version="v2.2",
                        tile_source=(
                            "offset_enrichment"
                        ),
                        value_0="ignore",
                        value_1="valid_label",
                    )

    if len(rows) != len(selection):
        raise RuntimeError(
            "Materialized manifest row count does not match "
            "the selected-candidate count."
        )

    manifest_dataframe = (
        pd.DataFrame(rows)
    )

    # This comparison should be exact because the simulator and
    # materializer read identical source-mask windows.
    if validation_mismatch_count:
        raise RuntimeError(
            f"{validation_mismatch_count} generated windows do not "
            "match the simulation statistics. "
            "No V2.2 dataset should be frozen until investigated."
        )

    if args.dry_run:

        print()
        print("Dry-run validation")
        print("------------------")
        print(
            f"Candidates validated: {len(rows)}"
        )
        print(
            "Simulation-statistic mismatches: "
            f"{validation_mismatch_count}"
        )
        print()
        print(
            "Result: dry-run validation PASSED. "
            "No raster files were created."
        )

        return

    manifest_dataframe.to_csv(
        manifest_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    source_checksums: dict[
        str,
        dict[str, str],
    ] = {}

    for city_id, paths in (
        checked_source_files.items()
    ):

        source_checksums[
            city_id
        ] = {
            "image_sha256": (
                sha256_file(
                    Path(paths["image"])
                )
            ),
            "semantic_sha256": (
                sha256_file(
                    Path(paths["semantic"])
                )
            ),
            "validity_sha256": (
                sha256_file(
                    Path(paths["validity"])
                )
            ),
        }

    summary = {
        "dataset_version": "v2.2",
        "component": "offset_enrichment",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "selection_source": str(
            args.selection
        ),
        "tiling_config": str(
            args.config
        ),
        "output_root": str(
            args.output_root
        ),
        "selected_candidate_count": int(
            len(selection)
        ),
        "materialized_tile_count": int(
            len(manifest_dataframe)
        ),
        "city_count": int(
            manifest_dataframe[
                "city_id"
            ].nunique()
        ),
        "split": "train",
        "tile_size": tile_size,
        "grid_type": "128_pixel_offset",
        "simulation_statistic_mismatches": int(
            validation_mismatch_count
        ),
        "tiles_with_padding": int(
            (
                manifest_dataframe[
                    "padding_pixel_count"
                ]
                > 0
            ).sum()
        ),
        "tiles_with_zero_valid_pixels": int(
            (
                manifest_dataframe[
                    "valid_pixel_count"
                ]
                == 0
            ).sum()
        ),
        "tiles_containing_each_class": {
            class_name: int(
                manifest_dataframe[
                    f"contains_{class_name}"
                ].sum()
            )
            for class_name in (
                TRAINABLE_CLASSES.values()
            )
        },
        "aggregate_class_pixels": {
            class_name: int(
                manifest_dataframe[
                    f"{class_name}_pixel_count"
                ].sum()
            )
            for class_name in (
                TRAINABLE_CLASSES.values()
            )
        },
        "aggregate_valid_pixels": int(
            manifest_dataframe[
                "valid_pixel_count"
            ].sum()
        ),
        "manifest": str(
            manifest_path
        ),
        "source_checksums": (
            source_checksums
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Materialization completed")
    print("-------------------------")
    print(
        f"Tiles generated: "
        f"{len(manifest_dataframe)}"
    )
    print(
        f"Cities: "
        f"{manifest_dataframe['city_id'].nunique()}"
    )
    print(
        "Simulation-statistic mismatches: "
        f"{validation_mismatch_count}"
    )
    print(
        f"Tiles with padding: "
        f"{summary['tiles_with_padding']}"
    )
    print(
        "Tiles with zero valid pixels: "
        f"{summary['tiles_with_zero_valid_pixels']}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        f"Summary: {summary_path}"
    )

    print()
    print(
        "Result: Dataset V2.2 enrichment tiles "
        "materialized successfully."
    )


if __name__ == "__main__":
    main()