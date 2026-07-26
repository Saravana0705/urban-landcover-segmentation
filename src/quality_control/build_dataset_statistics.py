"""Generate official Dataset Version 1 statistics and imbalance metadata.

This script reads only frozen Dataset V1 metadata. It does not modify or
rescan the SAR images or masks.

Inputs
------
metadata/dataset_v1/dataset_manifest.csv
metadata/dataset_v1/dataset_manifest_summary.json
metadata/dataset_v1/normalization/training_normalization.json

Outputs
-------
metadata/dataset_v1/statistics/dataset_statistics.json
metadata/dataset_v1/statistics/dataset_statistics.csv
metadata/dataset_v1/statistics/class_distribution_detailed.csv
metadata/dataset_v1/statistics/class_weights_candidates.json
metadata/dataset_v1/statistics/class_weights_candidates.csv
metadata/dataset_v1/statistics/DATASET_V1_STATISTICS.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_COLUMNS = {
    1: ("buildings", "buildings_pixel_count"),
    2: ("roads", "roads_pixel_count"),
    3: ("vegetation", "vegetation_pixel_count"),
    4: ("bare_land", "bare_land_pixel_count"),
    5: ("water", "water_pixel_count"),
}

EXPECTED_SPLITS = ("train", "val", "test")


def require_file(path: Path, label: str) -> None:
    """Require a non-empty input file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    require_file(path, "JSON file")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return data


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def normalized_mean_one(values: np.ndarray) -> np.ndarray:
    """Normalize positive weights to arithmetic mean 1."""
    values = np.asarray(values, dtype=np.float64)

    if np.any(~np.isfinite(values)):
        raise ValueError("Weights contain non-finite values.")

    if np.any(values <= 0):
        raise ValueError("Weights must be positive.")

    return values / values.mean()


def inverse_frequency_weights(
    frequencies: np.ndarray,
) -> np.ndarray:
    """Return mean-one inverse-frequency weights."""
    return normalized_mean_one(1.0 / frequencies)


def median_frequency_weights(
    frequencies: np.ndarray,
) -> np.ndarray:
    """Return mean-one median-frequency-balanced weights."""
    median = float(np.median(frequencies))
    return normalized_mean_one(median / frequencies)


def logarithmic_inverse_weights(
    frequencies: np.ndarray,
    constant: float = 1.02,
) -> np.ndarray:
    """Return ENet-style logarithmic inverse-frequency weights."""
    if constant <= 1.0:
        raise ValueError("Logarithmic constant must be greater than 1.")

    raw = 1.0 / np.log(constant + frequencies)
    return normalized_mean_one(raw)


def effective_number_weights(
    pixel_counts: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Return class-balanced weights using effective sample number."""
    if not 0.0 < beta < 1.0:
        raise ValueError("Beta must be between 0 and 1.")

    counts = np.asarray(pixel_counts, dtype=np.float64)
    numerator = 1.0 - beta
    denominator = 1.0 - np.power(beta, counts)

    raw = numerator / denominator
    return normalized_mean_one(raw)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Dataset V1 statistics and class-imbalance metadata."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "metadata/dataset_v1/dataset_manifest.csv"
        ),
    )

    parser.add_argument(
        "--manifest-summary",
        type=Path,
        default=Path(
            "metadata/dataset_v1/"
            "dataset_manifest_summary.json"
        ),
    )

    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path(
            "metadata/dataset_v1/normalization/"
            "training_normalization.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "metadata/dataset_v1/statistics"
        ),
    )

    parser.add_argument(
        "--effective-number-beta",
        type=float,
        default=0.9999999,
        help=(
            "Beta used only to report a candidate effective-number "
            "weighting scheme."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate Dataset V1 statistics."""
    args = parse_arguments()

    require_file(args.manifest, "dataset manifest")
    require_file(
        args.manifest_summary,
        "dataset manifest summary",
    )
    require_file(
        args.normalization,
        "training normalization metadata",
    )

    manifest = pd.read_csv(args.manifest)
    manifest_summary = load_json(args.manifest_summary)
    normalization = load_json(args.normalization)

    required_manifest_columns = {
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "tile_size",
        "padding_pixel_count",
        "valid_pixel_count",
        "unlabeled_pixel_count",
        "semantic_nodata_pixel_count",
        "buildings_pixel_count",
        "roads_pixel_count",
        "vegetation_pixel_count",
        "bare_land_pixel_count",
        "water_pixel_count",
        "contains_buildings",
        "contains_roads",
        "contains_vegetation",
        "contains_bare_land",
        "contains_water",
        "tile_qa_status",
    }

    missing = required_manifest_columns.difference(
        manifest.columns
    )

    if missing:
        raise ValueError(
            f"Dataset manifest is missing columns: {sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError("Dataset manifest is empty.")

    if manifest["tile_id"].duplicated().any():
        raise RuntimeError(
            "Dataset manifest contains duplicate tile IDs."
        )

    if set(manifest["split"].astype(str)) != set(EXPECTED_SPLITS):
        raise RuntimeError(
            "Manifest does not contain exactly train, val and test."
        )

    if not (
        manifest["tile_qa_status"].astype(str) == "PASS"
    ).all():
        raise RuntimeError(
            "At least one manifest row is not tile-QA PASS."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics_json_path = (
        args.output_dir / "dataset_statistics.json"
    )
    statistics_csv_path = (
        args.output_dir / "dataset_statistics.csv"
    )
    detailed_class_path = (
        args.output_dir
        / "class_distribution_detailed.csv"
    )
    weights_json_path = (
        args.output_dir
        / "class_weights_candidates.json"
    )
    weights_csv_path = (
        args.output_dir
        / "class_weights_candidates.csv"
    )
    markdown_path = (
        args.output_dir / "DATASET_V1_STATISTICS.md"
    )

    total_tile_pixels = int(
        (
            manifest["tile_size"].astype(np.int64)
            ** 2
        ).sum()
    )

    split_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    print("\nDataset V1 statistics")
    print("---------------------")
    print(f"Manifest rows: {len(manifest):,}")

    for split_name in (*EXPECTED_SPLITS, "all"):
        if split_name == "all":
            subset = manifest
        else:
            subset = manifest.loc[
                manifest["split"].astype(str)
                == split_name
            ]

        tile_count = len(subset)
        city_count = int(subset["city_id"].nunique())
        split_total_tile_pixels = int(
            (
                subset["tile_size"].astype(np.int64)
                ** 2
            ).sum()
        )

        valid_pixels = int(
            subset["valid_pixel_count"].sum()
        )
        unlabeled_pixels = int(
            subset["unlabeled_pixel_count"].sum()
        )
        semantic_nodata_pixels = int(
            subset[
                "semantic_nodata_pixel_count"
            ].sum()
        )
        padding_pixels = int(
            subset["padding_pixel_count"].sum()
        )

        class_pixel_total = int(
            sum(
                subset[column].sum()
                for _, column in CLASS_COLUMNS.values()
            )
        )

        split_rows.append(
            {
                "scope": split_name,
                "city_count": city_count,
                "tile_count": tile_count,
                "total_tile_pixels": split_total_tile_pixels,
                "valid_training_pixels": valid_pixels,
                "unlabeled_pixels": unlabeled_pixels,
                "semantic_nodata_pixels": semantic_nodata_pixels,
                "padding_pixels": padding_pixels,
                "zero_valid_tile_count": int(
                    (
                        subset["valid_pixel_count"] == 0
                    ).sum()
                ),
                "valid_fraction_of_tile_grid": (
                    valid_pixels / split_total_tile_pixels
                    if split_total_tile_pixels > 0
                    else 0.0
                ),
                "class_pixel_sum_matches_valid": (
                    class_pixel_total == valid_pixels
                ),
            }
        )

        for class_id, (
            class_name,
            pixel_column,
        ) in CLASS_COLUMNS.items():
            pixels = int(subset[pixel_column].sum())

            class_rows.append(
                {
                    "scope": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixel_count": pixels,
                    "percentage_of_valid_pixels": (
                        pixels / valid_pixels * 100.0
                        if valid_pixels > 0
                        else 0.0
                    ),
                    "frequency_fraction": (
                        pixels / valid_pixels
                        if valid_pixels > 0
                        else 0.0
                    ),
                    "mapped_area_km2_at_10m": (
                        pixels * 100.0 / 1_000_000.0
                    ),
                    "tiles_containing_class": int(
                        subset[
                            f"contains_{class_name}"
                        ].astype(bool).sum()
                    ),
                    "cities_containing_class": int(
                        subset.loc[
                            subset[
                                f"contains_{class_name}"
                            ].astype(bool),
                            "city_id",
                        ].nunique()
                    ),
                }
            )

    split_statistics = pd.DataFrame(split_rows)
    class_distribution = pd.DataFrame(class_rows)

    if not split_statistics[
        "class_pixel_sum_matches_valid"
    ].all():
        raise RuntimeError(
            "Class pixel totals do not equal valid-pixel totals."
        )

    training_distribution = (
        class_distribution.loc[
            class_distribution["scope"] == "train"
        ]
        .sort_values("class_id")
        .reset_index(drop=True)
    )

    training_counts = training_distribution[
        "pixel_count"
    ].to_numpy(dtype=np.float64)

    training_frequencies = training_distribution[
        "frequency_fraction"
    ].to_numpy(dtype=np.float64)

    if np.any(training_counts <= 0):
        raise RuntimeError(
            "At least one training class has zero pixels."
        )

    candidate_weights = {
        "inverse_frequency_mean_one":
            inverse_frequency_weights(
                training_frequencies
            ),
        "median_frequency_balancing_mean_one":
            median_frequency_weights(
                training_frequencies
            ),
        "logarithmic_inverse_c1_02_mean_one":
            logarithmic_inverse_weights(
                training_frequencies,
                constant=1.02,
            ),
        "effective_number_mean_one":
            effective_number_weights(
                training_counts,
                beta=args.effective_number_beta,
            ),
    }

    weight_rows: list[dict[str, Any]] = []

    for index, class_row in (
        training_distribution.iterrows()
    ):
        output_row = {
            "class_id": int(class_row["class_id"]),
            "class_name": str(
                class_row["class_name"]
            ),
            "training_pixel_count": int(
                class_row["pixel_count"]
            ),
            "training_frequency_fraction": float(
                class_row["frequency_fraction"]
            ),
            "training_percentage": float(
                class_row[
                    "percentage_of_valid_pixels"
                ]
            ),
        }

        for scheme_name, values in (
            candidate_weights.items()
        ):
            output_row[scheme_name] = float(
                values[index]
            )

        weight_rows.append(output_row)

    weights_dataframe = pd.DataFrame(weight_rows)

    imbalance_ratio = float(
        training_counts.max() / training_counts.min()
    )

    rarest_index = int(np.argmin(training_counts))
    dominant_index = int(np.argmax(training_counts))

    normalization_summary = {}

    for band_name in ("VV", "VH"):
        clipped = normalization["bands"][
            band_name
        ]["db_clipped_p01_p99"]

        normalization_summary[band_name] = {
            "clip_lower_db": float(
                clipped["clip_lower_db"]
            ),
            "clip_upper_db": float(
                clipped["clip_upper_db"]
            ),
            "mean_db_clipped": float(
                clipped["mean"]
            ),
            "std_db_clipped": float(
                clipped["std_population"]
            ),
            "valid_pixel_count": int(
                clipped["valid_pixel_count"]
            ),
        }

    dataset_statistics = {
        "dataset_version": "1.0-pre-freeze",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(
            args.manifest
        ),
        "manifest_summary": str(
            args.manifest_summary
        ),
        "manifest_summary_sha256": sha256_file(
            args.manifest_summary
        ),
        "normalization": str(args.normalization),
        "normalization_sha256": sha256_file(
            args.normalization
        ),
        "city_count": int(
            manifest["city_id"].nunique()
        ),
        "tile_count": len(manifest),
        "total_tile_pixels": total_tile_pixels,
        "tile_size": int(
            manifest["tile_size"].iloc[0]
        ),
        "input_channels": [
            "Sigma0_VV",
            "Sigma0_VH",
        ],
        "semantic_classes": [
            {
                "class_id": class_id,
                "class_name": class_name,
            }
            for class_id, (
                class_name,
                _,
            ) in CLASS_COLUMNS.items()
        ],
        "splits": {
            row["scope"]: {
                key: (
                    bool(value)
                    if isinstance(
                        value,
                        (np.bool_, bool),
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (np.integer,),
                    )
                    else float(value)
                    if isinstance(
                        value,
                        (np.floating,),
                    )
                    else value
                )
                for key, value in row.items()
                if key != "scope"
            }
            for row in split_rows
        },
        "training_class_imbalance": {
            "dominant_class": str(
                training_distribution.iloc[
                    dominant_index
                ]["class_name"]
            ),
            "rarest_class": str(
                training_distribution.iloc[
                    rarest_index
                ]["class_name"]
            ),
            "dominant_to_rarest_pixel_ratio":
                imbalance_ratio,
            "candidate_weight_schemes": [
                "inverse_frequency_mean_one",
                "median_frequency_balancing_mean_one",
                "logarithmic_inverse_c1_02_mean_one",
                "effective_number_mean_one",
            ],
            "effective_number_beta": (
                args.effective_number_beta
            ),
            "selection_status": (
                "not_selected; compare during loss-function design"
            ),
        },
        "normalization_policy": {
            "scope": "training_cities_only",
            "validation_and_test_used": False,
            "input_conversion": (
                "10 * log10(max(linear_sigma0, 1e-10))"
            ),
            "clipping": "training p01 to p99 by band",
            "standardization": (
                "training clipped mean and population std"
            ),
            "bands": normalization_summary,
        },
        "mandatory_checks": {
            "city_count_is_20": (
                manifest["city_id"].nunique()
                == 20
            ),
            "tile_count_is_2880": (
                len(manifest) == 2880
            ),
            "tile_ids_unique": bool(
                manifest["tile_id"].is_unique
            ),
            "all_tile_qa_pass": bool(
                (
                    manifest["tile_qa_status"]
                    == "PASS"
                ).all()
            ),
            "all_class_sums_match_valid": bool(
                split_statistics[
                    "class_pixel_sum_matches_valid"
                ].all()
            ),
            "normalization_training_only": bool(
                normalization[
                    "validation_and_test_used"
                ]
                is False
            ),
            "manifest_matches_summary": bool(
                len(manifest)
                == int(
                    manifest_summary["tile_count"]
                )
            ),
        },
    }

    dataset_statistics[
        "ready_for_dataset_freeze"
    ] = all(
        dataset_statistics[
            "mandatory_checks"
        ].values()
    )

    with statistics_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    split_statistics.to_csv(
        statistics_csv_path,
        index=False,
    )

    class_distribution.to_csv(
        detailed_class_path,
        index=False,
    )

    weights_dataframe.to_csv(
        weights_csv_path,
        index=False,
    )

    weights_json = {
        "dataset_version": "1.0-pre-freeze",
        "scope": "training_split_only",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "effective_number_beta": (
            args.effective_number_beta
        ),
        "selection_status": (
            "candidate schemes only; final loss weighting "
            "has not yet been selected"
        ),
        "classes": weight_rows,
    }

    with weights_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            weights_json,
            file,
            indent=2,
            ensure_ascii=False,
        )

    train_classes = training_distribution[
        [
            "class_id",
            "class_name",
            "pixel_count",
            "percentage_of_valid_pixels",
            "mapped_area_km2_at_10m",
            "tiles_containing_class",
            "cities_containing_class",
        ]
    ]

    markdown_lines = [
        "# Dataset Version 1 Statistics",
        "",
        f"- Dataset status: **pre-freeze**",
        f"- Cities: **{manifest['city_id'].nunique()}**",
        f"- Tiles: **{len(manifest)}**",
        f"- Tile size: **{int(manifest['tile_size'].iloc[0])} × "
        f"{int(manifest['tile_size'].iloc[0])} pixels**",
        "- Spatial resolution: **10 m**",
        "- Input channels: **Sigma0 VV and VH**",
        "- Semantic classes: **buildings, roads, vegetation, "
        "bare land, water**",
        "",
        "## Split Summary",
        "",
        split_statistics.to_markdown(index=False),
        "",
        "## Training Class Distribution",
        "",
        train_classes.to_markdown(index=False),
        "",
        "## Training-Class Imbalance",
        "",
        f"- Dominant class: **"
        f"{dataset_statistics['training_class_imbalance']['dominant_class']}**",
        f"- Rarest class: **"
        f"{dataset_statistics['training_class_imbalance']['rarest_class']}**",
        f"- Dominant-to-rarest pixel ratio: **"
        f"{imbalance_ratio:.4f}:1**",
        "- Candidate weighting schemes were calculated, but no final "
        "scheme has yet been selected.",
        "",
        "## Training-Only Normalization",
        "",
        "| Band | Lower clip (dB) | Upper clip (dB) | Mean (dB) | Std (dB) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| VV | {normalization_summary['VV']['clip_lower_db']:.6f} "
            f"| {normalization_summary['VV']['clip_upper_db']:.6f} "
            f"| {normalization_summary['VV']['mean_db_clipped']:.6f} "
            f"| {normalization_summary['VV']['std_db_clipped']:.6f} |"
        ),
        (
            f"| VH | {normalization_summary['VH']['clip_lower_db']:.6f} "
            f"| {normalization_summary['VH']['clip_upper_db']:.6f} "
            f"| {normalization_summary['VH']['mean_db_clipped']:.6f} "
            f"| {normalization_summary['VH']['std_db_clipped']:.6f} |"
        ),
        "",
        "## Mandatory Checks",
        "",
    ]

    for check_name, passed in dataset_statistics[
        "mandatory_checks"
    ].items():
        markdown_lines.append(
            f"- `{check_name}`: **{passed}**"
        )

    markdown_lines.extend(
        [
            "",
            f"Ready for Dataset V1 freeze: **"
            f"{dataset_statistics['ready_for_dataset_freeze']}**",
            "",
        ]
    )

    markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print("\nSplit summary")
    print(split_statistics.to_string(index=False))

    print("\nTraining class distribution")
    print(
        training_distribution[
            [
                "class_id",
                "class_name",
                "pixel_count",
                "percentage_of_valid_pixels",
            ]
        ].to_string(index=False)
    )

    print("\nTraining class imbalance")
    print(
        "  Dominant class: "
        f"{dataset_statistics['training_class_imbalance']['dominant_class']}"
    )
    print(
        "  Rarest class: "
        f"{dataset_statistics['training_class_imbalance']['rarest_class']}"
    )
    print(
        "  Dominant-to-rarest ratio: "
        f"{imbalance_ratio:.4f}:1"
    )

    print("\nMandatory checks")
    for name, passed in dataset_statistics[
        "mandatory_checks"
    ].items():
        print(f"  {name}: {passed}")

    print(
        "\nReady for Dataset V1 freeze: "
        f"{dataset_statistics['ready_for_dataset_freeze']}"
    )
    print(f"\nStatistics JSON: {statistics_json_path}")
    print(f"Statistics CSV: {statistics_csv_path}")
    print(f"Class distribution: {detailed_class_path}")
    print(f"Candidate weights JSON: {weights_json_path}")
    print(f"Candidate weights CSV: {weights_csv_path}")
    print(f"Markdown report: {markdown_path}")

    if not dataset_statistics[
        "ready_for_dataset_freeze"
    ]:
        raise RuntimeError(
            "Dataset statistics failed mandatory checks."
        )

    print(
        "\nResult: Dataset V1 statistics generated successfully."
    )


if __name__ == "__main__":
    main()
