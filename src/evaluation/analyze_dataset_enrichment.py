from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_PATH = Path("metadata/dataset_v1/dataset_manifest.csv")
OUTPUT_DIR = Path("reports/dataset_v1_enrichment_analysis")

CLASS_PIXEL_COLUMNS = {
    "buildings": "buildings_pixel_count",
    "roads": "roads_pixel_count",
    "vegetation": "vegetation_pixel_count",
    "bare_land": "bare_land_pixel_count",
    "water": "water_pixel_count",
}


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "valid_pixel_count",
        *CLASS_PIXEL_COLUMNS.values(),
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "Dataset manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        "valid_pixel_count",
        *CLASS_PIXEL_COLUMNS.values(),
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    return df


def prepare_training_tiles(df: pd.DataFrame) -> pd.DataFrame:
    train = df[
        (df["split"].astype(str).str.lower() == "train")
        & (df["valid_pixel_count"] > 0)
    ].copy()

    train = train.sort_values(
        ["city_id", "tile_id"]
    ).reset_index(drop=True)

    valid_pixels = train["valid_pixel_count"].clip(lower=1)

    for class_name, pixel_column in CLASS_PIXEL_COLUMNS.items():
        train[f"{class_name}_fraction"] = (
            train[pixel_column] / valid_pixels
        )

    class_fraction_columns = [
        f"{name}_fraction"
        for name in CLASS_PIXEL_COLUMNS
    ]

    train["dominant_class"] = (
        train[class_fraction_columns]
        .idxmax(axis=1)
        .str.replace("_fraction", "", regex=False)
    )

    train["dominant_fraction"] = (
        train[class_fraction_columns].max(axis=1)
    )

    train["non_vegetation_fraction"] = (
        1.0 - train["vegetation_fraction"]
    ).clip(lower=0.0)

    train["urban_fraction"] = (
        train["buildings_fraction"]
        + train["roads_fraction"]
        + train["bare_land_fraction"]
    )

    train["minority_fraction"] = (
        train["roads_fraction"]
        + train["bare_land_fraction"]
        + train["water_fraction"]
    )

    train["class_count_ge_1pct"] = sum(
        train[f"{name}_fraction"] >= 0.01
        for name in CLASS_PIXEL_COLUMNS
    )

    return train


def add_enrichment_flags(train: pd.DataFrame) -> pd.DataFrame:
    result = train.copy()

    result["vegetation_dominated"] = (
        result["vegetation_fraction"] >= 0.90
    )

    result["extreme_vegetation"] = (
        result["vegetation_fraction"] >= 0.97
    )

    result["building_rich"] = (
        result["buildings_fraction"] >= 0.05
    )

    result["road_rich"] = (
        result["roads_fraction"] >= 0.05
    )

    result["bare_land_present"] = (
        result["bare_land_fraction"] >= 0.01
    )

    result["bare_land_rich"] = (
        result["bare_land_fraction"] >= 0.05
    )

    result["water_rich"] = (
        result["water_fraction"] >= 0.05
    )

    result["multi_class_urban"] = (
        result["class_count_ge_1pct"] >= 3
    )

    return result


def calculate_enrichment_score(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diagnostic ranking only.

    This score is NOT yet used for Dataset V2 selection.
    It helps identify potentially informative training tiles.
    """

    result = train.copy()

    score = (
        1.25 * np.minimum(
            result["buildings_fraction"] / 0.10,
            1.0,
        )
        + 1.50 * np.minimum(
            result["roads_fraction"] / 0.10,
            1.0,
        )
        + 2.50 * np.minimum(
            result["bare_land_fraction"] / 0.05,
            1.0,
        )
        + 1.00 * np.minimum(
            result["water_fraction"] / 0.10,
            1.0,
        )
        + 0.50 * np.minimum(
            result["non_vegetation_fraction"] / 0.30,
            1.0,
        )
        + 0.50 * np.minimum(
            result["class_count_ge_1pct"] / 4.0,
            1.0,
        )
    )

    vegetation_penalty = np.where(
        result["vegetation_fraction"] >= 0.97,
        1.0,
        np.where(
            result["vegetation_fraction"] >= 0.90,
            0.5,
            0.0,
        ),
    )

    result["enrichment_score"] = (
        score - vegetation_penalty
    ).clip(lower=0.0)

    return result


def build_dataset_summary(
    train: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for class_name, pixel_column in CLASS_PIXEL_COLUMNS.items():
        fraction_column = f"{class_name}_fraction"

        rows.append(
            {
                "class": class_name,
                "total_pixels": int(
                    train[pixel_column].sum()
                ),
                "tiles_with_class": int(
                    (train[pixel_column] > 0).sum()
                ),
                "tiles_ge_1pct": int(
                    (train[fraction_column] >= 0.01).sum()
                ),
                "tiles_ge_5pct": int(
                    (train[fraction_column] >= 0.05).sum()
                ),
                "tiles_ge_10pct": int(
                    (train[fraction_column] >= 0.10).sum()
                ),
                "mean_fraction": float(
                    train[fraction_column].mean()
                ),
                "median_fraction": float(
                    train[fraction_column].median()
                ),
                "max_fraction": float(
                    train[fraction_column].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_city_summary(
    train: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        city_id,
        city_name,
    ), group in train.groupby(
        ["city_id", "city_name"],
        sort=True,
    ):
        rows.append(
            {
                "city_id": city_id,
                "city_name": city_name,
                "tile_count": len(group),
                "vegetation_dominated_tiles": int(
                    group["vegetation_dominated"].sum()
                ),
                "extreme_vegetation_tiles": int(
                    group["extreme_vegetation"].sum()
                ),
                "building_rich_tiles": int(
                    group["building_rich"].sum()
                ),
                "road_rich_tiles": int(
                    group["road_rich"].sum()
                ),
                "bare_land_present_tiles": int(
                    group["bare_land_present"].sum()
                ),
                "bare_land_rich_tiles": int(
                    group["bare_land_rich"].sum()
                ),
                "water_rich_tiles": int(
                    group["water_rich"].sum()
                ),
                "multi_class_urban_tiles": int(
                    group["multi_class_urban"].sum()
                ),
                "mean_vegetation_fraction": float(
                    group["vegetation_fraction"].mean()
                ),
                "mean_buildings_fraction": float(
                    group["buildings_fraction"].mean()
                ),
                "mean_roads_fraction": float(
                    group["roads_fraction"].mean()
                ),
                "mean_bare_land_fraction": float(
                    group["bare_land_fraction"].mean()
                ),
                "mean_water_fraction": float(
                    group["water_fraction"].mean()
                ),
                "mean_enrichment_score": float(
                    group["enrichment_score"].mean()
                ),
                "max_enrichment_score": float(
                    group["enrichment_score"].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def write_markdown_summary(
    train: pd.DataFrame,
    class_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
) -> None:
    path = OUTPUT_DIR / "DATASET_V1_ENRICHMENT_SUMMARY.md"

    vegetation_90 = int(
        train["vegetation_dominated"].sum()
    )

    vegetation_97 = int(
        train["extreme_vegetation"].sum()
    )

    road_rich = int(train["road_rich"].sum())

    bare_present = int(
        train["bare_land_present"].sum()
    )

    bare_rich = int(train["bare_land_rich"].sum())

    water_rich = int(train["water_rich"].sum())

    building_rich = int(
        train["building_rich"].sum()
    )

    text = f"""# Dataset V1 Enrichment Analysis

## Training dataset

- Training tiles: {len(train)}
- Training cities: {train["city_id"].nunique()}
- Vegetation >= 90%: {vegetation_90}
- Vegetation >= 97%: {vegetation_97}
- Building-rich tiles (>= 5%): {building_rich}
- Road-rich tiles (>= 5%): {road_rich}
- Bare-land-present tiles (>= 1%): {bare_present}
- Bare-land-rich tiles (>= 5%): {bare_rich}
- Water-rich tiles (>= 5%): {water_rich}

## Purpose

This report diagnoses Dataset V1 before constructing Dataset V2.

Dataset V2 must not modify the validation or test sets. Candidate
enrichment should be generated only from training-city imagery.

The enrichment score produced here is diagnostic only and must not
yet be treated as the final Dataset V2 selection rule.
"""

    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    train = prepare_training_tiles(manifest)
    train = add_enrichment_flags(train)
    train = calculate_enrichment_score(train)

    class_summary = build_dataset_summary(train)
    city_summary = build_city_summary(train)

    candidate_columns = [
        "tile_id",
        "city_id",
        "city_name",
        "valid_pixel_count",
        "buildings_fraction",
        "roads_fraction",
        "vegetation_fraction",
        "bare_land_fraction",
        "water_fraction",
        "urban_fraction",
        "minority_fraction",
        "class_count_ge_1pct",
        "dominant_class",
        "dominant_fraction",
        "enrichment_score",
        "building_rich",
        "road_rich",
        "bare_land_present",
        "bare_land_rich",
        "water_rich",
        "multi_class_urban",
        "vegetation_dominated",
        "extreme_vegetation",
        "image_path",
        "semantic_mask_path",
        "validity_mask_path",
    ]

    candidate_columns = [
        column
        for column in candidate_columns
        if column in train.columns
    ]

    ranked_candidates = (
        train[candidate_columns]
        .sort_values(
            "enrichment_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    vegetation_tiles = (
        train[
            train["vegetation_dominated"]
        ][candidate_columns]
        .sort_values(
            "vegetation_fraction",
            ascending=False,
        )
    )

    bare_land_candidates = (
        train[
            train["bare_land_present"]
        ][candidate_columns]
        .sort_values(
            [
                "bare_land_fraction",
                "enrichment_score",
            ],
            ascending=False,
        )
    )

    road_candidates = (
        train[
            train["roads_fraction"] >= 0.03
        ][candidate_columns]
        .sort_values(
            [
                "roads_fraction",
                "enrichment_score",
            ],
            ascending=False,
        )
    )

    class_summary.to_csv(
        OUTPUT_DIR / "class_distribution_summary.csv",
        index=False,
    )

    city_summary.to_csv(
        OUTPUT_DIR / "city_enrichment_summary.csv",
        index=False,
    )

    ranked_candidates.to_csv(
        OUTPUT_DIR / "ranked_training_tiles.csv",
        index=False,
    )

    vegetation_tiles.to_csv(
        OUTPUT_DIR / "vegetation_dominated_tiles.csv",
        index=False,
    )

    bare_land_candidates.to_csv(
        OUTPUT_DIR / "bare_land_candidates.csv",
        index=False,
    )

    road_candidates.to_csv(
        OUTPUT_DIR / "road_candidates.csv",
        index=False,
    )

    write_markdown_summary(
        train,
        class_summary,
        city_summary,
    )

    print("\nDataset V1 enrichment analysis")
    print("------------------------------")
    print(f"Training tiles: {len(train)}")
    print(
        f"Training cities: "
        f"{train['city_id'].nunique()}"
    )

    print("\nVegetation dominance")
    print(
        "Vegetation >= 90%:",
        int(train["vegetation_dominated"].sum()),
    )
    print(
        "Vegetation >= 97%:",
        int(train["extreme_vegetation"].sum()),
    )

    print("\nUrban/minority-rich tiles")
    print(
        "Buildings >= 5%:",
        int(train["building_rich"].sum()),
    )
    print(
        "Roads >= 5%:",
        int(train["road_rich"].sum()),
    )
    print(
        "Bare land >= 1%:",
        int(train["bare_land_present"].sum()),
    )
    print(
        "Bare land >= 5%:",
        int(train["bare_land_rich"].sum()),
    )
    print(
        "Water >= 5%:",
        int(train["water_rich"].sum()),
    )

    print("\nTop enrichment candidates")
    print(
        ranked_candidates[
            [
                "tile_id",
                "city_name",
                "buildings_fraction",
                "roads_fraction",
                "bare_land_fraction",
                "water_fraction",
                "vegetation_fraction",
                "enrichment_score",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nReports written to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()