from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("metadata/dataset_v1/dataset_manifest.csv")
OUTPUT_DIR = Path("reports/minority_class_analysis")

CLASSES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]

THRESHOLDS = [0.001, 0.01, 0.05, 0.10]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    # Numeric conversion for safety
    numeric_cols = ["valid_pixel_count"] + [
        f"{cls}_pixel_count" for cls in CLASSES
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---------------------------------------------------------
    # 1. Dataset split overview
    # ---------------------------------------------------------
    split_summary = (
        df.groupby("split")
        .agg(
            tiles=("tile_id", "count"),
            cities=("city_id", "nunique"),
            valid_pixels=("valid_pixel_count", "sum"),
        )
        .reset_index()
    )

    split_summary.to_csv(
        OUTPUT_DIR / "tile_split_overview.csv",
        index=False,
    )

    print("\nDataset split overview")
    print("----------------------")
    print(split_summary.to_string(index=False))

    # ---------------------------------------------------------
    # Analyse TRAINING split only
    # ---------------------------------------------------------
    train = df[df["split"].astype(str).str.lower() == "train"].copy()

    if train.empty:
        raise RuntimeError(
            "No training rows found in dataset_manifest.csv"
        )

    print(f"\nTraining tiles: {len(train)}")
    print(f"Training cities: {train['city_id'].nunique()}")

    # Calculate fraction of VALID pixels belonging to each class
    denominator = train["valid_pixel_count"].replace(0, pd.NA)

    for cls in CLASSES:
        train[f"{cls}_fraction"] = (
            train[f"{cls}_pixel_count"] / denominator
        ).fillna(0)

    # ---------------------------------------------------------
    # 2. Tile-level class distribution
    # ---------------------------------------------------------
    rows = []

    total_tiles = len(train)

    for cls in CLASSES:
        count_col = f"{cls}_pixel_count"
        fraction_col = f"{cls}_fraction"

        present = train[count_col] > 0
        present_fractions = train.loc[present, fraction_col]

        row = {
            "class": cls,
            "total_training_tiles": total_tiles,
            "tiles_containing_class": int(present.sum()),
            "pct_tiles_containing_class":
                100.0 * present.mean(),
            "total_class_pixels":
                int(train[count_col].sum()),
            "mean_fraction_all_tiles":
                train[fraction_col].mean(),
            "mean_fraction_when_present":
                present_fractions.mean()
                if len(present_fractions) else 0,
            "median_fraction_when_present":
                present_fractions.median()
                if len(present_fractions) else 0,
            "max_fraction":
                train[fraction_col].max(),
        }

        for threshold in THRESHOLDS:
            pct = threshold * 100

            row[f"tiles_ge_{pct:g}pct"] = int(
                (train[fraction_col] >= threshold).sum()
            )

            row[f"pct_tiles_ge_{pct:g}pct"] = (
                100.0
                * (train[fraction_col] >= threshold).mean()
            )

        rows.append(row)

    class_summary = pd.DataFrame(rows)

    class_summary.to_csv(
        OUTPUT_DIR / "training_tile_class_distribution.csv",
        index=False,
    )

    print("\nTraining tile class distribution")
    print("--------------------------------")
    display_cols = [
        "class",
        "tiles_containing_class",
        "pct_tiles_containing_class",
        "mean_fraction_when_present",
        "median_fraction_when_present",
        "max_fraction",
        "pct_tiles_ge_1pct",
        "pct_tiles_ge_5pct",
        "pct_tiles_ge_10pct",
    ]

    print(class_summary[display_cols].to_string(index=False))

    # ---------------------------------------------------------
    # 3. City-level class distribution
    # ---------------------------------------------------------
    city_rows = []

    for (city_id, city_name), city_df in train.groupby(
        ["city_id", "city_name"]
    ):
        row = {
            "city_id": city_id,
            "city_name": city_name,
            "training_tiles": len(city_df),
        }

        for cls in CLASSES:
            count_col = f"{cls}_pixel_count"
            fraction_col = f"{cls}_fraction"

            row[f"{cls}_tiles"] = int(
                (city_df[count_col] > 0).sum()
            )

            row[f"{cls}_tile_pct"] = (
                100.0
                * (city_df[count_col] > 0).mean()
            )

            row[f"{cls}_pixels"] = int(
                city_df[count_col].sum()
            )

            row[f"{cls}_mean_fraction"] = (
                city_df[fraction_col].mean()
            )

        city_rows.append(row)

    city_summary = pd.DataFrame(city_rows).sort_values("city_id")

    city_summary.to_csv(
        OUTPUT_DIR / "training_city_class_distribution.csv",
        index=False,
    )

    # Compact minority-class view
    minority_cols = [
        "city_id",
        "city_name",
        "training_tiles",
        "roads_tiles",
        "roads_tile_pct",
        "roads_pixels",
        "bare_land_tiles",
        "bare_land_tile_pct",
        "bare_land_pixels",
        "water_tiles",
        "water_tile_pct",
        "water_pixels",
    ]

    print("\nMinority-class distribution by training city")
    print("--------------------------------------------")
    print(city_summary[minority_cols].to_string(index=False))

    # ---------------------------------------------------------
    # 4. Minority-rich tile candidates
    # ---------------------------------------------------------
    minority = train[
        [
            "tile_id",
            "city_id",
            "city_name",
            "valid_pixel_count",
            "roads_pixel_count",
            "bare_land_pixel_count",
            "water_pixel_count",
            "roads_fraction",
            "bare_land_fraction",
            "water_fraction",
        ]
    ].copy()

    # Simple diagnostic score -- NOT yet a training sampler.
    minority["minority_fraction"] = (
        minority["roads_fraction"]
        + minority["bare_land_fraction"]
        + minority["water_fraction"]
    )

    minority = minority.sort_values(
        "minority_fraction",
        ascending=False,
    )

    minority.to_csv(
        OUTPUT_DIR / "training_minority_tile_candidates.csv",
        index=False,
    )

    print("\nTop 20 minority-rich training tiles")
    print("-----------------------------------")
    print(minority.head(20).to_string(index=False))

    print(f"\nReports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()