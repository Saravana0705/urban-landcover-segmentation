from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("metadata/dataset_v1/dataset_manifest.csv")
OUTPUT_DIR = Path("reports/dataset_v1_validity_analysis")


CLASS_PIXEL_COLUMNS = {
    "buildings": "buildings_pixel_count",
    "roads": "roads_pixel_count",
    "vegetation": "vegetation_pixel_count",
    "bare_land": "bare_land_pixel_count",
    "water": "water_pixel_count",
}


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "tile_size",
        "valid_pixel_count",
        "unlabeled_pixel_count",
        *CLASS_PIXEL_COLUMNS.values(),
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required manifest columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 1. Tile-level statistics
    # ---------------------------------------------------------

    df["total_tile_pixels"] = df["tile_size"] * df["tile_size"]

    df["invalid_pixel_count"] = (
        df["total_tile_pixels"] - df["valid_pixel_count"]
    )

    df["valid_fraction"] = (
        df["valid_pixel_count"] / df["total_tile_pixels"]
    )

    df["invalid_fraction"] = (
        df["invalid_pixel_count"] / df["total_tile_pixels"]
    )

    # Fractions relative to supervised/valid pixels
    for class_name, pixel_col in CLASS_PIXEL_COLUMNS.items():
        df[f"{class_name}_fraction_valid"] = (
            df[pixel_col] / df["valid_pixel_count"].replace(0, pd.NA)
        ).fillna(0.0)

    # Also compute total semantic pixels represented by five classes
    df["five_class_pixel_count"] = sum(
        df[col] for col in CLASS_PIXEL_COLUMNS.values()
    )

    df["class_accounting_difference"] = (
        df["valid_pixel_count"] - df["five_class_pixel_count"]
    )

    # ---------------------------------------------------------
    # 2. Split summary
    # ---------------------------------------------------------

    split_rows = []

    for split, group in df.groupby("split"):
        total_pixels = int(group["total_tile_pixels"].sum())
        valid_pixels = int(group["valid_pixel_count"].sum())
        invalid_pixels = int(group["invalid_pixel_count"].sum())

        row = {
            "split": split,
            "city_count": group["city_id"].nunique(),
            "tile_count": len(group),
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "invalid_pixels": invalid_pixels,
            "valid_pct": 100.0 * safe_divide(
                valid_pixels, total_pixels
            ),
            "invalid_pct": 100.0 * safe_divide(
                invalid_pixels, total_pixels
            ),
        }

        for class_name, pixel_col in CLASS_PIXEL_COLUMNS.items():
            class_pixels = int(group[pixel_col].sum())

            row[f"{class_name}_pixels"] = class_pixels
            row[f"{class_name}_pct_valid"] = (
                100.0 * safe_divide(class_pixels, valid_pixels)
            )

        split_rows.append(row)

    split_summary = pd.DataFrame(split_rows)

    # ---------------------------------------------------------
    # 3. City summary
    # ---------------------------------------------------------

    city_rows = []

    for (split, city_id, city_name), group in df.groupby(
        ["split", "city_id", "city_name"]
    ):
        total_pixels = int(group["total_tile_pixels"].sum())
        valid_pixels = int(group["valid_pixel_count"].sum())
        invalid_pixels = int(group["invalid_pixel_count"].sum())

        row = {
            "split": split,
            "city_id": city_id,
            "city_name": city_name,
            "tile_count": len(group),
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "invalid_pixels": invalid_pixels,
            "valid_pct": 100.0 * safe_divide(
                valid_pixels, total_pixels
            ),
            "invalid_pct": 100.0 * safe_divide(
                invalid_pixels, total_pixels
            ),
        }

        for class_name, pixel_col in CLASS_PIXEL_COLUMNS.items():
            class_pixels = int(group[pixel_col].sum())

            row[f"{class_name}_pixels"] = class_pixels
            row[f"{class_name}_pct_valid"] = (
                100.0 * safe_divide(class_pixels, valid_pixels)
            )

        city_rows.append(row)

    city_summary = pd.DataFrame(city_rows)

    # ---------------------------------------------------------
    # 4. Training-tile diagnostics for Dataset V2 planning
    # ---------------------------------------------------------

    train_df = df[df["split"] == "train"].copy()

    train_df["vegetation_dominant_90"] = (
        train_df["vegetation_fraction_valid"] >= 0.90
    )

    train_df["vegetation_dominant_95"] = (
        train_df["vegetation_fraction_valid"] >= 0.95
    )

    train_df["urban_rich"] = (
        train_df["buildings_fraction_valid"]
        + train_df["roads_fraction_valid"]
        >= 0.15
    )

    train_df["road_rich"] = (
        train_df["roads_fraction_valid"] >= 0.10
    )

    train_df["building_rich"] = (
        train_df["buildings_fraction_valid"] >= 0.10
    )

    train_df["bare_land_rich"] = (
        train_df["bare_land_fraction_valid"] >= 0.05
    )

    train_df["water_rich"] = (
        train_df["water_fraction_valid"] >= 0.10
    )

    # Mixed informative tile:
    # at least 3 of the 5 semantic classes occupy >= 5% each
    fraction_columns = [
        f"{class_name}_fraction_valid"
        for class_name in CLASS_PIXEL_COLUMNS
    ]

    train_df["classes_ge_5pct"] = (
        train_df[fraction_columns]
        .ge(0.05)
        .sum(axis=1)
    )

    train_df["mixed_informative"] = (
        train_df["classes_ge_5pct"] >= 3
    )

    # ---------------------------------------------------------
    # 5. Print main findings
    # ---------------------------------------------------------

    print()
    print("Dataset V1 validity and class-balance analysis")
    print("==============================================")
    print()

    print("Split-level summary")
    print("-------------------")

    display_columns = [
        "split",
        "city_count",
        "tile_count",
        "valid_pct",
        "invalid_pct",
        "buildings_pct_valid",
        "roads_pct_valid",
        "vegetation_pct_valid",
        "bare_land_pct_valid",
        "water_pct_valid",
    ]

    print(
        split_summary[display_columns]
        .to_string(index=False)
    )

    print()
    print("Training-tile composition")
    print("-------------------------")

    print(f"Training tiles: {len(train_df)}")
    print(
        "Vegetation >= 90%: "
        f"{train_df['vegetation_dominant_90'].sum()}"
    )
    print(
        "Vegetation >= 95%: "
        f"{train_df['vegetation_dominant_95'].sum()}"
    )
    print(
        "Urban-rich: "
        f"{train_df['urban_rich'].sum()}"
    )
    print(
        "Road-rich: "
        f"{train_df['road_rich'].sum()}"
    )
    print(
        "Building-rich: "
        f"{train_df['building_rich'].sum()}"
    )
    print(
        "Bare-land-rich: "
        f"{train_df['bare_land_rich'].sum()}"
    )
    print(
        "Water-rich: "
        f"{train_df['water_rich'].sum()}"
    )
    print(
        "Mixed informative: "
        f"{train_df['mixed_informative'].sum()}"
    )

    print()
    print("Class-accounting check")
    print("----------------------")

    mismatch_tiles = df[
        df["class_accounting_difference"] != 0
    ]

    print(
        f"Tiles where valid pixels != "
        f"sum of five semantic classes: {len(mismatch_tiles)}"
    )

    if not mismatch_tiles.empty:
        print()
        print(
            mismatch_tiles[
                [
                    "tile_id",
                    "city_name",
                    "split",
                    "valid_pixel_count",
                    "five_class_pixel_count",
                    "class_accounting_difference",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 6. Save reports
    # ---------------------------------------------------------

    split_summary.to_csv(
        OUTPUT_DIR / "split_validity_summary.csv",
        index=False,
    )

    city_summary.to_csv(
        OUTPUT_DIR / "city_validity_summary.csv",
        index=False,
    )

    train_df.to_csv(
        OUTPUT_DIR / "training_tile_balance.csv",
        index=False,
    )

    mismatch_tiles.to_csv(
        OUTPUT_DIR / "class_accounting_mismatches.csv",
        index=False,
    )

    print()
    print(
        f"Reports written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()