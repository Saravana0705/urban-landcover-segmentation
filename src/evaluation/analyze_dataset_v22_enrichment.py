from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "dataset_v21"
    / "dataset_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "dataset_v22_enrichment_analysis"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CLASS_COLUMNS = {
    "buildings": "buildings_pct_valid",
    "roads": "roads_pct_valid",
    "vegetation": "vegetation_pct_valid",
    "bare_land": "bare_land_pct_valid",
    "water": "water_pct_valid",
}


# Thresholds expressed as fraction of VALID pixels.
#
# These are intentionally exploratory thresholds.
# They are NOT final Dataset V2.2 selection rules.
THRESHOLDS = {
    "building_rich": 0.10,
    "road_rich": 0.08,
    "bare_land_moderate": 0.02,
    "bare_land_rich": 0.05,
    "water_rich": 0.05,
    "urban_mixed": 0.15,
}


MIN_VALID_FRACTION = 0.50


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def normalize_fraction_column(series: pd.Series) -> pd.Series:
    """
    Convert percentage-like columns to fractions when necessary.

    Supports either:
        0.10  -> 10%
    or:
        10.0  -> 10%
    """

    values = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if values.max() > 1.0:
        values = values / 100.0

    return values.clip(lower=0.0, upper=1.0)


def find_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:

    for name in candidates:
        if name in dataframe.columns:
            return name

    if required:
        raise KeyError(
            "Could not find any of these columns in manifest: "
            f"{candidates}\n\n"
            f"Available columns:\n{list(dataframe.columns)}"
        )

    return None


def prepare_manifest(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy()

    split_col = find_column(
        df,
        (
            "split",
            "dataset_split",
        ),
    )

    city_id_col = find_column(
        df,
        (
            "city_id",
            "city",
        ),
    )

    city_name_col = find_column(
        df,
        (
            "city_name",
            "name",
        ),
        required=False,
    )

    valid_col = find_column(
        df,
        (
            "valid_fraction",
            "valid_pct",
            "validity_fraction",
        ),
        required=False,
    )

    df["analysis_split"] = (
        df[split_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["analysis_city_id"] = df[city_id_col].astype(str)

    if city_name_col is not None:
        df["analysis_city_name"] = df[city_name_col].astype(str)
    else:
        df["analysis_city_name"] = df["analysis_city_id"]

    if valid_col is not None:
        df["analysis_valid_fraction"] = normalize_fraction_column(
            df[valid_col]
        )
    else:
        # If Dataset V2.1 already contains only usable pixels/tiles,
        # absence of this field should not block exploratory analysis.
        df["analysis_valid_fraction"] = 1.0

    return df


def resolve_class_columns(
    dataframe: pd.DataFrame,
) -> dict[str, str]:

    aliases = {
        "buildings": (
            "buildings_pct_valid",
            "building_pct_valid",
            "buildings_fraction_valid",
            "building_fraction_valid",
            "buildings_fraction",
            "building_fraction",
        ),
        "roads": (
            "roads_pct_valid",
            "road_pct_valid",
            "roads_fraction_valid",
            "road_fraction_valid",
            "roads_fraction",
            "road_fraction",
        ),
        "vegetation": (
            "vegetation_pct_valid",
            "vegetation_fraction_valid",
            "vegetation_fraction",
        ),
        "bare_land": (
            "bare_land_pct_valid",
            "bareland_pct_valid",
            "bare_land_fraction_valid",
            "bare_land_fraction",
            "bareland_fraction",
        ),
        "water": (
            "water_pct_valid",
            "water_fraction_valid",
            "water_fraction",
        ),
    }

    resolved = {}

    for class_name, candidates in aliases.items():
        resolved[class_name] = find_column(
            dataframe,
            candidates,
        )

    return resolved


def add_class_fractions(
    dataframe: pd.DataFrame,
    class_columns: dict[str, str],
) -> pd.DataFrame:

    df = dataframe.copy()

    for class_name, column_name in class_columns.items():
        df[f"{class_name}_fraction"] = normalize_fraction_column(
            df[column_name]
        )

    return df


def add_enrichment_flags(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    df = dataframe.copy()

    df["building_rich"] = (
        df["buildings_fraction"]
        >= THRESHOLDS["building_rich"]
    )

    df["road_rich"] = (
        df["roads_fraction"]
        >= THRESHOLDS["road_rich"]
    )

    df["bare_land_moderate"] = (
        df["bare_land_fraction"]
        >= THRESHOLDS["bare_land_moderate"]
    )

    df["bare_land_rich"] = (
        df["bare_land_fraction"]
        >= THRESHOLDS["bare_land_rich"]
    )

    df["water_rich"] = (
        df["water_fraction"]
        >= THRESHOLDS["water_rich"]
    )

    df["urban_fraction"] = (
        df["buildings_fraction"]
        + df["roads_fraction"]
    )

    df["urban_mixed"] = (
        df["urban_fraction"]
        >= THRESHOLDS["urban_mixed"]
    )

    # Number of minority / urban classes with meaningful presence.
    meaningful_flags = pd.DataFrame(
        {
            "buildings": (
                df["buildings_fraction"] >= 0.05
            ),
            "roads": (
                df["roads_fraction"] >= 0.05
            ),
            "bare_land": (
                df["bare_land_fraction"] >= 0.02
            ),
            "water": (
                df["water_fraction"] >= 0.03
            ),
        },
        index=df.index,
    )

    df["minority_class_count"] = meaningful_flags.sum(axis=1)

    df["mixed_minority"] = (
        df["minority_class_count"] >= 2
    )

    df["vegetation_extreme"] = (
        df["vegetation_fraction"] >= 0.95
    )

    df["useful_enrichment_candidate"] = (
        (
            df["building_rich"]
            | df["road_rich"]
            | df["bare_land_moderate"]
            | df["water_rich"]
            | df["urban_mixed"]
            | df["mixed_minority"]
        )
        & (
            df["analysis_valid_fraction"]
            >= MIN_VALID_FRACTION
        )
    )

    return df


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def build_class_summary(
    training_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for class_name in CLASS_COLUMNS:

        column = f"{class_name}_fraction"

        rows.append(
            {
                "class_name": class_name,
                "mean_fraction": training_df[column].mean(),
                "median_fraction": training_df[column].median(),
                "max_fraction": training_df[column].max(),
                "tiles_with_class": int(
                    (training_df[column] > 0).sum()
                ),
                "tiles_ge_1pct": int(
                    (training_df[column] >= 0.01).sum()
                ),
                "tiles_ge_5pct": int(
                    (training_df[column] >= 0.05).sum()
                ),
                "tiles_ge_10pct": int(
                    (training_df[column] >= 0.10).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_city_summary(
    training_df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for (city_id, city_name), city_df in training_df.groupby(
        ["analysis_city_id", "analysis_city_name"],
        dropna=False,
    ):

        record = {
            "city_id": city_id,
            "city_name": city_name,
            "tile_count": len(city_df),
            "candidate_tiles": int(
                city_df["useful_enrichment_candidate"].sum()
            ),
            "building_rich_tiles": int(
                city_df["building_rich"].sum()
            ),
            "road_rich_tiles": int(
                city_df["road_rich"].sum()
            ),
            "bare_land_moderate_tiles": int(
                city_df["bare_land_moderate"].sum()
            ),
            "bare_land_rich_tiles": int(
                city_df["bare_land_rich"].sum()
            ),
            "water_rich_tiles": int(
                city_df["water_rich"].sum()
            ),
            "urban_mixed_tiles": int(
                city_df["urban_mixed"].sum()
            ),
            "mixed_minority_tiles": int(
                city_df["mixed_minority"].sum()
            ),
        }

        for class_name in CLASS_COLUMNS:
            record[f"{class_name}_mean_fraction"] = (
                city_df[f"{class_name}_fraction"].mean()
            )

        records.append(record)

    result = pd.DataFrame(records)

    return result.sort_values(
        [
            "bare_land_rich_tiles",
            "road_rich_tiles",
            "building_rich_tiles",
            "water_rich_tiles",
        ],
        ascending=False,
    )


def build_candidate_summary(
    training_df: pd.DataFrame,
) -> pd.DataFrame:

    categories = {
        "building_rich": "building_rich",
        "road_rich": "road_rich",
        "bare_land_moderate": "bare_land_moderate",
        "bare_land_rich": "bare_land_rich",
        "water_rich": "water_rich",
        "urban_mixed": "urban_mixed",
        "mixed_minority": "mixed_minority",
        "vegetation_extreme": "vegetation_extreme",
        "useful_enrichment_candidate":
            "useful_enrichment_candidate",
    }

    rows = []

    for category, column in categories.items():

        count = int(training_df[column].sum())

        rows.append(
            {
                "category": category,
                "tile_count": count,
                "percentage_of_training_tiles": (
                    100.0 * count / len(training_df)
                    if len(training_df)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print()
    print("Dataset V2.2 enrichment analysis")
    print("================================")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset V2.1 manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    df = prepare_manifest(df)

    class_columns = resolve_class_columns(df)

    print("Resolved semantic-class columns")
    print("-------------------------------")

    for class_name, column_name in class_columns.items():
        print(f"{class_name:12s}: {column_name}")

    print()

    df = add_class_fractions(
        df,
        class_columns,
    )

    training_df = df[
        df["analysis_split"] == "train"
    ].copy()

    if training_df.empty:
        raise RuntimeError(
            "No training rows found in Dataset V2.1 manifest."
        )

    training_df = add_enrichment_flags(training_df)

    class_summary = build_class_summary(training_df)

    city_summary = build_city_summary(training_df)

    candidate_summary = build_candidate_summary(training_df)

    candidate_tiles = training_df[
        training_df["useful_enrichment_candidate"]
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_summary.to_csv(
        OUTPUT_DIR / "v22_candidate_by_class.csv",
        index=False,
    )

    city_summary.to_csv(
        OUTPUT_DIR / "v22_candidate_by_city.csv",
        index=False,
    )

    candidate_summary.to_csv(
        OUTPUT_DIR / "v22_candidate_summary.csv",
        index=False,
    )

    candidate_tiles.to_csv(
        OUTPUT_DIR / "v22_candidate_tiles.csv",
        index=False,
    )

    print("Dataset V2.1 training-set summary")
    print("---------------------------------")
    print(f"Training tiles: {len(training_df)}")
    print()

    print("Class distribution")
    print("------------------")

    for class_name in CLASS_COLUMNS:

        mean_fraction = training_df[
            f"{class_name}_fraction"
        ].mean()

        print(
            f"{class_name:12s}: "
            f"{mean_fraction * 100:6.3f}%"
        )

    print()

    print("Existing enrichment opportunities")
    print("---------------------------------")

    for _, row in candidate_summary.iterrows():
        print(
            f"{row['category']:28s}: "
            f"{int(row['tile_count']):5d} "
            f"({row['percentage_of_training_tiles']:6.2f}%)"
        )

    print()
    print(
        "Useful V2.2 candidate tiles: "
        f"{len(candidate_tiles)}"
    )

    print()
    print("Top cities for minority-class enrichment")
    print("----------------------------------------")

    display_columns = [
        "city_id",
        "city_name",
        "tile_count",
        "candidate_tiles",
        "road_rich_tiles",
        "bare_land_moderate_tiles",
        "bare_land_rich_tiles",
        "building_rich_tiles",
        "water_rich_tiles",
    ]

    print(
        city_summary[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        f"Reports written to: "
        f"{OUTPUT_DIR.relative_to(PROJECT_ROOT)}"
    )

    print()
    print(
        "NOTE: This script performs analysis only. "
        "No Dataset V2.2 tiles have been generated."
    )


if __name__ == "__main__":
    main()