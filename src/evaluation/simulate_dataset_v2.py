from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_PATH = Path(
    "reports/dataset_v1_validity_analysis/training_tile_balance.csv"
)

OUTPUT_DIR = Path(
    "reports/dataset_v2_simulation"
)


# ---------------------------------------------------------------------
# Dataset V2 policy
# ---------------------------------------------------------------------

MIN_VALID_FRACTION = 0.50

VEGETATION_DOMINANT_THRESHOLD = 0.95
VEGETATION_DOMINANT_KEEP_FRACTION = 0.30

BUILDINGS_RICH_THRESHOLD = 0.10
ROADS_RICH_THRESHOLD = 0.10
BARE_LAND_RICH_THRESHOLD = 0.05
WATER_RICH_THRESHOLD = 0.10

MIXED_CLASS_THRESHOLD = 0.05
MIN_MIXED_CLASSES = 3

RANDOM_SEED = 20260725


CLASS_NAMES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]


def load_training_tiles() -> pd.DataFrame:
    """
    Load the Dataset V1 training-tile diagnostic table.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()

    df = df.sort_values(
        ["city_id", "tile_id"]
    ).reset_index(drop=True)

    return df


def add_v2_policy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Dataset V2 candidate-selection rules without modifying files.
    """
    df = df.copy()

    # --------------------------------------------------------------
    # A. Validity
    # --------------------------------------------------------------

    df["passes_validity"] = (
        df["valid_fraction"] >= MIN_VALID_FRACTION
    )

    # --------------------------------------------------------------
    # B. Minority / urban richness
    # --------------------------------------------------------------

    df["v2_building_rich"] = (
        df["buildings_fraction_valid"]
        >= BUILDINGS_RICH_THRESHOLD
    )

    df["v2_road_rich"] = (
        df["roads_fraction_valid"]
        >= ROADS_RICH_THRESHOLD
    )

    df["v2_bare_land_rich"] = (
        df["bare_land_fraction_valid"]
        >= BARE_LAND_RICH_THRESHOLD
    )

    df["v2_water_rich"] = (
        df["water_fraction_valid"]
        >= WATER_RICH_THRESHOLD
    )

    df["minority_rich"] = (
        df[
            [
                "v2_building_rich",
                "v2_road_rich",
                "v2_bare_land_rich",
                "v2_water_rich",
            ]
        ]
        .any(axis=1)
    )

    # --------------------------------------------------------------
    # C. Mixed informative tiles
    # --------------------------------------------------------------

    fraction_columns = [
        "buildings_fraction_valid",
        "roads_fraction_valid",
        "vegetation_fraction_valid",
        "bare_land_fraction_valid",
        "water_fraction_valid",
    ]

    df["v2_classes_ge_5pct"] = (
        df[fraction_columns]
        .ge(MIXED_CLASS_THRESHOLD)
        .sum(axis=1)
    )

    df["v2_mixed_informative"] = (
        df["v2_classes_ge_5pct"]
        >= MIN_MIXED_CLASSES
    )

    # Protected tiles are deliberately retained whenever valid.
    df["protected_tile"] = (
        df["minority_rich"]
        | df["v2_mixed_informative"]
    )

    # --------------------------------------------------------------
    # D. Extreme vegetation dominance
    # --------------------------------------------------------------

    df["extreme_vegetation"] = (
        df["vegetation_fraction_valid"]
        >= VEGETATION_DOMINANT_THRESHOLD
    )

    # Candidate vegetation tiles that are not protected.
    vegetation_candidates = df[
        df["passes_validity"]
        & df["extreme_vegetation"]
        & ~df["protected_tile"]
    ].copy()

    # Deterministic selection within each city.
    kept_vegetation_indices: set[int] = set()

    for city_id, city_df in vegetation_candidates.groupby(
        "city_id",
        sort=True,
    ):
        city_df = city_df.sort_values("tile_id")

        n_tiles = len(city_df)

        if n_tiles == 0:
            continue

        n_keep = max(
            1,
            round(
                n_tiles
                * VEGETATION_DOMINANT_KEEP_FRACTION
            ),
        )

        sampled = city_df.sample(
            n=n_keep,
            random_state=RANDOM_SEED,
        )

        kept_vegetation_indices.update(
            sampled.index.tolist()
        )

    df["vegetation_control_keep"] = False

    if kept_vegetation_indices:
        df.loc[
            list(kept_vegetation_indices),
            "vegetation_control_keep",
        ] = True

    # --------------------------------------------------------------
    # Final V2 selection
    # --------------------------------------------------------------

    normal_valid_tile = (
        df["passes_validity"]
        & ~df["extreme_vegetation"]
    )

    protected_valid_tile = (
        df["passes_validity"]
        & df["protected_tile"]
    )

    sampled_vegetation_tile = (
        df["passes_validity"]
        & df["vegetation_control_keep"]
    )

    df["selected_v2"] = (
        normal_valid_tile
        | protected_valid_tile
        | sampled_vegetation_tile
    )

    # --------------------------------------------------------------
    # Selection reason
    # --------------------------------------------------------------

    def selection_reason(row: pd.Series) -> str:
        if not row["passes_validity"]:
            return "excluded_low_validity"

        if row["protected_tile"]:
            reasons = []

            if row["v2_building_rich"]:
                reasons.append("building_rich")

            if row["v2_road_rich"]:
                reasons.append("road_rich")

            if row["v2_bare_land_rich"]:
                reasons.append("bare_land_rich")

            if row["v2_water_rich"]:
                reasons.append("water_rich")

            if row["v2_mixed_informative"]:
                reasons.append("mixed_informative")

            return "+".join(reasons)

        if row["extreme_vegetation"]:
            if row["vegetation_control_keep"]:
                return "vegetation_control_kept"

            return "vegetation_control_removed"

        return "standard_valid_tile"

    df["v2_selection_reason"] = df.apply(
        selection_reason,
        axis=1,
    )

    return df


def summarize_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """
    Calculate tile-level and class-level summary statistics.
    """
    summary: dict = {
        "dataset": dataset_name,
        "tiles": int(len(df)),
        "cities": int(df["city_id"].nunique()),
    }

    if len(df) == 0:
        return summary

    total_valid = df["valid_pixel_count"].sum()

    summary["valid_pixels"] = int(total_valid)

    summary["mean_valid_fraction"] = float(
        df["valid_fraction"].mean()
    )

    summary["vegetation_ge_90pct_tiles"] = int(
        (
            df["vegetation_fraction_valid"]
            >= 0.90
        ).sum()
    )

    summary["vegetation_ge_95pct_tiles"] = int(
        (
            df["vegetation_fraction_valid"]
            >= 0.95
        ).sum()
    )

    summary["mixed_informative_tiles"] = int(
        df["v2_mixed_informative"].sum()
    )

    for class_name in CLASS_NAMES:
        pixel_column = f"{class_name}_pixel_count"

        pixels = df[pixel_column].sum()

        summary[f"{class_name}_pixels"] = int(
            pixels
        )

        if total_valid > 0:
            summary[
                f"{class_name}_pct_valid"
            ] = float(
                pixels
                / total_valid
                * 100.0
            )
        else:
            summary[
                f"{class_name}_pct_valid"
            ] = 0.0

    return summary


def city_summary(
    original: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare V1 and V2 representation city by city.
    """
    rows = []

    for city_id in sorted(
        original["city_id"].unique()
    ):
        before = original[
            original["city_id"] == city_id
        ]

        after = selected[
            selected["city_id"] == city_id
        ]

        city_name = before[
            "city_name"
        ].iloc[0]

        row = {
            "city_id": city_id,
            "city_name": city_name,
            "v1_tiles": len(before),
            "v2_tiles": len(after),
        }

        if len(before) > 0:
            row["tile_retention_pct"] = (
                len(after)
                / len(before)
                * 100.0
            )
        else:
            row["tile_retention_pct"] = 0.0

        for class_name in CLASS_NAMES:
            pixel_column = (
                f"{class_name}_pixel_count"
            )

            before_pixels = before[
                pixel_column
            ].sum()

            after_pixels = after[
                pixel_column
            ].sum()

            row[
                f"v1_{class_name}_pixels"
            ] = int(before_pixels)

            row[
                f"v2_{class_name}_pixels"
            ] = int(after_pixels)

            if before_pixels > 0:
                row[
                    f"{class_name}_retention_pct"
                ] = (
                    after_pixels
                    / before_pixels
                    * 100.0
                )
            else:
                row[
                    f"{class_name}_retention_pct"
                ] = 100.0

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(
    original_summary: dict,
    v2_summary: dict,
    policy_df: pd.DataFrame,
) -> None:
    """
    Print concise V1 versus candidate-V2 diagnostics.
    """
    print()
    print("Dataset V2 simulation")
    print("=====================")
    print()

    print("Selection policy")
    print("----------------")
    print(
        f"Minimum valid fraction: "
        f"{MIN_VALID_FRACTION:.2f}"
    )
    print(
        "Extreme vegetation threshold: "
        f"{VEGETATION_DOMINANT_THRESHOLD:.2f}"
    )
    print(
        "Extreme vegetation keep fraction: "
        f"{VEGETATION_DOMINANT_KEEP_FRACTION:.2f}"
    )
    print(
        "Buildings-rich threshold: "
        f"{BUILDINGS_RICH_THRESHOLD:.2f}"
    )
    print(
        "Road-rich threshold: "
        f"{ROADS_RICH_THRESHOLD:.2f}"
    )
    print(
        "Bare-land-rich threshold: "
        f"{BARE_LAND_RICH_THRESHOLD:.2f}"
    )
    print(
        "Water-rich threshold: "
        f"{WATER_RICH_THRESHOLD:.2f}"
    )

    print()
    print("Tile selection")
    print("--------------")

    total = len(policy_df)

    selected = int(
        policy_df["selected_v2"].sum()
    )

    removed = total - selected

    print(f"Dataset V1 tiles: {total}")
    print(f"Candidate V2 tiles: {selected}")
    print(f"Removed tiles: {removed}")

    if total > 0:
        print(
            "Retention: "
            f"{selected / total * 100:.2f}%"
        )

    print()
    print("Removal reasons")
    print("---------------")

    reason_counts = (
        policy_df[
            "v2_selection_reason"
        ]
        .value_counts()
    )

    print(reason_counts.to_string())

    print()
    print("V1 -> V2 class distribution")
    print("---------------------------")

    comparison_rows = []

    for class_name in CLASS_NAMES:
        v1_pct = original_summary[
            f"{class_name}_pct_valid"
        ]

        v2_pct = v2_summary[
            f"{class_name}_pct_valid"
        ]

        comparison_rows.append(
            {
                "class": class_name,
                "v1_pct_valid": v1_pct,
                "v2_pct_valid": v2_pct,
                "change_pct_points": (
                    v2_pct - v1_pct
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("Vegetation dominance")
    print("--------------------")

    print(
        "V1 vegetation >= 90% tiles: "
        f"{original_summary['vegetation_ge_90pct_tiles']}"
    )
    print(
        "V2 vegetation >= 90% tiles: "
        f"{v2_summary['vegetation_ge_90pct_tiles']}"
    )

    print(
        "V1 vegetation >= 95% tiles: "
        f"{original_summary['vegetation_ge_95pct_tiles']}"
    )
    print(
        "V2 vegetation >= 95% tiles: "
        f"{v2_summary['vegetation_ge_95pct_tiles']}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original = load_training_tiles()

    policy_df = add_v2_policy_columns(
        original
    )

    selected = (
        policy_df[
            policy_df["selected_v2"]
        ]
        .copy()
        .sort_values(
            ["city_id", "tile_id"]
        )
        .reset_index(drop=True)
    )

    original_summary = summarize_dataset(
        policy_df,
        "dataset_v1",
    )

    v2_summary = summarize_dataset(
        selected,
        "dataset_v2_candidate",
    )

    comparison = pd.DataFrame(
        [
            original_summary,
            v2_summary,
        ]
    )

    cities = city_summary(
        policy_df,
        selected,
    )

    # --------------------------------------------------------------
    # Write reports
    # --------------------------------------------------------------

    policy_df.to_csv(
        OUTPUT_DIR
        / "dataset_v2_selection_audit.csv",
        index=False,
    )

    selected.to_csv(
        OUTPUT_DIR
        / "dataset_v2_candidate_manifest.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "dataset_v1_vs_v2_summary.csv",
        index=False,
    )

    cities.to_csv(
        OUTPUT_DIR
        / "dataset_v2_city_retention.csv",
        index=False,
    )

    print_summary(
        original_summary,
        v2_summary,
        policy_df,
    )

    print()
    print(
        f"Reports written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()