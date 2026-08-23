"""Select a class-aware Dataset V2.2 enrichment subset.

This script reads the previously simulated offset candidates and selects
a controlled minority-class-focused subset.

It DOES NOT create raster tiles.

Selection priorities:
1. Bare-land-rich
2. Bare-land-moderate + mixed-minority
3. Road-rich + mixed-minority
4. Water-rich + mixed-minority
5. Building-rich + mixed-minority
6. Remaining useful candidates

A per-city cap is used to preserve geographic diversity.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CANDIDATE_PATH = Path(
    "reports/dataset_v22_offset_enrichment/offset_candidates.csv"
)

V21_DISTRIBUTION_PATH = Path(
    "reports/dataset_v22_offset_enrichment/"
    "projected_class_distribution.csv"
)

OUTPUT_DIR = Path(
    "reports/dataset_v22_class_aware_selection"
)


# ---------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------

TARGET_TILE_COUNT = 500

# ~36 tiles/city would be perfectly even for 14 cities.
# Allow some flexibility for genuinely richer cities while preventing
# Berlin/Hamburg/Cologne/etc. from dominating the dataset.
MAX_TILES_PER_CITY = 45


CLASS_NAMES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]


# ---------------------------------------------------------------------
# Priority tiers
# ---------------------------------------------------------------------

def assign_priority_tier(row: pd.Series) -> int:
    """Assign enrichment priority.

    Lower values represent higher priority.
    """

    # Priority 1:
    # Keep every genuinely bare-land-rich candidate whenever possible.
    if bool(row["bare_land_rich"]):
        return 1

    # Priority 2:
    # Moderate bare land becomes especially valuable when accompanied
    # by other minority classes.
    if (
        bool(row["bare_land_moderate"])
        and bool(row["mixed_minority"])
    ):
        return 2

    # Priority 3:
    # Road-rich mixed contexts.
    if (
        bool(row["road_rich"])
        and bool(row["mixed_minority"])
    ):
        return 3

    # Priority 4:
    # Water-rich mixed contexts.
    if (
        bool(row["water_rich"])
        and bool(row["mixed_minority"])
    ):
        return 4

    # Priority 5:
    # Building-rich mixed contexts.
    if (
        bool(row["building_rich"])
        and bool(row["mixed_minority"])
    ):
        return 5

    # Priority 6:
    # Remaining useful candidates.
    return 6


def build_enrichment_reason(row: pd.Series) -> str:
    """Return a human-readable audit reason."""

    reasons: list[str] = []

    if bool(row["bare_land_rich"]):
        reasons.append("bare_land_rich")
    elif bool(row["bare_land_moderate"]):
        reasons.append("bare_land_moderate")

    if bool(row["road_rich"]):
        reasons.append("road_rich")

    if bool(row["water_rich"]):
        reasons.append("water_rich")

    if bool(row["building_rich"]):
        reasons.append("building_rich")

    if bool(row["mixed_minority"]):
        reasons.append("mixed_minority")

    if not reasons:
        reasons.append("other_useful")

    return "+".join(reasons)


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

def select_candidates(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    useful = candidates.loc[
        candidates["useful_candidate"].astype(bool)
    ].copy()

    useful["priority_tier"] = useful.apply(
        assign_priority_tier,
        axis=1,
    )

    useful["selection_reason"] = useful.apply(
        build_enrichment_reason,
        axis=1,
    )

    # Primary ordering:
    # priority tier first, then enrichment score.
    #
    # Within equal scores, favour bare land first, then roads,
    # water and buildings.
    useful = useful.sort_values(
        [
            "priority_tier",
            "enrichment_score",
            "bare_land_fraction_valid",
            "roads_fraction_valid",
            "water_fraction_valid",
            "buildings_fraction_valid",
            "valid_fraction",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    city_counts: dict[str, int] = defaultdict(int)

    selected_indices: list[int] = []

    for index, row in useful.iterrows():

        city_id = str(row["city_id"])

        if city_counts[city_id] >= MAX_TILES_PER_CITY:
            continue

        selected_indices.append(index)

        city_counts[city_id] += 1

        if len(selected_indices) >= TARGET_TILE_COUNT:
            break

    selected = useful.loc[
        selected_indices
    ].copy()

    selected["selected_v22"] = True

    selected["selection_rank"] = np.arange(
        1,
        len(selected) + 1,
    )

    useful["selected_v22"] = useful.index.isin(
        selected_indices
    )

    return useful, selected


# ---------------------------------------------------------------------
# Projected class distribution
# ---------------------------------------------------------------------

def load_v21_pixel_counts() -> dict[str, int]:

    if not V21_DISTRIBUTION_PATH.exists():
        raise FileNotFoundError(
            "Previous V2.2 projected distribution file not found:\n"
            f"{V21_DISTRIBUTION_PATH}"
        )

    distribution = pd.read_csv(
        V21_DISTRIBUTION_PATH
    )

    required = {
        "class_name",
        "v21_pixels",
    }

    missing = required.difference(
        distribution.columns
    )

    if missing:
        raise ValueError(
            f"Missing V2.1 distribution columns: {sorted(missing)}"
        )

    return {
        str(row["class_name"]): int(row["v21_pixels"])
        for _, row in distribution.iterrows()
    }


def build_projected_distribution(
    selected: pd.DataFrame,
    baseline_counts: dict[str, int],
) -> pd.DataFrame:

    added_counts: dict[str, int] = {}

    for class_name in CLASS_NAMES:

        column = f"{class_name}_pixel_count"

        if column not in selected.columns:
            raise ValueError(
                f"Selected candidates missing column: {column}"
            )

        added_counts[class_name] = int(
            selected[column].sum()
        )

    baseline_total = sum(
        baseline_counts.values()
    )

    added_total = sum(
        added_counts.values()
    )

    projected_total = (
        baseline_total + added_total
    )

    rows = []

    for class_name in CLASS_NAMES:

        before = int(
            baseline_counts[class_name]
        )

        added = int(
            added_counts[class_name]
        )

        after = before + added

        rows.append(
            {
                "class_name": class_name,
                "v21_pixels": before,
                "v22_added_pixels": added,
                "projected_v22_pixels": after,
                "v21_fraction": (
                    before / baseline_total
                ),
                "projected_v22_fraction": (
                    after / projected_total
                ),
                "relative_pixel_increase": (
                    added / before
                    if before > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summary reports
# ---------------------------------------------------------------------

def build_city_summary(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    return (
        selected
        .groupby(
            ["city_id", "city_name"],
            as_index=False,
        )
        .agg(
            selected_tiles=(
                "candidate_id",
                "count",
            ),
            bare_land_rich_tiles=(
                "bare_land_rich",
                "sum",
            ),
            bare_land_moderate_tiles=(
                "bare_land_moderate",
                "sum",
            ),
            road_rich_tiles=(
                "road_rich",
                "sum",
            ),
            water_rich_tiles=(
                "water_rich",
                "sum",
            ),
            building_rich_tiles=(
                "building_rich",
                "sum",
            ),
            mixed_minority_tiles=(
                "mixed_minority",
                "sum",
            ),
            mean_enrichment_score=(
                "enrichment_score",
                "mean",
            ),
        )
        .sort_values(
            "city_id"
        )
        .reset_index(drop=True)
    )


def build_priority_summary(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    return (
        selected
        .groupby(
            "priority_tier",
            as_index=False,
        )
        .agg(
            selected_tiles=(
                "candidate_id",
                "count",
            )
        )
        .sort_values(
            "priority_tier"
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print()
    print("Dataset V2.2 class-aware enrichment selection")
    print("=============================================")

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate CSV not found:\n{CANDIDATE_PATH}"
        )

    candidates = pd.read_csv(
        CANDIDATE_PATH
    )

    print()
    print("Selection policy")
    print("----------------")
    print(
        f"Offset candidates: {len(candidates)}"
    )
    print(
        f"Useful candidates: "
        f"{int(candidates['useful_candidate'].sum())}"
    )
    print(
        f"Target enrichment tiles: {TARGET_TILE_COUNT}"
    )
    print(
        f"Maximum tiles per city: {MAX_TILES_PER_CITY}"
    )

    useful, selected = select_candidates(
        candidates
    )

    if len(selected) < TARGET_TILE_COUNT:
        print()
        print(
            "WARNING: Selection target could not be reached "
            "under the current city cap."
        )

    baseline_counts = load_v21_pixel_counts()

    projected_distribution = (
        build_projected_distribution(
            selected,
            baseline_counts,
        )
    )

    city_summary = build_city_summary(
        selected
    )

    priority_summary = build_priority_summary(
        selected
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_path = (
        OUTPUT_DIR
        / "dataset_v22_selected_candidates.csv"
    )

    useful_path = (
        OUTPUT_DIR
        / "dataset_v22_candidate_selection_audit.csv"
    )

    city_path = (
        OUTPUT_DIR
        / "dataset_v22_city_summary.csv"
    )

    priority_path = (
        OUTPUT_DIR
        / "dataset_v22_priority_summary.csv"
    )

    distribution_path = (
        OUTPUT_DIR
        / "dataset_v22_projected_distribution.csv"
    )

    selected.to_csv(
        selected_path,
        index=False,
    )

    useful.to_csv(
        useful_path,
        index=False,
    )

    city_summary.to_csv(
        city_path,
        index=False,
    )

    priority_summary.to_csv(
        priority_path,
        index=False,
    )

    projected_distribution.to_csv(
        distribution_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------

    print()
    print("Selection result")
    print("----------------")

    print(
        f"Selected enrichment tiles: {len(selected)}"
    )

    print(
        f"Bare-land rich: "
        f"{int(selected['bare_land_rich'].sum())}"
    )

    print(
        f"Bare-land moderate: "
        f"{int(selected['bare_land_moderate'].sum())}"
    )

    print(
        f"Road rich: "
        f"{int(selected['road_rich'].sum())}"
    )

    print(
        f"Water rich: "
        f"{int(selected['water_rich'].sum())}"
    )

    print(
        f"Building rich: "
        f"{int(selected['building_rich'].sum())}"
    )

    print(
        f"Mixed minority: "
        f"{int(selected['mixed_minority'].sum())}"
    )

    print()
    print("Priority-tier selection")
    print("-----------------------")

    print(
        priority_summary.to_string(
            index=False
        )
    )

    print()
    print("Per-city contribution")
    print("---------------------")

    print(
        city_summary[
            [
                "city_id",
                "city_name",
                "selected_tiles",
                "bare_land_rich_tiles",
                "bare_land_moderate_tiles",
                "road_rich_tiles",
                "water_rich_tiles",
                "building_rich_tiles",
                "mixed_minority_tiles",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Projected class distribution")
    print("----------------------------")

    display = projected_distribution.copy()

    display["v21_fraction"] *= 100.0
    display["projected_v22_fraction"] *= 100.0
    display["relative_pixel_increase"] *= 100.0

    print(
        display[
            [
                "class_name",
                "v21_fraction",
                "projected_v22_fraction",
                "relative_pixel_increase",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("Reports written")
    print("---------------")
    print(selected_path)
    print(useful_path)
    print(city_path)
    print(priority_path)
    print(distribution_path)

    print()
    print(
        "NOTE: Selection simulation only. "
        "No raster tiles have been generated."
    )


if __name__ == "__main__":
    main()