from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "reports/dataset_v1_enrichment_analysis/ranked_training_tiles.csv"
)

OUTPUT_DIR = Path(
    "reports/dataset_v1_label_qa"
)

OUTPUT_CSV = OUTPUT_DIR / "label_qa_tiles.csv"


def load_candidates() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "tile_id",
        "city_id",
        "city_name",
        "buildings_fraction",
        "roads_fraction",
        "vegetation_fraction",
        "bare_land_fraction",
        "water_fraction",
        "enrichment_score",
        "image_path",
        "semantic_mask_path",
        "validity_mask_path",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def add_group(
    selected: list[pd.DataFrame],
    df: pd.DataFrame,
    condition: pd.Series,
    sort_columns: list[str],
    n: int,
    qa_group: str,
    ascending: bool | list[bool] = False,
) -> None:
    subset = (
        df.loc[condition]
        .sort_values(
            sort_columns,
            ascending=ascending,
        )
        .copy()
    )

    if subset.empty:
        return

    # Avoid duplicates already selected.
    already_selected = set()

    for frame in selected:
        if not frame.empty:
            already_selected.update(
                frame["tile_id"].astype(str)
            )

    subset = subset[
        ~subset["tile_id"]
        .astype(str)
        .isin(already_selected)
    ]

    subset = subset.head(n).copy()

    if subset.empty:
        return

    subset["qa_group"] = qa_group
    selected.append(subset)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_candidates()

    selected: list[pd.DataFrame] = []

    # 1. Strong bare-land examples.
    add_group(
        selected=selected,
        df=df,
        condition=df["bare_land_fraction"] >= 0.05,
        sort_columns=[
            "bare_land_fraction",
            "enrichment_score",
        ],
        n=10,
        qa_group="bare_land_high",
        ascending=[False, False],
    )

    # 2. Moderate bare-land examples.
    add_group(
        selected=selected,
        df=df,
        condition=(
            (df["bare_land_fraction"] >= 0.01)
            & (df["bare_land_fraction"] < 0.05)
        ),
        sort_columns=[
            "bare_land_fraction",
            "enrichment_score",
        ],
        n=10,
        qa_group="bare_land_moderate",
        ascending=[False, False],
    )

    # 3. Road-rich examples.
    add_group(
        selected=selected,
        df=df,
        condition=df["roads_fraction"] >= 0.10,
        sort_columns=[
            "roads_fraction",
            "enrichment_score",
        ],
        n=5,
        qa_group="road_high",
        ascending=[False, False],
    )

    # 4. Building-rich examples.
    add_group(
        selected=selected,
        df=df,
        condition=df["buildings_fraction"] >= 0.15,
        sort_columns=[
            "buildings_fraction",
            "enrichment_score",
        ],
        n=5,
        qa_group="building_high",
        ascending=[False, False],
    )

    # 5. Mixed urban examples.
    mixed_condition = (
        (df["buildings_fraction"] >= 0.05)
        & (df["roads_fraction"] >= 0.05)
        & (
            (df["bare_land_fraction"] >= 0.01)
            | (df["water_fraction"] >= 0.05)
        )
    )

    add_group(
        selected=selected,
        df=df,
        condition=mixed_condition,
        sort_columns=["enrichment_score"],
        n=5,
        qa_group="mixed_urban",
        ascending=False,
    )

    # 6. Vegetation-dominated controls.
    add_group(
        selected=selected,
        df=df,
        condition=df["vegetation_fraction"] >= 0.97,
        sort_columns=["vegetation_fraction"],
        n=5,
        qa_group="vegetation_control",
        ascending=False,
    )

    if not selected:
        raise RuntimeError(
            "No QA samples could be selected."
        )

    qa = pd.concat(
        selected,
        ignore_index=True,
    )

    preferred_columns = [
        "qa_group",
        "tile_id",
        "city_id",
        "city_name",
        "buildings_fraction",
        "roads_fraction",
        "vegetation_fraction",
        "bare_land_fraction",
        "water_fraction",
        "enrichment_score",
        "image_path",
        "semantic_mask_path",
        "validity_mask_path",
    ]

    qa = qa[
        [
            column
            for column in preferred_columns
            if column in qa.columns
        ]
    ]

    qa.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nDataset V1 label-QA selection")
    print("-----------------------------")

    print(f"Selected tiles: {len(qa)}")

    print("\nQA groups")
    print(
        qa["qa_group"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCities represented")
    print(
        qa["city_name"]
        .value_counts()
        .to_string()
    )

    print("\nSelected tiles")
    print(
        qa[
            [
                "qa_group",
                "tile_id",
                "city_name",
                "buildings_fraction",
                "roads_fraction",
                "bare_land_fraction",
                "water_fraction",
                "vegetation_fraction",
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\nQA manifest written to:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()