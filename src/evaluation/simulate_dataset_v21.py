from pathlib import Path

import pandas as pd


INPUT_AUDIT = Path(
    "reports/dataset_v2_simulation/dataset_v2_selection_audit.csv"
)

OUTPUT_DIR = Path("reports/dataset_v21_simulation")


MIN_TILE_RETENTION = 0.70
MIN_CLASS_RETENTION = 0.80

CLASS_COLUMNS = {
    "buildings": "buildings_pixel_count",
    "roads": "roads_pixel_count",
    "bare_land": "bare_land_pixel_count",
    "water": "water_pixel_count",
}


def retention_fraction(
    selected: pd.DataFrame,
    original: pd.DataFrame,
    column: str,
) -> float:
    original_total = original[column].sum()

    if original_total <= 0:
        return 1.0

    selected_total = selected[column].sum()

    return selected_total / original_total


def city_constraints_satisfied(
    city_df: pd.DataFrame,
) -> bool:
    selected = city_df[city_df["selected_v21"]]

    tile_retention = len(selected) / len(city_df)

    if tile_retention < MIN_TILE_RETENTION:
        return False

    for column in CLASS_COLUMNS.values():
        retention = retention_fraction(
            selected=selected,
            original=city_df,
            column=column,
        )

        if retention < MIN_CLASS_RETENTION:
            return False

    return True


def calculate_restore_score(
    row: pd.Series,
    deficits: dict[str, float],
) -> float:
    """
    Rank currently removed tiles according to how useful they are
    for repairing the city's remaining V2.1 retention deficits.

    Larger score = better restoration candidate.
    """

    score = 0.0

    for class_name, column in CLASS_COLUMNS.items():
        deficit = deficits[class_name]

        if deficit > 0:
            score += (
                float(row[column])
                * deficit
            )

    # Prefer informative/mixed tiles where possible.
    if bool(row.get("v2_mixed_informative", False)):
        score *= 1.20

    if bool(row.get("minority_rich", False)):
        score *= 1.15

    # Slightly prefer tiles with more usable pixels.
    score *= max(
        float(row.get("valid_fraction", 1.0)),
        0.01,
    )

    return score


def calculate_city_deficits(
    city_df: pd.DataFrame,
) -> dict[str, float]:
    selected = city_df[city_df["selected_v21"]]

    deficits = {}

    for class_name, column in CLASS_COLUMNS.items():
        retention = retention_fraction(
            selected=selected,
            original=city_df,
            column=column,
        )

        deficits[class_name] = max(
            0.0,
            MIN_CLASS_RETENTION - retention,
        )

    return deficits


def restore_city_tiles(
    city_df: pd.DataFrame,
) -> pd.DataFrame:
    city_df = city_df.copy()

    while not city_constraints_satisfied(city_df):
        selected = city_df[city_df["selected_v21"]]
        removed = city_df[~city_df["selected_v21"]].copy()

        if removed.empty:
            break

        tile_retention = len(selected) / len(city_df)

        deficits = calculate_city_deficits(city_df)

        # If class constraints are already satisfied but tile count
        # is still too low, rank tiles by overall informativeness.
        if (
            tile_retention < MIN_TILE_RETENTION
            and all(value <= 0 for value in deficits.values())
        ):
            removed["restore_score"] = (
                removed["buildings_pixel_count"]
                + removed["roads_pixel_count"]
                + removed["bare_land_pixel_count"]
                + removed["water_pixel_count"]
            )

            removed["restore_score"] *= (
                removed["valid_fraction"].clip(lower=0.01)
            )

        else:
            removed["restore_score"] = removed.apply(
                calculate_restore_score,
                axis=1,
                deficits=deficits,
            )

        best_index = removed["restore_score"].idxmax()

        city_df.loc[best_index, "selected_v21"] = True
        city_df.loc[
            best_index,
            "v21_selection_reason",
        ] = "restored_city_retention_guard"

    return city_df


def build_city_retention_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (city_id, city_name), city_df in df.groupby(
        ["city_id", "city_name"],
        sort=True,
    ):
        selected_v2 = city_df[city_df["selected_v2"]]
        selected_v21 = city_df[city_df["selected_v21"]]

        row = {
            "city_id": city_id,
            "city_name": city_name,
            "v1_tiles": len(city_df),
            "v2_tiles": len(selected_v2),
            "v21_tiles": len(selected_v21),
            "v21_tile_retention_pct": (
                100.0 * len(selected_v21) / len(city_df)
            ),
        }

        for class_name, column in CLASS_COLUMNS.items():
            v1_pixels = city_df[column].sum()
            v2_pixels = selected_v2[column].sum()
            v21_pixels = selected_v21[column].sum()

            row[f"v1_{class_name}_pixels"] = v1_pixels
            row[f"v2_{class_name}_pixels"] = v2_pixels
            row[f"v21_{class_name}_pixels"] = v21_pixels

            if v1_pixels > 0:
                retention = (
                    100.0 * v21_pixels / v1_pixels
                )
            else:
                retention = 100.0

            row[
                f"v21_{class_name}_retention_pct"
            ] = retention

        rows.append(row)

    return pd.DataFrame(rows)


def build_dataset_summary(
    original: pd.DataFrame,
    selected_v21: pd.DataFrame,
) -> pd.DataFrame:
    class_columns = {
        "buildings": "buildings_pixel_count",
        "roads": "roads_pixel_count",
        "vegetation": "vegetation_pixel_count",
        "bare_land": "bare_land_pixel_count",
        "water": "water_pixel_count",
    }

    rows = []

    for dataset_name, frame in [
        ("dataset_v1", original),
        ("dataset_v21_candidate", selected_v21),
    ]:
        valid_pixels = frame["valid_pixel_count"].sum()

        row = {
            "dataset": dataset_name,
            "tiles": len(frame),
            "cities": frame["city_id"].nunique(),
            "valid_pixels": valid_pixels,
            "mean_valid_fraction": frame[
                "valid_fraction"
            ].mean(),
            "vegetation_ge_90pct_tiles": int(
                (
                    frame["vegetation_fraction_valid"]
                    >= 0.90
                ).sum()
            ),
            "vegetation_ge_95pct_tiles": int(
                (
                    frame["vegetation_fraction_valid"]
                    >= 0.95
                ).sum()
            ),
        }

        for class_name, column in class_columns.items():
            pixels = frame[column].sum()

            row[f"{class_name}_pixels"] = pixels

            row[f"{class_name}_pct_valid"] = (
                100.0 * pixels / valid_pixels
                if valid_pixels > 0
                else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_AUDIT)

    df = df[
        df["split"].astype(str).str.lower() == "train"
    ].copy()

    df["selected_v21"] = df["selected_v2"].astype(bool)

    df["v21_selection_reason"] = df[
        "v2_selection_reason"
    ].astype(str)

    restored_frames = []

    for _, city_df in df.groupby(
        "city_id",
        sort=True,
    ):
        restored_city = restore_city_tiles(city_df)
        restored_frames.append(restored_city)

    result = pd.concat(
        restored_frames,
        ignore_index=True,
    )

    selected_v21 = result[
        result["selected_v21"]
    ].copy()

    city_report = build_city_retention_report(result)

    summary = build_dataset_summary(
        original=result,
        selected_v21=selected_v21,
    )

    restored_tiles = result[
        result["selected_v21"]
        & ~result["selected_v2"]
    ].copy()

    audit_path = (
        OUTPUT_DIR
        / "dataset_v21_selection_audit.csv"
    )

    manifest_path = (
        OUTPUT_DIR
        / "dataset_v21_candidate_manifest.csv"
    )

    city_report_path = (
        OUTPUT_DIR
        / "dataset_v21_city_retention.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "dataset_v1_vs_v21_summary.csv"
    )

    restored_path = (
        OUTPUT_DIR
        / "dataset_v21_restored_tiles.csv"
    )

    result.to_csv(
        audit_path,
        index=False,
    )

    selected_v21.to_csv(
        manifest_path,
        index=False,
    )

    city_report.to_csv(
        city_report_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    restored_tiles.to_csv(
        restored_path,
        index=False,
    )

    print()
    print("Dataset V2.1 simulation")
    print("=======================")

    print()
    print("Retention safeguards")
    print("--------------------")
    print(
        f"Minimum tile retention per city: "
        f"{MIN_TILE_RETENTION:.0%}"
    )
    print(
        f"Minimum minority-class retention "
        f"per city: {MIN_CLASS_RETENTION:.0%}"
    )

    print()
    print("Dataset size")
    print("------------")
    print(f"Dataset V1 tiles: {len(result)}")
    print(
        f"Dataset V2 tiles: "
        f"{int(result['selected_v2'].sum())}"
    )
    print(
        f"Dataset V2.1 tiles: "
        f"{len(selected_v21)}"
    )
    print(
        f"Restored tiles: "
        f"{len(restored_tiles)}"
    )
    print(
        f"V2.1 retention: "
        f"{100.0 * len(selected_v21) / len(result):.2f}%"
    )

    print()
    print("V1 -> V2.1 class distribution")
    print("-----------------------------")

    display_columns = [
        "dataset",
        "tiles",
        "buildings_pct_valid",
        "roads_pct_valid",
        "vegetation_pct_valid",
        "bare_land_pct_valid",
        "water_pct_valid",
        "vegetation_ge_90pct_tiles",
        "vegetation_ge_95pct_tiles",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
        )
    )

    print()
    print("Per-city V2.1 retention")
    print("-----------------------")

    report_columns = [
        "city_id",
        "city_name",
        "v1_tiles",
        "v21_tiles",
        "v21_tile_retention_pct",
        "v21_buildings_retention_pct",
        "v21_roads_retention_pct",
        "v21_bare_land_retention_pct",
        "v21_water_retention_pct",
    ]

    print(
        city_report[report_columns].to_string(
            index=False,
        )
    )

    violations = city_report[
        (
            city_report["v21_tile_retention_pct"]
            < 100.0 * MIN_TILE_RETENTION
        )
        | (
            city_report[
                "v21_buildings_retention_pct"
            ]
            < 100.0 * MIN_CLASS_RETENTION
        )
        | (
            city_report[
                "v21_roads_retention_pct"
            ]
            < 100.0 * MIN_CLASS_RETENTION
        )
        | (
            city_report[
                "v21_bare_land_retention_pct"
            ]
            < 100.0 * MIN_CLASS_RETENTION
        )
        | (
            city_report[
                "v21_water_retention_pct"
            ]
            < 100.0 * MIN_CLASS_RETENTION
        )
    ]

    print()

    if violations.empty:
        print(
            "PASS: all city-level V2.1 "
            "retention safeguards satisfied."
        )
    else:
        print(
            "WARNING: some city-level safeguards "
            "could not be satisfied."
        )
        print(
            violations[report_columns].to_string(
                index=False,
            )
        )

    print()
    print(
        f"Reports written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()