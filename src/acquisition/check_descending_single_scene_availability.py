from __future__ import annotations

from pathlib import Path

import pandas as pd


CANDIDATES_FILE = Path("metadata/catalogue_candidates.csv")
OUTPUT_FILE = Path(
    "metadata/descending_single_scene_diagnostics.csv"
)

TARGET_CITIES = {
    "DE05": "Munich",
    "DE14": "Berlin",
    "DE20": "Rostock",
}

MINIMUM_COVERAGE = 0.99
REQUIRED_PLATFORM = "sentinel-1a"
REQUIRED_MODE = "IW"
REQUIRED_DIRECTION = "descending"


def to_boolean(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
    )


def main() -> None:
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {CANDIDATES_FILE}"
        )

    candidates = pd.read_csv(CANDIDATES_FILE)

    required_columns = {
        "city_id",
        "city_name",
        "stac_item_id",
        "acquisition_datetime",
        "platform",
        "instrument_mode",
        "orbit_direction",
        "relative_orbit",
        "coverage_fraction",
        "coverage_percent",
        "has_vv_vh",
        "product_download_url",
        "product_filename",
        "selection_score",
    }

    missing_columns = required_columns.difference(
        candidates.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in catalogue_candidates.csv: "
            + ", ".join(sorted(missing_columns))
        )

    candidates["has_vv_vh_bool"] = to_boolean(
        candidates["has_vv_vh"]
    )

    candidates["platform_normalized"] = (
        candidates["platform"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    candidates["mode_normalized"] = (
        candidates["instrument_mode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    candidates["direction_normalized"] = (
        candidates["orbit_direction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    diagnostic_rows: list[pd.DataFrame] = []

    for city_id, city_name in TARGET_CITIES.items():
        city_candidates = candidates.loc[
            candidates["city_id"] == city_id
        ].copy()

        valid_descending = city_candidates.loc[
            (
                city_candidates["platform_normalized"]
                == REQUIRED_PLATFORM
            )
            & (
                city_candidates["mode_normalized"]
                == REQUIRED_MODE
            )
            & (
                city_candidates["direction_normalized"]
                == REQUIRED_DIRECTION
            )
            & city_candidates["has_vv_vh_bool"]
            & (
                city_candidates["coverage_fraction"]
                >= MINIMUM_COVERAGE
            )
        ].copy()

        valid_descending = valid_descending.sort_values(
            by=[
                "coverage_fraction",
                "selection_score",
                "acquisition_datetime",
            ],
            ascending=[False, False, True],
        )

        print("=" * 80)
        print(f"{city_id} — {city_name}")
        print(
            "All catalogue candidates:",
            len(city_candidates),
        )
        print(
            "Valid Sentinel-1A descending single scenes:",
            len(valid_descending),
        )

        if valid_descending.empty:
            print(
                "RESULT: No valid descending single scene "
                "with at least 99% AOI coverage."
            )
        else:
            print(
                "RESULT: Valid descending single scenes exist."
            )

            preview_columns = [
                "stac_item_id",
                "acquisition_datetime",
                "relative_orbit",
                "coverage_percent",
                "selection_score",
                "product_filename",
            ]

            print(
                valid_descending[
                    preview_columns
                ].head(10).to_string(index=False)
            )

            valid_descending["diagnostic_rank"] = range(
                1,
                len(valid_descending) + 1,
            )

            diagnostic_rows.append(valid_descending)

    if diagnostic_rows:
        output = pd.concat(
            diagnostic_rows,
            ignore_index=True,
        )

        output_columns = [
            "city_id",
            "city_name",
            "diagnostic_rank",
            "stac_item_id",
            "acquisition_datetime",
            "platform",
            "instrument_mode",
            "orbit_direction",
            "relative_orbit",
            "absolute_orbit",
            "incidence_angle",
            "coverage_fraction",
            "coverage_percent",
            "selection_score",
            "product_download_url",
            "product_filename",
            "product_size_bytes",
            "product_size_gb",
            "product_checksum",
        ]

        available_columns = [
            column
            for column in output_columns
            if column in output.columns
        ]

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output[available_columns].to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print("=" * 80)
        print(
            f"Diagnostic results saved to: {OUTPUT_FILE}"
        )
    else:
        print("=" * 80)
        print(
            "No valid descending single scenes were found "
            "for any target city."
        )


if __name__ == "__main__":
    main()