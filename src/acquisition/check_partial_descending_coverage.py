from __future__ import annotations

from pathlib import Path

import pandas as pd


CANDIDATES_FILE = Path("metadata/catalogue_candidates.csv")
OUTPUT_FILE = Path(
    "metadata/partial_descending_candidates.csv"
)

TARGET_CITIES = ["DE05", "DE14", "DE20"]


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
    candidates = pd.read_csv(CANDIDATES_FILE)

    candidates["has_vv_vh_bool"] = to_boolean(
        candidates["has_vv_vh"]
    )

    filtered = candidates.loc[
        candidates["city_id"].isin(TARGET_CITIES)
        & (
            candidates["platform"]
            .astype(str)
            .str.lower()
            == "sentinel-1a"
        )
        & (
            candidates["instrument_mode"]
            .astype(str)
            .str.upper()
            == "IW"
        )
        & (
            candidates["orbit_direction"]
            .astype(str)
            .str.lower()
            == "descending"
        )
        & candidates["has_vv_vh_bool"]
    ].copy()

    filtered["acquisition_date"] = pd.to_datetime(
        filtered["acquisition_datetime"],
        utc=True,
    ).dt.date

    filtered = filtered.sort_values(
        by=[
            "city_id",
            "coverage_fraction",
            "acquisition_datetime",
        ],
        ascending=[True, False, True],
    )

    columns = [
        "city_id",
        "city_name",
        "stac_item_id",
        "acquisition_datetime",
        "acquisition_date",
        "relative_orbit",
        "absolute_orbit",
        "coverage_fraction",
        "coverage_percent",
        "incidence_angle",
        "product_filename",
        "product_download_url",
    ]

    filtered[columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    for city_id in TARGET_CITIES:
        city_rows = filtered.loc[
            filtered["city_id"] == city_id
        ]

        print("=" * 80)
        print(city_id)
        print(
            city_rows[
                [
                    "acquisition_datetime",
                    "relative_orbit",
                    "coverage_percent",
                    "stac_item_id",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("=" * 80)
    print(f"Saved partial candidates to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()