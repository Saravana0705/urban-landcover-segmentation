from __future__ import annotations

from pathlib import Path

import pandas as pd


SELECTED_SCENES = Path("metadata/selected_scenes.csv")

EXPECTED_CITY_COUNT = 20
MINIMUM_COVERAGE_PERCENT = 99.0

REQUIRED_PLATFORM = "sentinel-1a"
REQUIRED_INSTRUMENT_MODE = "IW"
PREFERRED_ORBIT_DIRECTION = "descending"

ALLOWED_ASCENDING_CITIES = {
    "DE05",  # Munich
    "DE14",  # Berlin
    "DE20",  # Rostock
}

REQUIRED_COLUMNS = {
    "city_id",
    "city_name",
    "stac_item_id",
    "acquisition_datetime",
    "platform",
    "instrument_mode",
    "orbit_direction",
    "relative_orbit",
    "coverage_percent",
    "has_vv_vh",
    "product_download_url",
    "product_filename",
    "product_size_bytes",
    "product_size_gb",
    "product_checksum",
}


def to_boolean(series: pd.Series) -> pd.Series:
    """
    Convert common CSV boolean representations into real booleans.
    """
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


def validate_required_columns(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(
        scenes.columns
    )

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def validate_city_count(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    if len(scenes) != EXPECTED_CITY_COUNT:
        errors.append(
            f"Expected {EXPECTED_CITY_COUNT} selected scenes, "
            f"but found {len(scenes)}."
        )


def validate_unique_cities(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    duplicated_cities = scenes.loc[
        scenes["city_id"].duplicated(),
        "city_id",
    ].tolist()

    if duplicated_cities:
        errors.append(
            f"Duplicate city selections: {duplicated_cities}"
        )


def validate_platform(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    invalid_platforms = scenes.loc[
        scenes["platform"]
        .astype(str)
        .str.strip()
        .str.lower()
        != REQUIRED_PLATFORM,
        [
            "city_id",
            "city_name",
            "platform",
        ],
    ]

    if not invalid_platforms.empty:
        errors.append(
            "Non-Sentinel-1A selections:\n"
            + invalid_platforms.to_string(index=False)
        )


def validate_instrument_mode(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    invalid_modes = scenes.loc[
        scenes["instrument_mode"]
        .astype(str)
        .str.strip()
        .str.upper()
        != REQUIRED_INSTRUMENT_MODE,
        [
            "city_id",
            "city_name",
            "instrument_mode",
        ],
    ]

    if not invalid_modes.empty:
        errors.append(
            "Non-IW selections:\n"
            + invalid_modes.to_string(index=False)
        )


def validate_polarisations(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    missing_polarisations = scenes.loc[
        ~to_boolean(scenes["has_vv_vh"]),
        [
            "city_id",
            "city_name",
            "has_vv_vh",
        ],
    ]

    if not missing_polarisations.empty:
        errors.append(
            "Scenes missing required VV/VH polarisations:\n"
            + missing_polarisations.to_string(
                index=False
            )
        )


def validate_coverage(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    insufficient_coverage = scenes.loc[
        scenes["coverage_percent"]
        < MINIMUM_COVERAGE_PERCENT,
        [
            "city_id",
            "city_name",
            "coverage_percent",
        ],
    ]

    if not insufficient_coverage.empty:
        errors.append(
            "Scenes below minimum AOI coverage:\n"
            + insufficient_coverage.to_string(
                index=False
            )
        )


def validate_orbit_policy(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    normalized_direction = (
        scenes["orbit_direction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_directions = {
        "ascending",
        "descending",
    }

    invalid_direction_values = scenes.loc[
        ~normalized_direction.isin(
            allowed_directions
        ),
        [
            "city_id",
            "city_name",
            "orbit_direction",
        ],
    ]

    if not invalid_direction_values.empty:
        errors.append(
            "Invalid orbit-direction values:\n"
            + invalid_direction_values.to_string(
                index=False
            )
        )

    unexpected_non_descending = scenes.loc[
        (
            normalized_direction
            != PREFERRED_ORBIT_DIRECTION
        )
        & (
            ~scenes["city_id"].isin(
                ALLOWED_ASCENDING_CITIES
            )
        ),
        [
            "city_id",
            "city_name",
            "orbit_direction",
        ],
    ]

    if not unexpected_non_descending.empty:
        errors.append(
            "Unexpected non-descending selections:\n"
            + unexpected_non_descending.to_string(
                index=False
            )
        )

    exception_rows = scenes.loc[
        scenes["city_id"].isin(
            ALLOWED_ASCENDING_CITIES
        ),
        [
            "city_id",
            "city_name",
            "orbit_direction",
        ],
    ].copy()

    missing_exception_cities = (
        ALLOWED_ASCENDING_CITIES
        - set(exception_rows["city_id"])
    )

    if missing_exception_cities:
        errors.append(
            "Configured ascending exception cities are "
            "missing from selected scenes: "
            + ", ".join(
                sorted(missing_exception_cities)
            )
        )

    incorrect_exception_directions = (
        exception_rows.loc[
            exception_rows["orbit_direction"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "ascending"
        ]
    )

    if not incorrect_exception_directions.empty:
        errors.append(
            "Configured ascending exception cities "
            "are not using ascending scenes:\n"
            + incorrect_exception_directions.to_string(
                index=False
            )
        )

    actual_ascending_cities = set(
        scenes.loc[
            normalized_direction == "ascending",
            "city_id",
        ]
    )

    unexpected_ascending_cities = (
        actual_ascending_cities
        - ALLOWED_ASCENDING_CITIES
    )

    if unexpected_ascending_cities:
        errors.append(
            "Ascending scenes found for unapproved cities: "
            + ", ".join(
                sorted(
                    unexpected_ascending_cities
                )
            )
        )


def validate_download_metadata(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    required_download_columns = [
        "product_download_url",
        "product_filename",
        "product_size_bytes",
        "product_checksum",
    ]

    for column in required_download_columns:
        missing_values = scenes.loc[
            scenes[column].isna()
            | (
                scenes[column]
                .astype(str)
                .str.strip()
                == ""
            ),
            [
                "city_id",
                "city_name",
            ],
        ]

        if not missing_values.empty:
            errors.append(
                f"Missing {column} for:\n"
                + missing_values.to_string(
                    index=False
                )
            )


def validate_positive_file_sizes(
    scenes: pd.DataFrame,
    errors: list[str],
) -> None:
    invalid_sizes = scenes.loc[
        pd.to_numeric(
            scenes["product_size_bytes"],
            errors="coerce",
        ).fillna(0)
        <= 0,
        [
            "city_id",
            "city_name",
            "product_size_bytes",
        ],
    ]

    if not invalid_sizes.empty:
        errors.append(
            "Invalid product file sizes:\n"
            + invalid_sizes.to_string(
                index=False
            )
        )


def validate_acquisition_dates(
    scenes: pd.DataFrame,
    errors: list[str],
) -> pd.Series:
    acquisition_dates = pd.to_datetime(
        scenes["acquisition_datetime"],
        utc=True,
        errors="coerce",
    )

    invalid_dates = scenes.loc[
        acquisition_dates.isna(),
        [
            "city_id",
            "city_name",
            "acquisition_datetime",
        ],
    ]

    if not invalid_dates.empty:
        errors.append(
            "Invalid acquisition datetimes:\n"
            + invalid_dates.to_string(
                index=False
            )
        )

    return acquisition_dates


def print_summary(
    scenes: pd.DataFrame,
    acquisition_dates: pd.Series,
) -> None:
    total_size_gb = pd.to_numeric(
        scenes["product_size_gb"],
        errors="coerce",
    ).sum()

    direction_counts = (
        scenes["orbit_direction"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )

    platform_counts = (
        scenes["platform"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )

    mode_counts = (
        scenes["instrument_mode"]
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
        .to_dict()
    )

    ascending_cities = scenes.loc[
        scenes["orbit_direction"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "ascending",
        [
            "city_id",
            "city_name",
        ],
    ]

    print("Selected-scene validation passed.")
    print(f"Cities: {len(scenes)}")
    print(f"Orbit directions: {direction_counts}")
    print(f"Platforms: {platform_counts}")
    print(f"Instrument modes: {mode_counts}")

    if acquisition_dates.notna().any():
        print(
            "Acquisition-date range:",
            acquisition_dates.min().isoformat(),
            "to",
            acquisition_dates.max().isoformat(),
        )

    print(
        f"Estimated total download size: "
        f"{total_size_gb:.2f} GB"
    )

    print(
        f"Minimum AOI coverage: "
        f"{scenes['coverage_percent'].min():.2f}%"
    )

    if not ascending_cities.empty:
        print("Approved ascending exceptions:")

        for row in ascending_cities.itertuples(
            index=False
        ):
            print(
                f"  {row.city_id} — {row.city_name}"
            )


def main() -> None:
    if not SELECTED_SCENES.exists():
        raise FileNotFoundError(
            f"Selected-scenes file not found: "
            f"{SELECTED_SCENES}"
        )

    scenes = pd.read_csv(
        SELECTED_SCENES
    )

    errors: list[str] = []

    validate_required_columns(
        scenes=scenes,
        errors=errors,
    )

    if errors:
        print("\nVALIDATION FAILED\n")

        for error in errors:
            print(error)
            print("-" * 70)

        raise SystemExit(1)

    validate_city_count(
        scenes=scenes,
        errors=errors,
    )

    validate_unique_cities(
        scenes=scenes,
        errors=errors,
    )

    validate_platform(
        scenes=scenes,
        errors=errors,
    )

    validate_instrument_mode(
        scenes=scenes,
        errors=errors,
    )

    validate_polarisations(
        scenes=scenes,
        errors=errors,
    )

    validate_coverage(
        scenes=scenes,
        errors=errors,
    )

    validate_orbit_policy(
        scenes=scenes,
        errors=errors,
    )

    validate_download_metadata(
        scenes=scenes,
        errors=errors,
    )

    validate_positive_file_sizes(
        scenes=scenes,
        errors=errors,
    )

    acquisition_dates = (
        validate_acquisition_dates(
            scenes=scenes,
            errors=errors,
        )
    )

    if errors:
        print("\nVALIDATION FAILED\n")

        for error in errors:
            print(error)
            print("-" * 70)

        raise SystemExit(1)

    print_summary(
        scenes=scenes,
        acquisition_dates=acquisition_dates,
    )


if __name__ == "__main__":
    main()