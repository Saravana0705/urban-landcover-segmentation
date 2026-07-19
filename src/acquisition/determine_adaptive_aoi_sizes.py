from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box, shape


LOGGER = logging.getLogger(__name__)

CITIES_FILE = Path("config/cities.csv")
RAW_STAC_DIRECTORY = Path("metadata/raw_stac_responses")

OUTPUT_FILE = Path(
    "metadata/adaptive_aoi_size_results.csv"
)

UPDATED_CITIES_FILE = Path(
    "config/cities_adaptive.csv"
)

TARGET_CITY_IDS = {
    "DE05",
    "DE14",
    "DE20",
}

CANDIDATE_SIZES_KM = [
    30,
    28,
    26,
    24,
    22,
    20,
]

MINIMUM_COVERAGE = 0.99
REQUIRED_PLATFORM = "sentinel-1a"
REQUIRED_ORBIT_DIRECTION = "descending"
REQUIRED_INSTRUMENT_MODE = "IW"
REQUIRED_POLARISATIONS = {"VV", "VH"}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def create_square_aoi(
    longitude: float,
    latitude: float,
    size_km: float,
    utm_epsg: int,
):
    """
    Create a square AOI centred on a city.

    The square is constructed in the local UTM CRS and then
    transformed to EPSG:4326 for comparison with STAC footprints.
    """
    wgs84 = CRS.from_epsg(4326)
    projected_crs = CRS.from_epsg(int(utm_epsg))

    transformer = Transformer.from_crs(
        wgs84,
        projected_crs,
        always_xy=True,
    )

    centre_x, centre_y = transformer.transform(
        longitude,
        latitude,
    )

    half_size_m = float(size_km) * 1000.0 / 2.0

    projected_geometry = box(
        centre_x - half_size_m,
        centre_y - half_size_m,
        centre_x + half_size_m,
        centre_y + half_size_m,
    )

    projected_gdf = gpd.GeoDataFrame(
        geometry=[projected_geometry],
        crs=f"EPSG:{utm_epsg}",
    )

    return projected_gdf.to_crs(
        "EPSG:4326"
    ).geometry.iloc[0]


def calculate_coverage_fraction(
    aoi_geometry,
    scene_geometry,
) -> float:
    if aoi_geometry.is_empty or scene_geometry.is_empty:
        return 0.0

    intersection = aoi_geometry.intersection(
        scene_geometry
    )

    if intersection.is_empty:
        return 0.0

    return float(
        intersection.area / aoi_geometry.area
    )


def normalize_polarisations(
    value: Any,
) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, list):
        return {
            str(item).strip().upper()
            for item in value
        }

    return {
        item.strip().upper()
        for item in str(value).split(",")
        if item.strip()
    }


def is_valid_scene(
    feature: dict[str, Any],
) -> bool:
    properties = feature.get("properties", {})

    platform = str(
        properties.get("platform", "")
    ).strip().lower()

    orbit_direction = str(
        properties.get("sat:orbit_state", "")
    ).strip().lower()

    instrument_mode = str(
        properties.get("sar:instrument_mode", "")
    ).strip().upper()

    polarisations = normalize_polarisations(
        properties.get("sar:polarizations")
    )

    return (
        platform == REQUIRED_PLATFORM
        and orbit_direction
        == REQUIRED_ORBIT_DIRECTION
        and instrument_mode
        == REQUIRED_INSTRUMENT_MODE
        and REQUIRED_POLARISATIONS.issubset(
            polarisations
        )
    )


def load_stac_features(
    city_id: str,
) -> list[dict[str, Any]]:
    path = (
        RAW_STAC_DIRECTORY
        / f"{city_id}_stac_features.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"STAC response not found for {city_id}: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        features = json.load(file)

    if not isinstance(features, list):
        raise ValueError(
            f"Unexpected STAC content for {city_id}."
        )

    return features


def evaluate_city(
    city: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    city_id = str(city["city_id"])
    city_name = str(city["city_name"])

    features = load_stac_features(city_id)

    valid_features = [
        feature
        for feature in features
        if is_valid_scene(feature)
        and feature.get("geometry")
    ]

    if not valid_features:
        raise RuntimeError(
            f"No valid Sentinel-1A descending IW VV/VH "
            f"features found for {city_id}."
        )

    result_rows: list[dict[str, Any]] = []
    selected_result: dict[str, Any] | None = None

    for size_km in CANDIDATE_SIZES_KM:
        aoi_geometry = create_square_aoi(
            longitude=float(city["center_lon"]),
            latitude=float(city["center_lat"]),
            size_km=float(size_km),
            utm_epsg=int(city["utm_epsg"]),
        )

        scene_results: list[dict[str, Any]] = []

        for feature in valid_features:
            scene_geometry = shape(
                feature["geometry"]
            )

            coverage_fraction = (
                calculate_coverage_fraction(
                    aoi_geometry,
                    scene_geometry,
                )
            )

            properties = feature.get(
                "properties",
                {},
            )

            scene_results.append(
                {
                    "stac_item_id": feature.get("id"),
                    "acquisition_datetime":
                        properties.get("datetime")
                        or properties.get(
                            "start_datetime"
                        ),
                    "relative_orbit":
                        properties.get(
                            "sat:relative_orbit"
                        ),
                    "absolute_orbit":
                        properties.get(
                            "sat:absolute_orbit"
                        ),
                    "incidence_angle":
                        properties.get(
                            "view:incidence_angle"
                        ),
                    "coverage_fraction":
                        coverage_fraction,
                    "coverage_percent":
                        coverage_fraction * 100.0,
                }
            )

        scene_results_df = pd.DataFrame(
            scene_results
        ).sort_values(
            by=[
                "coverage_fraction",
                "acquisition_datetime",
            ],
            ascending=[False, True],
        )

        best_scene = scene_results_df.iloc[0]

        size_result = {
            "city_id": city_id,
            "city_name": city_name,
            "tested_aoi_size_km": size_km,
            "best_stac_item_id":
                best_scene["stac_item_id"],
            "best_acquisition_datetime":
                best_scene[
                    "acquisition_datetime"
                ],
            "best_relative_orbit":
                best_scene["relative_orbit"],
            "best_absolute_orbit":
                best_scene["absolute_orbit"],
            "best_incidence_angle":
                best_scene["incidence_angle"],
            "best_coverage_fraction":
                best_scene[
                    "coverage_fraction"
                ],
            "best_coverage_percent":
                best_scene["coverage_percent"],
            "passes_minimum_coverage":
                bool(
                    best_scene[
                        "coverage_fraction"
                    ]
                    >= MINIMUM_COVERAGE
                ),
        }

        result_rows.append(size_result)

        LOGGER.info(
            "%s | %s km | best coverage %.2f%%",
            city_id,
            size_km,
            best_scene["coverage_percent"],
        )

        if (
            selected_result is None
            and size_result[
                "passes_minimum_coverage"
            ]
        ):
            selected_result = size_result

    if selected_result is None:
        raise RuntimeError(
            f"No tested AOI size achieved "
            f"{MINIMUM_COVERAGE * 100:.0f}% coverage "
            f"for {city_id}. Consider testing AOIs "
            f"smaller than {min(CANDIDATE_SIZES_KM)} km."
        )

    return result_rows, selected_result


def main() -> None:
    configure_logging()

    cities = pd.read_csv(CITIES_FILE)

    required_columns = {
        "city_id",
        "city_name",
        "aoi_size_km",
        "utm_epsg",
        "center_lat",
        "center_lon",
    }

    missing_columns = required_columns.difference(
        cities.columns
    )

    if missing_columns:
        raise ValueError(
            "cities.csv is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    detailed_results: list[dict[str, Any]] = []
    selected_results: list[dict[str, Any]] = []

    target_cities = cities.loc[
        cities["city_id"].isin(
            TARGET_CITY_IDS
        )
    ].copy()

    if len(target_cities) != len(
        TARGET_CITY_IDS
    ):
        found_ids = set(
            target_cities["city_id"]
        )

        missing_ids = (
            TARGET_CITY_IDS - found_ids
        )

        raise ValueError(
            "Target cities missing from registry: "
            + ", ".join(sorted(missing_ids))
        )

    for _, city in target_cities.iterrows():
        city_results, selected_result = (
            evaluate_city(city)
        )

        detailed_results.extend(
            city_results
        )

        selected_results.append(
            selected_result
        )

    results_df = pd.DataFrame(
        detailed_results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    selected_df = pd.DataFrame(
        selected_results
    )

    selected_size_mapping = dict(
        zip(
            selected_df["city_id"],
            selected_df[
                "tested_aoi_size_km"
            ],
        )
    )

    updated_cities = cities.copy()

    updated_cities[
        "original_aoi_size_km"
    ] = updated_cities["aoi_size_km"]

    updated_cities[
        "effective_aoi_size_km"
    ] = updated_cities["city_id"].map(
        selected_size_mapping
    ).fillna(
        updated_cities["aoi_size_km"]
    )

    updated_cities["aoi_adapted"] = (
        updated_cities[
            "effective_aoi_size_km"
        ]
        != updated_cities[
            "original_aoi_size_km"
        ]
    )

    # The AOI-generation script currently reads aoi_size_km.
    # Replace it with the selected effective size in the new file.
    updated_cities["aoi_size_km"] = (
        updated_cities[
            "effective_aoi_size_km"
        ]
    )

    updated_cities.to_csv(
        UPDATED_CITIES_FILE,
        index=False,
    )

    print("\nSelected adaptive AOI sizes:\n")

    print(
        selected_df[
            [
                "city_id",
                "city_name",
                "tested_aoi_size_km",
                "best_coverage_percent",
                "best_acquisition_datetime",
                "best_relative_orbit",
                "best_stac_item_id",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nDetailed results saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Updated city registry saved to: "
        f"{UPDATED_CITIES_FILE}"
    )


if __name__ == "__main__":
    main()