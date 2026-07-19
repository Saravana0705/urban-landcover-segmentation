from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box


LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "city_id",
    "city_name",
    "country",
    "split",
    "aoi_size_km",
    "utm_epsg",
    "center_lat",
    "center_lon",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_city_registry(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "cities.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("cities.csv contains no city records.")

    if df["city_id"].duplicated().any():
        duplicates = df.loc[df["city_id"].duplicated(), "city_id"].tolist()
        raise ValueError(f"Duplicate city_id values found: {duplicates}")

    invalid_sizes = df.loc[df["aoi_size_km"] <= 0, "city_id"].tolist()
    if invalid_sizes:
        raise ValueError(
            f"AOI size must be greater than zero for: {invalid_sizes}"
        )

    invalid_latitudes = df.loc[
        ~df["center_lat"].between(-90, 90), "city_id"
    ].tolist()

    invalid_longitudes = df.loc[
        ~df["center_lon"].between(-180, 180), "city_id"
    ].tolist()

    if invalid_latitudes:
        raise ValueError(
            f"Invalid latitude values for: {invalid_latitudes}"
        )

    if invalid_longitudes:
        raise ValueError(
            f"Invalid longitude values for: {invalid_longitudes}"
        )


def create_projected_aoi(
    longitude: float,
    latitude: float,
    aoi_size_km: float,
    utm_epsg: int,
):
    """
    Create a square AOI centred on longitude/latitude.

    The square is constructed in the city's projected UTM CRS so that
    its dimensions are defined accurately in metres.
    """
    source_crs = CRS.from_epsg(4326)
    target_crs = CRS.from_epsg(int(utm_epsg))

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    centre_x, centre_y = transformer.transform(
        longitude,
        latitude,
    )

    half_size_m = (float(aoi_size_km) * 1000.0) / 2.0

    return box(
        centre_x - half_size_m,
        centre_y - half_size_m,
        centre_x + half_size_m,
        centre_y + half_size_m,
    )


def generate_aois(
    cities_csv: Path,
    output_directory: Path,
    combined_output: Path,
) -> None:
    cities = pd.read_csv(cities_csv)
    validate_city_registry(cities)

    output_directory.mkdir(parents=True, exist_ok=True)
    combined_output.parent.mkdir(parents=True, exist_ok=True)

    combined_records: list[gpd.GeoDataFrame] = []

    for row in cities.itertuples(index=False):
        city_id = str(row.city_id)
        city_name = str(row.city_name)
        utm_epsg = int(row.utm_epsg)

        geometry = create_projected_aoi(
            longitude=float(row.center_lon),
            latitude=float(row.center_lat),
            aoi_size_km=float(row.aoi_size_km),
            utm_epsg=utm_epsg,
        )

        city_gdf = gpd.GeoDataFrame(
            {
                "city_id": [city_id],
                "city_name": [city_name],
                "country": [row.country],
                "split": [row.split],
                "aoi_size_km": [float(row.aoi_size_km)],
                "utm_epsg": [utm_epsg],
                "center_lat": [float(row.center_lat)],
                "center_lon": [float(row.center_lon)],
            },
            geometry=[geometry],
            crs=f"EPSG:{utm_epsg}",
        )

        city_output = output_directory / f"{city_id}_{city_name}_aoi.gpkg"

        city_gdf.to_file(
            city_output,
            layer="aoi",
            driver="GPKG",
        )

        LOGGER.info(
            "Created AOI for %s in EPSG:%s: %s",
            city_name,
            utm_epsg,
            city_output,
        )

        # Combined catalogue-search layer must use EPSG:4326.
        city_wgs84 = city_gdf.to_crs("EPSG:4326")
        combined_records.append(city_wgs84)

    combined_gdf = gpd.GeoDataFrame(
        pd.concat(combined_records, ignore_index=True),
        crs="EPSG:4326",
    )

    combined_gdf.to_file(
        combined_output,
        layer="city_aois",
        driver="GPKG",
    )

    geojson_output = combined_output.with_suffix(".geojson")

    combined_gdf.to_file(
        geojson_output,
        driver="GeoJSON",
    )

    summary_output = combined_output.parent / "city_aoi_summary.csv"

    summary_df = combined_gdf.drop(columns="geometry").copy()

    summary_df["min_lon"] = combined_gdf.bounds["minx"]
    summary_df["min_lat"] = combined_gdf.bounds["miny"]
    summary_df["max_lon"] = combined_gdf.bounds["maxx"]
    summary_df["max_lat"] = combined_gdf.bounds["maxy"]

    summary_df.to_csv(summary_output, index=False)

    LOGGER.info("Created combined AOI file: %s", combined_output)
    LOGGER.info("Created combined GeoJSON: %s", geojson_output)
    LOGGER.info("Created AOI summary: %s", summary_output)
    LOGGER.info("Successfully generated %d city AOIs.", len(combined_gdf))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standardized square AOIs for all project cities."
    )

    parser.add_argument(
        "--cities",
        type=Path,
        default=Path("config/cities.csv"),
        help="Path to the city registry CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aoi/cities"),
        help="Directory for individual AOI GeoPackages.",
    )

    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("data/aoi/all_city_aois.gpkg"),
        help="Combined EPSG:4326 GeoPackage output.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    generate_aois(
        cities_csv=args.cities,
        output_directory=args.output_dir,
        combined_output=args.combined_output,
    )


if __name__ == "__main__":
    main()