from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
import yaml
from shapely.geometry import shape
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return data


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ",".join(str(item) for item in value).upper()

    return str(value).upper()


def get_property(
    properties: dict[str, Any],
    possible_names: list[str],
) -> Any:
    for name in possible_names:
        if name in properties:
            return properties[name]

    return None


def calculate_coverage_fraction(
    aoi_geometry,
    scene_geometry,
) -> float:
    if aoi_geometry.is_empty or scene_geometry.is_empty:
        return 0.0

    intersection = aoi_geometry.intersection(scene_geometry)

    if intersection.is_empty:
        return 0.0

    return float(intersection.area / aoi_geometry.area)


def calculate_temporal_score(
    acquisition_datetime: datetime,
    target_datetime: datetime,
) -> float:
    difference_days = abs(
        (acquisition_datetime - target_datetime).total_seconds()
    ) / 86400.0

    return max(0.0, 1.0 - difference_days / 183.0)


def search_stac(
    session: requests.Session,
    search_url: str,
    geometry_geojson: dict[str, Any],
    start_date: str,
    end_date: str,
    collection: str,
    page_size: int,
    maximum_total_results: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve all matching STAC items by following pagination links.
    """
    payload = {
        "collections": [collection],
        "datetime": (
            f"{start_date}T00:00:00Z/"
            f"{end_date}T23:59:59Z"
        ),
        "intersects": geometry_geojson,
        "limit": page_size,
    }

    all_features: list[dict[str, Any]] = []
    next_url: str | None = search_url
    next_method = "POST"
    next_body: dict[str, Any] | None = payload

    while next_url:
        if next_method.upper() == "POST":
            response = session.post(
                next_url,
                json=next_body,
                timeout=120,
            )
        else:
            response = session.get(
                next_url,
                timeout=120,
            )

        response.raise_for_status()
        result = response.json()

        features = result.get("features", [])

        if not isinstance(features, list):
            raise ValueError(
                "Unexpected STAC response: 'features' is not a list."
            )

        all_features.extend(features)

        if (
            maximum_total_results is not None
            and len(all_features) >= maximum_total_results
        ):
            all_features = all_features[:maximum_total_results]
            break

        next_link = next(
            (
                link
                for link in result.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )

        if not next_link:
            break

        next_url = next_link.get("href")
        if not next_url:
            break

        next_method = str(next_link.get("method", "GET")).upper()
        next_body = next_link.get("body")

        if next_method == "POST" and next_body is None:
            next_body = payload

    return all_features


def parse_stac_feature(
    feature: dict[str, Any],
    city_record: pd.Series,
    aoi_geometry,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    properties = feature.get("properties", {})
    geometry_json = feature.get("geometry")

    if not geometry_json:
        return None

    scene_geometry = shape(geometry_json)

    coverage_fraction = calculate_coverage_fraction(
        aoi_geometry,
        scene_geometry,
    )

    instrument_mode = get_property(
        properties,
        [
            "sar:instrument_mode",
            "instrument_mode",
            "instrumentMode",
        ],
    )

    orbit_direction = get_property(
        properties,
        [
            "sat:orbit_state",
            "orbit_direction",
            "orbitDirection",
        ],
    )

    polarisations = get_property(
        properties,
        [
            "sar:polarizations",
            "polarizations",
            "polarisationChannels",
        ],
    )

    relative_orbit = get_property(
        properties,
        [
            "sat:relative_orbit",
            "relative_orbit",
            "relativeOrbitNumber",
        ],
    )

    absolute_orbit = get_property(
        properties,
        [
            "sat:absolute_orbit",
            "absolute_orbit",
            "absoluteOrbitNumber",
        ],
    )

    platform = get_property(
        properties,
        [
            "platform",
            "constellation",
        ],
    )

    incidence_angle = get_property(
        properties,
        ["view:incidence_angle"],
    )

    product_type = get_property(
        properties,
        ["product:type"],
    )

    datetime_text = (
        properties.get("datetime")
        or properties.get("start_datetime")
    )

    acquisition_datetime: datetime | None = None

    if datetime_text:
        acquisition_datetime = datetime.fromisoformat(
            datetime_text.replace("Z", "+00:00")
        )

    mode_text = normalize_text(instrument_mode)
    orbit_text = normalize_text(orbit_direction)
    polarisation_text = normalize_text(polarisations)

    required_polarisations = {
        value.upper()
        for value in config["search"]["required_polarisations"]
    }

    available_polarisations = {
        item.strip()
        for item in polarisation_text
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .split(",")
        if item.strip()
    }

    has_required_polarisations = required_polarisations.issubset(
        available_polarisations
    )

    preferred_mode = normalize_text(
        config["search"]["preferred_instrument_mode"]
    )

    preferred_orbit = normalize_text(
        config["search"]["preferred_orbit_direction"]
    )

    required_platform = normalize_text(
        config["search"]["required_platform"]
    )

    mode_matches = preferred_mode in mode_text
    orbit_matches = preferred_orbit in orbit_text
    platform_matches = (
        normalize_text(platform) == required_platform
    )

    target_year = (
        acquisition_datetime.year
        if acquisition_datetime
        else int(config["search"]["start_date"][:4])
    )

    target_datetime = datetime(
        year=target_year,
        month=int(config["selection"]["target_month"]),
        day=int(config["selection"]["target_day"]),
        tzinfo=timezone.utc,
    )

    temporal_score = (
        calculate_temporal_score(
            acquisition_datetime,
            target_datetime,
        )
        if acquisition_datetime
        else 0.0
    )

    score = (
        float(config["selection"]["coverage_weight"])
        * coverage_fraction
        + float(config["selection"]["temporal_weight"])
        * temporal_score
        + float(config["selection"]["orbit_weight"])
        * float(orbit_matches)
        + float(config["selection"]["mode_weight"])
        * float(mode_matches)
    )

    assets = feature.get("assets", {})

    product_asset = assets.get("Product", {})
    vv_asset = assets.get("vv", {})
    vh_asset = assets.get("vh", {})

    product_download_url = product_asset.get("href")
    product_size_bytes = product_asset.get("file:size")
    product_checksum = product_asset.get("file:checksum")
    product_filename = product_asset.get("file:local_path")

    vv_download_url = (
        vv_asset.get("alternate", {})
        .get("https", {})
        .get("href")
    )

    vh_download_url = (
        vh_asset.get("alternate", {})
        .get("https", {})
        .get("href")
    )

    return {
        "city_id": city_record["city_id"],
        "city_name": city_record["city_name"],
        "split": city_record["split"],
        "stac_item_id": feature.get("id"),
        "collection": feature.get("collection"),
        "acquisition_datetime": datetime_text,
        "platform": platform,
        "platform_matches": platform_matches,
        "instrument_mode": instrument_mode,
        "mode_matches": mode_matches,
        "orbit_direction": orbit_direction,
        "orbit_matches": orbit_matches,
        "relative_orbit": relative_orbit,
        "absolute_orbit": absolute_orbit,
        "incidence_angle": incidence_angle,
        "product_type": product_type,
        "polarisations": json.dumps(
            polarisations,
            ensure_ascii=False,
        ),
        "has_vv_vh": has_required_polarisations,
        "coverage_fraction": coverage_fraction,
        "coverage_percent": coverage_fraction * 100.0,
        "temporal_score": temporal_score,
        "selection_score": score,
        "product_download_url": product_download_url,
        "product_size_bytes": product_size_bytes,
        "product_size_gb": (
            float(product_size_bytes) / 1_000_000_000
            if product_size_bytes
            else None
        ),
        "product_checksum": product_checksum,
        "product_filename": product_filename,
        "vv_download_url": vv_download_url,
        "vh_download_url": vh_download_url,
    }


def select_best_scene(
    candidates: pd.DataFrame,
    minimum_coverage: float,
) -> pd.DataFrame:
    eligible = candidates.loc[
        candidates["has_vv_vh"].astype(bool)
        & candidates["mode_matches"].astype(bool)
        & candidates["platform_matches"].astype(bool)
        & (
            candidates["coverage_fraction"]
            >= minimum_coverage
        )
    ].copy()

    if eligible.empty:
        return eligible

    preferred_orbit = eligible.loc[
        eligible["orbit_matches"].astype(bool)
    ].copy()

    if not preferred_orbit.empty:
        eligible = preferred_orbit

    
    eligible = eligible.sort_values(
        by=[
            "selection_score",
            "coverage_fraction",
            "acquisition_datetime",
        ],
        ascending=[False, False, True],
    )

    return eligible.head(1)


def run_catalogue_search(
    cities_csv: Path,
    aois_file: Path,
    config_path: Path,
    candidates_output: Path,
    selected_output: Path,
    raw_output_directory: Path,
) -> None:
    cities = pd.read_csv(cities_csv)
    aois = gpd.read_file(aois_file)

    if aois.crs is None:
        raise ValueError("The AOI dataset has no CRS.")

    aois = aois.to_crs("EPSG:4326")
    config = load_yaml(config_path)

    stac_base_url = config["catalog"]["stac_url"].rstrip("/")
    search_url = f"{stac_base_url}/search"
    collection = config["catalog"]["collection"]

    start_date = config["search"]["start_date"]
    end_date = config["search"]["end_date"]
    page_size = int(config["search"]["page_size"])
    minimum_coverage = float(
        config["search"]["minimum_aoi_coverage"]
    )

    candidates_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    selected_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "SRH-Urban-SAR-Segmentation/0.1 "
                "(academic research)"
            ),
            "Accept": (
                "application/geo+json, application/json"
            ),
        }
    )

    all_candidates: list[dict[str, Any]] = []
    all_selected: list[pd.DataFrame] = []

    for _, city in tqdm(
        cities.iterrows(),
        total=len(cities),
        desc="Searching Sentinel-1 scenes",
    ):
        city_id = city["city_id"]

        city_aoi = aois.loc[
            aois["city_id"] == city_id
        ]

        if city_aoi.empty:
            LOGGER.warning(
                "No AOI found for %s",
                city_id,
            )
            continue

        geometry = city_aoi.geometry.iloc[0]
        geometry_geojson = geometry.__geo_interface__

        try:
            features = search_stac(
                session=session,
                search_url=search_url,
                geometry_geojson=geometry_geojson,
                start_date=start_date,
                end_date=end_date,
                collection=collection,
                page_size=page_size,
                maximum_total_results=None,
            )
        except requests.RequestException as error:
            LOGGER.error(
                "STAC search failed for %s: %s",
                city_id,
                error,
            )
            continue

        raw_response_path = (
            raw_output_directory
            / f"{city_id}_stac_features.json"
        )

        with raw_response_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                features,
                file,
                indent=2,
                ensure_ascii=False,
            )

        city_records: list[dict[str, Any]] = []

        for feature in features:
            record = parse_stac_feature(
                feature=feature,
                city_record=city,
                aoi_geometry=geometry,
                config=config,
            )

            if record is not None:
                city_records.append(record)
                all_candidates.append(record)

        if not city_records:
            LOGGER.warning(
                "No catalogue candidates found for %s.",
                city_id,
            )
            continue

        city_candidates = pd.DataFrame(city_records)

        selected = select_best_scene(
            candidates=city_candidates,
            minimum_coverage=minimum_coverage,
        )

        if selected.empty:
            LOGGER.warning(
                "No eligible scene selected for %s.",
                city_id,
            )
        else:
            all_selected.append(selected)

            selected_id = selected.iloc[0]["stac_item_id"]
            selected_coverage = selected.iloc[0][
                "coverage_percent"
            ]

            LOGGER.info(
                "Selected %s for %s with %.2f%% coverage.",
                selected_id,
                city_id,
                selected_coverage,
            )

    candidates_df = pd.DataFrame(all_candidates)

    if candidates_df.empty:
        raise RuntimeError(
            "No Sentinel-1 candidates were returned. "
            "Check the endpoint, collection ID and date range."
        )

    candidates_df = candidates_df.sort_values(
        by=[
            "city_id",
            "selection_score",
        ],
        ascending=[True, False],
    )

    candidates_df.to_csv(
        candidates_output,
        index=False,
    )

    if all_selected:
        selected_df = pd.concat(
            all_selected,
            ignore_index=True,
        )

        selected_df = selected_df.sort_values(
            by="city_id"
        )

        selected_df.to_csv(
            selected_output,
            index=False,
        )
    else:
        LOGGER.warning(
            "No scenes met all selection requirements."
        )

    LOGGER.info(
        "Saved %d candidate scenes to %s.",
        len(candidates_df),
        candidates_output,
    )

    LOGGER.info(
        "Selected scenes saved to %s.",
        selected_output,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search Sentinel-1 GRD scenes for all configured city AOIs."
        )
    )

    parser.add_argument(
        "--cities",
        type=Path,
        default=Path("config/cities.csv"),
    )

    parser.add_argument(
        "--aois",
        type=Path,
        default=Path("data/aoi/all_city_aois.gpkg"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/acquisition.yaml"),
    )

    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=Path(
            "metadata/catalogue_candidates.csv"
        ),
    )

    parser.add_argument(
        "--selected-output",
        type=Path,
        default=Path(
            "metadata/selected_scenes.csv"
        ),
    )

    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=Path(
            "metadata/raw_stac_responses"
        ),
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    run_catalogue_search(
        cities_csv=args.cities,
        aois_file=args.aois,
        config_path=args.config,
        candidates_output=args.candidates_output,
        selected_output=args.selected_output,
        raw_output_directory=args.raw_output_dir,
    )


if __name__ == "__main__":
    main()
