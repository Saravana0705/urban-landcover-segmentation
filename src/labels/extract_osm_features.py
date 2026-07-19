from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from pyrosm import OSM
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def load_city_registry(config: dict[str, Any]) -> pd.DataFrame:
    registry_path = Path(config["inputs"]["city_registry_file"])

    if not registry_path.exists():
        raise FileNotFoundError(f"City registry not found: {registry_path}")

    cities = pd.read_csv(registry_path)

    required_columns = {"city_id", "city_name", "utm_epsg"}
    missing = required_columns - set(cities.columns)

    if missing:
        raise ValueError(
            "City registry is missing columns: "
            + ", ".join(sorted(missing))
        )

    if cities["city_id"].duplicated().any():
        duplicates = cities.loc[
            cities["city_id"].duplicated(),
            "city_id",
        ].tolist()
        raise ValueError(f"Duplicate city IDs: {duplicates}")

    return cities.sort_values("city_id").reset_index(drop=True)


def load_exact_aoi(
    row: pd.Series,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, Path]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    epsg = int(row["utm_epsg"])

    aoi_path = (
        Path(config["inputs"]["city_aoi_directory"])
        / f"{city_id}_{city_name}_aoi.gpkg"
    )

    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")

    aoi = gpd.read_file(aoi_path, layer="aoi")

    if aoi.empty:
        raise ValueError(f"AOI is empty: {aoi_path}")

    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS: {aoi_path}")

    return aoi.to_crs(epsg=epsg), aoi_path.resolve()


def normalize_filter_values(values: Any) -> Any:
    if values == ["*"] or values == "*":
        return True
    return values


def load_class_mapping(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"OSM class mapping not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        mapping = yaml.safe_load(file)

    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid OSM class mapping: {path}")

    expected_classes = {
        "building",
        "road",
        "vegetation",
        "bare_land",
        "water",
    }
    missing = expected_classes - set(mapping)

    if missing:
        raise ValueError(
            "OSM class mapping is missing classes: "
            + ", ".join(sorted(missing))
        )

    return mapping


def build_custom_filter(class_config: dict[str, Any]) -> dict[str, Any]:
    include = class_config.get("include", {})

    if not isinstance(include, dict) or not include:
        raise ValueError(
            "Class include mapping must be a non-empty dictionary."
        )

    return {
        str(key): normalize_filter_values(value)
        for key, value in include.items()
    }


def empty_geodataframe(expected_epsg: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([], crs=f"EPSG:{expected_epsg}")
    )


def repair_and_clip(
    features: gpd.GeoDataFrame | None,
    aoi_projected: gpd.GeoDataFrame,
    expected_epsg: int,
    geometry_family: str,
) -> gpd.GeoDataFrame:
    if features is None or features.empty:
        return empty_geodataframe(expected_epsg)

    if features.crs is None:
        features = features.set_crs("EPSG:4326")

    features = features.to_crs(epsg=expected_epsg).copy()
    features = features.loc[
        features.geometry.notna() & ~features.geometry.is_empty
    ].copy()

    if features.empty:
        return empty_geodataframe(expected_epsg)

    features["geometry"] = features.geometry.make_valid()
    features = features.explode(index_parts=False, ignore_index=True)
    features = gpd.clip(
        features,
        aoi_projected,
        keep_geom_type=False,
    )

    if features.empty:
        return empty_geodataframe(expected_epsg)

    features = features.loc[
        features.geometry.notna() & ~features.geometry.is_empty
    ].copy()

    if features.empty:
        return empty_geodataframe(expected_epsg)

    features["geometry"] = features.geometry.make_valid()
    features = features.explode(index_parts=False, ignore_index=True)

    if geometry_family == "polygon":
        allowed = {"Polygon", "MultiPolygon"}
    elif geometry_family == "line":
        allowed = {"LineString", "MultiLineString"}
    else:
        raise ValueError(f"Unsupported geometry family: {geometry_family}")

    features = features.loc[
        features.geometry.geom_type.isin(allowed)
    ].copy()

    if features.empty:
        return empty_geodataframe(expected_epsg)

    features = features.reset_index(drop=True)
    features["feature_id"] = range(1, len(features) + 1)

    return features


def apply_exclusions(
    features: gpd.GeoDataFrame,
    class_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    exclude = class_config.get("exclude", {})

    if features.empty or not exclude:
        return features

    keep_mask = pd.Series(True, index=features.index, dtype=bool)

    for key, excluded_values in exclude.items():
        if key not in features.columns:
            continue

        excluded = {
            str(value).strip().lower()
            for value in excluded_values
        }

        values = (
            features[key]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        keep_mask &= ~values.isin(excluded)

    return features.loc[keep_mask].copy()


def extract_feature_class(
    osm: OSM,
    class_name: str,
    class_config: dict[str, Any],
    aoi_projected: gpd.GeoDataFrame,
    expected_epsg: int,
) -> gpd.GeoDataFrame:
    custom_filter = build_custom_filter(class_config)
    filter_keys = list(custom_filter)

    LOGGER.info(
        "Extracting %s with keys: %s",
        class_name,
        ", ".join(filter_keys),
    )

    if class_name == "building":
        raw = osm.get_buildings(
            custom_filter=custom_filter,
            extra_attributes=["building"],
        )
    else:
        keep_relations = class_name != "road"

        raw = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter,
            osm_keys_to_keep=filter_keys,
            filter_type="keep",
            tags_as_columns=filter_keys,
            keep_nodes=False,
            keep_ways=True,
            keep_relations=keep_relations,
            keep_other_tags=True,
        )

    geometry_family = "line" if class_name == "road" else "polygon"

    cleaned = repair_and_clip(
        features=raw,
        aoi_projected=aoi_projected,
        expected_epsg=expected_epsg,
        geometry_family=geometry_family,
    )
    cleaned = apply_exclusions(cleaned, class_config)

    if not cleaned.empty:
        cleaned["label_source"] = "OpenStreetMap"
        cleaned["raw_class"] = class_name

    return cleaned


def calculate_geometry_statistics(
    features: gpd.GeoDataFrame,
    geometry_family: str,
) -> dict[str, float | int]:
    if features.empty:
        return {
            "feature_count": 0,
            "invalid_geometry_count": 0,
            "empty_geometry_count": 0,
            "total_area_m2": 0.0,
            "total_length_m": 0.0,
        }

    invalid_count = int((~features.geometry.is_valid).sum())
    empty_count = int(features.geometry.is_empty.sum())

    if geometry_family == "polygon":
        total_area = float(features.geometry.area.sum())
        total_length = 0.0
    else:
        total_area = 0.0
        total_length = float(features.geometry.length.sum())

    return {
        "feature_count": int(len(features)),
        "invalid_geometry_count": invalid_count,
        "empty_geometry_count": empty_count,
        "total_area_m2": total_area,
        "total_length_m": total_length,
    }


def write_feature_file(
    features: gpd.GeoDataFrame,
    output_path: Path,
    layer_name: str,
    overwrite: bool,
) -> str:
    if features.empty:
        return "empty_not_written"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            return "existing_not_overwritten"

    features.to_file(
        output_path,
        layer=layer_name,
        driver="GPKG",
        index=False,
    )

    return "written"


def process_city(
    row: pd.Series,
    config: dict[str, Any],
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    expected_epsg = int(row["utm_epsg"])

    pbf_path = (
        Path(config["osm"]["city_pbf_directory"])
        / f"{city_id}_{city_name}.osm.pbf"
    )

    if not pbf_path.exists():
        raise FileNotFoundError(f"City OSM PBF not found: {pbf_path}")

    aoi_projected, aoi_path = load_exact_aoi(
        row=row,
        config=config,
    )

    LOGGER.info(
        "Opening cropped city PBF for %s (%s).",
        city_id,
        city_name,
    )

    osm = OSM(
        str(pbf_path),
        keep_metadata=False,
        engine="in_memory",
    )

    output_directory = (
        Path(config["osm"]["output_directory"])
        / f"{city_id}_{city_name}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    overwrite = bool(config["osm"].get("overwrite_existing", False))
    reports: list[dict[str, Any]] = []

    class_definitions = {
        "building": ("buildings.gpkg", "buildings", "polygon"),
        "road": ("roads.gpkg", "roads", "line"),
        "vegetation": ("vegetation.gpkg", "vegetation", "polygon"),
        "bare_land": ("bare_land.gpkg", "bare_land", "polygon"),
        "water": ("water.gpkg", "water", "polygon"),
    }

    for class_name, (
        filename,
        layer_name,
        geometry_family,
    ) in class_definitions.items():
        started = time.perf_counter()

        try:
            features = extract_feature_class(
                osm=osm,
                class_name=class_name,
                class_config=mapping[class_name],
                aoi_projected=aoi_projected,
                expected_epsg=expected_epsg,
            )

            statistics = calculate_geometry_statistics(
                features,
                geometry_family,
            )

            output_path = output_directory / filename
            write_status = write_feature_file(
                features=features,
                output_path=output_path,
                layer_name=layer_name,
                overwrite=overwrite,
            )

            reports.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                    "utm_epsg": expected_epsg,
                    "feature_class": class_name,
                    "status": "success",
                    "write_status": write_status,
                    "feature_count": statistics["feature_count"],
                    "invalid_geometry_count": statistics[
                        "invalid_geometry_count"
                    ],
                    "empty_geometry_count": statistics[
                        "empty_geometry_count"
                    ],
                    "total_area_m2": statistics["total_area_m2"],
                    "total_length_m": statistics["total_length_m"],
                    "aoi_file": str(aoi_path),
                    "input_pbf": str(pbf_path.resolve()),
                    "output_file": str(output_path),
                    "duration_seconds": time.perf_counter() - started,
                    "error": "",
                }
            )

            LOGGER.info(
                "%s %s: %d features (%s).",
                city_id,
                class_name,
                statistics["feature_count"],
                write_status,
            )

        except Exception as error:
            LOGGER.exception(
                "Extraction failed for %s %s.",
                city_id,
                class_name,
            )

            reports.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                    "utm_epsg": expected_epsg,
                    "feature_class": class_name,
                    "status": "failed",
                    "write_status": "not_written",
                    "feature_count": 0,
                    "invalid_geometry_count": 0,
                    "empty_geometry_count": 0,
                    "total_area_m2": 0.0,
                    "total_length_m": 0.0,
                    "aoi_file": str(aoi_path),
                    "input_pbf": str(pbf_path.resolve()),
                    "output_file": "",
                    "duration_seconds": time.perf_counter() - started,
                    "error": str(error),
                }
            )

    return reports


def save_report(
    reports: list[dict[str, Any]],
    report_path: Path,
) -> tuple[Path, Path]:
    report_df = pd.DataFrame(reports)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = (
        report_path.parent
        / f"{report_path.stem}_{timestamp}{report_path.suffix}"
    )

    report_df.to_csv(timestamped_path, index=False)
    report_df.to_csv(report_path, index=False)

    return timestamped_path, report_path


def run_extraction(
    labels_config_path: Path,
    mapping_path: Path,
    city_filter: str | None,
) -> None:
    config = load_yaml(labels_config_path)
    mapping = load_class_mapping(mapping_path)
    cities = load_city_registry(config)

    if city_filter:
        cities = cities.loc[cities["city_id"] == city_filter].copy()

        if cities.empty:
            raise ValueError(f"City not found: {city_filter}")

    all_reports: list[dict[str, Any]] = []

    for _, row in tqdm(
        cities.iterrows(),
        total=len(cities),
        desc="OSM extraction",
    ):
        all_reports.extend(
            process_city(
                row=row,
                config=config,
                mapping=mapping,
            )
        )

    report_path = Path(config["osm"]["report_file"])
    timestamped, latest = save_report(all_reports, report_path)

    failed = [
        report
        for report in all_reports
        if report["status"] != "success"
    ]

    print("\nOSM extraction summary")
    print("-" * 70)
    print(f"Cities requested: {len(cities)}")
    print(f"Class extractions requested: {len(all_reports)}")
    print(f"Successful: {len(all_reports) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Timestamped report: {timestamped}")
    print(f"Latest report: {latest}")

    if failed:
        raise SystemExit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract city-level OSM features from cropped city PBF files."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/labels.yaml"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/osm_class_mapping.yaml"),
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Optional city ID, for example DE01.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    run_extraction(
        labels_config_path=args.config,
        mapping_path=args.mapping,
        city_filter=args.city,
    )


if __name__ == "__main__":
    main()
