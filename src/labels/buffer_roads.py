"""Convert OSM road centre lines into variable-width road polygons."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry.base import BaseGeometry


TRUE_VALUES = {"yes", "true", "1"}
FALSE_VALUES = {"no", "false", "0"}
WIDTH_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def parse_tags(value: Any) -> dict[str, Any]:
    """Return the OSM tags field as a dictionary."""
    if isinstance(value, dict):
        return value

    if value is None:
        return {}

    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str):
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def parse_numeric_token(value: Any) -> float | None:
    """Extract the first finite numeric value from an OSM tag."""
    if value is None:
        return None

    text = str(value).strip().lower()

    if not text or text in {"none", "null", "nan", "unknown"}:
        return None

    match = WIDTH_PATTERN.search(text)

    if match is None:
        return None

    try:
        number = float(match.group())
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def parse_width_m(value: Any) -> float | None:
    """
    Parse an OSM width value and convert it to metres.

    Supports common values such as:
    - 7
    - 7.5
    - 7 m
    - 12 ft
    - 12'
    """
    if value is None:
        return None

    text = str(value).strip().lower()
    number = parse_numeric_token(text)

    if number is None:
        return None

    if "ft" in text or "'" in text or "feet" in text:
        return number * 0.3048

    return number


def parse_lanes(value: Any) -> float | None:
    """Parse a simple numeric OSM lanes value."""
    if value is None:
        return None

    text = str(value).strip().lower()

    # Ambiguous values such as "1;2" are rejected.
    if any(separator in text for separator in (";", "|", ",")):
        return None

    lanes = parse_numeric_token(text)

    if lanes is None or lanes <= 0:
        return None

    return lanes


def tag_is_true(value: Any) -> bool:
    """Interpret common OSM boolean-style tag values."""
    if value is None:
        return False

    return str(value).strip().lower() in TRUE_VALUES


def is_underground_tunnel(tags: dict[str, Any]) -> bool:
    """Return True for roads that should not appear on the surface mask."""
    tunnel_value = str(tags.get("tunnel", "")).strip().lower()

    if tunnel_value in FALSE_VALUES or not tunnel_value:
        return False

    # Covered roads remain visible as surface infrastructure.
    if tunnel_value == "building_passage":
        return False

    return True


def select_total_width(
    highway: str,
    tags: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, str]:
    """Resolve total road width and identify the source used."""
    resolution = config["width_resolution"]
    defaults = config["highway_default_widths_m"]

    minimum = float(resolution["minimum_width_m"])
    maximum = float(resolution["maximum_width_m"])
    fallback = float(resolution["fallback_width_m"])
    lane_width = float(resolution["lane_width_m"])

    lane_eligible = set(resolution["lane_eligible_highway_classes"])

    explicit_width = parse_width_m(tags.get("width"))

    if explicit_width is not None and explicit_width > 0:
        width = explicit_width
        source = "osm_width"
    else:
        lanes = parse_lanes(tags.get("lanes"))

        if lanes is not None and highway in lane_eligible:
            width = lanes * lane_width
            source = "osm_lanes"
        elif highway in defaults:
            width = float(defaults[highway])
            source = "highway_default"
        else:
            width = fallback
            source = "fallback"

    width = min(max(width, minimum), maximum)

    return width, source


def get_buffer_style(style_name: str) -> int:
    """Map YAML buffer style names to Shapely integer values."""
    styles = {
        "round": 1,
        "flat": 2,
        "square": 3,
        "mitre": 2,
        "bevel": 3,
    }

    if style_name not in styles:
        raise ValueError(f"Unsupported buffer style: {style_name}")

    return styles[style_name]


def valid_geometry(geometry: BaseGeometry | None) -> bool:
    """Check whether a geometry is present and non-empty."""
    return geometry is not None and not geometry.is_empty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate variable-width road polygons from OSM lines."
    )
    parser.add_argument("--roads", required=True, type=Path)
    parser.add_argument("--layer", default="roads")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--city-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("metadata/road_buffering"),
    )
    args = parser.parse_args()

    if not args.roads.exists():
        raise FileNotFoundError(f"Road input not found: {args.roads}")

    if not args.config.exists():
        raise FileNotFoundError(f"Configuration not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    roads = gpd.read_file(args.roads, layer=args.layer)

    if roads.empty:
        raise ValueError("Road input is empty.")

    if roads.crs is None:
        raise ValueError("Road input has no CRS.")

    if roads.crs.is_geographic:
        raise ValueError(
            "Road buffering requires a projected CRS measured in metres."
        )

    required_fields = {"highway", "tags", "geometry"}
    missing = required_fields.difference(roads.columns)

    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    roads = roads.copy()
    roads["_parsed_tags"] = roads["tags"].apply(parse_tags)
    roads["_is_tunnel"] = roads["_parsed_tags"].apply(
        is_underground_tunnel
    )
    roads["_is_bridge"] = roads["_parsed_tags"].apply(
        lambda tags: tag_is_true(tags.get("bridge"))
    )

    input_count = len(roads)

    if config["special_rules"]["exclude_tunnels"]:
        surface_roads = roads.loc[~roads["_is_tunnel"]].copy()
    else:
        surface_roads = roads.copy()

    excluded_tunnel_count = input_count - len(surface_roads)

    width_results = surface_roads.apply(
        lambda row: select_total_width(
            highway=str(row["highway"]),
            tags=row["_parsed_tags"],
            config=config,
        ),
        axis=1,
    )

    surface_roads["total_width_m"] = [
        result[0] for result in width_results
    ]
    surface_roads["width_source"] = [
        result[1] for result in width_results
    ]
    surface_roads["buffer_distance_m"] = (
        surface_roads["total_width_m"] / 2.0
    )

    buffer_config = config["buffering"]
    cap_style = get_buffer_style(buffer_config["cap_style"])
    join_style = get_buffer_style(buffer_config["join_style"])
    resolution = int(buffer_config["buffer_resolution"])

    surface_roads["geometry"] = [
        geometry.buffer(
            distance=distance,
            resolution=resolution,
            cap_style=cap_style,
            join_style=join_style,
        )
        for geometry, distance in zip(
            surface_roads.geometry,
            surface_roads["buffer_distance_m"],
            strict=True,
        )
    ]

    surface_roads = surface_roads.loc[
        surface_roads.geometry.apply(valid_geometry)
    ].copy()

    buffered_feature_count = len(surface_roads)

    # Dissolve into a single MultiPolygon to remove overlaps between
    # neighbouring and duplicated road segments.
    dissolved_geometry = surface_roads.geometry.union_all()

    output = gpd.GeoDataFrame(
        {
            "city_id": [args.city_id],
            "label_source": ["OpenStreetMap"],
            "raw_class": ["road"],
            "geometry": [dissolved_geometry],
        },
        crs=roads.crs,
    )

    output = output.loc[output.geometry.apply(valid_geometry)].copy()

    if output.empty:
        raise ValueError("Buffering produced no valid road polygons.")

    invalid_output = int((~output.geometry.is_valid).sum())

    if invalid_output > 0:
        output["geometry"] = output.geometry.make_valid()
        invalid_output = int((~output.geometry.is_valid).sum())

    if invalid_output > 0:
        raise ValueError(
            f"Output still contains {invalid_output} invalid geometries."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    layer_name = config["output"]["layer_name"]

    output.to_file(
        args.output,
        layer=layer_name,
        driver="GPKG",
    )

    source_counts = (
        surface_roads["width_source"]
        .value_counts()
        .rename_axis("width_source")
        .reset_index(name="feature_count")
    )
    source_counts["percentage"] = (
        source_counts["feature_count"]
        / len(surface_roads)
        * 100
    ).round(4)

    highway_width_summary = (
        surface_roads.groupby("highway", dropna=False)
        .agg(
            feature_count=("highway", "size"),
            mean_width_m=("total_width_m", "mean"),
            median_width_m=("total_width_m", "median"),
            min_width_m=("total_width_m", "min"),
            max_width_m=("total_width_m", "max"),
        )
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )

    report = {
        "city_id": args.city_id,
        "source_file": str(args.roads),
        "output_file": str(args.output),
        "crs": str(roads.crs),
        "input_line_features": input_count,
        "surface_line_features": int(len(surface_roads)),
        "excluded_tunnel_features": excluded_tunnel_count,
        "bridge_features_retained": int(
            surface_roads["_is_bridge"].sum()
        ),
        "buffered_line_features": buffered_feature_count,
        "output_polygon_features": int(len(output)),
        "output_geometry_type": str(
            output.geometry.geom_type.iloc[0]
        ),
        "output_valid": bool(output.geometry.is_valid.all()),
        "input_total_length_m": float(roads.geometry.length.sum()),
        "surface_input_length_m": float(
             roads.loc[~roads["_is_tunnel"]].geometry.length.sum()
        ),
        "output_road_area_m2": float(output.geometry.area.sum()),
        "minimum_width_m": float(
            surface_roads["total_width_m"].min()
        ),
        "maximum_width_m": float(
            surface_roads["total_width_m"].max()
        ),
        "mean_width_m": float(
            surface_roads["total_width_m"].mean()
        ),
        "median_width_m": float(
            surface_roads["total_width_m"].median()
        ),
        "width_source_counts": {
            row["width_source"]: int(row["feature_count"])
            for _, row in source_counts.iterrows()
        },
    }

    report_path = (
        args.report_dir
        / f"{args.city_id}_road_buffering_report.json"
    )
    source_path = (
        args.report_dir
        / f"{args.city_id}_road_width_source_counts.csv"
    )
    highway_path = (
        args.report_dir
        / f"{args.city_id}_road_width_summary_by_highway.csv"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    source_counts.to_csv(source_path, index=False)
    highway_width_summary.to_csv(highway_path, index=False)

    print("\nRoad buffering completed")
    print("------------------------")
    print(f"City: {args.city_id}")
    print(f"Input line features: {input_count:,}")
    print(f"Excluded tunnel features: {excluded_tunnel_count:,}")
    print(f"Surface features buffered: {len(surface_roads):,}")
    print(
        "Bridge features retained: "
        f"{int(surface_roads['_is_bridge'].sum()):,}"
    )
    print(f"Output geometry: {report['output_geometry_type']}")
    print(f"Output valid: {report['output_valid']}")
    print(
        f"Output road area: "
        f"{report['output_road_area_m2']:,.2f} m²"
    )
    print(
        f"Width range: "
        f"{report['minimum_width_m']:.2f}–"
        f"{report['maximum_width_m']:.2f} m"
    )
    print(f"Mean width: {report['mean_width_m']:.2f} m")
    print(f"Median width: {report['median_width_m']:.2f} m")

    print("\nWidth sources")
    print(source_counts.to_string(index=False))

    print(f"\nSaved polygon layer: {args.output}")
    print(f"Saved report: {report_path}")
    print(f"Saved width-source table: {source_path}")
    print(f"Saved highway summary: {highway_path}")


if __name__ == "__main__":
    main()