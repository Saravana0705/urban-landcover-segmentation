"""Audit OSM road attributes before converting road lines to polygons."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


TAG_FIELDS = ("width", "lanes", "bridge", "tunnel", "layer", "oneway")


def parse_tags(value: Any) -> dict[str, Any]:
    """Convert the tags field into a Python dictionary."""
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


def non_empty(value: Any) -> bool:
    """Return True when a tag value contains useful information."""
    if value is None:
        return False

    text = str(value).strip().lower()
    return text not in {"", "none", "null", "nan"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit highway classes and embedded OSM road tags."
    )
    parser.add_argument(
        "--roads",
        required=True,
        type=Path,
        help="Path to the roads GeoPackage.",
    )
    parser.add_argument(
        "--layer",
        default="roads",
        help="GeoPackage layer name. Default: roads",
    )
    parser.add_argument(
        "--city-id",
        required=True,
        help="City identifier, for example DE01.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/road_audit"),
        help="Directory for generated reports.",
    )
    args = parser.parse_args()

    if not args.roads.exists():
        raise FileNotFoundError(f"Road file not found: {args.roads}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.roads}")
    roads = gpd.read_file(args.roads, layer=args.layer)

    if roads.empty:
        raise ValueError("The road layer contains no features.")

    required_columns = {"geometry", "highway", "tags"}
    missing_columns = required_columns.difference(roads.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    parsed_tags = roads["tags"].apply(parse_tags)

    highway_counts = (
        roads["highway"]
        .fillna("<missing>")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("highway")
        .reset_index(name="feature_count")
    )

    highway_counts["percentage"] = (
        highway_counts["feature_count"] / len(roads) * 100
    ).round(4)

    tag_presence: dict[str, int] = {}

    for field in TAG_FIELDS:
        tag_presence[field] = int(
            parsed_tags.apply(
                lambda tags: non_empty(tags.get(field))
            ).sum()
        )

    geometry_counts = Counter(
        roads.geometry.geom_type.fillna("<missing>").astype(str)
    )

    feature_id_column = None

    if "id" in roads.columns:
        feature_id_column = "id"
    elif "osm_id" in roads.columns:
        feature_id_column = "osm_id"

    duplicate_id_rows = None
    unique_ids = None

    if feature_id_column is not None:
        non_null_ids = roads[feature_id_column].dropna()
        duplicate_id_rows = int(non_null_ids.duplicated(keep=False).sum())
        unique_ids = int(non_null_ids.nunique())

    invalid_geometries = int((~roads.geometry.is_valid).sum())
    empty_geometries = int(roads.geometry.is_empty.sum())
    missing_geometries = int(roads.geometry.isna().sum())

    report = {
        "city_id": args.city_id,
        "source_file": str(args.roads),
        "layer": args.layer,
        "crs": str(roads.crs),
        "feature_count": int(len(roads)),
        "geometry_types": dict(geometry_counts),
        "invalid_geometries": invalid_geometries,
        "empty_geometries": empty_geometries,
        "missing_geometries": missing_geometries,
        "feature_id_column": feature_id_column,
        "unique_feature_ids": unique_ids,
        "rows_with_duplicated_feature_id": duplicate_id_rows,
        "tag_presence_counts": tag_presence,
        "tag_presence_percentages": {
            field: round(count / len(roads) * 100, 4)
            for field, count in tag_presence.items()
        },
        "unique_highway_classes": int(
            roads["highway"].fillna("<missing>").astype(str).nunique()
        ),
    }

    highway_output = (
        args.output_dir
        / f"{args.city_id}_road_highway_counts.csv"
    )
    report_output = (
        args.output_dir
        / f"{args.city_id}_road_attribute_audit.json"
    )

    highway_counts.to_csv(highway_output, index=False)

    with report_output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nRoad attribute audit")
    print("--------------------")
    print(f"City: {args.city_id}")
    print(f"Features: {len(roads):,}")
    print(f"CRS: {roads.crs}")
    print(f"Geometry types: {dict(geometry_counts)}")
    print(f"Unique highway classes: {report['unique_highway_classes']}")
    print(f"Invalid geometries: {invalid_geometries:,}")
    print(f"Empty geometries: {empty_geometries:,}")
    print(f"Missing geometries: {missing_geometries:,}")

    if feature_id_column is not None:
        print(f"Feature ID field: {feature_id_column}")
        print(f"Unique IDs: {unique_ids:,}")
        print(
            "Rows belonging to duplicated IDs: "
            f"{duplicate_id_rows:,}"
        )

    print("\nEmbedded tag availability")

    for field in TAG_FIELDS:
        count = tag_presence[field]
        percentage = report["tag_presence_percentages"][field]
        print(f"  {field:<8}: {count:>8,} ({percentage:>7.4f}%)")

    print("\nMost common highway classes")
    print(highway_counts.head(20).to_string(index=False))

    print(f"\nSaved: {report_output}")
    print(f"Saved: {highway_output}")


if __name__ == "__main__":
    main()