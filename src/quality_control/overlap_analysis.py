"""Calculate pairwise spatial overlaps between vector label classes.

The script dissolves each class into a single spatial footprint before
calculating pairwise intersections. This prevents internal overlaps within
one class from being counted multiple times.

Outputs
-------
1. Pairwise overlap CSV
2. JSON summary containing source metadata and class-area statistics
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class LayerInput:
    """Configuration for one thematic vector layer."""

    name: str
    path: Path
    layer: str | None


# Thresholds are percentages of each class footprint.
#
# A pair is marked REVIEW when either class-specific percentage exceeds
# its corresponding threshold.
#
# These are screening thresholds, not automatic rejection criteria.
PAIR_THRESHOLDS_PERCENT: dict[tuple[str, str], tuple[float, float]] = {
    ("buildings", "roads"): (5.0, 10.0),
    ("buildings", "vegetation"): (5.0, 10.0),
    ("buildings", "water"): (1.0, 5.0),
    ("buildings", "bare_land"): (5.0, 10.0),
    ("roads", "vegetation"): (10.0, 10.0),
    ("roads", "water"): (5.0, 10.0),
    ("roads", "bare_land"): (10.0, 10.0),
    ("vegetation", "water"): (10.0, 10.0),
    ("vegetation", "bare_land"): (10.0, 10.0),
    ("water", "bare_land"): (10.0, 10.0),
}

DEFAULT_THRESHOLD_PERCENT = (10.0, 10.0)


def read_vector_layer(layer_input: LayerInput) -> gpd.GeoDataFrame:
    """Read and validate a vector layer."""
    if not layer_input.path.exists():
        raise FileNotFoundError(
            f"{layer_input.name} file not found: {layer_input.path}"
        )

    print(f"Reading {layer_input.name}: {layer_input.path}")

    if layer_input.layer:
        gdf = gpd.read_file(
            layer_input.path,
            layer=layer_input.layer,
        )
    else:
        gdf = gpd.read_file(layer_input.path)

    if gdf.empty:
        raise ValueError(
            f"The {layer_input.name} layer contains no features."
        )

    if gdf.crs is None:
        raise ValueError(
            f"The {layer_input.name} layer does not have a CRS."
        )

    if gdf.crs.is_geographic:
        raise ValueError(
            f"The {layer_input.name} layer uses geographic CRS "
            f"{gdf.crs}. Area calculations require a projected CRS."
        )

    missing_geometry_count = int(gdf.geometry.isna().sum())
    empty_geometry_count = int(gdf.geometry.is_empty.sum())

    if missing_geometry_count > 0:
        print(
            f"Warning: removing {missing_geometry_count:,} missing "
            f"geometries from {layer_input.name}."
        )

    if empty_geometry_count > 0:
        print(
            f"Warning: removing {empty_geometry_count:,} empty "
            f"geometries from {layer_input.name}."
        )

    gdf = gdf.loc[
        gdf.geometry.notna() & ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError(
            f"No usable geometries remain in {layer_input.name}."
        )

    invalid_count = int((~gdf.geometry.is_valid).sum())

    if invalid_count > 0:
        print(
            f"Repairing {invalid_count:,} invalid geometries in "
            f"{layer_input.name}."
        )
        gdf["geometry"] = gdf.geometry.make_valid()

        gdf = gdf.loc[
            gdf.geometry.notna() & ~gdf.geometry.is_empty
        ].copy()

        remaining_invalid = int((~gdf.geometry.is_valid).sum())

        if remaining_invalid > 0:
            raise ValueError(
                f"{layer_input.name} still contains "
                f"{remaining_invalid:,} invalid geometries after repair."
            )

    return gdf


def count_polygon_parts(geometry: BaseGeometry) -> int:
    """Count polygon components in an intersection geometry."""
    if geometry.is_empty:
        return 0

    if geometry.geom_type == "Polygon":
        return 1

    if geometry.geom_type == "MultiPolygon":
        return len(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        return sum(
            count_polygon_parts(part)
            for part in geometry.geoms
            if not part.is_empty
        )

    # Point and line intersections do not contribute area.
    return 0


def dissolve_class(
    class_name: str,
    gdf: gpd.GeoDataFrame,
) -> BaseGeometry:
    """Create one unioned footprint for a thematic class."""
    print(
        f"Dissolving {class_name} "
        f"({len(gdf):,} input features)..."
    )

    start_time = time.perf_counter()
    dissolved = gdf.geometry.union_all()
    elapsed = time.perf_counter() - start_time

    if dissolved is None or dissolved.is_empty:
        raise ValueError(
            f"Dissolving {class_name} produced an empty geometry."
        )

    if not dissolved.is_valid:
        dissolved = gpd.GeoSeries(
            [dissolved],
            crs=gdf.crs,
        ).make_valid().iloc[0]

    if not dissolved.is_valid:
        raise ValueError(
            f"The dissolved {class_name} geometry is invalid."
        )

    print(
        f"  Completed in {elapsed:.2f} seconds; "
        f"area = {dissolved.area:,.2f} m²"
    )

    return dissolved


def threshold_for_pair(
    class_a: str,
    class_b: str,
) -> tuple[float, float]:
    """Return class-specific overlap thresholds for one pair."""
    direct_key = (class_a, class_b)

    if direct_key in PAIR_THRESHOLDS_PERCENT:
        return PAIR_THRESHOLDS_PERCENT[direct_key]

    reverse_key = (class_b, class_a)

    if reverse_key in PAIR_THRESHOLDS_PERCENT:
        reverse_thresholds = PAIR_THRESHOLDS_PERCENT[reverse_key]
        return reverse_thresholds[1], reverse_thresholds[0]

    return DEFAULT_THRESHOLD_PERCENT


def analyse_pair(
    class_a: str,
    class_b: str,
    geometry_a: BaseGeometry,
    geometry_b: BaseGeometry,
    area_a: float,
    area_b: float,
) -> dict[str, Any]:
    """Calculate overlap statistics for two dissolved classes."""
    start_time = time.perf_counter()
    intersection = geometry_a.intersection(geometry_b)
    elapsed = time.perf_counter() - start_time

    overlap_area = float(intersection.area)

    percentage_a = (
        overlap_area / area_a * 100.0
        if area_a > 0
        else 0.0
    )
    percentage_b = (
        overlap_area / area_b * 100.0
        if area_b > 0
        else 0.0
    )

    threshold_a, threshold_b = threshold_for_pair(
        class_a,
        class_b,
    )

    status = (
        "REVIEW"
        if (
            percentage_a > threshold_a
            or percentage_b > threshold_b
        )
        else "OK"
    )

    return {
        "class_a": class_a,
        "class_b": class_b,
        "class_a_area_m2": round(area_a, 4),
        "class_b_area_m2": round(area_b, 4),
        "overlap_area_m2": round(overlap_area, 4),
        "overlap_area_km2": round(overlap_area / 1_000_000.0, 6),
        "overlap_percent_of_class_a": round(percentage_a, 6),
        "overlap_percent_of_class_b": round(percentage_b, 6),
        "threshold_percent_class_a": threshold_a,
        "threshold_percent_class_b": threshold_b,
        "overlap_component_count": count_polygon_parts(intersection),
        "intersection_geometry_type": intersection.geom_type,
        "processing_seconds": round(elapsed, 4),
        "status": status,
    }


def build_layer_inputs(args: argparse.Namespace) -> list[LayerInput]:
    """Convert command-line arguments into layer definitions."""
    return [
        LayerInput(
            name="buildings",
            path=args.buildings,
            layer=args.buildings_layer,
        ),
        LayerInput(
            name="roads",
            path=args.roads,
            layer=args.roads_layer,
        ),
        LayerInput(
            name="vegetation",
            path=args.vegetation,
            layer=args.vegetation_layer,
        ),
        LayerInput(
            name="bare_land",
            path=args.bare_land,
            layer=args.bare_land_layer,
        ),
        LayerInput(
            name="water",
            path=args.water,
            layer=args.water_layer,
        ),
    ]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pairwise overlaps between five vector-label classes."
        )
    )

    parser.add_argument(
        "--city-id",
        required=True,
        help="City identifier, for example DE01.",
    )

    parser.add_argument(
        "--buildings",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--roads",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--vegetation",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--bare-land",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--water",
        required=True,
        type=Path,
    )

    parser.add_argument("--buildings-layer")
    parser.add_argument("--roads-layer")
    parser.add_argument("--vegetation-layer")
    parser.add_argument("--bare-land-layer")
    parser.add_argument("--water-layer")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/overlap_analysis"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete overlap-analysis workflow."""
    args = parse_arguments()
    layer_inputs = build_layer_inputs(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded_layers: dict[str, gpd.GeoDataFrame] = {}
    reference_crs = None

    print("\nPairwise label-overlap analysis")
    print("--------------------------------")
    print(f"City: {args.city_id}")

    for layer_input in layer_inputs:
        gdf = read_vector_layer(layer_input)

        if reference_crs is None:
            reference_crs = gdf.crs
        elif gdf.crs != reference_crs:
            raise ValueError(
                f"CRS mismatch for {layer_input.name}: "
                f"{gdf.crs} != {reference_crs}"
            )

        loaded_layers[layer_input.name] = gdf

    dissolved_geometries: dict[str, BaseGeometry] = {}
    class_statistics: dict[str, dict[str, Any]] = {}

    print("\nCreating non-overcounted class footprints")

    for layer_input in layer_inputs:
        class_name = layer_input.name
        gdf = loaded_layers[class_name]

        dissolved = dissolve_class(class_name, gdf)
        dissolved_geometries[class_name] = dissolved

        raw_area = float(gdf.geometry.area.sum())
        dissolved_area = float(dissolved.area)

        internal_overlap_area = max(
            raw_area - dissolved_area,
            0.0,
        )

        class_statistics[class_name] = {
            "source_file": str(layer_input.path),
            "source_layer": layer_input.layer,
            "input_feature_count": int(len(gdf)),
            "input_geometry_types": sorted(
                gdf.geometry.geom_type.unique().tolist()
            ),
            "raw_summed_area_m2": round(raw_area, 4),
            "dissolved_area_m2": round(dissolved_area, 4),
            "internal_overlap_area_m2": round(
                internal_overlap_area,
                4,
            ),
            "dissolved_geometry_type": dissolved.geom_type,
            "dissolved_polygon_parts": count_polygon_parts(
                dissolved
            ),
        }

    results: list[dict[str, Any]] = []

    class_names = [
        "buildings",
        "roads",
        "vegetation",
        "bare_land",
        "water",
    ]

    print("\nCalculating pairwise intersections")

    for class_a, class_b in combinations(class_names, 2):
        print(f"  {class_a} ↔ {class_b}")

        geometry_a = dissolved_geometries[class_a]
        geometry_b = dissolved_geometries[class_b]

        area_a = float(geometry_a.area)
        area_b = float(geometry_b.area)

        result = analyse_pair(
            class_a=class_a,
            class_b=class_b,
            geometry_a=geometry_a,
            geometry_b=geometry_b,
            area_a=area_a,
            area_b=area_b,
        )
        results.append(result)

        print(
            f"    overlap = "
            f"{result['overlap_area_m2']:,.2f} m²; "
            f"{result['overlap_percent_of_class_a']:.4f}% "
            f"of {class_a}; "
            f"{result['overlap_percent_of_class_b']:.4f}% "
            f"of {class_b}; "
            f"status = {result['status']}"
        )

    results_df = pd.DataFrame(results)

    csv_path = (
        args.output_dir
        / f"{args.city_id}_pairwise_overlap.csv"
    )
    json_path = (
        args.output_dir
        / f"{args.city_id}_overlap_summary.json"
    )

    results_df.to_csv(csv_path, index=False)

    review_pairs = [
        {
            "class_a": row["class_a"],
            "class_b": row["class_b"],
            "overlap_percent_of_class_a":
                row["overlap_percent_of_class_a"],
            "overlap_percent_of_class_b":
                row["overlap_percent_of_class_b"],
        }
        for row in results
        if row["status"] == "REVIEW"
    ]

    summary = {
        "city_id": args.city_id,
        "crs": str(reference_crs),
        "area_unit": "square metres",
        "calculation_method": (
            "Pairwise intersection of dissolved class footprints"
        ),
        "class_priority_high_to_low": [
            "buildings",
            "roads",
            "water",
            "vegetation",
            "bare_land",
        ],
        "threshold_note": (
            "Thresholds are screening criteria. REVIEW does not "
            "automatically indicate an error."
        ),
        "class_statistics": class_statistics,
        "pair_count": len(results),
        "ok_pair_count": sum(
            result["status"] == "OK"
            for result in results
        ),
        "review_pair_count": sum(
            result["status"] == "REVIEW"
            for result in results
        ),
        "review_pairs": review_pairs,
        "pairwise_results": results,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nOverlap analysis completed")
    print("--------------------------")
    print(f"Pairs analysed: {len(results)}")
    print(f"OK pairs: {summary['ok_pair_count']}")
    print(f"REVIEW pairs: {summary['review_pair_count']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()