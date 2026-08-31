"""Read-only Dataset V3.1 road/bare-land policy simulation.

The script derives conservative label subsets from frozen Dataset V3. It
never writes semantic rasters or tiles. Removed road/bare pixels become
Ignore (255), never another semantic class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize

from src.labels.buffer_roads import (
    get_buffer_style,
    is_underground_tunnel,
    parse_tags,
    select_total_width,
)


IGNORE_INDEX = 255
CLASS_NAMES = {
    1: "buildings",
    2: "roads",
    3: "vegetation",
    4: "bare_land",
    5: "water",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate additive V3.1 label policies without writing labels."
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--road-config", type=Path, default=Path("config/road_widths.yaml")
    )
    parser.add_argument(
        "--policy-config", type=Path,
        default=Path("config/dataset_v31_policy.yaml"),
    )
    parser.add_argument(
        "--candidate-manifest", type=Path,
        default=Path("metadata/dataset_v3/candidates/dataset_v3_candidate_tiles.csv"),
    )
    parser.add_argument(
        "--selected-manifest", type=Path,
        default=Path("metadata/dataset_v3/selection/dataset_v3_selected_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("metadata/dataset_v3_1/policy_simulation"),
    )
    parser.add_argument(
        "--split", action="append", choices=["train", "val", "test"],
        help="Repeat as needed. Defaults to train and val; test is excluded.",
    )
    parser.add_argument("--city", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-metadata", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def semantic_path(city_id: str) -> Path:
    return Path("data/interim/labels_v3/fused") / city_id / f"{city_id}_v3_semantic.tif"


def ua_code_path(city_id: str) -> Path:
    return (
        Path("data/interim/labels_v3/urban_atlas")
        / city_id
        / f"{city_id}_urban_atlas_2021_code.tif"
    )


def roads_path(city: dict[str, Any]) -> Path:
    folder = f"{city['city_id']}_{city['city_name']}"
    return Path("data/vector/osm") / folder / "roads.gpkg"


def same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader) -> bool:
    return (
        reference.width == other.width
        and reference.height == other.height
        and reference.crs == other.crs
        and reference.transform == other.transform
    )


def road_policy_mask(
    city: dict[str, Any],
    reference: rasterio.DatasetReader,
    road_config: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    roads = gpd.read_file(roads_path(city), layer="roads")
    if roads.empty or roads.crs is None:
        raise ValueError(f"{city['city_id']} roads are empty or have no CRS")
    missing = {"highway", "tags", "geometry"}.difference(roads.columns)
    if missing:
        raise ValueError(f"{city['city_id']} roads lack fields: {sorted(missing)}")
    if roads.crs != reference.crs:
        roads = roads.to_crs(reference.crs)

    roads = roads.copy()
    roads["_tags"] = roads["tags"].apply(parse_tags)
    roads = roads.loc[~roads["_tags"].apply(is_underground_tunnel)].copy()
    roads["highway"] = roads["highway"].fillna("unknown").astype(str)
    resolved = roads.apply(
        lambda row: select_total_width(row["highway"], row["_tags"], road_config),
        axis=1,
    )
    roads["_width_m"] = [item[0] for item in resolved]

    allowed = set(policy["allowed_highway_classes"])
    minimum = float(policy["minimum_width_m"])
    eligible = roads.loc[
        roads["highway"].isin(allowed) & (roads["_width_m"] >= minimum)
    ].copy()

    buffering = road_config.get("buffering", {})
    cap_style = get_buffer_style(buffering.get("cap_style", "round"))
    join_style = get_buffer_style(buffering.get("join_style", "round"))
    resolution = int(buffering.get("buffer_resolution", 4))
    polygons = []
    for geometry, width in zip(eligible.geometry, eligible["_width_m"], strict=True):
        if geometry is None or geometry.is_empty:
            continue
        polygons.append(
            geometry.buffer(
                float(width) / 2.0,
                resolution=resolution,
                cap_style=cap_style,
                join_style=join_style,
            )
        )
    mask = rasterize(
        ((geometry, 1) for geometry in polygons),
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    details = {
        "surface_feature_count": int(len(roads)),
        "eligible_feature_count": int(len(eligible)),
        "eligible_rasterized_pixels": int(mask.sum()),
        "minimum_width_m": minimum,
        "allowed_highway_classes": sorted(allowed),
    }
    return mask, details


def apply_variant(
    baseline: np.ndarray,
    ua_code: np.ndarray,
    road_masks: dict[str, np.ndarray],
    variant: dict[str, Any],
    stable_bare_codes: set[int],
) -> np.ndarray:
    simulated = baseline.copy()
    road_policy = str(variant["road_policy"])
    if road_policy != "none":
        current_road = baseline == 2
        simulated[current_road & ~road_masks[road_policy]] = IGNORE_INDEX
    if bool(variant["stable_bare"]):
        current_bare = baseline == 4
        stable = np.isin(ua_code, sorted(stable_bare_codes))
        simulated[current_bare & ~stable] = IGNORE_INDEX
    return simulated


def class_counts(array: np.ndarray) -> dict[str, int]:
    result = {
        f"{name}_pixels": int((array == class_id).sum())
        for class_id, name in CLASS_NAMES.items()
    }
    result["ignore_pixels"] = int((array == IGNORE_INDEX).sum())
    result["valid_pixels"] = int(sum(result[f"{name}_pixels"] for name in CLASS_NAMES.values()))
    result["total_pixels"] = int(array.size)
    return result


def candidate_metrics(
    candidate: dict[str, str], simulated: np.ndarray, selected_ids: set[str]
) -> dict[str, Any]:
    row_off = int(candidate["row_offset"])
    col_off = int(candidate["col_offset"])
    source_h = int(candidate["source_height_pixels"])
    source_w = int(candidate["source_width_pixels"])
    source_pixels = int(candidate["source_pixel_count"])
    tile = simulated[row_off : row_off + source_h, col_off : col_off + source_w]
    if tile.size != source_pixels:
        raise RuntimeError(f"Candidate window accounting failed: {candidate['tile_id']}")
    counts = class_counts(tile)
    valid = counts["valid_pixels"]
    padding_fraction = float(candidate["padding_fraction"])
    valid_fraction_source = valid / source_pixels if source_pixels else 0.0
    result: dict[str, Any] = {
        "variant": "",
        "tile_id": candidate["tile_id"],
        "city_id": candidate["city_id"],
        "city_name": candidate["city_name"],
        "split": candidate["split"],
        "selected_in_v3": int(candidate["tile_id"] in selected_ids),
        "source_pixel_count": source_pixels,
        "padding_pixel_count": int(candidate["padding_pixel_count"]),
        "padding_fraction": padding_fraction,
        "valid_pixel_count": valid,
        "valid_fraction_source_pixels": valid_fraction_source,
        "ignore_pixel_count": counts["ignore_pixels"],
        "ignore_fraction_source_pixels": counts["ignore_pixels"] / source_pixels,
    }
    for class_id, name in CLASS_NAMES.items():
        pixels = counts[f"{name}_pixels"]
        result[f"{name}_pixel_count"] = pixels
        result[f"{name}_fraction_valid"] = pixels / valid if valid else 0.0
        result[f"{name}_fraction_source_pixels"] = pixels / source_pixels
    return result


def eligible(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        float(row["valid_fraction_source_pixels"])
        >= float(gate["minimum_valid_fraction_source_pixels"])
        and float(row["padding_fraction"])
        <= float(gate["maximum_padding_fraction"])
    )


def aggregate_policy(
    rows: list[dict[str, Any]], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    accum: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (str(row["variant"]), str(row["split"]))
        counter = accum[key]
        counter["candidate_tiles"] += 1
        counter["selected_v3_tiles"] += int(row["selected_in_v3"])
        row_eligible = eligible(row, gate)
        counter["coverage_eligible_tiles"] += int(row_eligible)
        counter["selected_v3_tiles_remaining_eligible"] += int(
            row_eligible and bool(row["selected_in_v3"])
        )
        if row_eligible:
            road_fraction = float(row["roads_fraction_valid"])
            bare_fraction = float(row["bare_land_fraction_valid"])
            for label, value, threshold in (
                ("roads_ge_5pct_tiles", road_fraction, 0.05),
                ("roads_ge_7_5pct_tiles", road_fraction, 0.075),
                ("roads_ge_10pct_tiles", road_fraction, 0.10),
                ("bare_ge_0_5pct_tiles", bare_fraction, 0.005),
                ("bare_ge_1pct_tiles", bare_fraction, 0.01),
                ("bare_ge_2pct_tiles", bare_fraction, 0.02),
            ):
                counter[label] += int(value >= threshold)
        for name in CLASS_NAMES.values():
            counter[f"{name}_pixels"] += int(row[f"{name}_pixel_count"])
        counter["valid_pixels"] += int(row["valid_pixel_count"])
        counter["ignore_pixels"] += int(row["ignore_pixel_count"])
        counter["source_pixels"] += int(row["source_pixel_count"])

    output: list[dict[str, Any]] = []
    for (variant, split), counter in sorted(accum.items()):
        valid = counter["valid_pixels"]
        source = counter["source_pixels"]
        row: dict[str, Any] = {
            "variant": variant,
            "split": split,
            **dict(counter),
            "valid_fraction_source_pixels": valid / source if source else 0.0,
            "ignore_fraction_source_pixels": counter["ignore_pixels"] / source if source else 0.0,
        }
        for name in CLASS_NAMES.values():
            row[f"{name}_fraction_valid"] = (
                counter[f"{name}_pixels"] / valid if valid else 0.0
            )
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.registry, "V3 city registry"),
        (args.road_config, "road-width configuration"),
        (args.policy_config, "V3.1 policy configuration"),
        (args.candidate_manifest, "V3 candidate manifest"),
    ):
        require_file(path, label)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    road_config = yaml.safe_load(args.road_config.read_text(encoding="utf-8"))
    policy_config = yaml.safe_load(args.policy_config.read_text(encoding="utf-8"))
    splits = set(args.split or ["train", "val"])
    cities = [city for city in registry["cities"] if city["split"] in splits]
    if args.city:
        requested = set(args.city)
        known = {city["city_id"] for city in cities}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown cities or cities outside selected splits: {sorted(unknown)}")
        cities = [city for city in cities if city["city_id"] in requested]
    if not cities:
        raise ValueError("No cities selected")

    candidate_rows = [
        row for row in read_csv(args.candidate_manifest)
        if row["split"] in splits and row["city_id"] in {c["city_id"] for c in cities}
    ]
    if not candidate_rows:
        raise ValueError("No candidate rows match the selected cities/splits")
    selected_ids: set[str] = set()
    if args.selected_manifest.is_file():
        selected_ids = {row["tile_id"] for row in read_csv(args.selected_manifest)}

    for city in cities:
        city_id = city["city_id"]
        for path, label in (
            (project_path(city["grid"]["reference_raster"]), "SAR reference"),
            (semantic_path(city_id), "V3 semantic"),
            (ua_code_path(city_id), "Urban Atlas code"),
            (roads_path(city), "OSM roads"),
        ):
            require_file(path, f"{city_id} {label}")

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "splits": sorted(splits),
            "test_inspected": "test" in splits,
            "city_ids": [city["city_id"] for city in cities],
            "candidate_rows": len(candidate_rows),
            "selected_manifest_available": args.selected_manifest.is_file(),
            "variants": [v["name"] for v in policy_config["simulation_variants"]],
            "labels_or_tiles_written": False,
        }, indent=2))
        return

    output_names = {
        "city": "dataset_v31_policy_city_summary.csv",
        "tiles": "dataset_v31_policy_candidate_tiles.csv",
        "aggregate": "dataset_v31_policy_aggregate_summary.csv",
        "audit": "dataset_v31_policy_simulation_audit.json",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [args.output_dir / name for name in output_names.values() if (args.output_dir / name).exists()]
    if collisions and not args.overwrite_metadata:
        raise FileExistsError(
            "Refusing to overwrite V3.1 simulation metadata: "
            + ", ".join(map(str, collisions))
        )

    variants = policy_config["simulation_variants"]
    road_policies = policy_config["road_policies"]
    stable_bare_codes = set(policy_config["stable_bare_land"]["retain_urban_atlas_codes"])
    candidates_by_city: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_city[row["city_id"]].append(row)

    city_output: list[dict[str, Any]] = []
    tile_output: list[dict[str, Any]] = []
    road_details: dict[str, Any] = {}
    for city in cities:
        city_id = city["city_id"]
        print(f"Simulating {city_id} ({city['split']})")
        reference_path = project_path(city["grid"]["reference_raster"])
        with rasterio.open(reference_path) as reference, rasterio.open(
            semantic_path(city_id)
        ) as semantic_ds, rasterio.open(ua_code_path(city_id)) as ua_ds:
            if not same_grid(reference, semantic_ds) or not same_grid(reference, ua_ds):
                raise ValueError(f"{city_id} policy inputs do not share the SAR grid")
            baseline = semantic_ds.read(1).astype(np.uint8)
            ua_code = ua_ds.read(1)
            allowed_values = set(np.unique(baseline).tolist())
            if not allowed_values.issubset({1, 2, 3, 4, 5, IGNORE_INDEX}):
                raise ValueError(f"{city_id} V3 semantic has invalid values: {sorted(allowed_values)}")
            masks: dict[str, np.ndarray] = {}
            road_details[city_id] = {}
            for name, road_policy in road_policies.items():
                masks[name], road_details[city_id][name] = road_policy_mask(
                    city, reference, road_config, road_policy
                )

        baseline_counts = class_counts(baseline)
        for variant in variants:
            simulated = apply_variant(
                baseline, ua_code, masks, variant, stable_bare_codes
            )
            counts = class_counts(simulated)
            if np.any((simulated != baseline) & (simulated != IGNORE_INDEX)):
                raise RuntimeError(f"{city_id} {variant['name']} introduced a non-Ignore label")
            city_row: dict[str, Any] = {
                "variant": variant["name"],
                "city_id": city_id,
                "city_name": city["city_name"],
                "split": city["split"],
                "road_policy": variant["road_policy"],
                "stable_bare": int(bool(variant["stable_bare"])),
                **counts,
                "removed_valid_pixels_vs_v3": baseline_counts["valid_pixels"] - counts["valid_pixels"],
                "retained_valid_fraction_vs_v3": counts["valid_pixels"] / baseline_counts["valid_pixels"] if baseline_counts["valid_pixels"] else 0.0,
                "retained_road_fraction_vs_v3": counts["roads_pixels"] / baseline_counts["roads_pixels"] if baseline_counts["roads_pixels"] else 0.0,
                "retained_bare_fraction_vs_v3": counts["bare_land_pixels"] / baseline_counts["bare_land_pixels"] if baseline_counts["bare_land_pixels"] else 0.0,
            }
            city_output.append(city_row)
            for candidate in candidates_by_city[city_id]:
                tile_row = candidate_metrics(candidate, simulated, selected_ids)
                tile_row["variant"] = variant["name"]
                tile_output.append(tile_row)

    aggregate_output = aggregate_policy(tile_output, policy_config["coverage_gate"])
    write_csv(args.output_dir / output_names["city"], city_output)
    write_csv(args.output_dir / output_names["tiles"], tile_output)
    write_csv(args.output_dir / output_names["aggregate"], aggregate_output)

    input_paths = [args.registry, args.road_config, args.policy_config, args.candidate_manifest]
    if args.selected_manifest.is_file():
        input_paths.append(args.selected_manifest)
    report = {
        "schema_version": "dataset-v3.1-policy-simulation-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "splits": sorted(splits),
            "city_ids": [city["city_id"] for city in cities],
            "test_inspected": "test" in splits,
        },
        "principles": {
            "five_class_objective_preserved": True,
            "removed_labels_become": "Ignore/Unknown (255)",
            "new_semantic_labels_added": False,
            "manual_qgis_required": False,
            "dataset_v3_modified": False,
        },
        "variants": variants,
        "road_policy_details_by_city": road_details,
        "stable_bare_retained_ua_codes": sorted(stable_bare_codes),
        "record_counts": {
            "city_summary_rows": len(city_output),
            "candidate_tile_rows": len(tile_output),
            "aggregate_rows": len(aggregate_output),
        },
        "inputs": {str(path): sha256_file(path) for path in input_paths},
        "outputs": {key: str(args.output_dir / name) for key, name in output_names.items() if key != "audit"},
        "safety": {
            "labels_written": False,
            "tiles_written": False,
            "dataset_v3_modified": False,
            "output_scope": str(args.output_dir),
        },
        "next_gate": "select_one_v31_policy_then_materialize_additively",
    }
    (args.output_dir / output_names["audit"]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "cities": len(cities),
        "variants": len(variants),
        "candidate_tile_rows": len(tile_output),
        "output_dir": str(args.output_dir),
        "labels_modified": False,
        "test_inspected": "test" in splits,
    }, indent=2))


if __name__ == "__main__":
    main()

