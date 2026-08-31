"""Read-only minority-label audit for additive Dataset V3.1 development.

This stage does not generate or modify labels.  It profiles:

* OSM road classes, estimated widths and Sentinel-1 contrast;
* V3 bare-land source combinations (OSM, Dynamic World, Urban Atlas);
* minority-class availability in the complete candidate grid; and
* a deterministic point sample for manual QGIS review.

The resulting evidence is used to freeze the V3.1 label policy before any
new semantic raster is written.
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
from rasterio.transform import xy
from rasterio.windows import Window

from src.data.build_dataset_v3_fusion import dynamic_world_evidence
from src.labels.buffer_roads import (
    is_underground_tunnel,
    parse_tags,
    select_total_width,
)


ACCEPTED_BARE_UA_CODES = {13100, 13300, 33000}
ROAD_CLASS_ID = 2
BARE_CLASS_ID = 4

MAJOR_ROADS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link",
}
REVIEW_ROADS = {
    "tertiary", "tertiary_link", "residential", "unclassified",
}
LIKELY_IGNORE_ROADS = {
    "service", "living_street", "pedestrian", "track", "path",
    "footway", "cycleway", "steps", "bridleway",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Dataset V3 road and bare-land evidence without writing labels."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--road-config",
        type=Path,
        default=Path("config/road_widths.yaml"),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path(
            "metadata/dataset_v3/candidates/dataset_v3_candidate_tiles.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v3_1/minority_audit"),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        help="Split to inspect; repeat as needed. Defaults to train and val only.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help="Optional city ID; repeat for multiple cities.",
    )
    parser.add_argument("--samples-per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Replace only files inside the V3.1 audit output directory.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return path


def exactly_one(patterns: Iterable[str], label: str) -> Path:
    hits = sorted(
        {
            path.resolve()
            for pattern in patterns
            for path in Path().glob(pattern)
            if path.is_file()
        }
    )
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {label}; found {len(hits)}: {hits}"
        )
    return hits[0]


def city_paths(city: dict[str, Any]) -> dict[str, Path]:
    city_id = city["city_id"]
    city_name = city["city_name"]
    folder = f"{city_id}_{city_name}"
    grid = city["grid"]
    return {
        "sar": project_path(grid["reference_raster"]),
        "osm_semantic": project_path(grid["osm_semantic"]),
        "roads": Path("data/vector/osm") / folder / "roads.gpkg",
        "dynamic_world": exactly_one(
            [
                f"data/raw/labels_v3/dynamic_world/{city_id}_*/"
                f"{city_id}_dynamic_world_v3_2025.tif",
                f"data/raw/labels_v3/dynamic_world/{city_id}_*/"
                f"{city_id}_dynamic_world_v3_pilot_2025.tif",
            ],
            f"{city_id} Dynamic World raster",
        ),
        "ua_code": Path("data/interim/labels_v3/urban_atlas")
        / city_id
        / f"{city_id}_urban_atlas_2021_code.tif",
        "ua_evidence": Path("data/interim/labels_v3/urban_atlas")
        / city_id
        / f"{city_id}_urban_atlas_2021_v3_evidence.tif",
    }


def same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader) -> bool:
    return (
        reference.crs == other.crs
        and reference.width == other.width
        and reference.height == other.height
        and reference.transform == other.transform
    )


def width_bin(width_m: float) -> str:
    if width_m < 10.0:
        return "subpixel_lt_10m"
    if width_m < 20.0:
        return "one_to_two_pixels_10_20m"
    return "two_plus_pixels_ge_20m"


def road_policy_group(highway: str) -> str:
    if highway in MAJOR_ROADS:
        return "retain_major_candidate"
    if highway in REVIEW_ROADS:
        return "manual_review_candidate"
    if highway in LIKELY_IGNORE_ROADS:
        return "likely_ignore_candidate"
    return "unknown_review_candidate"


def dilate_one_pixel(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for row_shift in range(3):
        for col_shift in range(3):
            result |= padded[
                row_shift : row_shift + mask.shape[0],
                col_shift : col_shift + mask.shape[1],
            ]
    return result


def contrast_stats(values: np.ndarray, ring_values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    ring_values = ring_values[np.isfinite(ring_values)]
    if values.size == 0 or ring_values.size == 0:
        return {
            "road_median": "", "ring_median": "", "median_difference": "",
            "robust_effect": "",
        }
    road_median = float(np.median(values))
    ring_median = float(np.median(ring_values))
    pooled = np.concatenate([values, ring_values])
    iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
    effect = (road_median - ring_median) / iqr if iqr > 0 else ""
    return {
        "road_median": road_median,
        "ring_median": ring_median,
        "median_difference": road_median - ring_median,
        "robust_effect": effect,
    }


def sample_indices(rng: np.random.Generator, count: int, wanted: int) -> np.ndarray:
    if count <= 0 or wanted <= 0:
        return np.array([], dtype=np.int64)
    return rng.choice(count, size=min(count, wanted), replace=False)


def audit_roads(
    city: dict[str, Any],
    paths: dict[str, Path],
    road_config: dict[str, Any],
    rng: np.random.Generator,
    samples_per_stratum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    city_id = city["city_id"]
    split = city["split"]
    roads = gpd.read_file(paths["roads"], layer="roads")
    if roads.empty or roads.crs is None:
        raise ValueError(f"{city_id} road layer is empty or has no CRS")
    required = {"highway", "tags", "geometry"}
    missing = required.difference(roads.columns)
    if missing:
        raise ValueError(f"{city_id} road layer lacks fields: {sorted(missing)}")

    with rasterio.open(paths["sar"]) as sar, rasterio.open(
        paths["osm_semantic"]
    ) as osm:
        if not same_grid(sar, osm):
            raise ValueError(f"{city_id} SAR and OSM semantic grids differ")
        if roads.crs != sar.crs:
            roads = roads.to_crs(sar.crs)
        sar_values = sar.read([1, 2]).astype(np.float32)
        osm_semantic = osm.read(1)
        transform = sar.transform
        out_shape = (sar.height, sar.width)
        crs_text = str(sar.crs)

    roads = roads.copy()
    roads["_tags"] = roads["tags"].apply(parse_tags)
    roads = roads.loc[
        ~roads["_tags"].apply(is_underground_tunnel)
    ].copy()
    roads["highway"] = roads["highway"].fillna("unknown").astype(str)
    resolved = roads.apply(
        lambda row: select_total_width(row["highway"], row["_tags"], road_config),
        axis=1,
    )
    roads["width_m"] = [item[0] for item in resolved]
    roads["width_source"] = [item[1] for item in resolved]
    roads["width_bin"] = roads["width_m"].map(width_bin)
    roads["policy_group"] = roads["highway"].map(road_policy_group)
    roads["length_m"] = roads.geometry.length

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    grouped = roads.groupby(["highway", "width_bin", "policy_group"], dropna=False)
    for (highway, width_group, policy_group), group in grouped:
        polygons = [
            geometry.buffer(float(width) / 2.0)
            for geometry, width in zip(group.geometry, group["width_m"], strict=True)
            if geometry is not None and not geometry.is_empty
        ]
        mask = rasterize(
            ((geometry, 1) for geometry in polygons),
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        ring = dilate_one_pixel(mask) & ~mask & (osm_semantic != ROAD_CLASS_ID)
        overlap = mask & (osm_semantic == ROAD_CLASS_ID)
        vv = contrast_stats(sar_values[0][mask], sar_values[0][ring])
        vh = contrast_stats(sar_values[1][mask], sar_values[1][ring])
        rows.append(
            {
                "city_id": city_id,
                "city_name": city["city_name"],
                "split": split,
                "highway": highway,
                "width_bin": width_group,
                "policy_group": policy_group,
                "feature_count": int(len(group)),
                "total_length_m": float(group["length_m"].sum()),
                "median_width_m": float(group["width_m"].median()),
                "min_width_m": float(group["width_m"].min()),
                "max_width_m": float(group["width_m"].max()),
                "rasterized_pixel_count": int(mask.sum()),
                "existing_osm_road_overlap_pixels": int(overlap.sum()),
                "existing_osm_road_overlap_fraction": (
                    float(overlap.sum() / mask.sum()) if mask.any() else ""
                ),
                "vv_road_median": vv["road_median"],
                "vv_ring_median": vv["ring_median"],
                "vv_median_difference": vv["median_difference"],
                "vv_robust_effect": vv["robust_effect"],
                "vh_road_median": vh["road_median"],
                "vh_ring_median": vh["ring_median"],
                "vh_median_difference": vh["median_difference"],
                "vh_robust_effect": vh["robust_effect"],
            }
        )

        chosen = sample_indices(rng, len(group), samples_per_stratum)
        for local_index in chosen:
            feature = group.iloc[int(local_index)]
            point = feature.geometry.interpolate(0.5, normalized=True)
            row, col = rasterio.transform.rowcol(transform, point.x, point.y)
            if not (0 <= row < out_shape[0] and 0 <= col < out_shape[1]):
                continue
            samples.append(
                {
                    "sample_type": "road",
                    "city_id": city_id,
                    "city_name": city["city_name"],
                    "split": split,
                    "stratum": f"{policy_group}|{width_group}|{highway}",
                    "x": float(point.x),
                    "y": float(point.y),
                    "crs": crs_text,
                    "pixel_row": int(row),
                    "pixel_col": int(col),
                    "highway": highway,
                    "estimated_width_m": float(feature["width_m"]),
                    "bare_tier": "",
                    "osm_bare": "",
                    "dw_bare": "",
                    "ua_bare": "",
                    "ua_code": "",
                    "vv": float(sar_values[0, row, col]),
                    "vh": float(sar_values[1, row, col]),
                    "review_decision": "",
                    "review_notes": "",
                }
            )
    return rows, samples


def bare_tier_masks(
    osm_semantic: np.ndarray,
    dw: np.ndarray,
    ua_evidence: np.ndarray,
    ua_code: np.ndarray,
) -> dict[str, np.ndarray]:
    osm_bare = osm_semantic == BARE_CLASS_ID
    dw_bare = dw == BARE_CLASS_ID
    ua_bare = ua_evidence == BARE_CLASS_ID
    accepted_code = np.isin(ua_code, list(ACCEPTED_BARE_UA_CODES))
    current = osm_bare & accepted_code
    ua_dw = ua_bare & dw_bare
    osm_dw = osm_bare & dw_bare
    proposed = current | ua_dw | osm_dw
    authoritative_conflict = np.isin(osm_semantic, [1, 2, 5])
    return {
        "current_v3": current,
        "tier1_ua_dw": ua_dw,
        "tier2_osm_dw": osm_dw,
        "proposed_union": proposed,
        "proposed_new": proposed & ~current,
        "proposed_conflict_with_osm_building_road_water": (
            proposed & authoritative_conflict
        ),
    }


def audit_bare(
    city: dict[str, Any],
    paths: dict[str, Path],
    rng: np.random.Generator,
    samples_per_stratum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    city_id = city["city_id"]
    datasets = {
        key: rasterio.open(paths[key])
        for key in ("sar", "osm_semantic", "dynamic_world", "ua_code", "ua_evidence")
    }
    try:
        reference = datasets["sar"]
        for key, dataset in datasets.items():
            if key != "sar" and not same_grid(reference, dataset):
                raise ValueError(f"{city_id} {key} grid differs from SAR")
        sar_values = reference.read([1, 2]).astype(np.float32)
        osm = datasets["osm_semantic"].read(1)
        ua_code = datasets["ua_code"].read(1)
        ua = datasets["ua_evidence"].read(1)
        full_window = Window(0, 0, reference.width, reference.height)
        dw = dynamic_world_evidence(datasets["dynamic_world"], full_window)
        transform = reference.transform
        crs = str(reference.crs)
    finally:
        for dataset in datasets.values():
            dataset.close()

    osm_bare = osm == BARE_CLASS_ID
    dw_bare = dw == BARE_CLASS_ID
    ua_bare = ua == BARE_CLASS_ID
    accepted_code = np.isin(ua_code, list(ACCEPTED_BARE_UA_CODES))
    tiers = bare_tier_masks(osm, dw, ua, ua_code)

    combination_counts: Counter[tuple[int, int, int, int]] = Counter()
    packed = (
        osm_bare.astype(np.uint8)
        + 2 * dw_bare.astype(np.uint8)
        + 4 * ua_bare.astype(np.uint8)
        + 8 * accepted_code.astype(np.uint8)
    )
    values, counts = np.unique(packed, return_counts=True)
    for value, count in zip(values, counts, strict=True):
        combination_counts[
            (
                int(bool(value & 1)), int(bool(value & 2)),
                int(bool(value & 4)), int(bool(value & 8)),
            )
        ] = int(count)

    rows: list[dict[str, Any]] = []
    for (osm_flag, dw_flag, ua_flag, code_flag), count in sorted(
        combination_counts.items()
    ):
        if not any((osm_flag, dw_flag, ua_flag, code_flag)):
            continue
        rows.append(
            {
                "city_id": city_id,
                "city_name": city["city_name"],
                "split": city["split"],
                "osm_bare": osm_flag,
                "dw_bare": dw_flag,
                "ua_bare": ua_flag,
                "accepted_ua_code": code_flag,
                "pixel_count": count,
                "fraction_city_pixels": count / osm.size,
            }
        )

    samples: list[dict[str, Any]] = []
    sample_masks = {
        "current_v3": tiers["current_v3"],
        "tier1_ua_dw_new": tiers["tier1_ua_dw"] & ~tiers["current_v3"],
        "tier2_osm_dw_new": tiers["tier2_osm_dw"] & ~tiers["current_v3"],
        "single_source_osm": osm_bare & ~dw_bare & ~ua_bare,
        "single_source_dw": dw_bare & ~osm_bare & ~ua_bare,
        "single_source_ua": ua_bare & ~osm_bare & ~dw_bare,
        "proposed_osm_conflict": tiers[
            "proposed_conflict_with_osm_building_road_water"
        ],
    }
    for tier_name, mask in sample_masks.items():
        locations = np.argwhere(mask)
        for chosen in sample_indices(rng, len(locations), samples_per_stratum):
            row, col = locations[int(chosen)]
            x_coord, y_coord = xy(transform, int(row), int(col), offset="center")
            samples.append(
                {
                    "sample_type": "bare_land",
                    "city_id": city_id,
                    "city_name": city["city_name"],
                    "split": city["split"],
                    "stratum": tier_name,
                    "x": float(x_coord),
                    "y": float(y_coord),
                    "crs": crs,
                    "pixel_row": int(row),
                    "pixel_col": int(col),
                    "highway": "",
                    "estimated_width_m": "",
                    "bare_tier": tier_name,
                    "osm_bare": int(osm_bare[row, col]),
                    "dw_bare": int(dw_bare[row, col]),
                    "ua_bare": int(ua_bare[row, col]),
                    "ua_code": int(ua_code[row, col]),
                    "vv": float(sar_values[0, row, col]),
                    "vh": float(sar_values[1, row, col]),
                    "review_decision": "",
                    "review_notes": "",
                }
            )

    summary = {
        "city_id": city_id,
        "city_name": city["city_name"],
        "split": city["split"],
        "total_pixels": int(osm.size),
    }
    for name, mask in tiers.items():
        summary[f"{name}_pixels"] = int(mask.sum())
        summary[f"{name}_fraction"] = float(mask.mean())
    rows.append({**summary, "record_type": "tier_summary"})
    return rows, samples


def audit_candidates(path: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    accum: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in allowed_splits:
                continue
            key = (split, row["city_id"])
            valid = float(row["valid_fraction_source_pixels"])
            padding = float(row["padding_fraction"])
            eligible = valid >= 0.40 and padding <= 0.25
            counter = accum[key]
            counter["candidate_tiles"] += 1
            if not eligible:
                continue
            counter["coverage_eligible_tiles"] += 1
            bare = float(row["bare_land_fraction_valid"])
            roads = float(row["roads_fraction_valid"])
            for label, threshold in (
                ("bare_ge_0_5pct", 0.005),
                ("bare_ge_1pct", 0.01),
                ("bare_ge_2pct", 0.02),
                ("bare_ge_5pct", 0.05),
                ("roads_ge_5pct", 0.05),
                ("roads_ge_7_5pct", 0.075),
                ("roads_ge_10pct", 0.10),
                ("roads_ge_20pct", 0.20),
            ):
                value = bare if label.startswith("bare") else roads
                if value >= threshold:
                    counter[label] += 1
    return [
        {"split": split, "city_id": city_id, **dict(counter)}
        for (split, city_id), counter in sorted(accum.items())
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    require_file(args.registry, "V3 city registry")
    require_file(args.road_config, "road-width configuration")
    require_file(args.candidate_manifest, "V3 candidate manifest")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    road_config = yaml.safe_load(args.road_config.read_text(encoding="utf-8"))
    splits = set(args.split or ["train", "val"])
    cities = [city for city in registry["cities"] if city["split"] in splits]
    if args.city:
        requested = set(args.city)
        known = {city["city_id"] for city in cities}
        unknown = requested - known
        if unknown:
            raise ValueError(
                f"Requested cities are unknown or outside selected splits: {sorted(unknown)}"
            )
        cities = [city for city in cities if city["city_id"] in requested]
    if not cities:
        raise ValueError("No cities selected")

    resolved: dict[str, dict[str, Path]] = {}
    for city in cities:
        paths = city_paths(city)
        for key, path in paths.items():
            require_file(path, f"{city['city_id']} {key}")
        resolved[city["city_id"]] = paths

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "splits": sorted(splits),
                    "city_ids": [city["city_id"] for city in cities],
                    "candidate_manifest": str(args.candidate_manifest),
                    "output_dir": str(args.output_dir),
                    "selection_or_label_writes_performed": False,
                },
                indent=2,
            )
        )
        return

    output_names = {
        "road": "dataset_v31_road_highway_audit.csv",
        "bare": "dataset_v31_bare_source_audit.csv",
        "candidate": "dataset_v31_candidate_minority_summary.csv",
        "samples": "dataset_v31_qgis_review_sample.csv",
        "audit": "dataset_v31_minority_audit.json",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [
        args.output_dir / name
        for name in output_names.values()
        if (args.output_dir / name).exists()
    ]
    if collisions and not args.overwrite_metadata:
        raise FileExistsError(
            "Refusing to overwrite V3.1 audit metadata: "
            + ", ".join(map(str, collisions))
        )

    rng = np.random.default_rng(args.seed)
    road_rows: list[dict[str, Any]] = []
    bare_rows: list[dict[str, Any]] = []
    review_samples: list[dict[str, Any]] = []
    for city in cities:
        city_id = city["city_id"]
        print(f"Auditing {city_id} ({city['split']})")
        city_road, city_road_samples = audit_roads(
            city,
            resolved[city_id],
            road_config,
            rng,
            args.samples_per_stratum,
        )
        city_bare, city_bare_samples = audit_bare(
            city,
            resolved[city_id],
            rng,
            args.samples_per_stratum,
        )
        road_rows.extend(city_road)
        bare_rows.extend(city_bare)
        review_samples.extend(city_road_samples)
        review_samples.extend(city_bare_samples)

    candidate_rows = audit_candidates(args.candidate_manifest, splits)
    write_csv(args.output_dir / output_names["road"], road_rows)
    write_csv(args.output_dir / output_names["bare"], bare_rows)
    write_csv(args.output_dir / output_names["candidate"], candidate_rows)
    write_csv(args.output_dir / output_names["samples"], review_samples)

    tier_summaries = [row for row in bare_rows if row.get("record_type") == "tier_summary"]
    totals = Counter()
    for row in tier_summaries:
        for key, value in row.items():
            if key.endswith("_pixels") and isinstance(value, int):
                totals[key] += value
    report = {
        "schema_version": "dataset-v3.1-minority-audit-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "splits": sorted(splits),
            "city_ids": [city["city_id"] for city in cities],
            "test_inspected": "test" in splits,
        },
        "baseline_policy": {
            "roads": "all OSM roads are authoritative after V3 rasterization",
            "bare_land": "OSM bare AND Urban Atlas code in 13100/13300/33000",
        },
        "diagnostic_candidate_policy": {
            "road_major_classes": sorted(MAJOR_ROADS),
            "road_manual_review_classes": sorted(REVIEW_ROADS),
            "road_likely_ignore_classes": sorted(LIKELY_IGNORE_ROADS),
            "bare_tier_1": "confident Urban Atlas bare AND confident Dynamic World bare",
            "bare_tier_2": "OSM bare AND confident Dynamic World bare",
            "bare_existing": "OSM bare AND accepted Urban Atlas code",
            "warning": "These are audit strata, not an approved V3.1 label policy.",
        },
        "record_counts": {
            "road_summary_rows": len(road_rows),
            "bare_summary_rows": len(bare_rows),
            "candidate_summary_rows": len(candidate_rows),
            "qgis_review_samples": len(review_samples),
        },
        "bare_pixel_totals": dict(totals),
        "inputs": {
            "registry": {
                "path": str(args.registry),
                "sha256": sha256_file(args.registry),
            },
            "road_config": {
                "path": str(args.road_config),
                "sha256": sha256_file(args.road_config),
            },
            "candidate_manifest": {
                "path": str(args.candidate_manifest),
                "sha256": sha256_file(args.candidate_manifest),
            },
        },
        "outputs": {
            key: str(args.output_dir / name)
            for key, name in output_names.items()
            if key != "audit"
        },
        "safety": {
            "labels_written": False,
            "tiles_written": False,
            "dataset_v3_modified": False,
            "output_scope": str(args.output_dir),
        },
        "next_gate": "manual_qgis_review_and_v31_label_policy_decision",
    }
    audit_path = args.output_dir / output_names["audit"]
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "cities": len(cities),
                "road_summary_rows": len(road_rows),
                "bare_summary_rows": len(bare_rows),
                "qgis_review_samples": len(review_samples),
                "output_dir": str(args.output_dir),
                "labels_modified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
