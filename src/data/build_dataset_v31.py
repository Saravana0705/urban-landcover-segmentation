"""Build additive full-city Dataset V3.1 labels from frozen Dataset V3.

V3.1 is a conservative subset of V3. No new semantic label is introduced.
Rejected road and bare-land pixels become Ignore (255). Frozen V3 inputs are
read-only and all outputs are written to new V3.1 paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

from src.evaluation.simulate_dataset_v31_policy import (
    class_counts,
    project_path,
    road_policy_mask,
    same_grid,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--road-config", type=Path, default=Path("config/road_widths.yaml")
    )
    parser.add_argument(
        "--policy-config", type=Path,
        default=Path("config/dataset_v31_frozen_policy.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/interim/labels_v3_1/fused"),
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("metadata/dataset_v3_1/fusion"),
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--city", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip only cities having a complete five-raster output and reports.",
    )
    return parser.parse_args()


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


def v3_paths(city_id: str) -> dict[str, Path]:
    root = Path("data/interim/labels_v3/fused") / city_id
    return {
        "semantic": root / f"{city_id}_v3_semantic.tif",
        "validity": root / f"{city_id}_v3_validity.tif",
        "provenance": root / f"{city_id}_v3_provenance.tif",
        "support": root / f"{city_id}_v3_support_count.tif",
        "conflict": root / f"{city_id}_v3_source_conflict.tif",
    }


def v31_paths(output_root: Path, city_id: str) -> dict[str, Path]:
    root = output_root / city_id
    return {
        "semantic": root / f"{city_id}_v31_semantic.tif",
        "validity": root / f"{city_id}_v31_validity.tif",
        "provenance": root / f"{city_id}_v31_provenance.tif",
        "support": root / f"{city_id}_v31_support_count.tif",
        "conflict": root / f"{city_id}_v31_source_conflict.tif",
    }


def reports(report_dir: Path, city_id: str) -> dict[str, Path]:
    return {
        "json": report_dir / f"{city_id}_v31_policy_qa.json",
        "csv": report_dir / f"{city_id}_v31_class_distribution.csv",
    }


def all_city_artifacts(
    output_root: Path, report_dir: Path, city_id: str
) -> list[Path]:
    return [*v31_paths(output_root, city_id).values(), *reports(report_dir, city_id).values()]


def output_profile(reference: rasterio.DatasetReader, nodata: int) -> dict[str, Any]:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=nodata,
        compress="DEFLATE", predictor=2, tiled=True,
        blockxsize=256, blockysize=256,
    )
    return profile


def write_raster(
    path: Path,
    reference: rasterio.DatasetReader,
    array: np.ndarray,
    nodata: int,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output must be removed manually: {temporary}")
    with rasterio.open(temporary, "w", **output_profile(reference, nodata)) as dst:
        dst.write(array.astype(np.uint8), 1)
        dst.set_band_description(1, description)
    temporary.replace(path)


def write_class_csv(path: Path, city_id: str, counts: dict[str, int]) -> None:
    valid = counts["valid_pixels"]
    total = counts["total_pixels"]
    rows: list[dict[str, Any]] = []
    for class_id, name in CLASS_NAMES.items():
        pixels = counts[f"{name}_pixels"]
        rows.append({
            "city_id": city_id,
            "class_id": class_id,
            "class_name": name,
            "pixel_count": pixels,
            "fraction_all_pixels": pixels / total,
            "fraction_valid_pixels": pixels / valid if valid else 0.0,
        })
    rows.append({
        "city_id": city_id,
        "class_id": IGNORE_INDEX,
        "class_name": "ignore_unknown",
        "pixel_count": counts["ignore_pixels"],
        "fraction_all_pixels": counts["ignore_pixels"] / total,
        "fraction_valid_pixels": "",
    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_city(
    city: dict[str, Any],
    road_config: dict[str, Any],
    policy: dict[str, Any],
    output_root: Path,
    report_dir: Path,
) -> dict[str, Any]:
    city_id = str(city["city_id"])
    reference_path = project_path(city["grid"]["reference_raster"])
    ua_path = (
        Path("data/interim/labels_v3/urban_atlas")
        / city_id
        / f"{city_id}_urban_atlas_2021_code.tif"
    )
    inputs = {"sar": reference_path, "ua_code": ua_path, **v3_paths(city_id)}
    for name, path in inputs.items():
        require_file(path, f"{city_id} {name}")

    output_paths = v31_paths(output_root, city_id)
    report_paths = reports(report_dir, city_id)
    collisions = [path for path in [*output_paths.values(), *report_paths.values()] if path.exists()]
    if collisions:
        raise FileExistsError(
            f"Refusing to overwrite {city_id} V3.1 outputs: "
            + ", ".join(map(str, collisions))
        )

    datasets = {name: rasterio.open(path) for name, path in inputs.items()}
    try:
        reference = datasets["sar"]
        for name, dataset in datasets.items():
            if name != "sar" and not same_grid(reference, dataset):
                raise ValueError(f"{city_id} {name} is not aligned with SAR")
        baseline = datasets["semantic"].read(1).astype(np.uint8)
        baseline_validity = datasets["validity"].read(1).astype(np.uint8)
        provenance = datasets["provenance"].read(1).astype(np.uint8)
        support = datasets["support"].read(1).astype(np.uint8)
        conflict = datasets["conflict"].read(1).astype(np.uint8)
        ua_code = datasets["ua_code"].read(1)

        baseline_class = (baseline >= 1) & (baseline <= 5)
        if np.any((baseline_validity == 1) != baseline_class):
            raise ValueError(f"{city_id} V3 semantic/validity mismatch")
        if np.any((provenance > 0) & ~baseline_class) or np.any((support > 0) & ~baseline_class):
            raise ValueError(f"{city_id} V3 provenance/support outside valid labels")

        road_mask, road_details = road_policy_mask(
            city, reference, road_config, policy["road_policy"]
        )
        current_road = baseline == 2
        current_bare = baseline == 4
        stable_bare = np.isin(
            ua_code,
            policy["bare_land_policy"]["retain_urban_atlas_codes"],
        )
        removed_road = current_road & ~road_mask
        removed_bare = current_bare & ~stable_bare
        removed = removed_road | removed_bare

        semantic = baseline.copy()
        semantic[removed] = IGNORE_INDEX
        validity = ((semantic >= 1) & (semantic <= 5)).astype(np.uint8)
        provenance = provenance.copy()
        provenance[removed] = 0
        support = support.copy()
        support[removed] = 0

        if np.any((semantic != baseline) & (semantic != IGNORE_INDEX)):
            raise RuntimeError(f"{city_id} V3.1 introduced a non-Ignore semantic label")
        if np.any((provenance > 0) & (validity == 0)) or np.any((support > 0) & (validity == 0)):
            raise RuntimeError(f"{city_id} V3.1 support/provenance accounting failure")

        baseline_counts = class_counts(baseline)
        counts = class_counts(semantic)
        descriptions = {
            "semantic": "dataset_v3_1_semantic_ignore_255",
            "validity": "dataset_v3_1_training_validity",
            "provenance": "v3_1_retained_source_bits_osm1_dynamic_world2_urban_atlas4",
            "support": "v3_1_retained_agreeing_source_count",
            "conflict": "v3_source_conflict_diagnostic_unchanged",
        }
        arrays = {
            "semantic": semantic,
            "validity": validity,
            "provenance": provenance,
            "support": support,
            "conflict": conflict,
        }
        for name, path in output_paths.items():
            write_raster(
                path, reference, arrays[name],
                IGNORE_INDEX if name == "semantic" else 0,
                descriptions[name],
            )
    finally:
        for dataset in datasets.values():
            dataset.close()

    report_dir.mkdir(parents=True, exist_ok=True)
    write_class_csv(report_paths["csv"], city_id, counts)
    report = {
        "schema_version": "dataset-v3.1-label-policy-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "city_id": city_id,
        "city_name": city["city_name"],
        "split": city["split"],
        "policy": policy,
        "road_policy_details": road_details,
        "counts": counts,
        "removed": {
            "road_pixels": int(removed_road.sum()),
            "bare_land_pixels": int(removed_bare.sum()),
            "total_pixels": int(removed.sum()),
        },
        "retention": {
            "valid": counts["valid_pixels"] / baseline_counts["valid_pixels"],
            "roads": counts["roads_pixels"] / baseline_counts["roads_pixels"] if baseline_counts["roads_pixels"] else 0.0,
            "bare_land": counts["bare_land_pixels"] / baseline_counts["bare_land_pixels"] if baseline_counts["bare_land_pixels"] else 0.0,
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in output_paths.items()},
        "class_distribution_csv": str(report_paths["csv"]),
        "safety": {
            "parent_v3_modified": False,
            "new_labels_added": False,
            "removed_labels_become": "Ignore/Unknown (255)",
        },
    }
    report_paths["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.registry, "V3 city registry"),
        (args.road_config, "road-width configuration"),
        (args.policy_config, "frozen V3.1 policy"),
    ):
        require_file(path, label)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    road_config = yaml.safe_load(args.road_config.read_text(encoding="utf-8"))
    policy = yaml.safe_load(args.policy_config.read_text(encoding="utf-8"))
    cities = list(registry["cities"])
    if args.city:
        requested = set(args.city)
        known = {str(city["city_id"]) for city in cities}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown city IDs: {sorted(unknown)}")
        cities = [city for city in cities if city["city_id"] in requested]

    if policy.get("removed_label_policy") != "ignore_255":
        raise ValueError("V3.1 builder permits only removed_label_policy: ignore_255")
    if policy.get("new_labels_added") is not False or policy.get("class_taxonomy_changed") is not False:
        raise ValueError("V3.1 must remain a five-class conservative subset of V3")

    states: dict[str, str] = {}
    for city in cities:
        city_id = str(city["city_id"])
        artifacts = all_city_artifacts(args.output_root, args.report_dir, city_id)
        present = [path.exists() and path.stat().st_size > 0 for path in artifacts]
        if all(present):
            states[city_id] = "COMPLETE"
        elif any(present):
            states[city_id] = "PARTIAL_REFUSE"
        else:
            states[city_id] = "PENDING"
    partial = [city_id for city_id, state in states.items() if state == "PARTIAL_REFUSE"]
    if partial:
        raise RuntimeError(f"Partial V3.1 city outputs require inspection: {partial}")
    complete = [city_id for city_id, state in states.items() if state == "COMPLETE"]
    if complete and not args.resume:
        raise FileExistsError(
            f"Complete V3.1 outputs already exist for {complete}; use --resume to skip them"
        )

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "city_count": len(cities),
            "city_states": states,
            "test_cities_processed_blindly": True,
            "parent_v3_modified": False,
            "labels_or_tiles_written": False,
        }, indent=2))
        return

    reports_written: list[dict[str, Any]] = []
    skipped: list[str] = []
    for city in sorted(cities, key=lambda item: item["city_id"]):
        city_id = str(city["city_id"])
        if states[city_id] == "COMPLETE" and args.resume:
            print(f"Skipping complete {city_id}")
            skipped.append(city_id)
            continue
        print(f"Building {city_id} ({city['split']})")
        reports_written.append(
            build_city(city, road_config, policy, args.output_root, args.report_dir)
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    batch_path = args.report_dir / "dataset_v31_build_audit.json"
    if batch_path.exists():
        raise FileExistsError(f"Refusing to overwrite V3.1 batch audit: {batch_path}")
    batch = {
        "schema_version": "dataset-v3.1-build-audit-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_version": "v3.1",
        "parent_dataset_version": "v3",
        "requested_city_count": len(cities),
        "written_city_ids": [report["city_id"] for report in reports_written],
        "skipped_complete_city_ids": skipped,
        "policy_config": {"path": str(args.policy_config), "sha256": sha256_file(args.policy_config)},
        "output_root": str(args.output_root),
        "parent_v3_modified": False,
        "next_gate": "v3_1_candidate_analysis_and_selection",
    }
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "written_cities": len(reports_written),
        "skipped_cities": len(skipped),
        "output_root": str(args.output_root),
        "next": "Dataset V3.1 candidate analysis",
    }, indent=2))


if __name__ == "__main__":
    main()

