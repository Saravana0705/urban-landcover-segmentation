"""Profile every Dataset V3 tile candidate without materializing tile rasters.

This is an additive, read-only analysis stage.  It reads the frozen full-city
SAR grid and V3 fusion rasters, checks exact alignment, and writes auditable
CSV/JSON statistics under ``metadata/dataset_v3/candidates``.  It does not
select tiles and does not write anywhere inside V1/V2/V2.1/V2.2 datasets.
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

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window, bounds as window_bounds


IGNORE_INDEX = 255
CLASS_NAMES = {
    1: "buildings",
    2: "roads",
    3: "vegetation",
    4: "bare_land",
    5: "water",
}
QUANTILES = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
VALID_THRESHOLDS = (0.25, 0.40, 0.50, 0.60, 0.70)
CLASS_THRESHOLDS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def load_object(path: Path) -> dict[str, Any]:
    require_file(path, "configuration")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def transforms_equal(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b))


def assert_same_grid(reference: Any, other: Any, label: str) -> None:
    checks = {
        "crs": reference.crs == other.crs,
        "width": reference.width == other.width,
        "height": reference.height == other.height,
        "transform": transforms_equal(reference.transform, other.transform),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{label} is not aligned with the SAR reference: {failed}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, q)) if array.size else 0.0


def city_paths(city: dict[str, Any]) -> dict[str, Path]:
    city_id = str(city["city_id"])
    grid = city.get("grid", {})
    sar = Path(str(grid.get("reference_raster", "")))
    root = Path("data/interim/labels_v3/fused") / city_id
    return {
        "sar": sar,
        "semantic": root / f"{city_id}_v3_semantic.tif",
        "validity": root / f"{city_id}_v3_validity.tif",
        "provenance": root / f"{city_id}_v3_provenance.tif",
        "support": root / f"{city_id}_v3_support_count.tif",
        "conflict": root / f"{city_id}_v3_source_conflict.tif",
    }


def profile_city(
    city: dict[str, Any], tile_size: int, stride: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    city_id = str(city["city_id"])
    city_name = str(city["city_name"])
    split = str(city["split"]).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split for {city_id}: {split}")
    paths = city_paths(city)
    for key, path in paths.items():
        require_file(path, f"{city_id} {key}")

    rows: list[dict[str, Any]] = []
    city_counts: Counter[str] = Counter()
    with rasterio.open(paths["sar"]) as sar, \
            rasterio.open(paths["semantic"]) as semantic, \
            rasterio.open(paths["validity"]) as validity, \
            rasterio.open(paths["provenance"]) as provenance, \
            rasterio.open(paths["support"]) as support, \
            rasterio.open(paths["conflict"]) as conflict:
        if sar.count != 2:
            raise ValueError(f"{city_id} SAR must have two bands; found {sar.count}")
        for dataset, label in (
            (semantic, "semantic"), (validity, "validity"),
            (provenance, "provenance"), (support, "support"),
            (conflict, "conflict"),
        ):
            assert_same_grid(sar, dataset, f"{city_id} {label}")
            if dataset.count != 1:
                raise ValueError(f"{city_id} {label} must have one band")

        row_starts = list(range(0, sar.height, stride))
        col_starts = list(range(0, sar.width, stride))
        tile_pixels = tile_size * tile_size

        for tile_row, row_off in enumerate(row_starts):
            for tile_col, col_off in enumerate(col_starts):
                window = Window(col_off, row_off, tile_size, tile_size)
                sem = semantic.read(1, window=window, boundless=True,
                                    fill_value=IGNORE_INDEX).astype(np.uint8)
                val = validity.read(1, window=window, boundless=True,
                                    fill_value=0).astype(np.uint8)
                prov = provenance.read(1, window=window, boundless=True,
                                       fill_value=0).astype(np.uint8)
                sup = support.read(1, window=window, boundless=True,
                                   fill_value=0).astype(np.uint8)
                con = conflict.read(1, window=window, boundless=True,
                                    fill_value=0).astype(np.uint8)

                source_width = max(0, min(tile_size, sar.width - col_off))
                source_height = max(0, min(tile_size, sar.height - row_off))
                source_pixels = source_width * source_height
                padding_pixels = tile_pixels - source_pixels
                source_mask = np.zeros((tile_size, tile_size), dtype=bool)
                source_mask[:source_height, :source_width] = True
                valid_mask = val == 1
                class_mask = (sem >= 1) & (sem <= 5)

                if np.any(valid_mask != class_mask):
                    mismatch = int(np.count_nonzero(valid_mask != class_mask))
                    raise ValueError(
                        f"{city_id} r{tile_row:03d} c{tile_col:03d}: "
                        f"semantic/validity mismatch in {mismatch} pixels"
                    )
                if np.any(valid_mask & ~source_mask):
                    raise ValueError(f"{city_id}: padding contains valid labels")
                if np.any((sup > 0) & ~valid_mask) or np.any((prov > 0) & ~valid_mask):
                    raise ValueError(f"{city_id}: support/provenance exists outside valid labels")
                if np.any(sup > 3) or np.any(prov > 7) or np.any(con > 1):
                    raise ValueError(f"{city_id}: invalid support/provenance/conflict value")

                valid_pixels = int(valid_mask.sum())
                ignore_pixels = int(np.count_nonzero((sem == IGNORE_INDEX) & source_mask))
                conflict_pixels = int(np.count_nonzero((con == 1) & source_mask))
                left, bottom, right, top = window_bounds(window, sar.transform)
                row: dict[str, Any] = {
                    "tile_id": f"{city_id}_r{tile_row:03d}_c{tile_col:03d}",
                    "city_id": city_id, "city_name": city_name, "split": split,
                    "tile_row": tile_row, "tile_col": tile_col,
                    "row_offset": row_off, "col_offset": col_off,
                    "tile_size": tile_size, "stride": stride,
                    "source_width_pixels": source_width,
                    "source_height_pixels": source_height,
                    "source_pixel_count": source_pixels,
                    "padding_pixel_count": padding_pixels,
                    "padding_fraction": padding_pixels / tile_pixels,
                    "valid_pixel_count": valid_pixels,
                    "valid_fraction_all_pixels": valid_pixels / tile_pixels,
                    "valid_fraction_source_pixels": valid_pixels / source_pixels if source_pixels else 0.0,
                    "ignore_pixel_count": ignore_pixels,
                    "ignore_fraction_source_pixels": ignore_pixels / source_pixels if source_pixels else 0.0,
                    "conflict_pixel_count": conflict_pixels,
                    "conflict_fraction_source_pixels": conflict_pixels / source_pixels if source_pixels else 0.0,
                }
                for class_id, name in CLASS_NAMES.items():
                    count = int(np.count_nonzero((sem == class_id) & valid_mask))
                    row[f"{name}_pixel_count"] = count
                    row[f"{name}_fraction_valid"] = count / valid_pixels if valid_pixels else 0.0
                    row[f"{name}_fraction_source_pixels"] = count / source_pixels if source_pixels else 0.0
                    city_counts[f"class_{class_id}"] += count
                for level in (1, 2, 3):
                    count = int(np.count_nonzero((sup == level) & valid_mask))
                    row[f"support_{level}_pixel_count"] = count
                    row[f"support_{level}_fraction_valid"] = count / valid_pixels if valid_pixels else 0.0
                    city_counts[f"support_{level}"] += count
                for bit_value in range(1, 8):
                    row[f"provenance_{bit_value}_pixel_count"] = int(
                        np.count_nonzero((prov == bit_value) & valid_mask)
                    )
                row.update({
                    "bounds_left": left, "bounds_bottom": bottom,
                    "bounds_right": right, "bounds_top": top, "crs": str(sar.crs),
                    "source_sar_path": str(paths["sar"]),
                    "source_semantic_path": str(paths["semantic"]),
                    "source_validity_path": str(paths["validity"]),
                    "source_provenance_path": str(paths["provenance"]),
                    "source_support_path": str(paths["support"]),
                    "source_conflict_path": str(paths["conflict"]),
                })
                rows.append(row)
                city_counts["source"] += source_pixels
                city_counts["padding"] += padding_pixels
                city_counts["valid"] += valid_pixels
                city_counts["ignore"] += ignore_pixels
                city_counts["conflict"] += conflict_pixels

    if city_counts["valid"] + city_counts["ignore"] != city_counts["source"]:
        raise RuntimeError(f"{city_id}: source-pixel accounting failed")
    if sum(city_counts[f"class_{x}"] for x in CLASS_NAMES) != city_counts["valid"]:
        raise RuntimeError(f"{city_id}: five-class accounting failed")
    summary: dict[str, Any] = {
        "city_id": city_id, "city_name": city_name, "split": split,
        "candidate_tile_count": len(rows),
        "source_pixel_count": city_counts["source"],
        "padding_pixel_count": city_counts["padding"],
        "valid_pixel_count": city_counts["valid"],
        "ignore_pixel_count": city_counts["ignore"],
        "valid_fraction_source_pixels": city_counts["valid"] / city_counts["source"],
        "ignore_fraction_source_pixels": city_counts["ignore"] / city_counts["source"],
        "conflict_fraction_source_pixels": city_counts["conflict"] / city_counts["source"],
    }
    for class_id, name in CLASS_NAMES.items():
        summary[f"{name}_pixel_count"] = city_counts[f"class_{class_id}"]
        summary[f"{name}_fraction_valid"] = (
            city_counts[f"class_{class_id}"] / city_counts["valid"]
        )
    for level in (1, 2, 3):
        summary[f"support_{level}_fraction_valid"] = (
            city_counts[f"support_{level}"] / city_counts["valid"]
        )
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    return rows, summary, hashes


def quantile_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = ["all", "train", "val", "test"]
    metrics = ["valid_fraction_source_pixels", "padding_fraction",
               "conflict_fraction_source_pixels"]
    metrics += [f"{name}_fraction_valid" for name in CLASS_NAMES.values()]
    for group in groups:
        selected = candidates if group == "all" else [r for r in candidates if r["split"] == group]
        for metric in metrics:
            all_values = [float(r[metric]) for r in selected]
            positive_values = [value for value in all_values if value > 0]
            for population, values in (("all_tiles", all_values), ("positive_tiles", positive_values)):
                for q in QUANTILES:
                    output.append({
                        "split": group, "metric": metric, "population": population,
                        "quantile": q, "value": quantile(values, q),
                        "population_tile_count": len(values),
                    })
    return output


def sensitivity_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_rows = [r for r in candidates if r["split"] == split]
        for min_valid in VALID_THRESHOLDS:
            coverage_rows = [
                r for r in split_rows
                if float(r["valid_fraction_source_pixels"]) >= min_valid
                and float(r["padding_fraction"]) <= 0.25
            ]
            for name in CLASS_NAMES.values():
                field = f"{name}_fraction_valid"
                for class_threshold in CLASS_THRESHOLDS:
                    qualifying = [r for r in coverage_rows if float(r[field]) >= class_threshold]
                    output.append({
                        "split": split, "class_name": name,
                        "minimum_valid_fraction_source_pixels": min_valid,
                        "maximum_padding_fraction": 0.25,
                        "minimum_class_fraction_valid": class_threshold,
                        "split_candidate_tiles": len(split_rows),
                        "coverage_eligible_tiles": len(coverage_rows),
                        "qualifying_tiles": len(qualifying),
                        "qualifying_fraction_of_coverage_eligible": (
                            len(qualifying) / len(coverage_rows) if coverage_rows else 0.0
                        ),
                        "qualifying_city_count": len({r["city_id"] for r in qualifying}),
                    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("config/dataset_v3_city_registry.json"))
    parser.add_argument("--tiling-config", type=Path, default=Path("config/tiling.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("metadata/dataset_v3/candidates"))
    parser.add_argument("--force", action="store_true", help="Replace only prior candidate-analysis reports")
    args = parser.parse_args()

    registry = load_object(args.registry)
    tiling = load_object(args.tiling_config)
    cities = registry.get("cities")
    if not isinstance(cities, list) or not cities:
        raise ValueError("Registry must contain a non-empty cities list")
    city_ids = [str(city.get("city_id")) for city in cities]
    if len(city_ids) != len(set(city_ids)):
        raise ValueError("Registry contains duplicate city IDs")
    tile_size = int(tiling["tile_size"])
    stride = int(tiling["stride"])
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if str(tiling.get("edge_policy", "")).lower() != "pad":
        raise ValueError("V3 candidate analysis requires edge_policy: pad")

    outputs = {
        "candidates": args.output_dir / "dataset_v3_candidate_tiles.csv",
        "cities": args.output_dir / "dataset_v3_candidate_city_summary.csv",
        "quantiles": args.output_dir / "dataset_v3_candidate_quantiles.csv",
        "sensitivity": args.output_dir / "dataset_v3_threshold_sensitivity.csv",
        "audit": args.output_dir / "dataset_v3_candidate_audit.json",
    }
    collisions = [path for path in outputs.values() if path.exists()]
    if collisions and not args.force:
        raise FileExistsError(
            "Candidate reports already exist; use --force to replace only these reports: "
            + ", ".join(map(str, collisions))
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    city_summaries: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for position, city in enumerate(sorted(cities, key=lambda item: item["city_id"]), start=1):
        rows, summary, hashes = profile_city(city, tile_size, stride)
        candidates.extend(rows)
        city_summaries.append(summary)
        source_hashes[str(city["city_id"])] = hashes
        print(f"Profiled {position}/{len(cities)}: {city['city_id']} ({len(rows)} candidates)")

    expected_splits = {"train": 14, "val": 3, "test": 3}
    actual_splits = Counter(str(city["split"]).lower() for city in cities)
    if dict(actual_splits) != expected_splits:
        raise ValueError(f"Expected frozen 14/3/3 city split; found {dict(actual_splits)}")
    if len(candidates) != sum(int(row["candidate_tile_count"]) for row in city_summaries):
        raise RuntimeError("Candidate row accounting failed")

    write_csv(outputs["candidates"], candidates)
    write_csv(outputs["cities"], city_summaries)
    write_csv(outputs["quantiles"], quantile_rows(candidates))
    write_csv(outputs["sensitivity"], sensitivity_rows(candidates))
    audit = {
        "schema_version": "dataset-v3-candidate-analysis-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "mode": "read_only_pre_materialization_analysis",
        "tile_size": tile_size, "stride": stride, "edge_policy": "pad",
        "city_count": len(cities), "split_city_counts": dict(actual_splits),
        "candidate_tile_count": len(candidates),
        "selection_performed": False,
        "thresholds_frozen": False,
        "source_hashes": source_hashes,
        "outputs": {key: str(path) for key, path in outputs.items() if key != "audit"},
        "safety": "Reads full-city V3/SAR rasters; writes metadata reports only; no tile or V1/V2/V2.1/V2.2 path is modified.",
    }
    outputs["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "cities": len(cities), "candidate_tiles": len(candidates),
        "output_dir": str(args.output_dir), "selection_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
