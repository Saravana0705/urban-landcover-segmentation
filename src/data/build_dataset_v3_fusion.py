"""Build conservative, additive Dataset V3 fusion masks for the 20-city batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

IGNORE_INDEX = 255
CLASS_NAMES = {1: "buildings", 2: "roads", 3: "vegetation", 4: "bare_land", 5: "water"}
DW_TO_V3 = np.array([5, 3, 3, 3, 3, 3, 1, 4, 0], dtype=np.uint8)
ACCEPTED_BARE_UA_CODES = np.array([13100, 13300, 33000], dtype=np.uint16)
SOURCE_BITS = {"osm": 1, "dynamic_world": 2, "urban_atlas": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_same_grid(reference, other, name: str) -> None:
    if (reference.width, reference.height, reference.crs, reference.transform) != (
        other.width, other.height, other.crs, other.transform
    ):
        raise ValueError(f"{name} is not aligned with the Dynamic World grid")


def dynamic_world_evidence(dataset, window) -> np.ndarray:
    mode = dataset.read(1, window=window).astype(np.int16)
    probabilities = dataset.read(list(range(2, 11)), window=window).astype(np.float32)
    observations = dataset.read(20, window=window)
    agreement = dataset.read(21, window=window)
    valid_mode = (mode >= 0) & (mode <= 8)
    mean_argmax = np.argmax(probabilities, axis=0).astype(np.int16)
    assigned_probability = np.zeros(mode.shape, dtype=np.float32)
    assigned_probability[valid_mode] = np.take_along_axis(
        probabilities, mode.clip(0, 8)[None, ...], axis=0
    )[0][valid_mode]
    confident = (
        valid_mode & (mode == mean_argmax) & (assigned_probability >= 0.60)
        & (agreement >= 0.70) & (observations >= 5)
    )
    evidence = np.zeros(mode.shape, dtype=np.uint8)
    evidence[confident] = DW_TO_V3[mode[confident]]
    return evidence


def fuse_block(osm: np.ndarray, dw: np.ndarray, ua: np.ndarray, ua_codes: np.ndarray):
    """Fuse one aligned block; zero means absent evidence, never vegetation."""
    semantic = np.full(osm.shape, IGNORE_INDEX, dtype=np.uint8)
    validity = np.zeros(osm.shape, dtype=np.uint8)
    provenance = np.zeros(osm.shape, dtype=np.uint8)
    support_count = np.zeros(osm.shape, dtype=np.uint8)

    nonzero_count = (osm > 0).astype(np.uint8) + (dw > 0) + (ua > 0)
    source_min = np.minimum(np.where(osm > 0, osm, 6), np.where(dw > 0, dw, 6))
    source_min = np.minimum(source_min, np.where(ua > 0, ua, 6))
    source_max = np.maximum(osm, np.maximum(dw, ua))
    conflict = ((nonzero_count >= 2) & (source_min != source_max)).astype(np.uint8)

    def assign(mask: np.ndarray, class_id: int) -> None:
        available = mask & (semantic == IGNORE_INDEX)
        if not available.any():
            return
        semantic[available] = class_id
        validity[available] = 1
        osm_vote = available & (osm == class_id)
        dw_vote = available & (dw == class_id)
        ua_vote = available & (ua == class_id)
        provenance[osm_vote] |= SOURCE_BITS["osm"]
        provenance[dw_vote] |= SOURCE_BITS["dynamic_world"]
        provenance[ua_vote] |= SOURCE_BITS["urban_atlas"]
        support_count[available] = (
            osm_vote[available].astype(np.uint8) + dw_vote[available].astype(np.uint8)
            + ua_vote[available].astype(np.uint8)
        )

    # Geometry-specific authoritative sources first, then consensus land cover.
    assign(osm == 1, 1)
    assign(osm == 2, 2)
    assign(((osm == 5).astype(np.uint8) + (dw == 5) + (ua == 5)) >= 2, 5)
    assign((osm == 4) & np.isin(ua_codes, ACCEPTED_BARE_UA_CODES), 4)
    assign(((osm == 3).astype(np.uint8) + (dw == 3) + (ua == 3)) >= 2, 3)
    return semantic, validity, provenance, support_count, conflict


def output_profile(reference, nodata: int) -> dict:
    profile = reference.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=nodata,
                   compress="DEFLATE", predictor=2, tiled=True,
                   blockxsize=256, blockysize=256)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True, )
    parser.add_argument("--dynamic-world", required=True, type=Path)
    parser.add_argument("--osm-semantic", required=True, type=Path)
    parser.add_argument("--osm-validity", required=True, type=Path)
    parser.add_argument("--ua-code", required=True, type=Path)
    parser.add_argument("--ua-evidence", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/labels_v3/fused"))
    parser.add_argument("--report-dir", type=Path, default=Path("metadata/dataset_v3/fusion"))
    args = parser.parse_args()

    inputs = [args.dynamic_world, args.osm_semantic, args.osm_validity, args.ua_code, args.ua_evidence]
    for path in inputs:
        if not path.is_file():
            parser.error(f"Input not found: {path}")

    out_dir = args.output_root / args.city_id
    outputs = {
        "semantic": out_dir / f"{args.city_id}_v3_semantic.tif",
        "validity": out_dir / f"{args.city_id}_v3_validity.tif",
        "provenance": out_dir / f"{args.city_id}_v3_provenance.tif",
        "support_count": out_dir / f"{args.city_id}_v3_support_count.tif",
        "conflict": out_dir / f"{args.city_id}_v3_source_conflict.tif",
    }
    report_json = args.report_dir / f"{args.city_id}_v3_fusion_qa.json"
    report_csv = args.report_dir / f"{args.city_id}_v3_class_distribution.csv"
    collisions = [p for p in [*outputs.values(), report_json, report_csv] if p.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite existing V3 batch outputs: " + ", ".join(map(str, collisions)))
    out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    counts, provenance_counts = Counter(), Counter()
    with rasterio.open(args.dynamic_world) as dw, rasterio.open(args.osm_semantic) as osm_sem, \
            rasterio.open(args.osm_validity) as osm_val, rasterio.open(args.ua_code) as ua_code, \
            rasterio.open(args.ua_evidence) as ua_evidence:
        for dataset, name in [(osm_sem, "OSM semantic"), (osm_val, "OSM validity"),
                              (ua_code, "Urban Atlas code"), (ua_evidence, "Urban Atlas evidence")]:
            assert_same_grid(dw, dataset, name)
        writers = {key: rasterio.open(path, "w", **output_profile(
            dw, IGNORE_INDEX if key == "semantic" else 0)) for key, path in outputs.items()}
        descriptions = {
            "semantic": "dataset_v3_semantic_ignore_255",
            "validity": "dataset_v3_training_validity",
            "provenance": "source_bits_osm1_dynamic_world2_urban_atlas4",
            "support_count": "agreeing_source_count_for_assigned_class",
            "conflict": "nonzero_source_class_disagreement",
        }
        grid = {"crs": str(dw.crs), "width": dw.width, "height": dw.height,
                "transform": list(dw.transform)[:6]}
        try:
            for key, writer in writers.items():
                writer.set_band_description(1, descriptions[key])
            for _, window in dw.block_windows(1):
                osm_raw = osm_sem.read(1, window=window).astype(np.uint8)
                osm = np.where(osm_val.read(1, window=window).astype(bool)
                               & (osm_raw >= 1) & (osm_raw <= 5), osm_raw, 0).astype(np.uint8)
                arrays = fuse_block(osm, dynamic_world_evidence(dw, window),
                                    ua_evidence.read(1, window=window).astype(np.uint8),
                                    ua_code.read(1, window=window).astype(np.uint16))
                for key, array in zip(outputs, arrays):
                    writers[key].write(array, 1, window=window)
                semantic, validity, provenance, support, conflict = arrays
                counts["total"] += semantic.size
                counts["ignore"] += int((semantic == IGNORE_INDEX).sum())
                counts["valid"] += int(validity.sum())
                counts["conflict"] += int(conflict.sum())
                for class_id in CLASS_NAMES:
                    counts[f"class_{class_id}"] += int((semantic == class_id).sum())
                for value, count in zip(*np.unique(provenance, return_counts=True)):
                    provenance_counts[int(value)] += int(count)
                for level in (1, 2, 3):
                    counts[f"support_{level}"] += int((support == level).sum())
        finally:
            for writer in writers.values():
                writer.close()

    if counts["valid"] + counts["ignore"] != counts["total"]:
        raise RuntimeError("Semantic accounting failure")
    rows = []
    for class_id, name in CLASS_NAMES.items():
        pixels = counts[f"class_{class_id}"]
        rows.append({"city_id": args.city_id, "class_id": class_id, "class_name": name,
                     "pixel_count": pixels, "fraction_all_pixels": pixels / counts["total"],
                     "fraction_valid_pixels": pixels / counts["valid"] if counts["valid"] else 0.0})
    rows.append({"city_id": args.city_id, "class_id": IGNORE_INDEX,
                 "class_name": "ignore_unknown", "pixel_count": counts["ignore"],
                 "fraction_all_pixels": counts["ignore"] / counts["total"],
                 "fraction_valid_pixels": ""})
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    report = {
        "schema_version": "dataset-v3-fusion-0.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "city_id": args.city_id,
        "qa_status": "PASS", "grid": grid, "class_ids": CLASS_NAMES, "ignore_index": IGNORE_INDEX,
        "policy": {"buildings": "OSM primary", "roads": "OSM primary",
                   "water": "at least two of OSM, confident Dynamic World, Urban Atlas",
                   "bare_land": "OSM bare AND Urban Atlas code in 13100/13300/33000",
                   "vegetation": "at least two of OSM, confident Dynamic World, Urban Atlas",
                   "precedence": ["buildings", "roads", "water", "bare_land", "vegetation"],
                   "unresolved": "Ignore/Unknown (255); never default vegetation"},
        "dynamic_world_confidence": {"probability_min": 0.60, "agreement_min": 0.70,
                                     "observation_count_min": 5, "mode_equals_mean_argmax": True},
        "counts": dict(counts),
        "fractions_all_pixels": {"valid": counts["valid"] / counts["total"],
                                 "ignore": counts["ignore"] / counts["total"],
                                 "source_conflict": counts["conflict"] / counts["total"]},
        "provenance_bit_counts": {str(k): v for k, v in sorted(provenance_counts.items())},
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {key: str(path) for key, path in outputs.items()},
        "class_distribution_csv": str(report_csv),
        "safety": "Additive outputs only; no V1/V2/V2.2 path is written.",
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"city_id": args.city_id, "qa_status": "PASS",
                      "valid_fraction": report["fractions_all_pixels"]["valid"],
                      "ignore_fraction": report["fractions_all_pixels"]["ignore"],
                      "json": str(report_json), "csv": str(report_csv)}, indent=2))


if __name__ == "__main__":
    main()
