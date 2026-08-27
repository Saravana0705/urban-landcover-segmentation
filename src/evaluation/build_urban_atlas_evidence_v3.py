"""Build non-final Urban Atlas evidence rasters and three-source batch statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds


# Direct/supporting evidence only. Ambiguous classes deliberately remain 0.
UA_TO_V3 = {
    12210: 2, 12220: 2, 12230: 2,  # transport support; OSM remains primary
    14110: 3, 14120: 3, 14130: 3,
    21000: 3, 22000: 3, 23000: 3, 31000: 3, 32000: 3,
    13100: 4, 13300: 4, 33000: 4,
    50000: 5,
}
DW_TO_V3 = np.array([5, 3, 3, 3, 3, 3, 1, 4, 0], dtype=np.uint8)
CLASS_NAMES = {1: "buildings", 2: "roads", 3: "vegetation", 4: "bare_land", 5: "water"}


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
        raise ValueError(f"{name} is not aligned to the Dynamic World grid")


def write_raster(path: Path, array: np.ndarray, reference, dtype: str, nodata: int, description: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff", count=1, dtype=dtype, nodata=nodata,
        compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(dtype), 1)
        destination.set_band_description(1, description)


def build(args) -> tuple[dict, list[dict]]:
    with rasterio.open(args.dynamic_world) as dw:
        with rasterio.open(args.semantic) as semantic, rasterio.open(args.validity) as validity:
            assert_same_grid(dw, semantic, "OSM semantic raster")
            assert_same_grid(dw, validity, "OSM validity raster")

            target_bounds_3035 = transform_bounds(
                dw.crs, "EPSG:3035", *dw.bounds, densify_pts=21
            )
            ua = pyogrio.read_dataframe(
                args.urban_atlas,
                bbox=target_bounds_3035,
                columns=["code_2021", "class_2021"],
            )
            if str(ua.crs).upper() != "EPSG:3035":
                raise ValueError(f"Expected Urban Atlas EPSG:3035, found {ua.crs}")
            ua = ua[ua.geometry.notna()].copy()
            ua["code_2021"] = ua["code_2021"].astype(str).str.strip().astype(int)
            ua = ua.to_crs(dw.crs)

            code_shapes = ((geometry, int(code)) for geometry, code in zip(ua.geometry, ua["code_2021"]))
            ua_codes = rasterize(
                code_shapes,
                out_shape=(dw.height, dw.width),
                transform=dw.transform,
                fill=0,
                dtype="uint16",
                all_touched=False,
            )
            ua_evidence = np.zeros(ua_codes.shape, dtype=np.uint8)
            for ua_code, v3_class in UA_TO_V3.items():
                ua_evidence[ua_codes == ua_code] = v3_class

            output_dir = args.output_root / args.city_id
            code_path = output_dir / f"{args.city_id}_urban_atlas_2021_code.tif"
            evidence_path = output_dir / f"{args.city_id}_urban_atlas_2021_v3_evidence.tif"
            write_raster(code_path, ua_codes, dw, "uint16", 0, "urban_atlas_code_2021")
            write_raster(evidence_path, ua_evidence, dw, "uint8", 0, "urban_atlas_v3_supporting_evidence")

            ua_dw = np.zeros((6, 6), dtype=np.int64)
            ua_osm = np.zeros((6, 6), dtype=np.int64)
            dw_osm = np.zeros((6, 6), dtype=np.int64)
            class_counts = {class_id: {key: 0 for key in (
                "ua", "dw_confident", "osm", "ua_dw_agree", "ua_osm_agree",
                "dw_osm_agree", "triple_agree"
            )} for class_id in CLASS_NAMES}

            for _, window in dw.block_windows(1):
                row0, col0 = int(window.row_off), int(window.col_off)
                row1, col1 = row0 + int(window.height), col0 + int(window.width)
                ua_block = ua_evidence[row0:row1, col0:col1]
                mode = dw.read(1, window=window).astype(np.int16)
                probabilities = dw.read(list(range(2, 11)), window=window).astype(np.float32)
                observations = dw.read(20, window=window)
                agreement = dw.read(21, window=window)
                mean_argmax = np.argmax(probabilities, axis=0).astype(np.int16)
                mode_valid = (mode >= 0) & (mode <= 8)
                mode_probability = np.zeros(mode.shape, dtype=np.float32)
                mode_probability[mode_valid] = np.take_along_axis(
                    probabilities, mode.clip(0, 8)[None, ...], axis=0
                )[0][mode_valid]
                dw_confident = (
                    mode_valid & (mean_argmax == mode) & (mode_probability >= 0.60)
                    & (agreement >= 0.70) & (observations >= 5)
                )
                dw_target = np.zeros(mode.shape, dtype=np.uint8)
                dw_target[dw_confident] = DW_TO_V3[mode[dw_confident]]
                osm = semantic.read(1, window=window).astype(np.uint8)
                osm_valid = validity.read(1, window=window).astype(bool) & (osm >= 1) & (osm <= 5)
                osm_target = np.where(osm_valid, osm, 0).astype(np.uint8)

                use = (ua_block <= 5) & (dw_target <= 5)
                np.add.at(ua_dw, (ua_block[use], dw_target[use]), 1)
                use = (ua_block <= 5) & (osm_target <= 5)
                np.add.at(ua_osm, (ua_block[use], osm_target[use]), 1)
                use = (dw_target <= 5) & (osm_target <= 5)
                np.add.at(dw_osm, (dw_target[use], osm_target[use]), 1)

                for class_id in CLASS_NAMES:
                    ua_class = ua_block == class_id
                    dw_class = dw_target == class_id
                    osm_class = osm_target == class_id
                    values = class_counts[class_id]
                    values["ua"] += int(ua_class.sum())
                    values["dw_confident"] += int(dw_class.sum())
                    values["osm"] += int(osm_class.sum())
                    values["ua_dw_agree"] += int((ua_class & dw_class).sum())
                    values["ua_osm_agree"] += int((ua_class & osm_class).sum())
                    values["dw_osm_agree"] += int((dw_class & osm_class).sum())
                    values["triple_agree"] += int((ua_class & dw_class & osm_class).sum())

    total_pixels = int(ua_codes.size)
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        row = {"class_id": class_id, "class_name": class_name, **class_counts[class_id]}
        for key in list(class_counts[class_id]):
            row[f"{key}_fraction_all_pixels"] = row[key] / total_pixels
        rows.append(row)

    report = {
        "audit_version": "v3-three-source-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": args.city_id,
        "qa_status": "PASS",
        "total_pixels": total_pixels,
        "urban_atlas_source": str(args.urban_atlas),
        "urban_atlas_source_sha256": sha256(args.urban_atlas),
        "dynamic_world_source": str(args.dynamic_world),
        "osm_semantic_source": str(args.semantic),
        "osm_validity_source": str(args.validity),
        "urban_atlas_code_raster": str(code_path),
        "urban_atlas_evidence_raster": str(evidence_path),
        "direct_urban_atlas_crosswalk": {str(k): v for k, v in UA_TO_V3.items()},
        "dynamic_world_confidence_rule": {
            "modal_mean_argmax_consensus": True,
            "assigned_probability_min": 0.60,
            "temporal_agreement_min": 0.70,
            "observation_count_min": 5,
        },
        "pairwise_cross_tabs": {
            "urban_atlas_rows_dynamic_world_columns": ua_dw.tolist(),
            "urban_atlas_rows_osm_columns": ua_osm.tolist(),
            "dynamic_world_rows_osm_columns": dw_osm.tolist(),
        },
        "notes": [
            "These are evidence rasters and agreement statistics, not final V3 labels.",
            "Urban Atlas transport is supporting evidence only; OSM remains the road geometry source.",
            "Urban Atlas classes 13400, 14200 and 40000 remain unresolved in this direct crosswalk.",
            "Class 0 means no direct evidence, not vegetation.",
        ],
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True, )
    parser.add_argument("--urban-atlas", required=True, type=Path)
    parser.add_argument("--dynamic-world", required=True, type=Path)
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--validity", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/labels_v3/urban_atlas"))
    parser.add_argument("--report-dir", type=Path, default=Path("metadata/dataset_v3/three_source_audit"))
    args = parser.parse_args()
    for path in (args.urban_atlas, args.dynamic_world, args.semantic, args.validity):
        if not path.is_file():
            parser.error(f"Input not found: {path}")

    report, rows = build(args)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"{args.city_id}_three_source_audit.json"
    csv_path = args.report_dir / f"{args.city_id}_three_source_class_summary.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite an existing three-source report")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"city_id": args.city_id, "qa_status": "PASS", "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
