"""Read-only per-code Urban Atlas agreement audit for V3 pilot decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


DW_TO_V3 = np.array([5, 3, 3, 3, 3, 3, 1, 4, 0], dtype=np.uint8)
UA_NAMES = {
    11100: "Continuous urban fabric", 11210: "Discontinuous dense urban fabric",
    11220: "Discontinuous medium density urban fabric", 11230: "Discontinuous low density urban fabric",
    11240: "Discontinuous very low density urban fabric", 11300: "Isolated structures",
    12100: "Industrial/commercial/public/military/private units", 12210: "Fast transit roads",
    12220: "Other roads", 12230: "Railways", 12300: "Port areas", 12400: "Airports",
    13100: "Mineral extraction and dump sites", 13300: "Construction sites",
    13400: "Land without current use", 14110: "Green urban areas - public",
    14120: "Green urban areas - private", 14130: "Green urban areas - unknown access",
    14200: "Sports and leisure facilities", 21000: "Arable land", 22000: "Permanent crops",
    23000: "Pastures", 24000: "Complex/mixed cultivation", 31000: "Forests",
    32000: "Herbaceous vegetation", 33000: "Open spaces with little/no vegetation",
    40000: "Wetlands", 50000: "Water",
}


def same_grid(reference, other, name):
    if (reference.width, reference.height, reference.crs, reference.transform) != (
        other.width, other.height, other.crs, other.transform
    ):
        raise ValueError(f"{name} is not aligned to the Dynamic World grid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True, choices=["DE03", "DE14"])
    parser.add_argument("--ua-code-raster", required=True, type=Path)
    parser.add_argument("--dynamic-world", required=True, type=Path)
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--validity", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("metadata/dataset_v3/pilot/ua_code_agreement"))
    args = parser.parse_args()

    counts = defaultdict(lambda: {
        "total": 0,
        **{f"osm_{i}": 0 for i in range(6)},
        **{f"dw_{i}": 0 for i in range(6)},
        "osm_bare_and_dw_bare": 0,
    })
    with rasterio.open(args.dynamic_world) as dw, rasterio.open(args.ua_code_raster) as ua, \
            rasterio.open(args.semantic) as semantic, rasterio.open(args.validity) as validity:
        same_grid(dw, ua, "Urban Atlas code raster")
        same_grid(dw, semantic, "OSM semantic raster")
        same_grid(dw, validity, "OSM validity raster")
        for _, window in dw.block_windows(1):
            ua_code = ua.read(1, window=window).astype(np.int32)
            mode = dw.read(1, window=window).astype(np.int16)
            probabilities = dw.read(list(range(2, 11)), window=window).astype(np.float32)
            observations = dw.read(20, window=window)
            agreement = dw.read(21, window=window)
            mean_argmax = np.argmax(probabilities, axis=0).astype(np.int16)
            valid_mode = (mode >= 0) & (mode <= 8)
            assigned_probability = np.zeros(mode.shape, dtype=np.float32)
            assigned_probability[valid_mode] = np.take_along_axis(
                probabilities, mode.clip(0, 8)[None, ...], axis=0
            )[0][valid_mode]
            confident = (
                valid_mode & (mean_argmax == mode) & (assigned_probability >= 0.60)
                & (agreement >= 0.70) & (observations >= 5)
            )
            dw_target = np.zeros(mode.shape, dtype=np.uint8)
            dw_target[confident] = DW_TO_V3[mode[confident]]
            osm = semantic.read(1, window=window).astype(np.uint8)
            osm_target = np.where(validity.read(1, window=window).astype(bool), osm, 0)
            osm_target[(osm_target < 0) | (osm_target > 5)] = 0

            for code in np.unique(ua_code):
                code = int(code)
                if code == 0:
                    continue
                selected = ua_code == code
                entry = counts[code]
                entry["total"] += int(selected.sum())
                for class_id in range(6):
                    entry[f"osm_{class_id}"] += int((selected & (osm_target == class_id)).sum())
                    entry[f"dw_{class_id}"] += int((selected & (dw_target == class_id)).sum())
                entry["osm_bare_and_dw_bare"] += int(
                    (selected & (osm_target == 4) & (dw_target == 4)).sum()
                )

    rows = []
    for code, values in sorted(counts.items()):
        total = values["total"]
        row = {
            "city_id": args.city_id,
            "code_2021": code,
            "class_2021": UA_NAMES.get(code, "<UNMAPPED_NAME>"),
            **values,
            "osm_bare_fraction": values["osm_4"] / total,
            "dw_bare_fraction": values["dw_4"] / total,
            "osm_bare_and_dw_bare_fraction": values["osm_bare_and_dw_bare"] / total,
        }
        rows.append(row)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / f"{args.city_id}_ua_code_agreement.csv"
    json_path = args.report_dir / f"{args.city_id}_ua_code_agreement.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite existing UA code-agreement reports")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "audit_version": "v3-pilot-ua-code-agreement-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": args.city_id,
        "qa_status": "PASS",
        "dynamic_world_rule": {"probability_min": 0.60, "agreement_min": 0.70, "observations_min": 5},
        "csv": str(csv_path),
        "notes": [
            "Class 0 means absent/unconfident evidence, never vegetation.",
            "This audit separates Urban Atlas bare candidate codes before any final fusion decision.",
        ],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"city_id": args.city_id, "qa_status": "PASS", "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
