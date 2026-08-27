"""Read-only Urban Atlas 2021 area audit within exact V3 pilot SAR footprints."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pyogrio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform


CITIES = {
    "DE03": {
        "sar_crs": "EPSG:32632",
        "bounds": [441391.24642737134, 5413394.106587356, 471391.24642737134, 5443394.106587356],
    },
    "DE14": {
        "sar_crs": "EPSG:32633",
        "bounds": [376779.25925275, 5805072.1592110265, 406779.25925275, 5835072.1592110265],
    },
}
UA_CRS = "EPSG:3035"


def analyze(vector_path: Path, city_id: str) -> tuple[dict, list[dict]]:
    city = CITIES[city_id]
    sar_box = box(*city["bounds"])
    transformer = Transformer.from_crs(city["sar_crs"], UA_CRS, always_xy=True)
    aoi = transform(transformer.transform, sar_box)

    frame = pyogrio.read_dataframe(
        vector_path,
        bbox=aoi.bounds,
        columns=["code_2021", "class_2021"],
    )
    source_feature_count = int(len(frame))
    null_geometry_count = int(frame.geometry.isna().sum())
    invalid_geometry_count = int((~frame.geometry.is_valid & frame.geometry.notna()).sum())
    frame = frame[frame.geometry.notna()].copy()
    frame["clipped_area_m2"] = frame.geometry.intersection(aoi).area
    frame = frame[frame["clipped_area_m2"] > 0].copy()

    grouped = (
        frame.groupby(["code_2021", "class_2021"], dropna=False)
        .agg(feature_count=("clipped_area_m2", "size"), area_m2=("clipped_area_m2", "sum"))
        .reset_index()
        .sort_values("area_m2", ascending=False)
    )
    aoi_area_m2 = float(aoi.area)
    covered_area_m2 = float(grouped["area_m2"].sum())
    rows = []
    for record in grouped.to_dict("records"):
        area_m2 = float(record["area_m2"])
        rows.append({
            "city_id": city_id,
            "code_2021": str(record["code_2021"]),
            "class_2021": str(record["class_2021"]),
            "intersecting_feature_count": int(record["feature_count"]),
            "area_m2": area_m2,
            "area_hectares": area_m2 / 10000.0,
            "fraction_of_sar_aoi": area_m2 / aoi_area_m2,
            "fraction_of_ua_covered_area": area_m2 / covered_area_m2 if covered_area_m2 else 0.0,
        })

    report = {
        "audit_version": "v3-pilot-ua-aoi-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": city_id,
        "vector_path": str(vector_path),
        "source_crs": UA_CRS,
        "sar_crs": city["sar_crs"],
        "sar_bounds": city["bounds"],
        "aoi_area_m2_in_equal_area_crs": aoi_area_m2,
        "urban_atlas_covered_area_m2": covered_area_m2,
        "urban_atlas_coverage_fraction": covered_area_m2 / aoi_area_m2,
        "bbox_candidate_feature_count": source_feature_count,
        "intersecting_feature_count": int(len(frame)),
        "null_geometry_count": null_geometry_count,
        "invalid_geometry_count": invalid_geometry_count,
        "class_count_in_aoi": int(len(rows)),
        "qa_status": "PASS" if rows and invalid_geometry_count == 0 else "REVIEW",
        "note": "Areas are geometric intersections in EPSG:3035; source vectors are not modified.",
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True, choices=sorted(CITIES))
    parser.add_argument("--vector", required=True, type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("metadata/dataset_v3/pilot/urban_atlas_aoi"),
    )
    args = parser.parse_args()
    if not args.vector.is_file():
        parser.error(f"Vector not found: {args.vector}")

    report, rows = analyze(args.vector, args.city_id)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"{args.city_id}_urban_atlas_aoi_audit.json"
    csv_path = args.report_dir / f"{args.city_id}_urban_atlas_aoi_class_areas.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "city_id": args.city_id,
        "qa_status": report["qa_status"],
        "coverage_fraction": report["urban_atlas_coverage_fraction"],
        "class_count_in_aoi": report["class_count_in_aoi"],
        "json": str(json_path),
        "csv": str(csv_path),
    }, indent=2))


if __name__ == "__main__":
    main()
