"""Read-only schema and class audit for Urban Atlas 2021 pilot vectors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import fiona
except ImportError:  # pragma: no cover
    fiona = None

try:
    import pyogrio
except ImportError:  # pragma: no cover
    pyogrio = None

if fiona is None and pyogrio is None:  # pragma: no cover
    raise SystemExit("Either pyogrio or Fiona is required; neither is installed.")


CLASS_FIELD_HINTS = ("class", "code", "label", "item")


def normalize(value):
    if value is None:
        return "<NULL>"
    return str(value).strip() or "<EMPTY>"


def audit_with_fiona(vector_path: Path, city_id: str) -> tuple[dict, list[dict]]:
    categorical_counts: dict[str, Counter[str]] = {}
    geometry_counts: Counter[str] = Counter()
    null_geometry_count = 0

    with fiona.open(vector_path) as source:
        properties = dict(source.schema.get("properties", {}))
        candidate_fields = [
            name for name in properties
            if any(hint in name.lower() for hint in CLASS_FIELD_HINTS)
        ]
        categorical_counts = {name: Counter() for name in candidate_fields}

        for feature in source:
            geometry = feature.get("geometry")
            if geometry is None:
                null_geometry_count += 1
            else:
                geometry_counts[geometry.get("type", "<UNKNOWN>")] += 1
            feature_properties = feature.get("properties") or {}
            for field in candidate_fields:
                categorical_counts[field][normalize(feature_properties.get(field))] += 1

        report = {
            "audit_version": "v3-pilot-ua-schema-0.1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "city_id": city_id,
            "vector_path": str(vector_path),
            "driver": source.driver,
            "feature_count": len(source),
            "declared_geometry": source.schema.get("geometry"),
            "geometry_counts": dict(sorted(geometry_counts.items())),
            "null_geometry_count": null_geometry_count,
            "crs": source.crs.to_string() if source.crs else None,
            "crs_wkt": source.crs_wkt,
            "bounds": list(source.bounds),
            "property_schema": properties,
            "candidate_class_fields": candidate_fields,
            "candidate_field_cardinality": {
                field: len(counts) for field, counts in categorical_counts.items()
            },
            "qa_status": "PASS" if len(source) > 0 and candidate_fields else "REVIEW",
        }

    rows = []
    for field, counts in categorical_counts.items():
        for value, count in counts.most_common():
            rows.append({
                "city_id": city_id,
                "field": field,
                "value": value,
                "feature_count": count,
            })
    report["reader_backend"] = "fiona"
    return report, rows


def audit_with_pyogrio(vector_path: Path, city_id: str) -> tuple[dict, list[dict]]:
    info = pyogrio.read_info(vector_path)
    fields = [str(field) for field in info.get("fields", [])]
    dtypes = [str(dtype) for dtype in info.get("dtypes", [])]
    properties = dict(zip(fields, dtypes))
    candidate_fields = [
        name for name in fields
        if any(hint in name.lower() for hint in CLASS_FIELD_HINTS)
    ]
    frame = pyogrio.read_dataframe(
        vector_path,
        columns=candidate_fields,
        read_geometry=False,
    )
    rows = []
    cardinality = {}
    for field in candidate_fields:
        counts = frame[field].map(normalize).value_counts(dropna=False)
        cardinality[field] = int(len(counts))
        for value, count in counts.items():
            rows.append({
                "city_id": city_id,
                "field": field,
                "value": str(value),
                "feature_count": int(count),
            })

    feature_count = int(info.get("features", len(frame)))
    geometry_type = str(info.get("geometry_type"))
    report = {
        "audit_version": "v3-pilot-ua-schema-0.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": city_id,
        "vector_path": str(vector_path),
        "reader_backend": "pyogrio",
        "driver": str(info.get("driver")),
        "feature_count": feature_count,
        "declared_geometry": geometry_type,
        "geometry_counts": {geometry_type: feature_count},
        "null_geometry_count": None,
        "crs": str(info.get("crs")),
        "crs_wkt": None,
        "bounds": [float(value) for value in info.get("total_bounds", [])],
        "property_schema": properties,
        "candidate_class_fields": candidate_fields,
        "candidate_field_cardinality": cardinality,
        "qa_status": "PASS" if feature_count > 0 and candidate_fields else "REVIEW",
    }
    return report, rows


def audit(vector_path: Path, city_id: str) -> tuple[dict, list[dict]]:
    if pyogrio is not None:
        return audit_with_pyogrio(vector_path, city_id)
    return audit_with_fiona(vector_path, city_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True, choices=["DE03", "DE14"])
    parser.add_argument("--vector", required=True, type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("metadata/dataset_v3/pilot/urban_atlas"),
    )
    args = parser.parse_args()
    if not args.vector.is_file():
        parser.error(f"Vector not found: {args.vector}")

    report, rows = audit(args.vector, args.city_id)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"{args.city_id}_urban_atlas_schema_audit.json"
    csv_path = args.report_dir / f"{args.city_id}_urban_atlas_class_counts.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["city_id", "field", "value", "feature_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "city_id": args.city_id,
        "qa_status": report["qa_status"],
        "feature_count": report["feature_count"],
        "candidate_class_fields": report["candidate_class_fields"],
        "json": str(json_path),
        "csv": str(csv_path),
    }, indent=2))


if __name__ == "__main__":
    main()
