"""Validate the Earth Engine acquisition audit before starting image tasks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt/acquisition/"
            "dataset_v3_mt_acquisition_audit_2025.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt/acquisition/"
            "dataset_v3_mt_acquisition_audit_qa.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audit_csv.is_file() or args.audit_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Acquisition audit missing: {args.audit_csv}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    frame = pd.read_csv(args.audit_csv)
    required = {
        "city_id", "city_name", "split", "registry_anchor_time",
        "anchor_candidate_count",
        "selected_anchor_product", "selected_anchor_time_ms", "orbit_pass",
        "relative_orbit", "q1_observation_count", "q2_observation_count",
        "q3_observation_count", "q4_observation_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Audit CSV missing columns: {sorted(missing)}")

    failures: list[str] = []
    expected_ids = [f"DE{number:02d}" for number in range(1, 21)]
    actual_ids = sorted(frame["city_id"].astype(str).tolist())
    if len(frame) != 20 or actual_ids != expected_ids:
        failures.append(f"Expected DE01-DE20 once each; found {actual_ids}")
    if frame["city_id"].duplicated().any():
        failures.append("Duplicate city IDs found")

    city_results = []
    for _, row in frame.sort_values("city_id").iterrows():
        city_failures = []
        if int(row["anchor_candidate_count"]) < 1:
            city_failures.append("no anchor candidate")
        registry_time = pd.to_datetime(row["registry_anchor_time"], utc=True)
        selected_time = pd.to_datetime(
            int(float(row["selected_anchor_time_ms"])), unit="ms", utc=True
        )
        anchor_difference_ms = abs(
            (selected_time - registry_time).total_seconds() * 1000
        )
        if anchor_difference_ms > 1000:
            city_failures.append(
                f"selected anchor differs by {anchor_difference_ms:.3f} ms"
            )
        if not str(row["selected_anchor_product"]).strip():
            city_failures.append("selected anchor product is empty")
        orbit_pass = str(row["orbit_pass"]).upper()
        if orbit_pass not in {"ASCENDING", "DESCENDING"}:
            city_failures.append(f"invalid pass {row['orbit_pass']}")
        if int(row["relative_orbit"]) <= 0:
            city_failures.append("invalid relative orbit")
        counts = {
            quarter: int(row[f"{quarter}_observation_count"])
            for quarter in ("q1", "q2", "q3", "q4")
        }
        for quarter, count in counts.items():
            if count < 1:
                city_failures.append(f"{quarter} has zero observations")
        city_results.append({
            "city_id": str(row["city_id"]),
            "status": "PASS" if not city_failures else "FAIL",
            "failures": city_failures,
            "orbit_pass": orbit_pass,
            "relative_orbit": int(row["relative_orbit"]),
            "anchor_time_difference_ms": anchor_difference_ms,
            "quarterly_observation_counts": counts,
        })
        failures.extend(f"{row['city_id']}: {value}" for value in city_failures)

    status = "PASS" if not failures else "FAIL"
    pass_counts = {
        str(key).upper(): int(value)
        for key, value in frame["orbit_pass"].value_counts().items()
    }
    report = {
        "schema_version": "dataset-v3-mt-acquisition-audit-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "city_count": len(frame),
        "orbit_pass_counts": pass_counts,
        "failures": failures,
        "cities": city_results,
        "image_exports_authorized": status == "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "city_count": len(frame),
        "image_exports_authorized": status == "PASS",
        "report": str(args.output),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
