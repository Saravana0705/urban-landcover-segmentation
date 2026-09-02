"""Select the safe V3-MT-D1 temporal cadence from the GEE audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MONTHS = tuple(f"m{month:02d}" for month in range(1, 13))
TWO_MONTH_PERIODS = tuple(
    (f"BIMONTH_{start:02d}_{start + 1:02d}", (f"m{start:02d}", f"m{start + 1:02d}"))
    for start in range(1, 13, 2)
)

# Frozen V3-MT anchor-orbit contract. DE05, DE14 and DE20 legitimately use
# ascending acquisitions; all other cities use descending acquisitions.
EXPECTED_ORBITS = {
    "DE01": ("DESCENDING", 139),
    "DE02": ("DESCENDING", 139),
    "DE03": ("DESCENDING", 66),
    "DE04": ("DESCENDING", 66),
    "DE05": ("ASCENDING", 117),
    "DE06": ("DESCENDING", 168),
    "DE07": ("DESCENDING", 139),
    "DE08": ("DESCENDING", 139),
    "DE09": ("DESCENDING", 139),
    "DE10": ("DESCENDING", 139),
    "DE11": ("DESCENDING", 66),
    "DE12": ("DESCENDING", 139),
    "DE13": ("DESCENDING", 139),
    "DE14": ("ASCENDING", 146),
    "DE15": ("DESCENDING", 168),
    "DE16": ("DESCENDING", 95),
    "DE17": ("DESCENDING", 139),
    "DE18": ("DESCENDING", 139),
    "DE19": ("DESCENDING", 139),
    "DE20": ("ASCENDING", 146),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/acquisition/"
            "dataset_v3_mt_d1_monthly_acquisition_audit_2025.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d1/acquisition/"
            "dataset_v3_mt_d1_acquisition_audit_qa.json"
        ),
    )
    parser.add_argument(
        "--minimum-observations-per-composite", type=int, default=2
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    minimum = args.minimum_observations_per_composite
    if minimum < 1:
        raise ValueError("Minimum observations must be at least 1")
    if not args.audit_csv.is_file() or args.audit_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Acquisition audit missing: {args.audit_csv}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    frame = pd.read_csv(args.audit_csv)
    required = {
        "city_id", "city_name", "split", "registry_anchor_time",
        "anchor_candidate_count", "selected_anchor_product",
        "selected_anchor_time_ms", "orbit_pass", "relative_orbit",
        *(f"{month}_observation_count" for month in MONTHS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Audit CSV missing columns: {sorted(missing)}")

    structural_failures: list[str] = []
    expected_ids = [f"DE{number:02d}" for number in range(1, 21)]
    actual_ids = sorted(frame["city_id"].astype(str).tolist())
    if len(frame) != 20 or actual_ids != expected_ids:
        structural_failures.append(
            f"Expected DE01-DE20 once each; found {actual_ids}"
        )
    if frame["city_id"].duplicated().any():
        structural_failures.append("Duplicate city IDs found")
    split_counts = frame["split"].astype(str).str.lower().value_counts().to_dict()
    if split_counts != {"train": 14, "val": 3, "test": 3}:
        structural_failures.append(f"Unexpected split counts: {split_counts}")

    city_results: list[dict[str, object]] = []
    all_monthly_adequate = True
    all_bimonthly_adequate = True
    for _, row in frame.sort_values("city_id").iterrows():
        city_id = str(row["city_id"])
        city_failures: list[str] = []
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
        if pd.isna(row["selected_anchor_product"]) or not str(
            row["selected_anchor_product"]
        ).strip():
            city_failures.append("selected anchor product is empty")
        actual_pass = str(row["orbit_pass"]).upper()
        actual_relative_orbit = int(row["relative_orbit"])
        expected_pass, expected_relative_orbit = EXPECTED_ORBITS[city_id]
        if actual_pass != expected_pass:
            city_failures.append(
                f"pass {actual_pass}; expected frozen {expected_pass}"
            )
        if actual_relative_orbit != expected_relative_orbit:
            city_failures.append(
                "relative orbit "
                f"{actual_relative_orbit}; expected frozen "
                f"{expected_relative_orbit}"
            )

        monthly_counts = {
            month: int(row[f"{month}_observation_count"])
            for month in MONTHS
        }
        zero_months = [month for month, count in monthly_counts.items() if count < 1]
        if zero_months:
            city_failures.append(f"zero-observation months: {zero_months}")
        weak_months = [
            month for month, count in monthly_counts.items() if count < minimum
        ]
        bimonthly_counts = {
            name: sum(monthly_counts[month] for month in months)
            for name, months in TWO_MONTH_PERIODS
        }
        weak_bimonthly = [
            name for name, count in bimonthly_counts.items() if count < minimum
        ]
        all_monthly_adequate = all_monthly_adequate and not weak_months
        all_bimonthly_adequate = all_bimonthly_adequate and not weak_bimonthly
        city_results.append({
            "city_id": city_id,
            "status": "PASS" if not city_failures else "FAIL",
            "failures": city_failures,
            "orbit_pass": actual_pass,
            "relative_orbit": actual_relative_orbit,
            "anchor_time_difference_ms": anchor_difference_ms,
            "monthly_observation_counts": monthly_counts,
            "months_below_threshold": weak_months,
            "bimonthly_observation_counts": bimonthly_counts,
            "bimonthly_periods_below_threshold": weak_bimonthly,
        })
        structural_failures.extend(
            f"{city_id}: {failure}" for failure in city_failures
        )

    if structural_failures or not all_bimonthly_adequate:
        status, selected_cadence, period_count = "FAIL", None, 0
    elif all_monthly_adequate:
        status, selected_cadence, period_count = "PASS", "monthly", 12
    else:
        status, selected_cadence, period_count = "PASS", "bimonthly", 6

    report = {
        "schema_version": "dataset-v3-mt-d1-acquisition-audit-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "city_count": len(frame),
        "split_counts": split_counts,
        "minimum_observations_per_composite": minimum,
        "all_monthly_periods_adequate": all_monthly_adequate,
        "all_bimonthly_periods_adequate": all_bimonthly_adequate,
        "selected_cadence": selected_cadence,
        "selected_period_count": period_count,
        "selected_channel_count": period_count * 2,
        "structural_failures": structural_failures,
        "cities": city_results,
        "image_exports_authorized": status == "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "city_count": len(frame),
        "selected_cadence": selected_cadence,
        "selected_channel_count": period_count * 2,
        "image_exports_authorized": status == "PASS",
        "report": str(args.output),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
