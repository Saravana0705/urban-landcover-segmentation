"""Validate and select V3-MT-D2 ascending/descending orbit candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


QUARTERS = ("q1", "q2", "q3", "q4")
PASSES = ("ASCENDING", "DESCENDING")

# Frozen V3-MT primary-orbit contract. D2 must preserve this orbit and add the
# best adequate orbit from the opposite pass; it must not silently replace the
# original input source.
FROZEN_PRIMARY_ORBITS = {
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
            "metadata/dataset_v3_mt_d2/acquisition/"
            "dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v3_mt_d2/acquisition/"
            "dataset_v3_mt_d2_acquisition_audit_qa.json"
        ),
    )
    parser.add_argument(
        "--minimum-observations-per-quarter", type=int, default=2
    )
    parser.add_argument(
        "--minimum-scene-coverage-fraction", type=float, default=0.98
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def integer_value(value: Any, label: str) -> int:
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{label} is not an integer: {value}")
    return int(number)


def candidate_record(row: pd.Series) -> dict[str, Any]:
    counts = {
        quarter: integer_value(
            row[f"{quarter}_observation_count"],
            f"{row['city_id']} {row['orbit_pass']} {quarter}",
        )
        for quarter in QUARTERS
    }
    return {
        "orbit_pass": str(row["orbit_pass"]),
        "relative_orbit": integer_value(
            row["relative_orbit"], "relative_orbit"
        ),
        "annual_observation_count": integer_value(
            row["annual_observation_count"], "annual_observation_count"
        ),
        "quarterly_observation_counts": counts,
        "minimum_quarterly_observation_count": min(counts.values()),
        "minimum_scene_coverage_fraction": float(
            row["minimum_scene_coverage_fraction"]
        ),
        "mean_scene_coverage_fraction": float(
            row["mean_scene_coverage_fraction"]
        ),
    }


def is_adequate(candidate: dict[str, Any], minimum: int, coverage: float) -> bool:
    return bool(
        candidate["minimum_quarterly_observation_count"] >= minimum
        and candidate["minimum_scene_coverage_fraction"] >= coverage - 1e-9
    )


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda item: (
            -item["minimum_quarterly_observation_count"],
            -item["annual_observation_count"],
            -item["mean_scene_coverage_fraction"],
            item["relative_orbit"],
        ),
    )[0]


def main() -> None:
    args = parse_args()
    if args.minimum_observations_per_quarter < 1:
        raise ValueError("Minimum observations per quarter must be at least 1")
    if not 0 < args.minimum_scene_coverage_fraction <= 1:
        raise ValueError("Minimum coverage fraction must be in (0, 1]")
    if not args.audit_csv.is_file() or args.audit_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Acquisition audit missing: {args.audit_csv}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    frame = pd.read_csv(args.audit_csv)
    required = {
        "city_id", "city_name", "split", "orbit_pass", "relative_orbit",
        "coverage_unit",
        "annual_observation_count", "minimum_scene_coverage_fraction",
        "mean_scene_coverage_fraction",
        *(f"{quarter}_observation_count" for quarter in QUARTERS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Audit CSV missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["city_id"] = frame["city_id"].astype(str).str.strip().str.upper()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["orbit_pass"] = (
        frame["orbit_pass"].astype(str).str.strip().str.upper()
    )
    frame["coverage_unit"] = frame["coverage_unit"].astype(str).str.strip()

    failures: list[str] = []
    coverage_units = sorted(frame["coverage_unit"].unique().tolist())
    if coverage_units != ["same_day_slice_union"]:
        failures.append(
            "Expected mosaic-aware same_day_slice_union coverage; found "
            f"{coverage_units}"
        )
    expected_ids = list(FROZEN_PRIMARY_ORBITS)
    actual_ids = sorted(frame["city_id"].unique().tolist())
    if actual_ids != expected_ids:
        failures.append(
            f"Expected candidate rows for DE01-DE20; found {actual_ids}"
        )
    invalid_passes = sorted(set(frame["orbit_pass"]).difference(PASSES))
    if invalid_passes:
        failures.append(f"Unexpected orbit passes: {invalid_passes}")
    duplicate_key = ["city_id", "orbit_pass", "relative_orbit"]
    if frame.duplicated(duplicate_key).any():
        failures.append("Duplicate city/pass/relative-orbit candidate rows")

    city_identity = frame[["city_id", "city_name", "split"]].drop_duplicates()
    split_counts = (
        city_identity.drop_duplicates("city_id")["split"].value_counts().to_dict()
    )
    if split_counts != {"train": 14, "val": 3, "test": 3}:
        failures.append(f"Unexpected city split counts: {split_counts}")
    identity_counts = city_identity.groupby("city_id").size().to_dict()
    inconsistent = sorted(
        city_id for city_id, count in identity_counts.items() if count != 1
    )
    if inconsistent:
        failures.append(f"Inconsistent city identity rows: {inconsistent}")

    city_results: list[dict[str, Any]] = []
    for city_id in expected_ids:
        city_rows = frame[frame["city_id"] == city_id]
        city_failures: list[str] = []
        primary_pass, primary_orbit = FROZEN_PRIMARY_ORBITS[city_id]
        selected: dict[str, Any] = {}
        all_candidates: dict[str, list[dict[str, Any]]] = {}

        for pass_name in PASSES:
            pass_rows = city_rows[city_rows["orbit_pass"] == pass_name]
            candidates = [candidate_record(row) for _, row in pass_rows.iterrows()]
            all_candidates[pass_name] = candidates
            adequate = [
                item for item in candidates
                if is_adequate(
                    item,
                    args.minimum_observations_per_quarter,
                    args.minimum_scene_coverage_fraction,
                )
            ]
            if not adequate:
                city_failures.append(
                    f"no adequate {pass_name} orbit with >= "
                    f"{args.minimum_observations_per_quarter} observations in "
                    "every quarter"
                )
                continue

            if pass_name == primary_pass:
                matches = [
                    item for item in adequate
                    if item["relative_orbit"] == primary_orbit
                ]
                if not matches:
                    city_failures.append(
                        f"frozen primary {primary_pass} orbit {primary_orbit} "
                        "is missing or inadequate"
                    )
                    continue
                choice = matches[0]
                choice = {**choice, "selection_role": "frozen_primary"}
            else:
                choice = select_best(adequate)
                choice = {**choice, "selection_role": "added_opposite_pass"}
            selected[pass_name] = choice

        city_name = (
            str(city_rows.iloc[0]["city_name"]) if not city_rows.empty else None
        )
        split = str(city_rows.iloc[0]["split"]) if not city_rows.empty else None
        city_results.append({
            "city_id": city_id,
            "city_name": city_name,
            "split": split,
            "status": "PASS" if not city_failures else "FAIL",
            "failures": city_failures,
            "frozen_primary": {
                "orbit_pass": primary_pass,
                "relative_orbit": primary_orbit,
            },
            "selected_orbits": selected,
            "candidates": all_candidates,
        })
        failures.extend(f"{city_id}: {item}" for item in city_failures)

    status = "PASS" if not failures else "FAIL"
    report = {
        "schema_version": "dataset-v3-mt-d2-acquisition-audit-qa-0.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_version": "v3-mt-d2",
        "parent_dataset": "v3-mt",
        "experiment": "quarterly-ascending-descending-early-fusion",
        "coverage_unit": "same_day_slice_union",
        "city_count": len(city_results),
        "split_counts": split_counts,
        "minimum_observations_per_quarter": (
            args.minimum_observations_per_quarter
        ),
        "minimum_scene_coverage_fraction": (
            args.minimum_scene_coverage_fraction
        ),
        "selected_period_count": 4,
        "selected_pass_count": 2,
        "selected_channel_count": 16,
        "expected_channel_order": [
            f"{polarisation}_{pass_abbreviation}_Q{quarter}"
            for pass_abbreviation in ("ASC", "DESC")
            for quarter in range(1, 5)
            for polarisation in ("VV", "VH")
        ],
        "labels_modified": False,
        "split_assignments_modified": False,
        "failures": failures,
        "cities": city_results,
        "image_exports_authorized": status == "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "candidate_rows": len(frame),
        "city_count": len(city_results),
        "selected_channel_count": 16,
        "image_exports_authorized": status == "PASS",
        "report": str(args.output),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
