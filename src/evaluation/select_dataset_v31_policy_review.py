"""Select a compact, deterministic QGIS review set for Dataset V3.1.

This metadata-only stage reduces the broad minority-label audit sample to a
small policy-focused review.  It never reads or writes semantic rasters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "metadata/dataset_v3_1/minority_audit/dataset_v31_qgis_review_sample.csv"
)
DEFAULT_OUTPUT_DIR = Path("metadata/dataset_v3_1/policy_review")


ROAD_STRATA = {
    "road_major_resolvable": lambda r: (
        r["sample_type"] == "road"
        and r["policy_group"] == "retain_major_candidate"
        and r["width_bin"] != "subpixel_lt_10m"
    ),
    "road_major_subpixel": lambda r: (
        r["sample_type"] == "road"
        and r["policy_group"] == "retain_major_candidate"
        and r["width_bin"] == "subpixel_lt_10m"
    ),
    "road_local_resolvable": lambda r: (
        r["sample_type"] == "road"
        and r["policy_group"] == "manual_review_candidate"
        and r["width_bin"] != "subpixel_lt_10m"
    ),
    "road_local_subpixel": lambda r: (
        r["sample_type"] == "road"
        and r["policy_group"] == "manual_review_candidate"
        and r["width_bin"] == "subpixel_lt_10m"
    ),
    "road_low_priority": lambda r: (
        r["sample_type"] == "road"
        and r["policy_group"] == "likely_ignore_candidate"
    ),
}

BARE_STRATA = {
    "bare_current_v3": lambda r: (
        r["sample_type"] == "bare_land" and r["bare_tier"] == "current_v3"
    ),
    "bare_dw_confirmed_new": lambda r: (
        r["sample_type"] == "bare_land"
        and r["bare_tier"] in {"tier1_ua_dw_new", "tier2_osm_dw_new"}
    ),
    "bare_osm_only": lambda r: (
        r["sample_type"] == "bare_land"
        and r["bare_tier"] == "single_source_osm"
    ),
    "bare_ua_only": lambda r: (
        r["sample_type"] == "bare_land"
        and r["bare_tier"] == "single_source_ua"
    ),
}

TARGETS = {
    "road_major_resolvable": 6,
    "road_major_subpixel": 6,
    "road_local_resolvable": 6,
    "road_local_subpixel": 6,
    "road_low_priority": 6,
    "bare_current_v3": 6,
    "bare_dw_confirmed_new": 6,
    "bare_osm_only": 4,
    "bare_ua_only": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select 50 policy-focused QGIS points from the V3.1 audit."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_road_stratum(value: str) -> tuple[str, str, str]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Audit sample not found or empty: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Audit sample has no rows: {path}")
    required = {"sample_type", "city_id", "split", "stratum", "x", "y", "crs"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Audit sample lacks columns: {sorted(missing)}")
    for row in rows:
        policy, width, highway = split_road_stratum(row.get("stratum", ""))
        row["policy_group"] = policy
        row["width_bin"] = width
        if not row.get("highway"):
            row["highway"] = highway
    return rows


def diversity_key(row: dict[str, str], stratum: str) -> tuple[str, ...]:
    if stratum.startswith("road_"):
        return (
            row.get("city_id", ""), row.get("highway", ""),
            row.get("width_bin", ""), row.get("pixel_row", ""),
        )
    return (
        row.get("city_id", ""), row.get("ua_code", ""),
        row.get("bare_tier", ""), row.get("pixel_row", ""),
    )


def round_robin_select(
    candidates: list[dict[str, str]], target: int, stratum: str
) -> list[dict[str, str]]:
    ordered = sorted(candidates, key=lambda row: diversity_key(row, stratum))
    by_city: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        by_city[row["city_id"]].append(row)
    selected: list[dict[str, str]] = []
    city_ids = sorted(by_city)
    if city_ids:
        token = hashlib.sha256(stratum.encode("utf-8")).hexdigest()
        offset_city = int(token[:8], 16) % len(city_ids)
        city_ids = city_ids[offset_city:] + city_ids[:offset_city]
    offset = 0
    while len(selected) < target and city_ids:
        remaining: list[str] = []
        for city_id in city_ids:
            items = by_city[city_id]
            if offset < len(items):
                selected.append(items[offset])
                if len(selected) == target:
                    break
            if offset + 1 < len(items):
                remaining.append(city_id)
        offset += 1
        city_ids = remaining
    return selected


def split_aware_select(
    candidates: list[dict[str, str]], target: int, stratum: str
) -> list[dict[str, str]]:
    """Reserve one validation example when available, then fill from train."""
    validation = [row for row in candidates if row.get("split") == "val"]
    training = [row for row in candidates if row.get("split") == "train"]
    selected: list[dict[str, str]] = []
    if validation and target:
        selected.extend(round_robin_select(validation, 1, stratum + "|val"))
    remaining = target - len(selected)
    selected.extend(round_robin_select(training, remaining, stratum + "|train"))
    remaining = target - len(selected)
    if remaining:
        used = {id(row) for row in selected}
        fallback = [row for row in candidates if id(row) not in used]
        selected.extend(round_robin_select(fallback, remaining, stratum + "|fallback"))
    return selected


def question_for(stratum: str) -> str:
    if stratum.startswith("road_"):
        return "Is a road surface visibly supportable at the 10 m SAR pixel scale?"
    return "Is this location genuinely bare/non-vegetated land for the SAR acquisition period?"


def allowed_decisions(stratum: str) -> str:
    if stratum.startswith("road_"):
        return "retain_road|ignore_unresolvable|not_road|uncertain"
    return "retain_bare|ignore_temporal_or_ambiguous|not_bare|uncertain"


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    definitions = {**ROAD_STRATA, **BARE_STRATA}
    selected: list[dict[str, str]] = []
    availability: dict[str, int] = {}
    for stratum, predicate in definitions.items():
        candidates = [row for row in rows if predicate(row)]
        availability[stratum] = len(candidates)
        wanted = TARGETS[stratum]
        chosen = split_aware_select(candidates, wanted, stratum)
        if len(chosen) < wanted:
            raise ValueError(
                f"Insufficient {stratum} samples: wanted {wanted}, found {len(chosen)}"
            )
        for row in chosen:
            item = dict(row)
            item["policy_review_stratum"] = stratum
            item["policy_question"] = question_for(stratum)
            item["allowed_decisions"] = allowed_decisions(stratum)
            item["review_decision"] = ""
            item["review_notes"] = ""
            selected.append(item)

    for index, row in enumerate(selected, start=1):
        row["review_order"] = str(index)

    audit: dict[str, Any] = {
        "schema_version": "dataset-v3.1-policy-review-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "target_by_stratum": TARGETS,
        "available_by_stratum": availability,
        "selected_by_stratum": dict(Counter(r["policy_review_stratum"] for r in selected)),
        "selected_by_split": dict(Counter(r["split"] for r in selected)),
        "selected_city_count": len({r["city_id"] for r in selected}),
        "test_split_inspected": any(r["split"] == "test" for r in selected),
        "labels_modified": False,
        "selection_or_materialization_performed": False,
    }

    if args.dry_run:
        print(json.dumps(audit, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dataset_v31_policy_review.csv"
    crs_paths = {
        "EPSG:32632": args.output_dir / "dataset_v31_policy_review_epsg32632.csv",
        "EPSG:32633": args.output_dir / "dataset_v31_policy_review_epsg32633.csv",
    }
    json_path = args.output_dir / "dataset_v31_policy_review_audit.json"
    collisions = [
        path for path in (csv_path, *crs_paths.values(), json_path) if path.exists()
    ]
    if collisions and not args.overwrite_metadata:
        raise FileExistsError(
            "Refusing to overwrite policy-review metadata: "
            + ", ".join(map(str, collisions))
        )

    fieldnames = list(selected[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    audit["crs_outputs"] = {}
    for crs, crs_path in crs_paths.items():
        crs_rows = [row for row in selected if row["crs"] == crs]
        with crs_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(crs_rows)
        audit["crs_outputs"][crs] = {
            "path": str(crs_path),
            "rows": len(crs_rows),
            "sha256": sha256_file(crs_path),
        }
    audit["output_csv"] = str(csv_path)
    audit["output_csv_sha256"] = sha256_file(csv_path)
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "selected_rows": len(selected),
        "selected_city_count": audit["selected_city_count"],
        "output_csv": str(csv_path), "crs_outputs": audit["crs_outputs"],
        "labels_modified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
