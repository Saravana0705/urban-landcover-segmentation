"""Validate D3 SLC pair invariants before any SNAP processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_mt_d3_slc_coherence_pilot.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = Path(config["outputs"]["pair_manifest"])
    pairs = pd.read_csv(manifest)
    baseline = pd.to_numeric(pairs["temporal_baseline_days"], errors="coerce")
    pairing = config["pairing"]
    checks = {
        "pilot_only": config.get("status") == "PILOT_ONLY",
        "full_export_blocked": config.get("full_export_authorized") is False,
        "cities_exact": set(pairs["city_id"]) == set(config["pilot_cities"]),
        "pair_ids_unique": bool(pairs["pair_id"].is_unique),
        "relative_orbits_present": bool(pairs["relative_orbit"].notna().all()),
        "same_platform_enforced": bool(
            pairs["platform"].astype(str).str.lower().eq(
                str(pairing.get("preferred_platform", "sentinel-1a")).lower()
            ).all()
        ),
        "maximum_pairs_per_city_respected": bool(
            pairs.groupby("city_id").size().le(
                int(pairing.get("maximum_pairs_per_city", 1))
            ).all()
        ),
        "baselines_valid": bool(baseline.between(float(pairing["minimum_temporal_baseline_days"]), float(pairing["maximum_temporal_baseline_days"]), inclusive="both").all()),
        "product_ids_present": bool(
            pairs[["master_item_id", "slave_item_id"]]
            .fillna("")
            .astype(str)
            .apply(lambda column: column.str.strip().ne(""))
            .all()
            .all()
        ),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "pair_count": len(pairs), "checks": checks}
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
