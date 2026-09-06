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
        "baselines_valid": bool(baseline.between(float(pairing["minimum_temporal_baseline_days"]), float(pairing["maximum_temporal_baseline_days"]), inclusive="both").all()),
        "urls_present": bool(pairs[["master_download_url", "slave_download_url"]].notna().all().all()),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "pair_count": len(pairs), "checks": checks}
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
