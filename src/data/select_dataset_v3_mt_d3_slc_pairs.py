"""Select physically valid same-track Sentinel-1 SLC coherence pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED = {
    "city_id", "stac_item_id", "acquisition_datetime", "orbit_direction",
    "relative_orbit", "coverage_fraction", "instrument_mode", "product_type",
    "platform", "has_vv_vh",
}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def select_pairs(catalog: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    missing = REQUIRED.difference(catalog.columns)
    if missing:
        raise ValueError("SLC catalogue is missing columns: " + ", ".join(sorted(missing)))
    pilot = set(config["pilot_cities"])
    pairing = config["pairing"]
    data = catalog.copy()
    data["acquisition_datetime"] = pd.to_datetime(data["acquisition_datetime"], utc=True)
    data["relative_orbit"] = pd.to_numeric(data["relative_orbit"], errors="coerce")
    data = data.loc[
        data["city_id"].isin(pilot)
        & (data["instrument_mode"].astype(str).str.upper() == "IW")
        & data["product_type"].astype(str).str.upper().str.contains("SLC", regex=False)
        & truthy(data["has_vv_vh"])
        & (pd.to_numeric(data["coverage_fraction"], errors="coerce") >= float(pairing["minimum_aoi_coverage"]))
        & data["relative_orbit"].notna()
    ].sort_values("acquisition_datetime")
    preferred_platform = str(pairing.get("preferred_platform", "sentinel-1a")).lower()
    if preferred_platform:
        data = data.loc[data["platform"].astype(str).str.lower() == preferred_platform]
    target_date = pd.Timestamp(str(pairing.get("target_date", "2025-07-15")), tz="UTC")
    preferred_baseline = float(pairing.get("preferred_temporal_baseline_days", 12))
    candidates_by_city: dict[str, list[dict[str, Any]]] = {
        str(city): [] for city in config["pilot_cities"]
    }
    group_columns = ["city_id", "platform", "orbit_direction", "relative_orbit"]
    for (city, platform, direction, orbit), group in data.groupby(group_columns, sort=True):
        records = list(group.to_dict("records"))
        for left_index, master in enumerate(records):
            for slave in records[left_index + 1:]:
                baseline = (slave["acquisition_datetime"] - master["acquisition_datetime"]).total_seconds() / 86400
                if float(pairing["minimum_temporal_baseline_days"]) <= baseline <= float(pairing["maximum_temporal_baseline_days"]):
                    midpoint = master["acquisition_datetime"] + (slave["acquisition_datetime"] - master["acquisition_datetime"]) / 2
                    candidates_by_city[str(city)].append({
                        "pair_id": f"{city}_{str(direction).upper()}_R{int(orbit):03d}_{master['acquisition_datetime']:%Y%m%d}_{slave['acquisition_datetime']:%Y%m%d}",
                        "city_id": city,
                        "platform": str(platform).lower(),
                        "orbit_direction": str(direction).upper(),
                        "relative_orbit": int(orbit),
                        "master_item_id": master["stac_item_id"],
                        "slave_item_id": slave["stac_item_id"],
                        "master_datetime": master["acquisition_datetime"].isoformat(),
                        "slave_datetime": slave["acquisition_datetime"].isoformat(),
                        "temporal_baseline_days": baseline,
                        "midpoint_distance_days": abs((midpoint - target_date).total_seconds()) / 86400,
                        "baseline_distance_days": abs(baseline - preferred_baseline),
                        "minimum_coverage_fraction": min(
                            1.0,
                            float(master["coverage_fraction"]),
                            float(slave["coverage_fraction"]),
                        ),
                        "master_download_url": master.get("product_download_url", ""),
                        "slave_download_url": slave.get("product_download_url", ""),
                    })
    rows: list[dict[str, Any]] = []
    for city in config["pilot_cities"]:
        candidates = candidates_by_city[str(city)]
        candidates.sort(
            key=lambda row: (
                round(row["baseline_distance_days"], 3),
                row["midpoint_distance_days"],
                -row["minimum_coverage_fraction"],
                row["pair_id"],
            )
        )
        rows.extend(candidates[: int(pairing.get("maximum_pairs_per_city", 1))])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_mt_d3_slc_coherence_pilot.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "PILOT_ONLY" or config.get("full_export_authorized") is not False:
        raise ValueError("D3 groundwork must remain PILOT_ONLY with full export disabled.")
    output = Path(config["outputs"]["pair_manifest"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    pairs = select_pairs(pd.read_csv(args.catalog), config)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output, index=False)
    city_counts = pairs.groupby("city_id").size().to_dict() if not pairs.empty else {}
    report = {
        "status": "PASS" if set(config["pilot_cities"]).issubset(city_counts) else "FAIL",
        "pilot_only": True,
        "full_export_authorized": False,
        "pair_count": len(pairs),
        "pairs_by_city": {str(k): int(v) for k, v in city_counts.items()},
        "pair_manifest": str(output),
    }
    audit = Path(config["outputs"]["pair_audit"])
    audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
