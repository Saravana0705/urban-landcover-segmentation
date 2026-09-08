"""Choose and validate a common Sentinel-1 IW burst footprint before SLC download."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
import yaml
from shapely.geometry import shape
from shapely.ops import unary_union


def query_bursts(session: requests.Session, base_url: str, product_name: str, timeout: int) -> list[dict[str, Any]]:
    escaped = product_name.replace("'", "''")
    response = session.get(
        f"{base_url.rstrip('/')}/Bursts",
        params={
            "$filter": f"ParentProductName eq '{escaped}' and PolarisationChannels eq 'VV'",
            "$top": "1000",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("value", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected Bursts response for {product_name}.")
    return rows


def projected_geometry(geometry, epsg: int):
    return gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(epsg=epsg).iloc[0]


def coverage(aoi, geometries: list[Any]) -> float:
    valid = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not valid:
        return 0.0
    return min(1.0, float(aoi.intersection(unary_union(valid)).area / aoi.area))


def burst_map(rows: list[dict[str, Any]], swath: str, epsg: int) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for row in rows:
        if str(row.get("SwathIdentifier", "")).upper() != swath:
            continue
        footprint = row.get("GeoFootprint")
        burst_id = row.get("BurstId")
        if footprint and burst_id is not None:
            result[int(burst_id)] = projected_geometry(shape(footprint), epsg)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_mt_d3_slc_coherence_pilot.yaml"))
    parser.add_argument("--cities", type=Path, default=Path("config/dataset_v3_mt_d3_slc_pilot_cities.csv"))
    parser.add_argument("--aois", type=Path, default=Path("data/aoi/all_city_aois.gpkg"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "PILOT_ONLY" or config.get("full_export_authorized") is not False:
        raise ValueError("Burst QA must remain inside the bounded D3 pilot.")
    output = Path(config["outputs"]["burst_coverage_manifest"])
    audit = Path(config["outputs"]["burst_coverage_audit"])
    if (output.exists() or audit.exists()) and not args.overwrite:
        raise FileExistsError("Burst QA output exists; pass --overwrite to replace it.")

    products = pd.read_csv(Path(config["outputs"]["download_manifest"]), keep_default_na=False)
    cities = pd.read_csv(args.cities).set_index("city_id")
    aois = gpd.read_file(args.aois).to_crs("EPSG:4326")
    session = requests.Session()
    cache: dict[str, list[dict[str, Any]]] = {}
    failures: list[str] = []
    rows: list[dict[str, Any]] = []

    for pair_id, pair in products.groupby("pair_id", sort=True):
        city_id = str(pair["city_id"].iloc[0])
        try:
            city = cities.loc[city_id]
            epsg = int(city["utm_epsg"])
            aoi_rows = aois.loc[aois["city_id"].astype(str) == city_id]
            if len(aoi_rows) != 1:
                raise ValueError(f"Expected one AOI for {city_id}; found {len(aoi_rows)}.")
            aoi = projected_geometry(aoi_rows.geometry.iloc[0], epsg)
            role_products = {str(row["role"]): str(row["product_name"]) for _, row in pair.iterrows()}
            if set(role_products) != {"master", "slave"}:
                raise ValueError(f"Pair {pair_id} does not have one master and one slave.")
            for product_name in role_products.values():
                if product_name not in cache:
                    cache[product_name] = query_bursts(
                        session, str(config["odata"]["catalogue_url"]), product_name,
                        int(config["odata"].get("timeout_seconds", 60)),
                    )

            pair_rows: list[dict[str, Any]] = []
            for swath in ("IW1", "IW2", "IW3"):
                master = burst_map(cache[role_products["master"]], swath, epsg)
                slave = burst_map(cache[role_products["slave"]], swath, epsg)
                common = sorted(set(master).intersection(slave))
                common_geometry = [master[burst_id].intersection(slave[burst_id]) for burst_id in common]
                pair_rows.append({
                    "pair_id": pair_id,
                    "city_id": city_id,
                    "swath": swath,
                    "master_burst_count": len(master),
                    "slave_burst_count": len(slave),
                    "common_burst_count": len(common),
                    "common_burst_ids": ";".join(str(value) for value in common),
                    "master_aoi_coverage": coverage(aoi, list(master.values())),
                    "slave_aoi_coverage": coverage(aoi, list(slave.values())),
                    "common_aoi_coverage": coverage(aoi, common_geometry),
                })
            best = max(
                pair_rows,
                key=lambda row: (
                    row["common_aoi_coverage"],
                    min(row["master_aoi_coverage"], row["slave_aoi_coverage"]),
                    row["common_burst_count"],
                    row["swath"],
                ),
            )
            for row in pair_rows:
                row["selected"] = row["swath"] == best["swath"]
            rows.extend(pair_rows)
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            failures.append(f"{pair_id}: {error}")

    result = pd.DataFrame(rows)
    selected = result.loc[result["selected"]].copy() if not result.empty else result
    minimum = float(config["pairing"]["minimum_aoi_coverage"])
    checks = {
        "pilot_only": config.get("status") == "PILOT_ONLY",
        "full_export_blocked": config.get("full_export_authorized") is False,
        "all_pairs_audited": len(failures) == 0 and len(selected) == len(config["pilot_cities"]),
        "one_swath_selected_per_pair": not selected.empty and bool(selected.groupby("pair_id").size().eq(1).all()),
        "common_bursts_present": not selected.empty and bool(selected["common_burst_count"].gt(0).all()),
        "master_coverage_valid": not selected.empty and bool(selected["master_aoi_coverage"].ge(minimum).all()),
        "slave_coverage_valid": not selected.empty and bool(selected["slave_aoi_coverage"].ge(minimum).all()),
        "common_coverage_valid": not selected.empty and bool(selected["common_aoi_coverage"].ge(minimum).all()),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "minimum_required_coverage": minimum,
        "selected_swath_by_pair": selected.set_index("pair_id")["swath"].to_dict() if not selected.empty else {},
        "selected_common_coverage_by_pair": selected.set_index("pair_id")["common_aoi_coverage"].astype(float).to_dict() if not selected.empty else {},
        "checks": checks,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
