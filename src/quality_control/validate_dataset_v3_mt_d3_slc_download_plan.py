"""Validate the bounded D3 pilot download manifest without downloading data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


REQUIRED = {
    "pair_id", "city_id", "role", "stac_item_id", "product_name",
    "odata_product_id", "content_length_bytes", "online", "product_download_url",
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_mt_d3_slc_coherence_pilot.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = pd.read_csv(Path(config["outputs"]["download_manifest"]))
    missing = REQUIRED.difference(manifest.columns)
    if missing:
        raise ValueError("Download manifest is missing columns: " + ", ".join(sorted(missing)))
    expected_products = 2 * len(config["pilot_cities"])
    sizes = pd.to_numeric(manifest["content_length_bytes"], errors="coerce")
    checks = {
        "pilot_only": config.get("status") == "PILOT_ONLY",
        "full_export_blocked": config.get("full_export_authorized") is False,
        "download_default_blocked": config.get("download", {}).get("authorized") is False,
        "cities_exact": set(manifest["city_id"]) == set(config["pilot_cities"]),
        "two_products_per_city": bool(manifest.groupby("city_id").size().eq(2).all()),
        "master_and_slave_per_pair": bool(
            manifest.groupby("pair_id")["role"].apply(lambda roles: set(roles) == {"master", "slave"}).all()
        ),
        "product_ids_unique": int(manifest["odata_product_id"].nunique()) == expected_products,
        "product_names_are_safe": bool(manifest["product_name"].astype(str).str.upper().str.endswith(".SAFE").all()),
        "products_online": bool(as_bool(manifest["online"]).all()),
        "sizes_positive": bool(sizes.notna().all() and sizes.gt(0).all()),
        "download_urls_present": bool(manifest["product_download_url"].fillna("").astype(str).str.strip().ne("").all()),
    }
    total_bytes = int(sizes.sum()) if sizes.notna().all() else 0
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pair_count": int(manifest["pair_id"].nunique()),
        "product_count": int(manifest["odata_product_id"].nunique()),
        "total_download_bytes": total_bytes,
        "total_download_gib": total_bytes / (1024 ** 3),
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
