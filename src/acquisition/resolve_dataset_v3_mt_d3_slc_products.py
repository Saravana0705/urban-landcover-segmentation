"""Resolve the four D3 pilot SLC products through the CDSE OData catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


PAIR_COLUMNS = {"pair_id", "city_id", "master_item_id", "slave_item_id"}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    if config.get("status") != "PILOT_ONLY" or config.get("full_export_authorized") is not False:
        raise ValueError("D3 resolution must remain PILOT_ONLY with full export disabled.")
    return config


def safe_product_name(item_id: str) -> str:
    name = str(item_id).strip()
    return name if name.upper().endswith(".SAFE") else f"{name}.SAFE"


def checksum_fields(value: Any) -> tuple[str, str]:
    checksums = value if isinstance(value, list) else []
    if not checksums:
        return "", ""
    first = checksums[0] if isinstance(checksums[0], dict) else {}
    return str(first.get("Algorithm", "")).strip(), str(first.get("Value", "")).strip()


def query_product(
    session: requests.Session,
    catalogue_url: str,
    collection_name: str,
    item_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    name = safe_product_name(item_id)
    escaped = name.replace("'", "''")
    response = session.get(
        f"{catalogue_url.rstrip('/')}/Products",
        params={
            "$filter": f"Collection/Name eq '{collection_name}' and Name eq '{escaped}'",
            "$top": "2",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    matches = response.json().get("value", [])
    exact = [row for row in matches if str(row.get("Name", "")) == name]
    if len(exact) != 1:
        raise RuntimeError(f"Expected one OData match for {name}; received {len(exact)}.")
    return exact[0]


def pair_product_references(pairs: pd.DataFrame) -> list[dict[str, str]]:
    missing = PAIR_COLUMNS.difference(pairs.columns)
    if missing:
        raise ValueError("Pair manifest is missing columns: " + ", ".join(sorted(missing)))
    rows: list[dict[str, str]] = []
    for pair in pairs.to_dict("records"):
        for role in ("master", "slave"):
            rows.append({
                "pair_id": str(pair["pair_id"]),
                "city_id": str(pair["city_id"]),
                "role": role,
                "stac_item_id": str(pair[f"{role}_item_id"]),
            })
    return rows


def resolve_products(
    pairs: pd.DataFrame,
    config: dict[str, Any],
    session: requests.Session,
) -> tuple[pd.DataFrame, list[str]]:
    odata = config["odata"]
    cache: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    output: list[dict[str, Any]] = []
    for reference in pair_product_references(pairs):
        item_id = reference["stac_item_id"]
        try:
            if item_id not in cache:
                cache[item_id] = query_product(
                    session,
                    str(odata["catalogue_url"]),
                    str(odata["collection_name"]),
                    item_id,
                    int(odata.get("timeout_seconds", 60)),
                )
            product = cache[item_id]
            algorithm, checksum = checksum_fields(product.get("Checksum"))
            product_id = str(product.get("Id", "")).strip()
            size = int(product.get("ContentLength") or 0)
            output.append({
                **reference,
                "product_name": str(product.get("Name", "")),
                "odata_product_id": product_id,
                "content_length_bytes": size,
                "content_length_gib": size / (1024 ** 3),
                "online": bool(product.get("Online", False)),
                "s3_path": str(product.get("S3Path", "")),
                "checksum_algorithm": algorithm,
                "checksum_value": checksum,
                "product_download_url": str(odata["download_url_template"]).format(product_id=product_id),
            })
        except (requests.RequestException, RuntimeError, ValueError, TypeError) as error:
            failures.append(f"{item_id}: {error}")
    return pd.DataFrame(output), failures


def build_report(manifest: pd.DataFrame, failures: list[str], config: dict[str, Any]) -> dict[str, Any]:
    unique_products = manifest.drop_duplicates("odata_product_id") if not manifest.empty else manifest
    required_count = 2 * len(config["pilot_cities"])
    total_bytes = int(unique_products["content_length_bytes"].sum()) if not unique_products.empty else 0
    checks = {
        "pilot_only": config.get("status") == "PILOT_ONLY",
        "full_export_blocked": config.get("full_export_authorized") is False,
        "download_default_blocked": config.get("download", {}).get("authorized") is False,
        "all_products_resolved": len(failures) == 0 and len(manifest) == required_count,
        "unique_products": not manifest.empty and int(manifest["odata_product_id"].nunique()) == required_count,
        "all_products_online": not manifest.empty and bool(manifest["online"].all()),
        "all_sizes_positive": not manifest.empty and bool((manifest["content_length_bytes"] > 0).all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pair_count": len(manifest) // 2,
        "product_count": len(unique_products),
        "total_download_bytes": total_bytes,
        "total_download_gib": total_bytes / (1024 ** 3),
        "products_by_city": manifest.groupby("city_id").size().astype(int).to_dict() if not manifest.empty else {},
        "checks": checks,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_mt_d3_slc_coherence_pilot.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    pair_path = Path(config["outputs"]["pair_manifest"])
    output = Path(config["outputs"]["download_manifest"])
    audit = Path(config["outputs"]["product_resolution_audit"])
    if (output.exists() or audit.exists()) and not args.overwrite:
        raise FileExistsError("Resolution output exists; pass --overwrite to replace it.")
    manifest, failures = resolve_products(pd.read_csv(pair_path), config, requests.Session())
    report = build_report(manifest, failures, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
