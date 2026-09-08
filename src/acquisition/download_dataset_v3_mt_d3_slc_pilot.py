"""Explicitly gated downloader for the four-product D3 SLC pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from src.acquisition.download_selected_scenes import download_file, get_access_token, validate_zip


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    if config.get("status") != "PILOT_ONLY" or config.get("full_export_authorized") is not False:
        raise ValueError("Only the bounded D3 pilot may be downloaded by this command.")
    return config


def checksum(path: Path, algorithm: str, chunk_bytes: int) -> str:
    digest = hashlib.new(algorithm.lower().replace("-", ""))
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_product(path: Path, row: pd.Series, config: dict[str, Any]) -> dict[str, bool]:
    download = config["download"]
    expected_size = int(row["content_length_bytes"])
    size_ok = not bool(download.get("verify_file_size", True)) or path.stat().st_size == expected_size
    algorithm = str(row.get("checksum_algorithm", "")).strip()
    expected_checksum = str(row.get("checksum_value", "")).strip()
    checksum_required = bool(download.get("verify_checksum_when_available", True)) and bool(algorithm and expected_checksum)
    checksum_ok = True
    if checksum_required:
        checksum_ok = checksum(path, algorithm, int(download.get("chunk_size_mb", 8)) * 1024 * 1024).lower() == expected_checksum.lower()
    zip_ok = not bool(download.get("verify_zip", True)) or validate_zip(path)
    return {"size_verified": size_ok, "checksum_verified": checksum_ok, "zip_verified": zip_ok}


def validate_scope(manifest: pd.DataFrame, config: dict[str, Any]) -> None:
    expected = 2 * len(config["pilot_cities"])
    required = {"city_id", "role", "product_name", "odata_product_id", "content_length_bytes", "online", "product_download_url"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError("Download manifest is missing columns: " + ", ".join(sorted(missing)))
    if len(manifest) != expected or manifest["odata_product_id"].nunique() != expected:
        raise ValueError(f"Refusing download: expected exactly {expected} unique pilot products.")
    if set(manifest["city_id"]) != set(config["pilot_cities"]):
        raise ValueError("Refusing download: manifest cities differ from pilot_cities.")
    if not manifest.groupby("city_id").size().eq(2).all():
        raise ValueError("Refusing download: each pilot city must have exactly two products.")
    online = manifest["online"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    if not bool(online.all()):
        raise ValueError("Refusing download: one or more products are not immediately online.")
    if not bool(pd.to_numeric(manifest["content_length_bytes"], errors="coerce").gt(0).all()):
        raise ValueError("Refusing download: product sizes are missing or invalid.")


def output_path_for(row: pd.Series, root: Path) -> Path:
    name = str(row["product_name"])
    archive_name = name[:-5] + ".zip" if name.upper().endswith(".SAFE") else name + ".zip"
    return root / str(row["city_id"]) / archive_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_mt_d3_slc_coherence_pilot.yaml"))
    parser.add_argument("--authorize-download", action="store_true", help="Required acknowledgement that the four resolved SLC archives are large.")
    parser.add_argument("--overwrite-audit", action="store_true")
    args = parser.parse_args()
    if not args.authorize_download:
        raise SystemExit("Download blocked. Review the resolution audit, then pass --authorize-download explicitly.")

    config = load_config(args.config)
    manifest = pd.read_csv(Path(config["outputs"]["download_manifest"]), keep_default_na=False)
    validate_scope(manifest, config)
    audit_path = Path(config["outputs"]["download_audit"])
    if audit_path.exists() and not args.overwrite_audit:
        raise FileExistsError(f"Download audit exists; pass --overwrite-audit: {audit_path}")

    root = Path(config["download"]["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    required_bytes = 0
    for _, row in manifest.iterrows():
        destination = output_path_for(row, root)
        if not destination.exists() or not all(verify_product(destination, row, config).values()):
            required_bytes += int(row["content_length_bytes"])
    free_bytes = shutil.disk_usage(root).free
    multiplier = float(config["download"].get("minimum_free_space_multiplier", 1.25))
    if free_bytes < int(required_bytes * multiplier):
        raise RuntimeError(
            f"Insufficient free disk space: {free_bytes / 1024**3:.2f} GiB available; "
            f"at least {required_bytes * multiplier / 1024**3:.2f} GiB required."
        )

    load_dotenv()
    username, password = os.getenv("CDSE_USERNAME"), os.getenv("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be defined in .env.")

    download = config["download"]
    session = requests.Session()
    results: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        destination = output_path_for(row, root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            checks = verify_product(destination, row, config)
            if all(checks.values()):
                results.append({"product_name": row["product_name"], "status": "skipped_verified", "path": str(destination), **checks})
                continue
            destination.unlink()

        error = ""
        checks = {"size_verified": False, "checksum_verified": False, "zip_verified": False}
        for attempt in range(1, int(download.get("maximum_retries", 3)) + 1):
            try:
                token = get_access_token(
                    session, str(download["token_url"]), str(download.get("client_id", "cdse-public")),
                    username, password, int(download.get("timeout_seconds", 300)),
                )
                download_file(
                    session=session, url=str(row["product_download_url"]), output_path=destination,
                    expected_size=int(row["content_length_bytes"]), token=token,
                    chunk_size_bytes=int(download.get("chunk_size_mb", 8)) * 1024 * 1024,
                    timeout_seconds=int(download.get("timeout_seconds", 300)),
                )
                checks = verify_product(destination, row, config)
                if not all(checks.values()):
                    raise RuntimeError(f"Downloaded product verification failed: {checks}")
                error = ""
                break
            except (requests.RequestException, RuntimeError, OSError, ValueError) as exc:
                error = str(exc)
                if destination.exists():
                    destination.unlink()
                if attempt < int(download.get("maximum_retries", 3)):
                    time.sleep(int(download.get("retry_wait_seconds", 15)))
        results.append({
            "product_name": row["product_name"],
            "status": "downloaded_verified" if not error else "failed",
            "path": str(destination), **checks, "error": error,
        })

    passed = all(row["status"] in {"downloaded_verified", "skipped_verified"} for row in results)
    report = {
        "status": "PASS" if passed else "FAIL", "pilot_only": True,
        "full_export_authorized": False, "explicit_download_authorization_received": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "product_count": len(results), "results": results,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
