"""Guarded registry-driven runner for Urban Atlas evidence and V3 fusion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def one(patterns: list[str], label: str) -> Path:
    hits = sorted({p.resolve() for pattern in patterns for p in Path().glob(pattern) if p.is_file()})
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {label}; found {len(hits)}: {hits}")
    return hits[0]


def paths(city: dict, need_ua_source: bool) -> dict[str, Path]:
    cid = city["city_id"]
    grid = city["grid"]
    result = {
        "dw": one([f"data/raw/labels_v3/dynamic_world/{cid}_*/{cid}_dynamic_world_v3_2025.tif",
                   f"data/raw/labels_v3/dynamic_world/{cid}_*/{cid}_dynamic_world_v3_pilot_2025.tif"], f"{cid} Dynamic World raster"),
        "osm_sem": Path(grid["osm_semantic"]), "osm_val": Path(grid["osm_validity"]),
        "ua_code": Path(f"data/interim/labels_v3/urban_atlas/{cid}/{cid}_urban_atlas_2021_code.tif"),
        "ua_evidence": Path(f"data/interim/labels_v3/urban_atlas/{cid}/{cid}_urban_atlas_2021_v3_evidence.tif"),
    }
    if need_ua_source:
        result["ua"] = one([f"data/raw/urban_atlas_2021/{cid}_*/extracted/**/*.fgb"], f"{cid} Urban Atlas FGB")
    return result


def command(module: str, args: list[object]) -> list[str]:
    return [sys.executable, "-m", module, *map(str, args)]


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--city", action="append", help="City ID; repeat for multiple cities")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--registry", type=Path, default=Path("config/dataset_v3_city_registry.json"))
    parser.add_argument("--stage", choices=["urban-atlas", "fusion", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    by_id = {x["city_id"]: x for x in registry["cities"]}
    selected = sorted(by_id) if args.all else args.city
    unknown = sorted(set(selected) - set(by_id))
    if unknown: parser.error(f"Unknown city IDs: {unknown}")

    plan, completed = [], []
    for cid in selected:
        ua_code = Path(f"data/interim/labels_v3/urban_atlas/{cid}/{cid}_urban_atlas_2021_code.tif")
        ua_evidence = Path(f"data/interim/labels_v3/urban_atlas/{cid}/{cid}_urban_atlas_2021_v3_evidence.tif")
        ua_complete = ua_code.is_file() and ua_evidence.is_file()
        ua_partial = ua_code.exists() != ua_evidence.exists()
        if ua_partial: raise RuntimeError(f"{cid} has partial Urban Atlas evidence outputs; inspect before continuing")
        p = paths(by_id[cid], args.stage in ("urban-atlas", "all") and not ua_complete)
        for key in ("osm_sem", "osm_val"):
            if not p[key].is_file(): raise FileNotFoundError(f"{cid} missing {key}: {p[key]}")
        commands = []
        if args.stage in ("urban-atlas", "all") and not ua_complete:
            commands.append(command("src.evaluation.build_urban_atlas_evidence_v3", [
                "--city-id", cid, "--urban-atlas", p["ua"], "--dynamic-world", p["dw"],
                "--semantic", p["osm_sem"], "--validity", p["osm_val"]]))
        if args.stage in ("fusion", "all"):
            commands.append(command("src.data.build_dataset_v3_fusion", [
                "--city-id", cid, "--dynamic-world", p["dw"], "--osm-semantic", p["osm_sem"],
                "--osm-validity", p["osm_val"], "--ua-code", p["ua_code"], "--ua-evidence", p["ua_evidence"]]))
        plan.append({"city_id": cid, "split": by_id[cid]["split"], "commands": commands})
        if not args.dry_run:
            for cmd in commands: subprocess.run(cmd, check=True)
            completed.append(cid)

    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "stage": args.stage, "plan": plan}, indent=2))
        return
    manifest_dir = Path("metadata/dataset_v3/batch_runs")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = manifest_dir / f"v3_batch_{args.stage}_{stamp}.json"
    if manifest.exists(): raise FileExistsError(f"Refusing to overwrite {manifest}")
    manifest.write_text(json.dumps({"schema_version": "dataset-v3-batch-run-0.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "stage": args.stage, "registry": str(args.registry), "completed_city_ids": completed,
                                    "safety": "Additive V3 paths only; V1/V2.1/V2.2 unchanged."}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "completed": completed, "manifest": str(manifest)}, indent=2))


if __name__ == "__main__": main()
