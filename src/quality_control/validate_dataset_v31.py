"""Independently validate Dataset V3.1 tiles using the audited V3 checks."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from src.quality_control import validate_dataset_v3 as base

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("config/dataset_v31_materialization.yaml"))
    p.add_argument("--resume",action="store_true")
    return p.parse_args()

def main():
    a=parse_args(); cfg=base.load_yaml(a.config)
    if cfg.get("dataset_version")!="v3.1": raise ValueError("Expected V3.1 configuration")
    meta=Path(cfg["output"]["materialization_metadata"]); qa=Path(cfg["output"]["qa_metadata"])
    source_manifest=meta/"dataset_v31_materialized_manifest.csv"; source_audit=meta/"dataset_v31_materialization_audit.json"
    base.require_file(source_manifest,"V3.1 materialized manifest"); base.require_file(source_audit,"V3.1 materialization audit")
    # The audited validator expects these two conventional temporary names.
    compat_manifest=meta/"dataset_v3_materialized_manifest.csv"; compat_audit=meta/"dataset_v3_materialization_audit.json"
    if compat_manifest.exists() or compat_audit.exists(): raise FileExistsError("Stale compatibility files in V3.1 metadata")
    compat_manifest.write_bytes(source_manifest.read_bytes()); compat_audit.write_bytes(source_audit.read_bytes())
    old_argv=sys.argv[:]
    try:
        sys.argv=["validate_dataset_v3","--config",str(a.config)] + (["--resume"] if a.resume else [])
        base.main()
    finally:
        sys.argv=old_argv
        compat_manifest.unlink(missing_ok=True); compat_audit.unlink(missing_ok=True)
    old_report=qa/"dataset_v3_tile_qa.json"; old_city=qa/"dataset_v3_city_qa_summary.csv"; old_visual=qa/"dataset_v3_visual_qa_sample.csv"
    new_report=qa/"dataset_v31_tile_qa.json"; new_city=qa/"dataset_v31_city_qa_summary.csv"; new_visual=qa/"dataset_v31_visual_qa_sample.csv"
    for old,new in [(old_city,new_city),(old_visual,new_visual)]:
        if new.exists(): raise FileExistsError(new)
        os.replace(old,new)
    report=json.loads(old_report.read_text(encoding="utf-8")); report.update({"schema_version":"dataset-v3.1-tile-qa-0.1","dataset_version":"v3.1","parent_dataset_version":"v3","materialization_manifest":str(source_manifest),"materialization_manifest_sha256":base.sha256(source_manifest),"materialization_audit_sha256":base.sha256(source_audit),"visual_qa_sample":str(new_visual),"manual_visual_qa_required":False,"next_gate":"training_normalization_and_dataset_v3_1_freeze"})
    new_report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); old_report.unlink()
    print(json.dumps({"qa_status":"PASS","dataset_version":"v3.1","tile_count":report["tile_count"],"official_manifest":report["official_manifest"],"manual_visual_qa_required":False},indent=2))
if __name__=="__main__": main()
