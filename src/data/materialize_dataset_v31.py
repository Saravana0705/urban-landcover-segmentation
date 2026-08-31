"""Materialize exactly the frozen Dataset V3.1 tile selection."""
from __future__ import annotations
import argparse, json, os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, transform as window_transform
from src.data import materialize_dataset_v3 as base

DESC = {
    "semantic": "dataset_v3_1_semantic_ignore_255",
    "validity": "dataset_v3_1_training_validity",
    "provenance": "v3_1_retained_source_bits_osm1_dynamic_world2_urban_atlas4",
    "support": "v3_1_retained_agreeing_source_count",
    "conflict": "v3_source_conflict_diagnostic_unchanged",
}

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("config/dataset_v31_materialization.yaml"))
    p.add_argument("--selected-manifest",type=Path,default=None)
    p.add_argument("--resume",action="store_true")
    return p.parse_args()

def main():
    a=args(); cfg=base.load_yaml(a.config)
    if cfg.get("dataset_version")!="v3.1": raise ValueError("Expected V3.1 configuration")
    selected=a.selected_manifest or Path(cfg["input"]["selected_manifest"])
    root=Path(cfg["output"]["tile_root"]); meta=Path(cfg["output"]["materialization_metadata"])
    base.require_within(root,Path(cfg["safety"]["permitted_data_write_prefix"]),"V3.1 tile root")
    base.require_within(meta,Path(cfg["safety"]["permitted_metadata_write_prefix"]),"V3.1 metadata")
    if not selected.is_file(): raise FileNotFoundError(selected)
    frame=pd.read_csv(selected); size=int(cfg["encoding"]["tile_size"]); base.validate_selection(frame,size)
    exp=cfg["expected"]
    if len(frame)!=int(exp["tile_count"]): raise ValueError(f"Expected {exp['tile_count']} rows; found {len(frame)}")
    splits={str(k):int(v) for k,v in frame.groupby("split").size().items()}
    if splits!={str(k):int(v) for k,v in exp["split_tile_counts"].items()}: raise ValueError(f"Split mismatch: {splits}")
    for role,(column,_,_,_) in base.ROLES.items():
        if role!="image":
            normalized=frame[column].astype(str).str.replace("\\","/",regex=False)
            if (~normalized.str.contains("data/interim/labels_v3_1/fused/",regex=False)).any():
                raise ValueError(f"{role} sources are not V3.1")
    frame=frame.sort_values(["split","city_id","tile_id"]).reset_index(drop=True)
    if root.exists() and not a.resume: raise FileExistsError(f"Tile root exists: {root}")
    meta.mkdir(parents=True,exist_ok=True)
    manifest=meta/"dataset_v31_materialized_manifest.csv"; report=meta/"dataset_v31_materialization_audit.json"
    if (manifest.exists() or report.exists()) and not a.resume: raise FileExistsError("V3.1 metadata exists")
    rows=[]; written=0; skipped=0; source_hashes={}
    for city_id,group in frame.groupby("city_id",sort=True):
        first=group.iloc[0]
        paths={r:base.project_path(first[c]) for r,(c,_,_,_) in base.ROLES.items()}
        for role,path in paths.items():
            if not path.is_file() or path.stat().st_size==0: raise FileNotFoundError(f"{city_id} {role}: {path}")
            source_hashes[str(path)]=base.sha256(path)
        with ExitStack() as stack:
            src={r:stack.enter_context(rasterio.open(p)) for r,p in paths.items()}; ref=src["image"]
            if ref.count!=2: raise ValueError(f"{city_id} SAR band count")
            for role,ds in src.items():
                if role!="image" and ds.count!=1: raise ValueError(f"{city_id} {role} band count")
                if not base.same_grid(ref,ds): raise ValueError(f"{city_id} {role} grid")
            for _,row in group.iterrows():
                outs=base.output_paths(root,row); exists=[p.exists() and p.stat().st_size>0 for p in outs.values()]
                if any(exists):
                    if not a.resume or not all(exists): raise FileExistsError(f"Partial group: {row['tile_id']}")
                    skipped+=1
                else:
                    win=Window(int(row["col_offset"]),int(row["row_offset"]),size,size)
                    transform=window_transform(win,ref.transform)
                    tags={"dataset_version":"v3.1","parent_dataset_version":"v3","city_id":str(row["city_id"]),"city_name":str(row["city_name"]),"split":str(row["split"]),"tile_id":str(row["tile_id"])}
                    for role,ds in src.items():
                        fill=base.ROLES[role][3]
                        if role=="image":
                            array=ds.read(window=win,out_shape=(2,size,size),boundless=True,fill_value=fill).astype(np.float32,copy=False); descriptions=["Sigma0_VV","Sigma0_VH"]
                        else:
                            array=ds.read(1,window=win,out_shape=(size,size),boundless=True,fill_value=fill).astype(np.uint8,copy=False); descriptions=[DESC[role]]
                        base.atomic_write(outs[role],array,base.make_profile(ds,role,transform,cfg),tags,descriptions)
                    written+=1
                out=row.to_dict(); out.update({"image_path":str(outs["image"]),"semantic_mask_path":str(outs["semantic"]),"validity_mask_path":str(outs["validity"]),"provenance_mask_path":str(outs["provenance"]),"support_mask_path":str(outs["support"]),"conflict_mask_path":str(outs["conflict"]),"materialization_status":"PASS"}); rows.append(out)
        print(f"{city_id}: {len(group)} tiles complete")
    result=pd.DataFrame(rows)
    tmp=manifest.with_suffix(".partial.csv"); result.to_csv(tmp,index=False); os.replace(tmp,manifest)
    payload={"schema_version":"dataset-v3.1-materialization-audit-0.1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","dataset_version":"v3.1","parent_dataset_version":"v3","selected_manifest":str(selected),"selected_manifest_sha256":base.sha256(selected),"tile_count":len(result),"written_tile_count_this_run":written,"resumed_tile_count":skipped,"raster_file_count":len(result)*len(base.ROLES),"selected_by_split":{str(k):int(v) for k,v in result.groupby("split").size().items()},"source_sha256":source_hashes,"materialized_manifest":str(manifest),"materialized_manifest_sha256":base.sha256(manifest),"parent_dataset_v3_modified":False,"qa_status":"PENDING_INDEPENDENT_TILE_QA"}
    tmp=report.with_suffix(".partial.json"); tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,report)
    print(json.dumps({"status":"PASS","tile_count":len(result),"raster_file_count":len(result)*6,"next":"python -m src.quality_control.validate_dataset_v31"},indent=2))
if __name__=="__main__": main()
