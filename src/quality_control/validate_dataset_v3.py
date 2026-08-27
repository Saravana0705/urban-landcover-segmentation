"""Independently validate materialized Dataset V3 tiles and freeze a QA manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml


CLASSES = {1: "buildings", 2: "roads", 3: "vegetation", 4: "bare_land", 5: "water"}
PATH_COLUMNS = {
    "image": "image_path",
    "semantic": "semantic_mask_path",
    "validity": "validity_mask_path",
    "provenance": "provenance_mask_path",
    "support": "support_mask_path",
    "conflict": "conflict_mask_path",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def require_within(path: Path, permitted_prefix: Path, label: str) -> None:
    if not path.resolve().is_relative_to(permitted_prefix.resolve()):
        raise ValueError(f"{label} must remain under {permitted_prefix}: {path}")


def sample_visual_qa(manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    reasons: dict[str, set[str]] = {}
    def add(rows: pd.DataFrame, reason: str) -> None:
        for tile_id in rows["tile_id"].astype(str):
            reasons.setdefault(tile_id, set()).add(reason)

    evaluation = manifest[manifest["split"].isin(["val", "test"])]
    for _, group in evaluation.groupby("city_id"):
        ranked = group.sort_values(["conflict_fraction_source_pixels", "valid_fraction_source_pixels"], ascending=False)
        add(ranked.head(1), "evaluation_city_representative")

    training = manifest[manifest["split"] == "train"]
    count = int(config["visual_qa_sampling"]["top_training_examples_per_minority_class"])
    for class_name in ("buildings", "roads", "bare_land", "water"):
        add(training.sort_values(f"{class_name}_fraction_valid", ascending=False).head(count), f"high_{class_name}")

    conflict_count = int(config["visual_qa_sampling"]["top_source_conflict_examples"])
    add(manifest.sort_values("conflict_fraction_source_pixels", ascending=False).head(conflict_count), "high_source_conflict")

    selected = manifest[manifest["tile_id"].astype(str).isin(reasons)].copy()
    selected["visual_qa_reason"] = selected["tile_id"].astype(str).map(lambda x: "+".join(sorted(reasons[x])))
    columns = [
        "tile_id", "city_id", "city_name", "split", "visual_qa_reason",
        "image_path", "semantic_mask_path", "validity_mask_path",
        "provenance_mask_path", "support_mask_path", "conflict_mask_path",
        "buildings_fraction_valid", "roads_fraction_valid", "vegetation_fraction_valid",
        "bare_land_fraction_valid", "water_fraction_valid", "conflict_fraction_source_pixels",
    ]
    return selected[columns].sort_values(["split", "city_id", "tile_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/dataset_v3_materialization.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Re-run validation and replace only V3 QA metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    materialization_dir = Path(config["output"]["materialization_metadata"])
    manifest_path = args.manifest or materialization_dir / "dataset_v3_materialized_manifest.csv"
    audit_path = materialization_dir / "dataset_v3_materialization_audit.json"
    selected_path = Path(config["input"]["selected_manifest"])
    qa_dir = Path(config["output"]["qa_metadata"])
    official_path = Path(config["output"]["official_manifest"])
    outputs = {
        "official": official_path,
        "report": qa_dir / "dataset_v3_tile_qa.json",
        "city": qa_dir / "dataset_v3_city_qa_summary.csv",
        "visual": qa_dir / "dataset_v3_visual_qa_sample.csv",
    }
    metadata_prefix = Path(config["safety"]["permitted_metadata_write_prefix"])
    for label, path in outputs.items():
        require_within(path, metadata_prefix, f"V3 QA output {label}")
    collisions = [str(path) for path in outputs.values() if path.exists()]
    if collisions and not args.resume:
        raise FileExistsError(f"V3 QA outputs already exist; use --resume after review: {collisions}")
    require_file(manifest_path, "materialized manifest")
    require_file(audit_path, "materialization audit")
    require_file(selected_path, "selected candidate manifest")

    frame = pd.read_csv(manifest_path)
    selected_source = pd.read_csv(selected_path)
    if frame.empty or frame["tile_id"].duplicated().any():
        raise ValueError("Materialized manifest is empty or has duplicate tile IDs")
    if set(frame["tile_id"].astype(str)) != set(selected_source["tile_id"].astype(str)):
        raise ValueError("Materialized tile IDs do not exactly match the selected manifest")
    expected_splits = {
        str(k): int(v) for k, v in selected_source.groupby("split").size().items()
    }
    actual_splits = frame.groupby("split").size().to_dict()
    if actual_splits != expected_splits:
        raise ValueError(f"Unexpected split counts: {actual_splits}; expected {expected_splits}")

    tile_size = int(config["encoding"]["tile_size"])
    passed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        tile_id = str(row["tile_id"])
        paths = {role: Path(str(row[column])) for role, column in PATH_COLUMNS.items()}
        try:
            for role, path in paths.items():
                require_file(path, f"{tile_id} {role}")
            with rasterio.open(paths["image"]) as image_ds, \
                    rasterio.open(paths["semantic"]) as semantic_ds, \
                    rasterio.open(paths["validity"]) as validity_ds, \
                    rasterio.open(paths["provenance"]) as provenance_ds, \
                    rasterio.open(paths["support"]) as support_ds, \
                    rasterio.open(paths["conflict"]) as conflict_ds:
                datasets = [image_ds, semantic_ds, validity_ds, provenance_ds, support_ds, conflict_ds]
                grids = {(ds.width, ds.height, str(ds.crs), tuple(ds.transform)[:6]) for ds in datasets}
                if len(grids) != 1 or image_ds.count != 2 or any(ds.count != 1 for ds in datasets[1:]):
                    raise ValueError("band count or grid alignment failure")
                if image_ds.width != tile_size or image_ds.height != tile_size:
                    raise ValueError("unexpected tile dimensions")
                if image_ds.dtypes != ("float32", "float32") or any(ds.dtypes[0] != "uint8" for ds in datasets[1:]):
                    raise ValueError("unexpected raster dtype")

                image = image_ds.read()
                semantic = semantic_ds.read(1)
                validity = validity_ds.read(1)
                provenance = provenance_ds.read(1)
                support = support_ds.read(1)
                conflict = conflict_ds.read(1)
                allowed = config["encoding"]
                expected_bounds = np.asarray([
                    row["bounds_left"], row["bounds_bottom"],
                    row["bounds_right"], row["bounds_top"],
                ], dtype=np.float64)
                actual_bounds = np.asarray([
                    image_ds.bounds.left, image_ds.bounds.bottom,
                    image_ds.bounds.right, image_ds.bounds.top,
                ], dtype=np.float64)
                tags = image_ds.tags()
                checks = {
                    "sar_finite": bool(np.isfinite(image).all()),
                    "crs_matches_manifest": str(image_ds.crs) == str(row["crs"]),
                    "bounds_match_manifest": bool(np.allclose(actual_bounds, expected_bounds, rtol=0.0, atol=1e-6)),
                    "pixel_size_is_10m": bool(abs(float(image_ds.transform.a) - 10.0) <= 1e-9 and abs(abs(float(image_ds.transform.e)) - 10.0) <= 1e-9),
                    "tile_tags_match": tags.get("tile_id") == tile_id and tags.get("city_id") == str(row["city_id"]) and tags.get("split") == str(row["split"]),
                    "semantic_nodata_255": semantic_ds.nodata == 255,
                    "validity_nodata_0": validity_ds.nodata == 0,
                    "semantic_values": bool(set(np.unique(semantic)).issubset({1, 2, 3, 4, 5, 255})),
                    "validity_values": bool(set(np.unique(validity)).issubset(set(allowed["validity_values"]))),
                    "provenance_values": bool(set(np.unique(provenance)).issubset(set(allowed["provenance_values"]))),
                    "support_values": bool(set(np.unique(support)).issubset(set(allowed["support_count_values"]))),
                    "conflict_values": bool(set(np.unique(conflict)).issubset(set(allowed["conflict_values"]))),
                    "valid_semantic_classes": bool(np.all(np.isin(semantic[validity == 1], [1, 2, 3, 4, 5]))),
                    "ignored_semantic_is_255": bool(np.all(semantic[validity == 0] == 255)),
                    "valid_has_provenance": bool(np.all(provenance[validity == 1] > 0)),
                    "valid_has_support": bool(np.all(support[validity == 1] >= 1)),
                    "valid_count": int((validity == 1).sum()) == int(row["valid_pixel_count"]),
                    "ignore_count": int((semantic == 255).sum()) == int(row["ignore_pixel_count"] + row["padding_pixel_count"]),
                    "conflict_count": int(conflict.sum()) == int(row["conflict_pixel_count"]),
                }
                for class_id, class_name in CLASSES.items():
                    checks[f"{class_name}_count"] = int((semantic == class_id).sum()) == int(row[f"{class_name}_pixel_count"])
                for level in (1, 2, 3):
                    checks[f"support_{level}_count"] = int((support == level).sum()) == int(row[f"support_{level}_pixel_count"])
                for value in range(1, 8):
                    checks[f"provenance_{value}_count"] = int((provenance == value).sum()) == int(row[f"provenance_{value}_pixel_count"])
                source_height = int(row["source_height_pixels"])
                source_width = int(row["source_width_pixels"])
                padding = np.ones((tile_size, tile_size), dtype=bool)
                padding[:source_height, :source_width] = False
                checks.update({
                    "padding_semantic": bool(np.all(semantic[padding] == 255)),
                    "padding_validity": bool(np.all(validity[padding] == 0)),
                    "padding_auxiliary": bool(np.all(provenance[padding] == 0) and np.all(support[padding] == 0) and np.all(conflict[padding] == 0)),
                    "padding_sar": bool(np.all(image[:, padding] == 0.0)),
                })
                failed = [name for name, result in checks.items() if not result]
                if failed:
                    raise ValueError("failed checks: " + ", ".join(failed))

            output_row = row.to_dict()
            output_row.update({f"{role}_sha256": sha256(path) for role, path in paths.items()})
            output_row["tile_qa_status"] = "PASS"
            passed_rows.append(output_row)
        except Exception as exc:
            failures.append({"tile_id": tile_id, "error": str(exc)})
        if position % 100 == 0 or position == len(frame):
            print(f"Validated {position}/{len(frame)} tiles; failures={len(failures)}")

    if failures:
        qa_dir.mkdir(parents=True, exist_ok=True)
        failure_path = qa_dir / "dataset_v3_tile_qa_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Dataset V3 tile QA failed for {len(failures)} tiles; see {failure_path}")

    official = pd.DataFrame(passed_rows).sort_values(["split", "city_id", "tile_id"])
    city_rows = []
    for (split, city_id, city_name), group in official.groupby(["split", "city_id", "city_name"]):
        city_rows.append({
            "split": split, "city_id": city_id, "city_name": city_name,
            "tile_count": int(len(group)), "valid_pixel_count": int(group["valid_pixel_count"].sum()),
            "ignore_pixel_count_source": int(group["ignore_pixel_count"].sum()),
            **{f"{name}_pixel_count": int(group[f"{name}_pixel_count"].sum()) for name in CLASSES.values()},
            "tile_qa_status": "PASS",
        })
    city = pd.DataFrame(city_rows)
    visual = sample_visual_qa(official, config)

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_official = official_path.with_suffix(".partial.csv")
    official.to_csv(temporary_official, index=False)
    os.replace(temporary_official, official_path)
    temporary_city = outputs["city"].with_suffix(".partial.csv")
    temporary_visual = outputs["visual"].with_suffix(".partial.csv")
    city.to_csv(temporary_city, index=False)
    visual.to_csv(temporary_visual, index=False)
    os.replace(temporary_city, outputs["city"])
    os.replace(temporary_visual, outputs["visual"])

    report = {
        "schema_version": "dataset-v3-tile-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_status": "PASS",
        "dataset_version": "v3",
        "tile_count": int(len(official)),
        "passed_tile_count": int(len(official)),
        "failed_tile_count": 0,
        "split_tile_counts": expected_splits,
        "city_counts": {k: int(v) for k, v in official.groupby("split")["city_id"].nunique().items()},
        "valid_pixel_count": int(official["valid_pixel_count"].sum()),
        "class_pixel_counts": {name: int(official[f"{name}_pixel_count"].sum()) for name in CLASSES.values()},
        "materialization_manifest": str(manifest_path),
        "materialization_manifest_sha256": sha256(manifest_path),
        "materialization_audit_sha256": sha256(audit_path),
        "selected_manifest_sha256": sha256(selected_path),
        "official_manifest": str(official_path),
        "official_manifest_sha256": sha256(official_path),
        "visual_qa_sample": str(outputs["visual"]),
        "visual_qa_tile_count": int(len(visual)),
        "ignore_index": 255,
        "model_target_mapping": {"source_1_to_5": "model_0_to_4", "source_255": "ignore_index_255"},
        "legacy_dataset_artifacts_modified": False,
        "next_gate": "targeted_visual_qa_then_training_normalization_and_freeze",
    }
    temporary_report = outputs["report"].with_suffix(".partial.json")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, outputs["report"])
    print(json.dumps({"qa_status": "PASS", "tile_count": len(official), "official_manifest": str(official_path), "visual_qa_tile_count": len(visual)}, indent=2))


if __name__ == "__main__":
    main()
