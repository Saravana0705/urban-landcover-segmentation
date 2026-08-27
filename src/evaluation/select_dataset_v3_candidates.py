"""Select Dataset V3 tiles using a versioned, class-aware policy.

This stage is metadata-only. It never writes, moves, or deletes image, semantic,
validity, or source-evidence rasters. Existing Dataset V1/V2/V2.2 artifacts are
therefore outside its write scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


CLASSES = ("buildings", "roads", "vegetation", "bare_land", "water")
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    if not isinstance(policy, dict):
        raise ValueError("Selection policy must be a YAML mapping")
    return policy


def validate_candidates(frame: pd.DataFrame, policy: dict[str, Any]) -> None:
    required = {
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "valid_fraction_source_pixels",
        "padding_fraction",
        "valid_pixel_count",
    }
    for class_name in CLASSES:
        required.add(f"{class_name}_pixel_count")
        required.add(f"{class_name}_fraction_valid")

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Candidate manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate manifest is empty")
    if frame["tile_id"].duplicated().any():
        duplicates = frame.loc[frame["tile_id"].duplicated(), "tile_id"].tolist()
        raise ValueError(f"Duplicate tile_id values: {duplicates[:10]}")

    unknown_splits = sorted(set(frame["split"]) - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"Unknown split values: {unknown_splits}")

    fraction_columns = [
        "valid_fraction_source_pixels",
        "padding_fraction",
        *[f"{name}_fraction_valid" for name in CLASSES],
    ]
    values = frame[fraction_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Non-finite values found in required fraction columns")
    if ((values < 0) | (values > 1)).any().any():
        raise ValueError("Required fractions must be between 0 and 1")

    count_columns = [f"{name}_pixel_count" for name in CLASSES]
    counts = frame[count_columns].apply(pd.to_numeric, errors="coerce")
    valid_counts = pd.to_numeric(frame["valid_pixel_count"], errors="coerce")
    if counts.isna().any().any() or valid_counts.isna().any():
        raise ValueError("Non-numeric values found in required pixel-count columns")
    if (counts < 0).any().any() or (valid_counts < 0).any():
        raise ValueError("Pixel counts cannot be negative")
    if not np.array_equal(counts.sum(axis=1).to_numpy(), valid_counts.to_numpy()):
        raise ValueError("Per-class pixel counts do not sum to valid_pixel_count")

    expected = policy["geographic_split_guard"]["expected_city_counts"]
    actual = frame.groupby("split")["city_id"].nunique().to_dict()
    for split in SPLITS:
        if int(actual.get(split, 0)) != int(expected[split]):
            raise ValueError(
                f"Expected {expected[split]} {split} cities but found "
                f"{actual.get(split, 0)}"
            )

    city_splits = frame.groupby("city_id")["split"].nunique()
    if (city_splits != 1).any():
        bad = city_splits[city_splits != 1].index.tolist()
        raise ValueError(f"Cities occur in more than one split: {bad}")


def select(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    coverage = policy["coverage_gate"]
    result["coverage_eligible"] = (
        result["valid_fraction_source_pixels"]
        >= float(coverage["minimum_valid_fraction_source_pixels"])
    ) & (
        result["padding_fraction"] <= float(coverage["maximum_padding_fraction"])
    )

    thresholds = policy["training_selection"]["thresholds_fraction_valid"]
    for class_name, threshold in thresholds.items():
        result[f"rule_{class_name}"] = (
            result[f"{class_name}_fraction_valid"] >= float(threshold)
        )

    rule_columns = [f"rule_{name}" for name in thresholds]
    train_class_eligible = result[rule_columns].any(axis=1)
    is_train = result["split"].eq("train")
    result["selected"] = result["coverage_eligible"] & (
        (~is_train) | train_class_eligible
    )

    def reason(row: pd.Series) -> str:
        if not row["coverage_eligible"]:
            return "coverage_gate_failed"
        if row["split"] != "train":
            return "coverage_only_evaluation"
        matches = [
            f"{name}>={float(threshold):.6f}"
            for name, threshold in thresholds.items()
            if row[f"rule_{name}"]
        ]
        return "+".join(matches) if matches else "no_training_minority_rule"

    result["selection_reason"] = result.apply(reason, axis=1)
    result["rejection_reason"] = np.where(
        result["selected"], "", result["selection_reason"]
    )
    return result


def city_summary(selected_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, city_id, city_name), group in selected_audit.groupby(
        ["split", "city_id", "city_name"], sort=True
    ):
        chosen = group[group["selected"]]
        rows.append(
            {
                "split": split,
                "city_id": city_id,
                "city_name": city_name,
                "candidate_tile_count": int(len(group)),
                "coverage_eligible_tile_count": int(group["coverage_eligible"].sum()),
                "selected_tile_count": int(len(chosen)),
                "selected_fraction_candidates": float(len(chosen) / len(group)),
                "selected_valid_pixel_count": int(chosen["valid_pixel_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def class_distribution(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [(split, selected[selected["split"] == split]) for split in SPLITS]
    groups.append(("all", selected))
    for split, group in groups:
        valid = int(group["valid_pixel_count"].sum())
        for class_name in CLASSES:
            pixels = int(group[f"{class_name}_pixel_count"].sum())
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "pixel_count": pixels,
                    "fraction_valid": float(pixels / valid) if valid else 0.0,
                    "selected_tile_count": int(len(group)),
                    "valid_pixel_count": valid,
                }
            )
    return pd.DataFrame(rows)


def ensure_outputs_available(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing selection metadata. Use --force only "
            f"after reviewing these files: {existing}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dataset_v3_selection.yaml"),
    )
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = read_policy(args.config)
    candidate_path = args.candidates or Path(policy["input"]["candidate_manifest"])
    output_dir = args.output_dir or Path(policy["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "selected": output_dir / "dataset_v3_selected_candidates.csv",
        "audit_csv": output_dir / "dataset_v3_candidate_selection_audit.csv",
        "city": output_dir / "dataset_v3_selection_city_summary.csv",
        "classes": output_dir / "dataset_v3_selected_class_distribution.csv",
        "audit_json": output_dir / "dataset_v3_selection_audit.json",
    }
    ensure_outputs_available(list(output_paths.values()), args.force)

    candidates = pd.read_csv(candidate_path)
    validate_candidates(candidates, policy)
    audit = select(candidates, policy)

    split_order = pd.Categorical(audit["split"], categories=SPLITS, ordered=True)
    audit = audit.assign(_split_order=split_order).sort_values(
        ["_split_order", "city_id", "tile_id"]
    ).drop(columns="_split_order")
    selected = audit[audit["selected"]].copy()
    selected.insert(0, "selection_rank_global", range(1, len(selected) + 1))
    selected.insert(
        1,
        "selection_rank_within_split",
        selected.groupby("split", sort=False).cumcount() + 1,
    )

    if policy["geographic_split_guard"]["require_each_city_to_contribute"]:
        missing_cities = sorted(set(audit["city_id"]) - set(selected["city_id"]))
        if missing_cities:
            raise ValueError(f"No selected tiles for cities: {missing_cities}")

    compact_columns = [
        "tile_id",
        "city_id",
        "city_name",
        "split",
        "coverage_eligible",
        *[f"rule_{name}" for name in policy["training_selection"]["thresholds_fraction_valid"]],
        "selected",
        "selection_reason",
        "rejection_reason",
    ]
    summary = city_summary(audit)
    distribution = class_distribution(selected)

    selected.to_csv(output_paths["selected"], index=False)
    audit[compact_columns].to_csv(output_paths["audit_csv"], index=False)
    summary.to_csv(output_paths["city"], index=False)
    distribution.to_csv(output_paths["classes"], index=False)

    split_counts = selected.groupby("split").size().reindex(SPLITS, fill_value=0)
    report = {
        "schema_version": "dataset-v3-selection-audit-0.1",
        "status": "PASS",
        "dataset_version": policy["dataset_version"],
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "policy_path": str(args.config),
        "policy_sha256": sha256(args.config),
        "candidate_tile_count": int(len(candidates)),
        "selected_tile_count": int(len(selected)),
        "selected_by_split": {key: int(value) for key, value in split_counts.items()},
        "selected_city_count_by_split": {
            split: int(selected.loc[selected["split"] == split, "city_id"].nunique())
            for split in SPLITS
        },
        "policy": policy,
        "leakage_guards": {
            "city_in_exactly_one_split": True,
            "training_class_rules_applied_to_validation_or_test": False,
            "validation_and_test_use_coverage_only": True,
        },
        "write_scope": "metadata/dataset_v3/selection only",
        "existing_dataset_rasters_modified": False,
    }
    with output_paths["audit_json"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    report["output_sha256"] = {
        key: sha256(path)
        for key, path in output_paths.items()
        if key != "audit_json"
    }
    with output_paths["audit_json"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps({
        "status": "PASS",
        "selected_tile_count": int(len(selected)),
        "selected_by_split": report["selected_by_split"],
        "output_directory": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
