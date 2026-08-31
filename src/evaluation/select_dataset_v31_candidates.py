"""Select Dataset V3.1 candidates using the frozen metadata-only policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation import select_dataset_v3_candidates as v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/dataset_v31_selection.yaml"),
    )
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = v3.read_policy(args.config)
    if policy.get("dataset_version") != "v3.1":
        raise ValueError("Expected Dataset V3.1 selection policy")
    candidate_path = args.candidates or Path(policy["input"]["candidate_manifest"])
    if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
        raise FileNotFoundError(f"V3.1 candidate manifest missing or empty: {candidate_path}")
    output_dir = args.output_dir or Path(policy["output"]["directory"])

    candidates = pd.read_csv(candidate_path)
    v3.validate_candidates(candidates, policy)
    expected = policy["expected_accounting"]
    if len(candidates) != int(expected["candidate_tiles"]):
        raise ValueError(
            f"Expected {expected['candidate_tiles']} candidates; found {len(candidates)}"
        )
    audit = v3.select(candidates, policy)
    selected = audit[audit["selected"]].copy()
    actual_counts = (
        selected.groupby("split").size().reindex(v3.SPLITS, fill_value=0).to_dict()
    )
    expected_counts = expected["selected_by_split"]
    if any(int(actual_counts[split]) != int(expected_counts[split]) for split in v3.SPLITS):
        raise RuntimeError(
            f"Frozen V3.1 selection accounting changed: expected {expected_counts}, "
            f"found {actual_counts}"
        )
    missing_cities = sorted(set(audit["city_id"]) - set(selected["city_id"]))
    if missing_cities:
        raise ValueError(f"No V3.1 selected tiles for cities: {missing_cities}")

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "candidate_tiles": int(len(candidates)),
            "selected_tiles": int(len(selected)),
            "selected_by_split": {k: int(v) for k, v in actual_counts.items()},
            "selected_city_count": int(selected["city_id"].nunique()),
            "validation_and_test_coverage_only": True,
            "metadata_written": False,
            "tiles_written": False,
        }, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "selected": output_dir / "dataset_v31_selected_candidates.csv",
        "audit_csv": output_dir / "dataset_v31_candidate_selection_audit.csv",
        "city": output_dir / "dataset_v31_selection_city_summary.csv",
        "classes": output_dir / "dataset_v31_selected_class_distribution.csv",
        "audit_json": output_dir / "dataset_v31_selection_audit.json",
    }
    v3.ensure_outputs_available(list(output_paths.values()), args.force)

    split_order = pd.Categorical(audit["split"], categories=v3.SPLITS, ordered=True)
    audit = audit.assign(_split_order=split_order).sort_values(
        ["_split_order", "city_id", "tile_id"]
    ).drop(columns="_split_order")
    selected = audit[audit["selected"]].copy()
    selected.insert(0, "selection_rank_global", range(1, len(selected) + 1))
    selected.insert(
        1, "selection_rank_within_split",
        selected.groupby("split", sort=False).cumcount() + 1,
    )

    compact_columns = [
        "tile_id", "city_id", "city_name", "split", "coverage_eligible",
        *[f"rule_{name}" for name in policy["training_selection"]["thresholds_fraction_valid"]],
        "selected", "selection_reason", "rejection_reason",
    ]
    summary = v3.city_summary(audit)
    distribution = v3.class_distribution(selected)
    selected.to_csv(output_paths["selected"], index=False)
    audit[compact_columns].to_csv(output_paths["audit_csv"], index=False)
    summary.to_csv(output_paths["city"], index=False)
    distribution.to_csv(output_paths["classes"], index=False)

    report = {
        "schema_version": "dataset-v3.1-selection-audit-0.1",
        "status": "PASS",
        "dataset_version": "v3.1",
        "parent_dataset_version": "v3",
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": v3.sha256(candidate_path),
        "policy_path": str(args.config),
        "policy_sha256": v3.sha256(args.config),
        "candidate_tile_count": int(len(candidates)),
        "selected_tile_count": int(len(selected)),
        "selected_by_split": {key: int(value) for key, value in actual_counts.items()},
        "selected_city_count_by_split": {
            split: int(selected.loc[selected["split"] == split, "city_id"].nunique())
            for split in v3.SPLITS
        },
        "policy": policy,
        "leakage_guards": {
            "city_in_exactly_one_split": True,
            "training_class_rules_applied_to_validation_or_test": False,
            "validation_and_test_use_coverage_only": True,
            "test_performance_metrics_used_for_selection": False,
        },
        "write_scope": str(output_dir),
        "existing_dataset_rasters_modified": False,
    }
    report["output_sha256"] = {
        key: v3.sha256(path)
        for key, path in output_paths.items()
        if key != "audit_json"
    }
    output_paths["audit_json"].write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "selected_tile_count": int(len(selected)),
        "selected_by_split": report["selected_by_split"],
        "output_directory": str(output_dir),
        "tiles_written": False,
    }, indent=2))


if __name__ == "__main__":
    main()

