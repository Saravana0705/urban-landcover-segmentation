"""Profile every Dataset V3.1 candidate tile without materializing tiles.

This additive stage reuses the audited Dataset V3 candidate-analysis logic
while routing all label inputs and metadata outputs to V3.1 paths. The frozen
five-class V3.1 label policy is not changed by this analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation import analyze_dataset_v3_candidates as v3


def v31_city_paths(city: dict[str, Any]) -> dict[str, Path]:
    city_id = str(city["city_id"])
    grid = city.get("grid", {})
    sar = Path(str(grid.get("reference_raster", "")).replace("\\", "/"))
    root = Path("data/interim/labels_v3_1/fused") / city_id
    return {
        "sar": sar,
        "semantic": root / f"{city_id}_v31_semantic.tif",
        "validity": root / f"{city_id}_v31_validity.tif",
        "provenance": root / f"{city_id}_v31_provenance.tif",
        "support": root / f"{city_id}_v31_support_count.tif",
        "conflict": root / f"{city_id}_v31_source_conflict.tif",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("config/dataset_v3_city_registry.json"),
    )
    parser.add_argument(
        "--tiling-config", type=Path, default=Path("config/tiling.yaml")
    )
    parser.add_argument(
        "--policy-config", type=Path,
        default=Path("config/dataset_v31_frozen_policy.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("metadata/dataset_v3_1/candidates"),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace only prior V3.1 candidate-analysis reports.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = v3.load_object(args.registry)
    tiling = v3.load_object(args.tiling_config)
    policy = v3.load_object(args.policy_config)
    cities = registry.get("cities")
    if not isinstance(cities, list) or not cities:
        raise ValueError("Registry must contain a non-empty cities list")
    city_ids = [str(city.get("city_id")) for city in cities]
    if len(city_ids) != len(set(city_ids)):
        raise ValueError("Registry contains duplicate city IDs")
    if policy.get("dataset_version") != "v3.1":
        raise ValueError("Expected the frozen Dataset V3.1 policy")
    if policy.get("new_labels_added") is not False:
        raise ValueError("Candidate analysis requires conservative V3.1 labels")

    tile_size = int(tiling["tile_size"])
    stride = int(tiling["stride"])
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if str(tiling.get("edge_policy", "")).lower() != "pad":
        raise ValueError("V3.1 candidate analysis requires edge_policy: pad")

    expected_splits = {"train": 14, "val": 3, "test": 3}
    actual_splits = Counter(str(city["split"]).lower() for city in cities)
    if dict(actual_splits) != expected_splits:
        raise ValueError(f"Expected frozen 14/3/3 city split; found {dict(actual_splits)}")

    for city in cities:
        for key, path in v31_city_paths(city).items():
            v3.require_file(path, f"{city['city_id']} {key}")

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "dataset_version": "v3.1",
            "city_count": len(cities),
            "split_city_counts": dict(actual_splits),
            "tile_size": tile_size,
            "stride": stride,
            "candidate_tiles_expected": sum(
                ((int(city["grid"]["height"]) + stride - 1) // stride)
                * ((int(city["grid"]["width"]) + stride - 1) // stride)
                for city in cities
            ),
            "selection_performed": False,
            "tiles_written": False,
        }, indent=2))
        return

    outputs = {
        "candidates": args.output_dir / "dataset_v31_candidate_tiles.csv",
        "cities": args.output_dir / "dataset_v31_candidate_city_summary.csv",
        "quantiles": args.output_dir / "dataset_v31_candidate_quantiles.csv",
        "sensitivity": args.output_dir / "dataset_v31_threshold_sensitivity.csv",
        "audit": args.output_dir / "dataset_v31_candidate_audit.json",
    }
    collisions = [path for path in outputs.values() if path.exists()]
    if collisions and not args.force:
        raise FileExistsError(
            "V3.1 candidate reports already exist; use --force to replace only these reports: "
            + ", ".join(map(str, collisions))
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The audited V3 profiler resolves its paths through this module global.
    # Replace only that resolver for this process; no project file is modified.
    original_city_paths = v3.city_paths
    v3.city_paths = v31_city_paths
    try:
        candidates: list[dict[str, Any]] = []
        city_summaries: list[dict[str, Any]] = []
        source_hashes: dict[str, dict[str, str]] = {}
        for position, city in enumerate(
            sorted(cities, key=lambda item: item["city_id"]), start=1
        ):
            rows, summary, hashes = v3.profile_city(city, tile_size, stride)
            candidates.extend(rows)
            city_summaries.append(summary)
            source_hashes[str(city["city_id"])] = hashes
            print(
                f"Profiled {position}/{len(cities)}: "
                f"{city['city_id']} ({len(rows)} candidates)"
            )
    finally:
        v3.city_paths = original_city_paths

    if len(candidates) != sum(int(row["candidate_tile_count"]) for row in city_summaries):
        raise RuntimeError("V3.1 candidate row accounting failed")
    if len(candidates) != 2880:
        raise RuntimeError(f"Expected 2,880 V3.1 candidates; found {len(candidates)}")

    v3.write_csv(outputs["candidates"], candidates)
    v3.write_csv(outputs["cities"], city_summaries)
    v3.write_csv(outputs["quantiles"], v3.quantile_rows(candidates))
    v3.write_csv(outputs["sensitivity"], v3.sensitivity_rows(candidates))
    audit = {
        "schema_version": "dataset-v3.1-candidate-analysis-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "mode": "read_only_pre_materialization_analysis",
        "dataset_version": "v3.1",
        "parent_dataset_version": "v3",
        "tile_size": tile_size,
        "stride": stride,
        "edge_policy": "pad",
        "city_count": len(cities),
        "split_city_counts": dict(actual_splits),
        "candidate_tile_count": len(candidates),
        "selection_performed": False,
        "label_policy_already_frozen_before_test_processing": True,
        "test_performance_metrics_inspected": False,
        "policy_config": {
            "path": str(args.policy_config),
            "sha256": v3.sha256_file(args.policy_config),
        },
        "source_hashes": source_hashes,
        "outputs": {
            key: str(path) for key, path in outputs.items() if key != "audit"
        },
        "safety": {
            "tiles_written": False,
            "dataset_v3_modified": False,
            "dataset_v3_1_labels_modified": False,
            "output_scope": str(args.output_dir),
        },
        "next_gate": "freeze_v3_1_selection_policy",
    }
    outputs["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "dataset_version": "v3.1",
        "cities": len(cities),
        "candidate_tiles": len(candidates),
        "selection_performed": False,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

