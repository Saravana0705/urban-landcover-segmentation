"""Corrected read-only distribution and OSM-agreement audit for V3 pilot evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


DW_NAMES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
]
OSM_NAMES = {0: "unknown", 1: "buildings", 2: "roads", 3: "vegetation", 4: "bare_land", 5: "water"}
DW_TO_TARGET = {0: 5, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1, 7: 4, 8: 0}
PROB_BINS = np.linspace(0.0, 1.0, 1001)


def quantiles_from_hist(hist: np.ndarray, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float | None]:
    total = int(hist.sum())
    if total == 0:
        return {str(q): None for q in quantiles}
    cumulative = np.cumsum(hist)
    result = {}
    for q in quantiles:
        index = min(int(np.searchsorted(cumulative, q * total, side="left")), len(hist) - 1)
        result[str(q)] = float((PROB_BINS[index] + PROB_BINS[index + 1]) / 2)
    return result


def ensure_same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader, label: str) -> None:
    if (reference.width, reference.height, reference.crs, reference.transform) != (
        other.width, other.height, other.crs, other.transform
    ):
        raise ValueError(f"{label} is not on the Dynamic World grid")


def analyze(dw_path: Path, semantic_path: Path | None, validity_path: Path | None) -> tuple[dict, list[dict]]:
    mode_counts: Counter[int] = Counter()
    observation_counts: Counter[int] = Counter()
    top1_hist = np.zeros(1000, dtype=np.int64)
    mode_probability_hist = np.zeros(1000, dtype=np.int64)
    margin_hist = np.zeros(1000, dtype=np.int64)
    agreement_hist = np.zeros(1000, dtype=np.int64)
    per_mode_top1 = {i: np.zeros(1000, dtype=np.int64) for i in range(9)}
    per_mode_agreement = {i: np.zeros(1000, dtype=np.int64) for i in range(9)}
    cross_tab = np.zeros((6, 6), dtype=np.int64)
    mode_mean_consensus_count = 0
    profiles = [
        (0.50, 0.60, 3), (0.60, 0.70, 3), (0.70, 0.80, 3),
        (0.60, 0.70, 5), (0.70, 0.80, 5),
    ]
    profile_counts = {p: Counter() for p in profiles}

    with rasterio.open(dw_path) as dw:
        if dw.count != 21:
            raise ValueError(f"Expected 21 Dynamic World bands, found {dw.count}")
        semantic = rasterio.open(semantic_path) if semantic_path else None
        validity = rasterio.open(validity_path) if validity_path else None
        try:
            if semantic:
                ensure_same_grid(dw, semantic, "semantic raster")
            if validity:
                ensure_same_grid(dw, validity, "validity raster")

            for _, window in dw.block_windows(1):
                mode = dw.read(1, window=window).astype(np.int16)
                probs = dw.read(list(range(2, 11)), window=window).astype(np.float32)
                observations = dw.read(20, window=window).astype(np.int16)
                agreement = np.clip(dw.read(21, window=window).astype(np.float32), 0, 1)
                sorted_probs = np.partition(probs, -2, axis=0)
                top1 = np.clip(sorted_probs[-1], 0, 1)
                margin = np.clip(sorted_probs[-1] - sorted_probs[-2], 0, 1)
                mean_argmax = np.argmax(probs, axis=0).astype(np.int16)
                mode_in_range = (mode >= 0) & (mode < len(DW_NAMES))
                mode_probability = np.zeros(mode.shape, dtype=np.float32)
                mode_probability[mode_in_range] = np.take_along_axis(
                    probs, mode.clip(0, len(DW_NAMES) - 1)[None, ...], axis=0
                )[0][mode_in_range]
                consensus = mode_in_range & (mean_argmax == mode)
                mode_mean_consensus_count += int(consensus.sum())

                values, counts = np.unique(mode, return_counts=True)
                mode_counts.update({int(v): int(c) for v, c in zip(values, counts)})
                values, counts = np.unique(observations, return_counts=True)
                observation_counts.update({int(v): int(c) for v, c in zip(values, counts)})
                top1_hist += np.histogram(top1, bins=PROB_BINS)[0]
                mode_probability_hist += np.histogram(mode_probability[mode_in_range], bins=PROB_BINS)[0]
                margin_hist += np.histogram(margin, bins=PROB_BINS)[0]
                agreement_hist += np.histogram(agreement, bins=PROB_BINS)[0]

                for dw_class in range(9):
                    selected = mode == dw_class
                    if selected.any():
                        per_mode_top1[dw_class] += np.histogram(top1[selected], bins=PROB_BINS)[0]
                        per_mode_agreement[dw_class] += np.histogram(agreement[selected], bins=PROB_BINS)[0]

                target = np.zeros_like(mode)
                for source_id, target_id in DW_TO_TARGET.items():
                    target[mode == source_id] = target_id
                for profile in profiles:
                    probability_min, agreement_min, observation_min = profile
                    accepted = (
                        consensus
                        & (mode_probability >= probability_min)
                        & (agreement >= agreement_min)
                        & (observations >= observation_min)
                    )
                    values, counts = np.unique(target[accepted], return_counts=True)
                    profile_counts[profile].update({int(v): int(c) for v, c in zip(values, counts)})

                if semantic:
                    osm = semantic.read(1, window=window).astype(np.int16)
                    valid = validity.read(1, window=window).astype(bool) if validity else (osm > 0)
                    use = valid & (osm >= 0) & (osm <= 5) & (target >= 0) & (target <= 5)
                    np.add.at(cross_tab, (osm[use], target[use]), 1)
        finally:
            if semantic:
                semantic.close()
            if validity:
                validity.close()

    total = int(sum(mode_counts.values()))
    per_class = []
    for class_id in range(9):
        count = int(mode_counts[class_id])
        per_class.append({
            "dw_class_id": class_id,
            "dw_class_name": DW_NAMES[class_id],
            "pixel_count": count,
            "pixel_fraction": count / total if total else 0.0,
            "top1_probability_quantiles": quantiles_from_hist(per_mode_top1[class_id]),
            "temporal_agreement_quantiles": quantiles_from_hist(per_mode_agreement[class_id]),
        })

    profile_rows = []
    for (p_min, a_min, o_min), counts in profile_counts.items():
        accepted_total = int(sum(counts.values()))
        for target_id in range(6):
            profile_rows.append({
                "probability_min": p_min, "agreement_min": a_min, "observation_min": o_min,
                "target_class_id": target_id, "target_class_name": OSM_NAMES[target_id],
                "accepted_pixel_count": int(counts[target_id]),
                "accepted_fraction_all_pixels": counts[target_id] / total if total else 0.0,
                "profile_total_accepted_fraction": accepted_total / total if total else 0.0,
            })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dynamic_world_raster": str(dw_path),
        "semantic_raster": str(semantic_path) if semantic_path else None,
        "validity_raster": str(validity_path) if validity_path else None,
        "pixel_count": total,
        "observation_count_distribution": dict(sorted(observation_counts.items())),
        "top1_probability_quantiles": quantiles_from_hist(top1_hist),
        "assigned_mode_probability_quantiles": quantiles_from_hist(mode_probability_hist),
        "top1_margin_quantiles": quantiles_from_hist(margin_hist),
        "mode_mean_consensus_pixel_count": mode_mean_consensus_count,
        "mode_mean_consensus_fraction": mode_mean_consensus_count / total if total else 0.0,
        "temporal_agreement_quantiles": quantiles_from_hist(agreement_hist),
        "dynamic_world_mode_classes": per_class,
        "osm_vs_dynamic_world_target_cross_tab": cross_tab.tolist() if semantic_path else None,
        "notes": [
            "Candidate profiles require modal-label/mean-argmax consensus and use the assigned modal class probability.",
            "Candidate profiles are sensitivity diagnostics, not frozen V3 thresholds.",
            "Dynamic World has no road class; roads must remain sourced from OSM/authoritative transport data.",
            "Snow/ice maps to Unknown and is never accepted as a target class.",
        ],
    }
    return report, profile_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-id", required=True)
    parser.add_argument("--dynamic-world", required=True, type=Path)
    parser.add_argument("--semantic", type=Path)
    parser.add_argument("--validity", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("metadata/dataset_v3/pilot/evidence_audit"))
    args = parser.parse_args()
    if bool(args.semantic) != bool(args.validity):
        parser.error("Provide both --semantic and --validity, or neither")

    report, rows = analyze(args.dynamic_world, args.semantic, args.validity)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"{args.city_id}_dynamic_world_evidence_audit.json"
    csv_path = args.report_dir / f"{args.city_id}_candidate_profile_sensitivity.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"city_id": args.city_id, "status": "PASS", "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
