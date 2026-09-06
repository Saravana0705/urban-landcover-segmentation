"""Audit the E7 training-only uncertainty transform without writing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from src.data.segmentation_dataset import Sentinel1UrbanDataset


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]["dataset_config_path"]
    manifest = config["dataset"]["manifest_path"]
    uncertainty = config["label_uncertainty"]
    plain = Sentinel1UrbanDataset("train", manifest, dataset_config)
    modified = Sentinel1UrbanDataset("train", manifest, dataset_config, label_uncertainty=uncertainty)
    limit = len(plain) if args.max_tiles <= 0 else min(args.max_tiles, len(plain))
    ignored_added = 0
    valid_original = 0
    original_selected_pixels = 0
    retained_selected_pixels = 0
    original_by_class = {1: 0, 3: 0}
    retained_by_class = {1: 0, 3: 0}
    source_paths: set[Path] = set()
    before: dict[str, str] = {}
    for index in range(limit):
        base_sample = plain[index]
        changed_sample = modified[index]
        valid_original += int((base_sample["target"] != 255).sum())
        ignored_added += int(changed_sample["uncertainty_mask"].sum())
        selected = (base_sample["target"] == 1) | (base_sample["target"] == 3)
        original_selected_pixels += int(selected.sum())
        retained_selected_pixels += int((selected & (changed_sample["target"] != 255)).sum())
        for class_id in original_by_class:
            class_pixels = base_sample["target"] == class_id
            original_by_class[class_id] += int(class_pixels.sum())
            retained_by_class[class_id] += int(
                (class_pixels & (changed_sample["target"] != 255)).sum()
            )
        path = Path(base_sample["paths"]["semantic"])
        source_paths.add(path)
        before[str(path)] = digest(path)
    unchanged = all(before[str(path)] == digest(path) for path in source_paths)
    validation_clean = Sentinel1UrbanDataset("val", manifest, dataset_config).label_uncertainty.enabled is False
    ignored_fraction = ignored_added / max(valid_original, 1)
    selected_retained_fraction = retained_selected_pixels / max(original_selected_pixels, 1)
    maximum_ignored_fraction = float(uncertainty.get("maximum_ignored_fraction", 0.15))
    minimum_selected_retained_fraction = float(
        uncertainty.get("minimum_selected_class_retained_fraction", 0.70)
    )
    fraction_gate_passed = ignored_fraction <= maximum_ignored_fraction
    retained_fraction_by_class = {
        "roads": retained_by_class[1] / max(original_by_class[1], 1),
        "bare_land": retained_by_class[3] / max(original_by_class[3], 1),
    }
    retention_gate_passed = (
        selected_retained_fraction >= minimum_selected_retained_fraction
        and all(
            value >= minimum_selected_retained_fraction
            for value in retained_fraction_by_class.values()
        )
    )
    passed = (
        unchanged
        and validation_clean
        and ignored_added > 0
        and fraction_gate_passed
        and retention_gate_passed
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "tiles_audited": limit,
        "source_labels_unchanged": unchanged,
        "validation_labels_unmodified": validation_clean,
        "ignored_boundary_pixels_added": ignored_added,
        "ignored_fraction_of_original_valid": ignored_fraction,
        "maximum_ignored_fraction": maximum_ignored_fraction,
        "fraction_gate_passed": fraction_gate_passed,
        "selected_class_pixels_retained": retained_selected_pixels,
        "selected_class_retained_fraction": selected_retained_fraction,
        "selected_class_retained_fraction_by_class": retained_fraction_by_class,
        "minimum_selected_class_retained_fraction": minimum_selected_retained_fraction,
        "retention_gate_passed": retention_gate_passed,
        "training_only_runtime_transform": True,
    }
    output = Path("metadata/model_development/d2_e7_label_uncertainty_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {output}")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
