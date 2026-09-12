"""Compile the final D2 benchmark, robustness and locked-test report evidence.

The input root may contain extracted directories, ZIP archives, or both. Files
are resolved by evidence family and content hash rather than fixed directory
depth. Conflicting copies are rejected. The script never trains, evaluates a
model, loads a checkpoint, or opens a dataset raster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_CODES = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
MODEL_ORDER = (
    "U-Net++",
    "U-Net",
    "Attention U-Net",
    "Swin Transformer",
    "DeepLabV3+",
    "SegFormer",
    "Mask2Former",
)
CLASS_ORDER = ("buildings", "roads", "vegetation", "bare_land", "water")
CONDITION_ORDER = ("clean", "light_speckle_l25", "moderate_speckle_l9")
CONDITION_LABELS = {
    "clean": "Clean",
    "light_speckle_l25": "Light speckle (L=25)",
    "moderate_speckle_l9": "Moderate speckle (L=9)",
}
TEST_CITIES = {"DE18": ("Freiburg", 120), "DE19": ("Kiel", 117), "DE20": ("Rostock", 102)}
CONTROLLED_WINNER_CODE = "E2"
CONTROLLED_WINNER_MIOU = 0.6860177249062984
FINAL_VALIDATION_MIOU = 0.6982309104696373
FINAL_CHECKPOINT_SHA256 = "9ea0b5220b96acc26cce52f289e2f5b35a085dab0b0927d0794490907f70c799"
EXPECTED_TEST_TILES = 339

FAMILY_TOKENS = {
    "controlled": ("d2_controlled_architectures",),
    "robustness": ("d2_final_robustness",),
    "test": ("d2_final_locked_test",),
}


@dataclass(frozen=True)
class Evidence:
    family: str
    filename: str
    source: str
    sha256: str
    content: bytes
    duplicate_locations: tuple[str, ...]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_family(label: str, family: str) -> bool:
    lowered = label.replace("\\", "/").lower()
    return any(token in lowered for token in FAMILY_TOKENS[family])


def locate_evidence(input_root: Path, filename: str, family: str) -> Evidence:
    """Find one logical artifact across extracted trees and ZIP members."""
    candidates: list[tuple[str, bytes]] = []
    for path in input_root.rglob(filename):
        if path.is_file() and _matches_family(path.as_posix(), family):
            candidates.append((str(path.resolve()), path.read_bytes()))
    for archive_path in input_root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    label = f"{archive_path.resolve()}::{member.filename}"
                    if (
                        not member.is_dir()
                        and Path(member.filename).name == filename
                        and _matches_family(label, family)
                    ):
                        candidates.append((label, archive.read(member)))
        except zipfile.BadZipFile as error:
            raise RuntimeError(f"Invalid ZIP archive: {archive_path}") from error
    if not candidates:
        raise FileNotFoundError(
            f"Missing {family} evidence '{filename}' below {input_root.resolve()}."
        )
    by_hash: dict[str, list[tuple[str, bytes]]] = {}
    for label, content in candidates:
        by_hash.setdefault(sha256_bytes(content), []).append((label, content))
    if len(by_hash) != 1:
        conflict = {digest: [label for label, _ in rows] for digest, rows in by_hash.items()}
        raise RuntimeError(
            f"Conflicting copies of {family}/{filename}: "
            + json.dumps(conflict, indent=2)
        )
    digest, rows = next(iter(by_hash.items()))
    rows.sort(key=lambda item: ("::" in item[0], len(item[0]), item[0]))
    return Evidence(
        family=family,
        filename=filename,
        source=rows[0][0],
        sha256=digest,
        content=rows[0][1],
        duplicate_locations=tuple(label for label, _ in rows),
    )


def read_csv(evidence: Evidence) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(evidence.content), encoding="utf-8-sig")


def read_json(evidence: Evidence) -> dict[str, Any]:
    payload = json.loads(evidence.content.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {evidence.source}")
    return payload


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def validate_sources(
    benchmark: pd.DataFrame,
    class_matrix: pd.DataFrame,
    noise: pd.DataFrame,
    city_consistency: pd.DataFrame,
    city_conditions: pd.DataFrame,
    inference: pd.DataFrame,
    test_summary: Mapping[str, Any],
    test_ledger: Mapping[str, Any],
    test_overall: pd.DataFrame,
    test_classes: pd.DataFrame,
    test_cities: pd.DataFrame,
) -> dict[str, bool]:
    require_columns(
        benchmark,
        (
            "experiment_code", "model", "validation_rank", "mean_iou",
            "mean_dice", "pixel_accuracy", "training_time_minutes",
            "trainable_parameters", "training_peak_gpu_allocated_mb",
            "checkpoint_available", "evaluation_split",
        ),
        "controlled benchmark",
    )
    require_columns(class_matrix, ("model", *CLASS_ORDER), "controlled class IoU")
    require_columns(
        noise,
        ("experiment_code", "model", "condition", "mean_iou", "delta_miou_from_clean", "evaluated_tiles"),
        "noise sensitivity",
    )
    require_columns(
        city_consistency,
        ("experiment_code", "model", "city_mean_miou", "city_std_miou", "city_miou_range"),
        "city consistency",
    )
    require_columns(
        city_conditions,
        ("experiment_code", "model", "condition", "city_id", "city_name", "tile_count", "mean_iou"),
        "city-condition metrics",
    )
    require_columns(
        inference,
        ("experiment_code", "model", "inference_images_per_second", "inference_milliseconds_per_image", "inference_peak_gpu_memory_allocated_mb"),
        "inference profile",
    )
    require_columns(test_overall, ("mean_iou", "mean_dice", "pixel_accuracy"), "test overall metrics")
    require_columns(test_classes, ("class_name", "iou", "dice", "precision", "recall"), "test class metrics")
    require_columns(test_cities, ("city_id", "city_name", "tile_count", "mean_iou"), "test city metrics")

    benchmark_codes = tuple(sorted(benchmark["experiment_code"].astype(str)))
    benchmark_ranks = tuple(sorted(benchmark["validation_rank"].astype(int)))
    winner = benchmark.sort_values("validation_rank").iloc[0]
    clean_e2 = noise.loc[
        (noise["experiment_code"].astype(str) == "E2") & (noise["condition"] == "clean")
    ]
    result = test_summary.get("result", {})
    aggregate = result.get("aggregate", {})
    city_contract = {
        str(row.city_id): (str(row.city_name), int(row.tile_count))
        for row in test_cities.itertuples(index=False)
    }
    checks = {
        "controlled_models_exact": benchmark_codes == MODEL_CODES and len(benchmark) == 7,
        "controlled_ranks_exact": benchmark_ranks == tuple(range(1, 8)),
        "controlled_split_validation": set(benchmark["evaluation_split"].astype(str)) == {"validation"},
        "controlled_winner_exact": (
            str(winner["experiment_code"]) == CONTROLLED_WINNER_CODE
            and str(winner["model"]) == "U-Net++"
            and close(winner["mean_iou"], CONTROLLED_WINNER_MIOU)
        ),
        "controlled_class_models_exact": set(class_matrix["model"].astype(str)) == set(MODEL_ORDER),
        "robustness_models_exact": set(noise["experiment_code"].astype(str)) == set(MODEL_CODES),
        "robustness_conditions_exact": set(noise["condition"].astype(str)) == set(CONDITION_ORDER),
        "robustness_rows_exact": len(noise) == 21,
        "robustness_validation_tiles_exact": set(noise["evaluated_tiles"].astype(int)) == {351},
        "robustness_e2_clean_reconciles": len(clean_e2) == 1 and close(clean_e2.iloc[0]["mean_iou"], CONTROLLED_WINNER_MIOU),
        "city_consistency_models_exact": set(city_consistency["experiment_code"].astype(str)) == set(MODEL_CODES),
        "city_condition_rows_exact": len(city_conditions) == 63,
        "inference_models_exact": set(inference["experiment_code"].astype(str)) == set(MODEL_CODES),
        "test_summary_pass": test_summary.get("status") == "PASS" and test_summary.get("all_checks_passed") is True,
        "test_method_locked": (
            test_summary.get("split") == "test"
            and test_summary.get("selection_locked_before_test") is True
            and close(test_summary.get("validation_reference_mean_iou"), FINAL_VALIDATION_MIOU)
            and test_summary.get("checkpoint_sha256") == FINAL_CHECKPOINT_SHA256
        ),
        "test_ledger_complete_once": (
            test_ledger.get("status") == "COMPLETED"
            and int(test_ledger.get("test_evaluation_count_this_output", -1)) == 1
            and test_ledger.get("training_performed") is False
            and test_ledger.get("checkpoint_updated") is False
        ),
        "test_tiles_exact": int(result.get("evaluated_tile_count", -1)) == EXPECTED_TEST_TILES,
        "test_reconciliation_passed": result.get("reconciliation_passed") is True,
        "test_overall_reconciles": (
            len(test_overall) == 1
            and close(test_overall.iloc[0]["mean_iou"], aggregate.get("mean_iou"))
            and close(test_overall.iloc[0]["mean_dice"], aggregate.get("mean_dice"))
            and close(test_overall.iloc[0]["pixel_accuracy"], aggregate.get("pixel_accuracy"))
        ),
        "test_classes_exact": set(test_classes["class_name"].astype(str)) == set(CLASS_ORDER),
        "test_cities_exact": city_contract == TEST_CITIES,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Final report source validation failed: " + ", ".join(failed))
    return checks


def build_architecture_table(benchmark: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "validation_rank", "experiment_code", "model", "architecture_family",
        "mean_iou", "mean_dice", "mean_precision", "mean_recall",
        "pixel_accuracy", "best_epoch_1_based", "completed_epochs",
        "early_stopped", "training_time_minutes", "trainable_parameters",
        "training_peak_gpu_allocated_mb", "checkpoint_available",
    ]
    result = benchmark.loc[:, fields].sort_values("validation_rank").copy()
    result["delta_miou_from_winner"] = result["mean_iou"] - result["mean_iou"].max()
    return result


def build_robustness_table(
    noise: pd.DataFrame, city: pd.DataFrame, inference: pd.DataFrame
) -> pd.DataFrame:
    values = noise.pivot(index=["experiment_code", "model"], columns="condition", values="mean_iou")
    values = values.rename(columns={
        "clean": "clean_mean_iou",
        "light_speckle_l25": "light_speckle_mean_iou",
        "moderate_speckle_l9": "moderate_speckle_mean_iou",
    }).reset_index()
    values["light_drop"] = values["light_speckle_mean_iou"] - values["clean_mean_iou"]
    values["moderate_drop"] = values["moderate_speckle_mean_iou"] - values["clean_mean_iou"]
    result = values.merge(city, on=["experiment_code", "model"], validate="one_to_one")
    result = result.merge(inference, on=["experiment_code", "model"], validate="one_to_one")
    result["clean_rank"] = result["clean_mean_iou"].rank(method="min", ascending=False).astype(int)
    fields = [
        "clean_rank", "experiment_code", "model", "clean_mean_iou",
        "light_speckle_mean_iou", "moderate_speckle_mean_iou", "light_drop",
        "moderate_drop", "city_mean_miou", "city_std_miou", "city_min_miou",
        "city_max_miou", "city_miou_range", "inference_images_per_second",
        "inference_milliseconds_per_image", "inference_peak_gpu_memory_allocated_mb",
        "inference_peak_gpu_memory_reserved_mb",
    ]
    return result.loc[:, fields].sort_values("clean_rank")


def build_final_method_table(test_summary: Mapping[str, Any]) -> pd.DataFrame:
    aggregate = test_summary["result"]["aggregate"]
    validation = float(test_summary["validation_reference_mean_iou"])
    test = float(aggregate["mean_iou"])
    return pd.DataFrame([
        {
            "method": test_summary["method"],
            "architecture": "U-Net++",
            "selected_checkpoint": "E9B",
            "validation_mean_iou": validation,
            "test_mean_iou": test,
            "validation_to_test_delta": test - validation,
            "validation_to_test_relative_change_percent": 100.0 * (test - validation) / validation,
            "test_mean_dice": aggregate["mean_dice"],
            "test_pixel_accuracy": aggregate["pixel_accuracy"],
            "test_tiles": test_summary["result"]["evaluated_tile_count"],
            "test_runtime_seconds": test_summary["result"]["runtime"]["duration_seconds"],
            "test_images_per_second": test_summary["result"]["runtime"]["images_per_second"],
            "test_peak_gpu_memory_allocated_mb": test_summary["result"]["runtime"]["peak_gpu_memory_allocated_mb"],
            "test_evaluation_count": test_summary["test_evaluation_count_this_output"],
        }
    ])


def configure_plots() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig: Any, directory: Path, name: str) -> list[Path]:
    paths = [directory / f"{name}.png", directory / f"{name}.svg"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_architecture(architecture: pd.DataFrame, directory: Path) -> list[Path]:
    data = architecture.sort_values("mean_iou")
    colors = ["#255F85" if code == "E2" else "#A8BBC8" for code in data["experiment_code"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(data["model"], data["mean_iou"], color=colors)
    ax.set_title("Controlled architecture validation performance")
    ax.set_xlabel("Mean IoU")
    ax.set_xlim(0, 0.75)
    ax.grid(axis="x", color="#D9E1E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in data["mean_iou"]], padding=3, fontsize=8)
    return save_figure(fig, directory, "architecture_validation_miou")


def plot_noise(noise: pd.DataFrame, directory: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 0.72, 7))
    x = np.arange(len(CONDITION_ORDER))
    for color, code in zip(colors, MODEL_CODES):
        model_data = noise.loc[noise["experiment_code"] == code].set_index("condition").loc[list(CONDITION_ORDER)]
        width = 2.5 if code == "E2" else 1.2
        alpha = 1.0 if code == "E2" else 0.72
        ax.plot(x, model_data["mean_iou"], marker="o", linewidth=width, color=color, alpha=alpha, label=model_data.iloc[0]["model"])
    ax.set_title("Validation performance under simulated speckle")
    ax.set_ylabel("Mean IoU")
    ax.set_xticks(x, [CONDITION_LABELS[value] for value in CONDITION_ORDER])
    ax.set_ylim(0.24, 0.72)
    ax.grid(axis="y", color="#D9E1E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="lower left")
    return save_figure(fig, directory, "noise_sensitivity_miou")


def plot_test_classes(test_classes: pd.DataFrame, directory: Path) -> list[Path]:
    data = test_classes.set_index("class_name").loc[list(CLASS_ORDER)].reset_index()
    labels = [value.replace("_", " ").title() for value in data["class_name"]]
    colors = ["#255F85", "#4F86A6", "#2F7D5B", "#C58B36", "#3D7EA6"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, data["iou"], color=colors)
    ax.set_title("Final locked test performance by class")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#D9E1E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in data["iou"]], padding=3, fontsize=8)
    return save_figure(fig, directory, "final_test_per_class_iou")


def plot_test_cities(test_cities: pd.DataFrame, directory: Path) -> list[Path]:
    data = test_cities.copy()
    data["label"] = data["city_name"] + " (" + data["city_id"] + ")"
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(data["label"], data["mean_iou"], color=["#A8BBC8", "#255F85", "#5D8C72"])
    ax.axhline(0.6, color="#9C3D3D", linestyle="--", linewidth=1.2, label="Project reference (0.60)")
    ax.set_title("Final locked test performance by city")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", color="#D9E1E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in data["mean_iou"]], padding=3, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save_figure(fig, directory, "final_test_per_city_miou")


def plot_validation_test(final_method: pd.DataFrame, directory: Path) -> list[Path]:
    row = final_method.iloc[0]
    values = [row["validation_mean_iou"], row["test_mean_iou"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bars = ax.bar(["Locked validation", "Final test"], values, color=["#255F85", "#5D8C72"])
    ax.axhline(0.6, color="#9C3D3D", linestyle="--", linewidth=1.2, label="Project reference (0.60)")
    ax.set_title("Locked final method: validation and test")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 0.76)
    ax.grid(axis="y", color="#D9E1E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return save_figure(fig, directory, "final_method_validation_vs_test")


def write_markdown_table(frame: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column).replace("_", " ") for column in frame.columns]
    rows = [[format_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_findings(
    path: Path,
    architecture: pd.DataFrame,
    robustness: pd.DataFrame,
    final_method: pd.DataFrame,
    test_classes: pd.DataFrame,
    test_cities: pd.DataFrame,
) -> None:
    winner = architecture.iloc[0]
    robust_winner = robustness.iloc[0]
    final = final_method.iloc[0]
    weakest_class = test_classes.sort_values("iou").iloc[0]
    strongest_class = test_classes.sort_values("iou", ascending=False).iloc[0]
    weakest_city = test_cities.sort_values("mean_iou").iloc[0]
    strongest_city = test_cities.sort_values("mean_iou", ascending=False).iloc[0]
    text = f"""# Final D2 Results Summary

## Controlled architecture benchmark

The controlled validation benchmark compared seven architectures under the
same Dataset V3-MT-D2 protocol. {winner['model']} ranked first with mean IoU
{winner['mean_iou']:.4f}. Its margin over the next-ranked model was
{winner['mean_iou'] - architecture.iloc[1]['mean_iou']:.4f}.

## Robustness

{robust_winner['model']} retained the highest absolute mean IoU under clean,
light-speckle and moderate-speckle conditions. Its moderate-speckle mean IoU
was {robust_winner['moderate_speckle_mean_iou']:.4f}, a change of
{robust_winner['moderate_drop']:.4f} from clean validation. Its clean
cross-city mean was {robust_winner['city_mean_miou']:.4f}, with a city range of
{robust_winner['city_miou_range']:.4f}.

## Validation-locked final test

The final method was fixed before test access: E9B U-Net++ with four-way TTA
and fixed class-probability multipliers. Locked validation mean IoU was
{final['validation_mean_iou']:.4f}; final test mean IoU was
{final['test_mean_iou']:.4f}, a difference of
{final['validation_to_test_delta']:.4f}. Test mean Dice was
{final['test_mean_dice']:.4f}, and pixel accuracy was
{final['test_pixel_accuracy']:.4f} across {int(final['test_tiles'])} tiles.

Test class IoU ranged from {weakest_class['iou']:.4f} for
{str(weakest_class['class_name']).replace('_', ' ')} to
{strongest_class['iou']:.4f} for
{str(strongest_class['class_name']).replace('_', ' ')}. City mean IoU ranged
from {weakest_city['mean_iou']:.4f} in {weakest_city['city_name']} to
{strongest_city['mean_iou']:.4f} in {strongest_city['city_name']}.

## Interpretation boundary

The test result is the final unbiased generalization estimate. It is reported
separately from validation and robustness evidence and was not used for model,
checkpoint, threshold, TTA, multiplier, architecture or dataset reselection.
"""
    path.write_text(text, encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("reports/final_benchmark"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/final_benchmark/d2_final_report_analysis"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.input_root.is_dir():
        parser.error(f"Input root does not exist: {args.input_root}")

    specifications = {
        "benchmark": ("controlled_model_benchmark.csv", "controlled"),
        "class_matrix": ("per_class_iou_matrix.csv", "controlled"),
        "noise": ("noise_sensitivity_summary.csv", "robustness"),
        "city_consistency": ("cross_city_consistency_summary.csv", "robustness"),
        "city_conditions": ("cross_city_condition_metrics.csv", "robustness"),
        "inference": ("inference_gpu_profile.csv", "robustness"),
        "test_summary": ("final_evaluation_summary.json", "test"),
        "test_ledger": ("test_access_ledger.json", "test"),
        "test_overall": ("overall_metrics.csv", "test"),
        "test_classes": ("per_class_metrics.csv", "test"),
        "test_cities": ("per_city_metrics.csv", "test"),
    }
    evidence = {
        name: locate_evidence(args.input_root, filename, family)
        for name, (filename, family) in specifications.items()
    }
    benchmark = read_csv(evidence["benchmark"])
    class_matrix = read_csv(evidence["class_matrix"])
    noise = read_csv(evidence["noise"])
    city_consistency = read_csv(evidence["city_consistency"])
    city_conditions = read_csv(evidence["city_conditions"])
    inference = read_csv(evidence["inference"])
    test_summary = read_json(evidence["test_summary"])
    test_ledger = read_json(evidence["test_ledger"])
    test_overall = read_csv(evidence["test_overall"])
    test_classes = read_csv(evidence["test_classes"])
    test_cities = read_csv(evidence["test_cities"])
    checks = validate_sources(
        benchmark, class_matrix, noise, city_consistency, city_conditions,
        inference, test_summary, test_ledger, test_overall, test_classes, test_cities,
    )
    validation_report = {
        "status": "PASS",
        "input_root": str(args.input_root.resolve()),
        "training_started": False,
        "checkpoint_loaded": False,
        "dataset_rasters_opened": False,
        "test_evaluation_performed": False,
        "source_count": len(evidence),
        "sources": [
            {
                "logical_name": name,
                "family": item.family,
                "filename": item.filename,
                "selected_source": item.source,
                "sha256": item.sha256,
                "identical_location_count": len(item.duplicate_locations),
            }
            for name, item in evidence.items()
        ],
        "checks": checks,
        "all_checks_passed": True,
    }
    if args.validate_only:
        print(json.dumps(validation_report, indent=2))
        return
    output = args.output_directory
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace only this analysis output."
            )
        shutil.rmtree(output)
    tables = output / "tables"
    figures = output / "figures"
    reports = output / "reports"
    tables.mkdir(parents=True)
    figures.mkdir()
    reports.mkdir()

    architecture = build_architecture_table(benchmark)
    robustness = build_robustness_table(noise, city_consistency, inference)
    final_method = build_final_method_table(test_summary)
    class_table = test_classes.set_index("class_name").loc[list(CLASS_ORDER)].reset_index()
    city_table = test_cities.sort_values("city_id").reset_index(drop=True)
    table_outputs = {
        "controlled_architecture_benchmark.csv": architecture,
        "controlled_per_class_iou.csv": class_matrix,
        "robustness_summary.csv": robustness,
        "robustness_noise_long.csv": noise,
        "robustness_city_condition_long.csv": city_conditions,
        "final_method_validation_test.csv": final_method,
        "final_test_per_class.csv": class_table,
        "final_test_per_city.csv": city_table,
    }
    written: list[Path] = []
    for filename, frame in table_outputs.items():
        path = tables / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    write_markdown_table(architecture, reports / "controlled_architecture_benchmark.md")
    write_markdown_table(robustness, reports / "robustness_summary.md")
    write_markdown_table(final_method, reports / "final_method_validation_test.md")
    write_markdown_table(class_table, reports / "final_test_per_class.md")
    write_markdown_table(city_table, reports / "final_test_per_city.md")
    written.extend(sorted(reports.glob("*.md")))
    findings_path = reports / "final_report_findings.md"
    write_findings(
        findings_path,
        architecture, robustness, final_method, class_table, city_table,
    )
    written.append(findings_path)
    configure_plots()
    written.extend(plot_architecture(architecture, figures))
    written.extend(plot_noise(noise, figures))
    written.extend(plot_test_classes(class_table, figures))
    written.extend(plot_test_cities(city_table, figures))
    written.extend(plot_validation_test(final_method, figures))

    validation_path = reports / "source_validation.json"
    validation_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    written.append(validation_path)
    manifest_path = reports / "analysis_manifest.json"
    inventory = [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(written))
        if path.is_file() and path != manifest_path
    ]
    manifest = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "input_root": str(args.input_root.resolve()),
        "output_directory": str(output.resolve()),
        "controlled_architecture_winner": "U-Net++ (E2)",
        "final_locked_method": "E9B U-Net++ + four-way TTA + class multipliers",
        "final_validation_mean_iou": FINAL_VALIDATION_MIOU,
        "final_test_mean_iou": float(final_method.iloc[0]["test_mean_iou"]),
        "test_evaluation_count": 1,
        "training_performed": False,
        "checkpoint_updated": False,
        "artifact_count": len(inventory),
        "artifacts": inventory,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output_directory": str(output.resolve()),
        "controlled_winner": "U-Net++ (E2)",
        "controlled_winner_mean_iou": CONTROLLED_WINNER_MIOU,
        "final_locked_validation_mean_iou": FINAL_VALIDATION_MIOU,
        "final_test_mean_iou": float(final_method.iloc[0]["test_mean_iou"]),
        "generated_tables": len(table_outputs),
        "generated_figures": len(list(figures.glob("*.png"))),
        "test_evaluation_performed": False,
        "all_checks_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
