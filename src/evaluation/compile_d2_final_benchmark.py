"""Compile the final seven-architecture Dataset V3-MT-D2 benchmark.

This command is evidence-only: it reads exported validation artifacts, never
loads imagery, never performs inference, and never accesses the test split.
Different ZIP extraction layouts are supported through recursive discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/model_experiments/benchmark_analysis_runs_latest")
DEFAULT_OUTPUT = Path("reports/final_benchmark/d2_controlled_architectures")
CLASS_ORDER = ("buildings", "roads", "vegetation", "bare_land", "water")
CORE_ARTIFACTS = {
    "benchmark_summary": "benchmark_summary*.json",
    "evaluation_metrics": "evaluation_metrics*.json",
    "overall_metrics": "overall_metrics*.csv",
    "per_class_metrics": "per_class_metrics*.csv",
    "confusion_matrix": "confusion_matrix*.csv",
    "training_history": "training_history*.csv",
    "experiment_state": "experiment_state*.json",
    "best_checkpoint": "best.pt",
}


@dataclass(frozen=True)
class Experiment:
    code: str
    folder: str
    expected_name: str
    model_name: str
    display_name: str
    family: str


EXPERIMENTS = (
    Experiment("E0", "E0_unet_v3_mt_d2_e0_cross_orbit_50ep", "unet_v3_mt_d2_e0_cross_orbit_50ep", "unet", "U-Net", "CNN encoder-decoder"),
    Experiment("E1", "E1_attention_unet_v3_mt_d2_s0_cross_orbit_50ep", "attention_unet_v3_mt_d2_s0_cross_orbit_50ep", "attention_unet", "Attention U-Net", "Attention-gated CNN"),
    Experiment("E2", "E2_unetpp_v3_mt_d2_s0_cross_orbit_50ep", "unetpp_v3_mt_d2_s0_cross_orbit_50ep", "unetpp", "U-Net++", "Nested-skip CNN"),
    Experiment("E3", "E3_swin_transformer_v3_mt_d2_s0_cross_orbit_50ep", "swin_transformer_v3_mt_d2_s0_cross_orbit_50ep", "swin_transformer", "Swin Transformer", "Hierarchical transformer"),
    Experiment("E4", "E4_deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep", "deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep", "deeplabv3plus", "DeepLabV3+", "Atrous-convolution CNN"),
    Experiment("E5", "E5_segformer_v3_mt_d2_s0_cross_orbit_50ep", "segformer_v3_mt_d2_s0_cross_orbit_50ep", "segformer", "SegFormer", "Hierarchical transformer"),
    Experiment("E6", "E6_mask2former_v3_mt_d2_s0_cross_orbit_50ep", "mask2former_v3_mt_d2_s0_cross_orbit_50ep", "mask2former", "Mask2Former", "Query-based transformer"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preferred_path(path: Path, root: Path) -> tuple[int, int, str]:
    relative = path.relative_to(root)
    parts = tuple(part.lower() for part in relative.parts)
    # Packaged canonical experiment trees take precedence over supplementary
    # evidence. Direct logs/metrics trees are the next preference.
    canonical = 0 if "experiment" in parts else 1
    supplementary = 1 if any("supplement" in part for part in parts) else 0
    return supplementary, canonical, relative.as_posix()


def discover(root: Path, pattern: str) -> tuple[Path | None, list[Path], bool]:
    matches = [path for path in root.rglob(pattern) if path.is_file()]
    if pattern == "best.pt":
        matches = [path for path in matches if "integration_check" not in path.as_posix().lower()]
    matches.sort(key=lambda path: preferred_path(path, root))
    if not matches:
        return None, [], False
    hashes = {sha256(path) for path in matches}
    return matches[0], matches, len(hashes) > 1


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def metric(data: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = number(data.get(name))
        if value is not None:
            return value
    return None


def same(a: float | None, b: float | None, tolerance: float = 1e-9) -> bool:
    return a is None or b is None or math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def inspect_experiment(spec: Experiment, benchmark_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, dict[str, Any], list[str]]:
    root = benchmark_root / spec.folder
    if not root.is_dir():
        raise FileNotFoundError(f"Missing experiment folder: {root}")

    selected: dict[str, Path | None] = {}
    evidence: dict[str, Any] = {}
    warnings: list[str] = []
    for category, pattern in CORE_ARTIFACTS.items():
        chosen, matches, conflict = discover(root, pattern)
        selected[category] = chosen
        evidence[category] = {
            "selected": str(chosen.resolve()) if chosen else None,
            "selected_sha256": sha256(chosen) if chosen else None,
            "candidate_count": len(matches),
            "distinct_content": conflict,
        }
        if conflict:
            warnings.append(f"{spec.code}: distinct duplicate files found for {category}; selected {chosen}")

    required = set(CORE_ARTIFACTS) - {"best_checkpoint"}
    missing = sorted(name for name in required if selected[name] is None)
    if missing:
        raise FileNotFoundError(f"{spec.code} is missing required artifacts: {missing}")
    if selected["best_checkpoint"] is None:
        if spec.code == "E0":
            warnings.append("E0: original best.pt unavailable; retained validation evidence is used without reconstruction")
        else:
            raise FileNotFoundError(f"{spec.code} is missing best.pt")

    benchmark = load_json(selected["benchmark_summary"])  # type: ignore[arg-type]
    evaluation = load_json(selected["evaluation_metrics"])  # type: ignore[arg-type]
    state = load_json(selected["experiment_state"])  # type: ignore[arg-type]
    overall = pd.read_csv(selected["overall_metrics"], encoding="utf-8-sig")  # type: ignore[arg-type]
    history = pd.read_csv(selected["training_history"], encoding="utf-8-sig")  # type: ignore[arg-type]
    if overall.empty or history.empty:
        raise ValueError(f"{spec.code}: empty overall metrics or training history")

    actual_name = str(benchmark.get("experiment_name", ""))
    actual_model = str(benchmark.get("model_name", ""))
    if actual_name != spec.expected_name:
        raise ValueError(f"{spec.code}: expected experiment {spec.expected_name!r}, found {actual_name!r}")
    if actual_model != spec.model_name:
        raise ValueError(f"{spec.code}: expected model {spec.model_name!r}, found {actual_model!r}")
    if str(benchmark.get("status", "")).lower() != "completed" or state.get("completed") is not True:
        raise ValueError(f"{spec.code}: run is not marked completed")

    overall_row = overall.iloc[0].to_dict()
    eval_miou = metric(evaluation, "mean_iou")
    if eval_miou is None:
        raise ValueError(f"{spec.code}: evaluation mean_iou is missing")
    for label, candidate in (
        ("benchmark_summary", metric(benchmark, "mean_iou")),
        ("overall_metrics", number(overall_row.get("mean_iou"))),
    ):
        if not same(eval_miou, candidate):
            raise ValueError(f"{spec.code}: mean_iou disagrees with {label}")

    classes = evaluation.get("per_class")
    if not isinstance(classes, dict) or tuple(classes.keys()) != CLASS_ORDER:
        raise ValueError(f"{spec.code}: unexpected or incomplete class order")
    class_rows: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        values = classes[class_name]
        class_rows.append({
            "experiment_code": spec.code,
            "model": spec.display_name,
            "class_name": class_name,
            "iou": metric(values, "iou"),
            "dice": metric(values, "dice"),
            "precision": metric(values, "precision"),
            "recall": metric(values, "recall"),
            "f1": metric(values, "f1"),
            "target_pixels": metric(values, "target_pixels"),
            "predicted_pixels": metric(values, "predicted_pixels"),
        })

    completed_epochs = int(benchmark.get("completed_epochs") or len(history))
    if completed_epochs != len(history):
        warnings.append(f"{spec.code}: completed_epochs={completed_epochs}, history rows={len(history)}")
    # Dice and pixel-level F1 are mathematically equivalent.  Some exported
    # runs leave F1 undefined when a target-present class has no predictions
    # (Mask2Former/bare_land) and then omit that class from macro F1.  Preserve
    # that raw value, but standardize the comparison over all five target-
    # present classes by using their Dice/F1 values, including zero failures.
    standardized_macro_f1 = float(np.mean([
        metric(classes[class_name], "dice") or 0.0
        for class_name in CLASS_ORDER
    ]))
    row = {
        "experiment_code": spec.code,
        "experiment_name": spec.expected_name,
        "model_name": spec.model_name,
        "model": spec.display_name,
        "architecture_family": spec.family,
        "validation_rank": 0,
        "mean_iou": eval_miou,
        "mean_dice": metric(evaluation, "mean_dice"),
        "standardized_macro_f1": standardized_macro_f1,
        "reported_mean_f1": metric(evaluation, "mean_f1", "macro_f1"),
        "mean_precision": metric(evaluation, "mean_precision", "macro_precision"),
        "mean_recall": metric(evaluation, "mean_recall", "macro_recall"),
        "pixel_accuracy": metric(evaluation, "pixel_accuracy"),
        "frequency_weighted_iou": metric(evaluation, "frequency_weighted_iou"),
        "best_epoch_1_based": int(benchmark.get("best_epoch")) if benchmark.get("best_epoch") is not None else None,
        "completed_epochs": completed_epochs,
        "early_stopped": completed_epochs < 50,
        "training_time_seconds": metric(benchmark, "training_time_seconds"),
        "training_time_minutes": (metric(benchmark, "training_time_seconds") or 0.0) / 60.0,
        "trainable_parameters": int(benchmark.get("trainable_parameters")) if benchmark.get("trainable_parameters") is not None else None,
        "training_peak_gpu_allocated_mb": metric(benchmark, "training_peak_gpu_memory_allocated_mb", "peak_gpu_memory_mb"),
        "training_peak_gpu_reserved_mb": metric(benchmark, "training_peak_gpu_memory_reserved_mb"),
        "inference_images_per_second": metric(benchmark, "inference_images_per_second"),
        "inference_milliseconds_per_image": metric(benchmark, "inference_milliseconds_per_image"),
        "checkpoint_available": selected["best_checkpoint"] is not None,
        "metrics_source": "exported best-checkpoint validation evaluation",
        "evaluation_split": "validation",
    }
    history = history.copy()
    history.insert(0, "model", spec.display_name)
    history.insert(0, "experiment_code", spec.code)
    evidence["root"] = str(root.resolve())
    evidence["warnings"] = warnings
    return row, class_rows, history, evidence, warnings


def save_bar(frame: pd.DataFrame, field: str, ylabel: str, path: Path, dpi: int, target: float | None = None) -> None:
    subset = frame[["model", field]].dropna()
    if subset.empty:
        return
    fig, axis = plt.subplots(figsize=(9.4, 5.2))
    colors = ["#2f6f9f" if model != "U-Net++" else "#d97706" for model in subset["model"]]
    bars = axis.bar(subset["model"], subset[field], color=colors)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=22)
    axis.grid(axis="y", alpha=0.25)
    if target is not None:
        axis.axhline(target, color="#8b1e3f", linestyle="--", linewidth=1.2, label=f"Target {target:.1f}")
        axis.legend()
    for bar, value in zip(bars, subset[field]):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_class_heatmap(class_frame: pd.DataFrame, path: Path, dpi: int) -> None:
    pivot = class_frame.pivot(index="model", columns="class_name", values="iou")
    pivot = pivot.reindex([spec.display_name for spec in EXPERIMENTS], columns=CLASS_ORDER)
    values = pivot.to_numpy(float)
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    image = axis.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(CLASS_ORDER)), [name.replace("_", " ").title() for name in CLASS_ORDER])
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=axis, label="IoU")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_convergence(history: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(9.4, 5.4))
    for model, group in history.groupby("model", sort=False):
        if "val_mean_iou" in group:
            axis.plot(np.arange(1, len(group) + 1), pd.to_numeric(group["val_mean_iou"], errors="coerce"), label=model, linewidth=1.4)
    axis.set_xlabel("Completed epoch")
    axis.set_ylabel("Validation mIoU")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_findings(frame: pd.DataFrame, coverage: pd.DataFrame, output: Path, warnings: list[str]) -> None:
    best = frame.sort_values("mean_iou", ascending=False).iloc[0]
    road = pd.read_csv(output / "per_class_metrics_long.csv").query("class_name == 'roads'").sort_values("iou", ascending=False).iloc[0]
    bare = pd.read_csv(output / "per_class_metrics_long.csv").query("class_name == 'bare_land'").sort_values("iou", ascending=False).iloc[0]
    lines = [
        "# Controlled Dataset V3-MT-D2 Architecture Benchmark",
        "",
        "All values in the main comparison are retained validation-set results from controlled runs on the same dataset, split, seed, loss, batch size and epoch budget. The test split was not accessed.",
        "",
        "## Primary result",
        "",
        f"- Best controlled architecture: **{best['model']}** with validation mIoU **{best['mean_iou']:.4f}**.",
        f"- Best road IoU: **{road['model']}**, **{road['iou']:.4f}**.",
        f"- Best bare-land IoU: **{bare['model']}**, **{bare['iou']:.4f}**.",
        "- Validation mIoU is the primary selection metric; pixel accuracy is secondary because vegetation dominates the supervised pixels.",
        "- Macro F1 is standardized over all five target-present classes. Exported macro F1 is retained separately for provenance.",
        "",
        "## Evidence limitations",
        "",
        "- E0's original checkpoint is unavailable. Its exported metrics, history, confusion matrix and runtime evidence are retained; no values were reconstructed.",
        "- Recorded GPU-memory values are peak **training** allocation, not inference memory.",
        "- Inference throughput is reported only when it exists in the original run artifact; missing values remain blank.",
        "- Geographic robustness and controlled SAR-noise sensitivity require separate validation-only evaluations and are not inferred from aggregate metrics.",
        "- The optimized E9B/calibrated-TTA solution is outside this controlled architecture table and should be reported separately as the final selected system.",
        "",
        "## Reproducibility",
        "",
        f"- Models compiled: {len(frame)}",
        f"- Core evidence-complete models: {int(coverage['core_metrics_complete'].sum())}/{len(coverage)}",
        "- Evaluation scope: validation only",
        "- Test split accessed: no",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in warnings])
    (output / "benchmark_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Benchmark root does not exist: {benchmark_root}")

    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    warnings: list[str] = []
    for spec in EXPERIMENTS:
        row, classes, history, evidence, experiment_warnings = inspect_experiment(spec, benchmark_root)
        rows.append(row)
        class_rows.extend(classes)
        histories.append(history)
        provenance[spec.code] = evidence
        warnings.extend(experiment_warnings)

    frame = pd.DataFrame(rows)
    frame["validation_rank"] = frame["mean_iou"].rank(method="min", ascending=False).astype(int)
    frame["delta_miou_from_unet"] = frame["mean_iou"] - float(frame.loc[frame["experiment_code"] == "E0", "mean_iou"].iloc[0])
    frame["delta_miou_from_best"] = frame["mean_iou"] - float(frame["mean_iou"].max())
    class_frame = pd.DataFrame(class_rows)
    history_frame = pd.concat(histories, ignore_index=True)

    supervised_counts = set(pd.to_numeric(class_frame["target_pixels"], errors="coerce").groupby(class_frame["class_name"]).sum().index)
    if supervised_counts != set(CLASS_ORDER):
        raise ValueError("Class evidence is incomplete")

    coverage_rows = []
    for spec in EXPERIMENTS:
        evidence = provenance[spec.code]
        present = {name: evidence[name]["selected"] is not None for name in CORE_ARTIFACTS}
        coverage_rows.append({
            "experiment_code": spec.code,
            "model": spec.display_name,
            **present,
            "core_metrics_complete": all(present[name] for name in CORE_ARTIFACTS if name != "best_checkpoint"),
            "per_city_robustness_available": False,
            "sar_noise_sensitivity_available": False,
        })
    coverage = pd.DataFrame(coverage_rows)

    status = {
        "status": "PASS",
        "experiment_count": len(frame),
        "core_metrics_complete": bool(coverage["core_metrics_complete"].all()),
        "missing_checkpoint_allowed": ["E0"] if not bool(coverage.loc[coverage["experiment_code"] == "E0", "best_checkpoint"].iloc[0]) else [],
        "selection_split": "validation",
        "test_split_loaded": False,
        "best_controlled_architecture": frame.sort_values("mean_iou", ascending=False).iloc[0]["model"],
        "best_validation_mean_iou": float(frame["mean_iou"].max()),
    }
    print(json.dumps(status, indent=2))
    if args.validate_only:
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.sort_values("validation_rank").to_csv(output / "controlled_model_benchmark.csv", index=False, encoding="utf-8-sig")
    class_frame.to_csv(output / "per_class_metrics_long.csv", index=False, encoding="utf-8-sig")
    class_frame.pivot(index="model", columns="class_name", values="iou").reindex(columns=CLASS_ORDER).to_csv(output / "per_class_iou_matrix.csv", encoding="utf-8-sig")
    history_frame.to_csv(output / "training_history_combined.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "evidence_coverage.csv", index=False, encoding="utf-8-sig")
    (output / "benchmark_provenance.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "controlled_validation_only",
        "test_split_loaded": False,
        "status": status,
        "artifacts": provenance,
    }, indent=2) + "\n", encoding="utf-8")

    save_bar(frame, "mean_iou", "Validation mIoU", output / "validation_miou_comparison", args.dpi, target=0.6)
    save_bar(frame, "training_time_minutes", "Training time (minutes)", output / "training_time_comparison", args.dpi)
    save_bar(frame, "training_peak_gpu_allocated_mb", "Peak allocated GPU memory during training (MB)", output / "training_gpu_memory_comparison", args.dpi)
    parameter_frame = frame.copy()
    parameter_frame["parameters_millions"] = parameter_frame["trainable_parameters"] / 1_000_000
    save_bar(parameter_frame, "parameters_millions", "Trainable parameters (millions)", output / "parameter_count_comparison", args.dpi)
    save_class_heatmap(class_frame, output / "per_class_iou_heatmap", args.dpi)
    save_convergence(history_frame, output / "validation_miou_convergence", args.dpi)
    write_findings(frame, coverage, output, warnings)
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
