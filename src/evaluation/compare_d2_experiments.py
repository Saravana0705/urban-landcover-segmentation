"""Create validation-only comparisons for controlled V3-MT-D2 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


BASELINE = "unet_v3_mt_d2_e0_cross_orbit_50ep"
CLASS_NAMES = ("buildings", "roads", "vegetation", "bare_land", "water")
DEFAULT_EXPERIMENTS = (
    BASELINE,
    "unet_v3_mt_d2_e1_minority_sampling_50ep",
    "orbit_aware_unet_v3_mt_d2_e2_50ep",
    "unet_v3_mt_d2_f1_cross_orbit_ratio_50ep",
    "swin_transformer_v3_mt_d2_s0_cross_orbit_50ep",
    "unetpp_v3_mt_d2_s0_cross_orbit_50ep",
    "attention_unet_v3_mt_d2_s0_cross_orbit_50ep",
)

CORE_CATEGORIES = {
    "experiment_state": ("experiment_state.json", "reports/experiment_state.json"),
    "evaluation_metrics": ("metrics/evaluation_metrics.json", "evaluation_metrics.json"),
    "overall_metrics": ("metrics/overall_metrics.csv", "overall_metrics.csv"),
    "per_class_metrics": ("metrics/per_class_metrics.csv", "per_class_metrics.csv"),
    "best_checkpoint": ("checkpoints/best.pt",),
    "training_history": (
        "training_history.json",
        "history/training_history.json",
        "logs/training_history.json",
    ),
}


def first_existing(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return None


def newest_match(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def load_first_csv_row(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.DictReader(stream), {})


def number(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value):
    parsed = number(value)
    return int(parsed) if parsed is not None else None


def model_from_name(name: str) -> str:
    if name.startswith("orbit_aware_unet"):
        return "orbit_aware_unet"
    if name.startswith("attention_unet"):
        return "attention_unet"
    if name.startswith("swin_transformer"):
        return "swin_transformer"
    if name.startswith("unetpp"):
        return "unetpp"
    return "unet"


def parse_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected EXPERIMENT_NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Expected EXPERIMENT_NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def read_experiment(name: str, root: Path) -> tuple[dict, list[dict], dict]:
    if not root.is_dir():
        return ({
            "experiment_name": name, "model": model_from_name(name),
            "status": "pending", "artifact_complete": False,
            "experiment_directory": str(root),
        }, [], {"status": "pending", "missing": list(CORE_CATEGORIES)})

    located = {key: first_existing(root, candidates)
               for key, candidates in CORE_CATEGORIES.items()}
    state = load_json(located["experiment_state"])
    evaluation = load_json(located["evaluation_metrics"])
    overall = load_first_csv_row(located["overall_metrics"])
    benchmark = load_json(newest_match(root, "benchmark_summary.json"))

    mean_iou = number(evaluation.get("mean_iou"))
    if mean_iou is None:
        mean_iou = number(overall.get("mean_iou"))
    completed = state.get("completed") is True or benchmark.get("status") == "completed"
    missing = [key for key, path in located.items() if path is None]
    artifact_complete = not missing
    status = "completed" if completed and mean_iou is not None else "incomplete"

    extra = state.get("extra") if isinstance(state.get("extra"), dict) else {}
    row = {
        "experiment_name": name,
        "model": benchmark.get("model_name") or model_from_name(name),
        "status": status,
        "artifact_complete": artifact_complete,
        "experiment_directory": str(root),
        "best_epoch": integer(benchmark.get("best_epoch", state.get("best_epoch"))),
        "validation_mean_iou": mean_iou,
        "delta_mean_iou_from_d2_e0": None,
        "pixel_accuracy": number(evaluation.get("pixel_accuracy", overall.get("pixel_accuracy"))),
        "mean_dice": number(evaluation.get("mean_dice", overall.get("mean_dice"))),
        "macro_precision": number(evaluation.get("macro_precision", overall.get("mean_precision"))),
        "macro_recall": number(evaluation.get("macro_recall", overall.get("mean_recall"))),
        "training_duration_seconds": number(
            benchmark.get("training_time_seconds", extra.get("duration_seconds"))
        ),
        "trainable_parameters": integer(benchmark.get("trainable_parameters")),
        "peak_gpu_memory_mb": number(
            benchmark.get("peak_gpu_memory_mb", benchmark.get("training_peak_gpu_memory_allocated_mb"))
        ),
    }

    per_class: list[dict] = []
    classes = evaluation.get("per_class")
    if isinstance(classes, dict):
        for class_name in CLASS_NAMES:
            metrics = classes.get(class_name, {})
            if isinstance(metrics, dict):
                per_class.append({
                    "experiment_name": name,
                    "class_name": class_name,
                    "iou": number(metrics.get("iou")),
                    "delta_iou_from_d2_e0": None,
                    "dice": number(metrics.get("dice")),
                    "precision": number(metrics.get("precision")),
                    "recall": number(metrics.get("recall")),
                })
    if not per_class and located["per_class_metrics"]:
        with located["per_class_metrics"].open("r", encoding="utf-8-sig", newline="") as stream:
            for metrics in csv.DictReader(stream):
                per_class.append({
                    "experiment_name": name,
                    "class_name": metrics.get("class_name"),
                    "iou": number(metrics.get("iou")),
                    "delta_iou_from_d2_e0": None,
                    "dice": number(metrics.get("dice")),
                    "precision": number(metrics.get("precision")),
                    "recall": number(metrics.get("recall")),
                })
    provenance = {
        "status": status,
        "experiment_directory": str(root),
        "missing": missing,
        "located": {key: str(path) if path else None for key, path in located.items()},
    }
    return row, per_class, provenance


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path,
                        default=Path("outputs/model_experiments"))
    parser.add_argument("--experiment", action="append", default=[],
                        help="Override a path using EXPERIMENT_NAME=PATH")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/model_comparisons/v3_mt_d2"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = dict(parse_override(value) for value in args.experiment)
    names = list(DEFAULT_EXPERIMENTS)
    for name in overrides:
        if name not in names:
            names.append(name)

    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    provenance: dict[str, dict] = {}
    for name in names:
        root = overrides.get(name, args.experiments_root / name).resolve()
        row, classes, evidence = read_experiment(name, root)
        summary_rows.append(row)
        class_rows.extend(classes)
        provenance[name] = evidence

    baseline_row = next(row for row in summary_rows if row["experiment_name"] == BASELINE)
    baseline_iou = baseline_row.get("validation_mean_iou")
    baseline_classes = {
        row["class_name"]: row["iou"] for row in class_rows
        if row["experiment_name"] == BASELINE
    }
    if baseline_iou is not None:
        for row in summary_rows:
            value = row.get("validation_mean_iou")
            if value is not None:
                row["delta_mean_iou_from_d2_e0"] = value - baseline_iou
    for row in class_rows:
        baseline = baseline_classes.get(row["class_name"])
        if baseline is not None and row.get("iou") is not None:
            row["delta_iou_from_d2_e0"] = row["iou"] - baseline

    output = args.output_dir.resolve()
    summary_columns = [
        "experiment_name", "model", "status", "artifact_complete",
        "best_epoch", "validation_mean_iou", "delta_mean_iou_from_d2_e0",
        "pixel_accuracy", "mean_dice", "macro_precision", "macro_recall",
        "training_duration_seconds", "trainable_parameters",
        "peak_gpu_memory_mb", "experiment_directory",
    ]
    class_columns = [
        "experiment_name", "class_name", "iou", "delta_iou_from_d2_e0",
        "dice", "precision", "recall",
    ]
    write_csv(output / "d2_experiment_summary.csv", summary_rows, summary_columns)
    write_csv(output / "d2_per_class_comparison.csv", class_rows, class_columns)
    report = {
        "status": "PASS" if baseline_iou is not None else "BASELINE_MISSING",
        "scope": "validation_only",
        "test_split_loaded": False,
        "reference_experiment": BASELINE,
        "reference_mean_iou": baseline_iou,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": provenance,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "d2_comparison_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "completed": sum(row["status"] == "completed" for row in summary_rows),
        "pending": sum(row["status"] == "pending" for row in summary_rows),
        "incomplete": sum(row["status"] == "incomplete" for row in summary_rows),
        "output_directory": str(output),
        "test_split_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()