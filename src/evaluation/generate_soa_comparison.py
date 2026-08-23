"""Generate a cautious literature/SoA comparison for SAR segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_REGISTRY = Path("metadata/model_development/experiment_registry.csv")
DEFAULT_LITERATURE = Path("metadata/literature/soa_segmentation_results.csv")
DEFAULT_OUTPUT = Path("reports/soa_comparison")

MODEL_ALIASES = {
    "unet": "U-Net",
    "u_net": "U-Net",
    "u-net": "U-Net",
    "attention_unet": "Attention U-Net",
    "attention u-net": "Attention U-Net",
    "unetpp": "U-Net++",
    "u-net++": "U-Net++",
    "deeplabv3plus": "DeepLabV3+",
    "deeplabv3+": "DeepLabV3+",
    "segformer": "SegFormer",
    "swin_transformer": "Swin Transformer",
    "swin transformer": "Swin Transformer",
    "mask2former": "Mask2Former",
}

COMPARABILITY_ORDER = {
    "A_direct": 1,
    "B_near": 2,
    "C_contextual": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--literature", type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-pattern", default="extended")
    return parser.parse_args()


def first_column(frame: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"None of these columns exist: {list(names)}")


def canonical_model(value: object) -> str:
    text = str(value).strip()
    return MODEL_ALIASES.get(text.lower(), MODEL_ALIASES.get(text.lower().replace("-", "_"), text))


def numeric(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    return pd.to_numeric(frame[first_column(frame, names)], errors="coerce")


def load_this_work(path: Path, pattern: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    experiment_col = first_column(frame, ("experiment_name", "experiment", "name"))
    model_col = first_column(frame, ("model_display_name", "model_name", "model"))

    selected = frame[
        frame[experiment_col].astype(str).str.contains(pattern, case=False, na=False)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No experiment rows matched {pattern!r}")

    selected["model"] = selected[model_col].map(canonical_model)
    selected["our_miou"] = numeric(selected, ("mean_iou", "miou"))
    selected["our_dice"] = numeric(selected, ("mean_dice", "macro_f1", "mean_f1", "dice"))
    selected["our_pixel_accuracy"] = numeric(
        selected, ("pixel_accuracy", "overall_accuracy", "accuracy")
    )
    selected = selected.drop_duplicates("model", keep="last")

    return selected[
        ["model", "our_miou", "our_dice", "our_pixel_accuracy", experiment_col]
    ].rename(columns={experiment_col: "our_experiment"})


def load_literature(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "study_id", "year", "citation", "model", "sensor_input", "task",
        "dataset", "class_count", "metric_name", "metric_value",
        "comparability_tier", "source_url", "verification_status", "notes",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing literature columns: {sorted(missing)}")

    frame["model"] = frame["model"].map(canonical_model)
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["comparability_rank"] = (
        frame["comparability_tier"].map(COMPARABILITY_ORDER).fillna(99).astype(int)
    )
    return frame


def fmt(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.3f}"


def make_markdown(comparison: pd.DataFrame, literature: pd.DataFrame) -> str:
    lines = [
        "# State-of-the-Art Contextual Comparison",
        "",
        "Cross-dataset values are contextual and are not treated as a controlled leaderboard.",
        "",
        "## Comparability tiers",
        "",
        "- **A_direct:** Sentinel-1/SAR-only land-cover segmentation with a similar class structure.",
        "- **B_near:** SAR segmentation with different classes, geography, inputs or labels.",
        "- **C_contextual:** Different modality/task; architectural context only.",
        "",
        "## This work versus selected literature",
        "",
        "| Model | This work mIoU | Literature mIoU | Tier | Literature context |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {fmt(row['our_miou'])} | "
            f"{fmt(row['literature_miou'])} | "
            f"{row.get('comparability_tier', '—')} | "
            f"{str(row.get('literature_context', '—')).replace('|', '/')} |"
        )

    lines += [
        "",
        "## Evidence register",
        "",
        "| Study | Year | Model | Sensor/input | Task | Dataset | Classes | Metric | Value | Tier |",
        "|---|---:|---|---|---|---|---:|---|---:|---|",
    ]
    ordered = literature.sort_values(
        ["comparability_rank", "year", "model"], ascending=[True, False, True]
    )
    for _, row in ordered.iterrows():
        classes = "—" if pd.isna(row["class_count"]) else int(row["class_count"])
        lines.append(
            f"| {str(row['citation']).replace('|', '/')} | {int(row['year'])} | "
            f"{row['model']} | {str(row['sensor_input']).replace('|', '/')} | "
            f"{str(row['task']).replace('|', '/')} | "
            f"{str(row['dataset']).replace('|', '/')} | {classes} | "
            f"{row['metric_name']} | {fmt(row['metric_value'])} | "
            f"{row['comparability_tier']} |"
        )

    lines += [
        "",
        "## Interpretation checklist",
        "",
        "1. Compare like-for-like metrics only.",
        "2. Explain differences in sensor inputs, labels, classes and spatial resolution.",
        "3. Discuss roads and bare land separately because they are minority classes.",
        "4. Use converged runs before making architectural claims.",
        "5. Do not call a cross-dataset difference a performance gap without qualification.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ours = load_this_work(args.experiment_registry, args.experiment_pattern)
    literature = load_literature(args.literature)

    miou = literature[
        literature["metric_name"].str.lower().isin({"miou", "mean_iou"})
        & literature["metric_value"].notna()
    ].copy()
    miou = miou.sort_values(
        ["model", "comparability_rank", "metric_value"],
        ascending=[True, True, False],
    )
    best = miou.drop_duplicates("model", keep="first").copy()
    best["literature_context"] = (
        best["dataset"].astype(str) + "; " + best["task"].astype(str)
    )

    comparison = ours.merge(
        best[
            [
                "model", "metric_value", "comparability_tier",
                "literature_context", "citation", "source_url"
            ]
        ].rename(columns={"metric_value": "literature_miou"}),
        on="model",
        how="left",
    )
    comparison["absolute_gap"] = (
        comparison["our_miou"] - comparison["literature_miou"]
    )

    comparison.to_csv(args.output_dir / "soa_model_comparison.csv", index=False)
    literature.drop(columns=["comparability_rank"]).to_csv(
        args.output_dir / "soa_evidence_register.csv", index=False
    )
    (args.output_dir / "soa_comparison_report.md").write_text(
        make_markdown(comparison, literature), encoding="utf-8"
    )
    manifest = {
        "models_in_this_work": len(ours),
        "literature_rows": len(literature),
        "warning": "Cross-dataset values are contextual, not a controlled leaderboard.",
    }
    (args.output_dir / "soa_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"This-work models: {len(ours)}")
    print(f"Literature evidence rows: {len(literature)}")
    print(f"Output directory: {args.output_dir}")
    print("Important: treat cross-dataset values as contextual, not a leaderboard.")


if __name__ == "__main__":
    main()
