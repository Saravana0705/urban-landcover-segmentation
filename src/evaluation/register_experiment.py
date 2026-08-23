"""Register and benchmark an already completed segmentation experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.evaluation.benchmarking import (
    benchmark_inference, build_registry_row, load_best_checkpoint,
    peak_gpu_memory_mb, upsert_registry, write_benchmark_json,
)
from src.evaluation.evaluate_segmentation import (
    evaluate_segmentation_model, write_metrics_outputs,
)
from src.evaluation.recover_training_summary import recover_training_summary
from src.models.model_factory import build_model
from src.training.train_segmentation import (
    build_project_dataloaders, model_display_name, nested_get,
    normalise_model_name, read_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    experiment_name = str(nested_get(config, "experiment.name", "experiment_name"))
    experiment_root = Path(nested_get(
        config, "experiment.output_directory", "output_directory",
        default=f"outputs/model_experiments/{experiment_name}",
    ))
    report_path = Path("metadata/model_development") / f"{normalise_model_name(config)}_training_summary.json"
    if not report_path.exists():
        print(f"Training summary missing; recovering from experiment artifacts: {report_path}")
        report_path = recover_training_summary(args.config)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    checkpoint_path = experiment_root / "checkpoints/best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    device_value = args.device or str(nested_get(config, "training.device", "device", default="auto"))
    device = resolve_device(device_value)
    model = build_model(config).to(device)
    checkpoint = load_best_checkpoint(model, checkpoint_path, device)
    loaders = build_project_dataloaders(config, integration_check=False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    benchmark = benchmark_inference(
        model=model, loader=loaders["val"], device=device,
        warmup_batches=int(nested_get(config, "benchmark.warmup_batches", default=3)),
        measured_batches=int(nested_get(config, "benchmark.measured_batches", default=20)),
    )
    memory = peak_gpu_memory_mb(device)
    evaluation = evaluate_segmentation_model(
        model=model, loader=loaders["val"], device=device
    )
    metrics_paths = write_metrics_outputs(
        evaluation, experiment_root / "metrics"
    )
    checkpoint = dict(checkpoint)
    checkpoint["metrics"] = evaluation
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    row = build_registry_row(
        experiment_name=experiment_name,
        model_name=normalise_model_name(config),
        model_display_name=model_display_name(config),
        model_class=type(model).__name__,
        config_path=str(args.config),
        trainable_parameters=parameters,
        experiment_root=experiment_root,
        summary=summary,
        checkpoint=checkpoint,
        benchmark=benchmark,
        peak_memory_mb=memory,
    )
    registry = Path("metadata/model_development/experiment_registry.csv")
    upsert_registry(registry, row)
    write_benchmark_json(experiment_root / "logs/benchmark_summary.json", row)
    print(f"Registered: {experiment_name}")
    print(f"Inference: {benchmark['inference_images_per_second']:.2f} images/s")
    print(f"Registry: {registry}")
    print(f"Overall metrics: {metrics_paths['overall_metrics']}")
    print(f"Per-class metrics: {metrics_paths['per_class_metrics']}")


if __name__ == "__main__":
    main()
