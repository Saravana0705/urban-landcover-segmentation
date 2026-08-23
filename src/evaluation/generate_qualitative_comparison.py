"""Generate identical-tile qualitative comparisons for registered models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import ListedColormap

from src.data.segmentation_dataset import Sentinel1UrbanDataset
from src.evaluation.benchmarking import load_best_checkpoint
from src.models.model_factory import build_model

CLASS_NAMES = ["Buildings", "Roads", "Vegetation", "Bare land", "Water"]
CLASS_COLOURS = ["#d73027", "#fdae61", "#1a9850", "#d9b38c", "#4575b4"]
IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("metadata/model_development/experiment_registry.csv"),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--indices", type=int, nargs="+", default=(0, 72, 143))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/final_benchmark/qualitative"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return payload


def display_sar(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy().astype(np.float32)
    # Robust false-colour display from normalized VV and VH.
    vv, vh = array[0], array[1]
    ratio = vv - vh
    channels = np.stack((vv, vh, ratio), axis=-1)
    low = np.nanpercentile(channels, 2, axis=(0, 1), keepdims=True)
    high = np.nanpercentile(channels, 98, axis=(0, 1), keepdims=True)
    return np.clip((channels - low) / np.maximum(high - low, 1e-6), 0, 1)


def mask_for_display(mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(mask == IGNORE_INDEX, mask)


def main() -> None:
    args = parse_args()
    if not args.registry.exists():
        raise FileNotFoundError(f"Registry not found: {args.registry}")
    registry = pd.read_csv(args.registry)
    registry = registry.loc[registry["status"].astype(str).str.lower() == "completed"].copy()
    if registry.empty:
        raise ValueError("No completed models are registered.")

    dataset = Sentinel1UrbanDataset(split=args.split)
    invalid = [index for index in args.indices if index < 0 or index >= len(dataset)]
    if invalid:
        raise IndexError(f"Sample indices outside {args.split} dataset: {invalid}")

    device = resolve_device(args.device)
    models: list[tuple[str, torch.nn.Module]] = []
    skipped: list[str] = []
    for row in registry.to_dict(orient="records"):
        config_path = Path(str(row["config_path"]))
        checkpoint_path = Path(str(row["best_checkpoint"]))
        if not config_path.exists() or not checkpoint_path.exists():
            skipped.append(str(row["model_display_name"]))
            continue
        model = build_model(load_config(config_path)).to(device)
        load_best_checkpoint(model, checkpoint_path, device)
        model.eval()
        models.append((str(row["model_display_name"]), model))

    if not models:
        raise RuntimeError("No registered model could be loaded.")

    args.output.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(CLASS_COLOURS)
    manifest: list[dict[str, object]] = []

    for sample_index in args.indices:
        sample = dataset[sample_index]
        image = sample["image"].unsqueeze(0).to(device, dtype=torch.float32)
        target = sample["target"].detach().cpu().numpy()
        predictions: list[tuple[str, np.ndarray]] = []
        with torch.inference_mode():
            for display_name, model in models:
                prediction = model(image).argmax(dim=1)[0].detach().cpu().numpy()
                prediction[target == IGNORE_INDEX] = IGNORE_INDEX
                predictions.append((display_name, prediction))

        column_count = 2 + len(predictions)
        fig, axes = plt.subplots(1, column_count, figsize=(3.2 * column_count, 3.8))
        axes[0].imshow(display_sar(sample["image"]))
        axes[0].set_title("Sentinel-1 VV/VH")
        axes[1].imshow(mask_for_display(target), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[1].set_title("Ground truth")
        for axis, (display_name, prediction) in zip(axes[2:], predictions):
            axis.imshow(mask_for_display(prediction), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            axis.set_title(display_name)
        for axis in axes:
            axis.axis("off")
        fig.suptitle(
            f"{sample['city_id']} — {sample['city_name']} — {sample['tile_id']}",
            fontsize=12,
        )
        fig.tight_layout()
        base = args.output / f"{args.split}_{sample_index:04d}_{sample['tile_id']}"
        fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "split": args.split,
                "sample_index": sample_index,
                "tile_id": sample["tile_id"],
                "city_id": sample["city_id"],
                "city_name": sample["city_name"],
                "png": str(base.with_suffix(".png")),
                "svg": str(base.with_suffix(".svg")),
            }
        )

    (args.output / "qualitative_manifest.json").write_text(
        json.dumps(
            {
                "models": [name for name, _ in models],
                "skipped_models": skipped,
                "samples": manifest,
                "class_names": CLASS_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Loaded models: {len(models)}")
    print(f"Generated figures: {len(manifest)}")
    print(f"Output directory: {args.output}")
    if skipped:
        print("Skipped models: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
