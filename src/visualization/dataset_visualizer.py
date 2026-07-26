"""Visual QA utility for frozen Sentinel-1 Dataset Version 1.

Saves six-panel inspection figures containing normalized VV, normalized VH,
a VV/VH false-colour composite, semantic labels, validity mask, and a semantic
overlay on VV. Source tiles and frozen metadata are never modified.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch

from src.data.segmentation_dataset import Sentinel1UrbanDataset
from src.data.transforms import build_evaluation_transform, build_train_transform

CLASS_NAMES = ["buildings", "roads", "vegetation", "bare land", "water"]
CLASS_COLORS = ["#d7191c", "#fdae61", "#1a9641", "#a6611a", "#2c7bb6"]
IGNORE_COLOR = "#d9d9d9"


def robust_limits(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, [2.0, 98.0])
    if np.isclose(vmin, vmax):
        delta = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.1
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def scale01(array: np.ndarray) -> np.ndarray:
    vmin, vmax = robust_limits(array)
    return np.clip((array - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)


def false_colour(image: np.ndarray) -> np.ndarray:
    vv = scale01(image[0])
    vh = scale01(image[1])
    diff = scale01(image[0] - image[1])
    return np.stack([vv, vh, diff], axis=-1)


def semantic_display(target: np.ndarray, ignore_index: int) -> np.ndarray:
    display = np.zeros(target.shape, dtype=np.uint8)
    supervised = target != ignore_index
    display[supervised] = (target[supervised] + 1).astype(np.uint8)
    return display


def semantic_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap([IGNORE_COLOR, *CLASS_COLORS])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    return cmap, norm


def overlay_on_vv(vv: np.ndarray, display: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = scale01(vv)
    base_rgb = np.repeat(base[..., None], 3, axis=-1)
    lookup = np.asarray(
        [
            [0, 0, 0],
            [215, 25, 28],
            [253, 174, 97],
            [26, 150, 65],
            [166, 97, 26],
            [44, 123, 182],
        ],
        dtype=np.float32,
    ) / 255.0
    output = base_rgb.copy()
    supervised = display > 0
    output[supervised] = (
        (1.0 - alpha) * base_rgb[supervised]
        + alpha * lookup[display[supervised]]
    )
    return np.clip(output, 0.0, 1.0)


def summarize(sample: dict[str, Any]) -> dict[str, Any]:
    image = sample["image"].cpu().numpy()
    target = sample["target"].cpu().numpy()
    validity = sample["validity"].cpu().numpy()
    ignore_index = int(sample["ignore_index"])
    supervised = target != ignore_index
    total = int(target.size)
    return {
        "tile_id": sample["tile_id"],
        "city_id": sample["city_id"],
        "city_name": sample["city_name"],
        "split": sample["split"],
        "image_shape": list(image.shape),
        "target_shape": list(target.shape),
        "image_all_finite": bool(np.isfinite(image).all()),
        "target_values": sorted(int(v) for v in np.unique(target)),
        "valid_pixel_count": int(validity.sum()),
        "supervised_pixel_count": int(supervised.sum()),
        "ignored_pixel_count": int((target == ignore_index).sum()),
        "supervised_fraction": float(supervised.sum() / total),
        "validity_fraction": float(validity.sum() / total),
        "per_class_pixel_counts": {
            CLASS_NAMES[class_id]: int((target == class_id).sum())
            for class_id in range(5)
        },
        "invalid_pixels_are_ignored": bool(np.all(target[~validity] == ignore_index)),
        "paths": sample["paths"],
    }


def save_figure(sample: dict[str, Any], output_path: Path, mode: str, dpi: int) -> dict[str, Any]:
    image = sample["image"].cpu().numpy()
    target = sample["target"].cpu().numpy()
    validity = sample["validity"].cpu().numpy()
    ignore_index = int(sample["ignore_index"])

    if image.shape[0] != 2:
        raise ValueError(f"Expected two SAR channels, found {image.shape[0]}.")

    display = semantic_display(target, ignore_index)
    cmap, norm = semantic_cmap()
    vv_limits = robust_limits(image[0])
    vh_limits = robust_limits(image[1])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

    im = axes[0, 0].imshow(image[0], cmap="gray", vmin=vv_limits[0], vmax=vv_limits[1], interpolation="nearest")
    axes[0, 0].set_title("Normalized VV")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(image[1], cmap="gray", vmin=vh_limits[0], vmax=vh_limits[1], interpolation="nearest")
    axes[0, 1].set_title("Normalized VH")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(false_colour(image), interpolation="nearest")
    axes[0, 2].set_title("False colour: VV / VH / VV−VH")

    im = axes[1, 0].imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1, 0].set_title("Semantic ground truth")
    cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=np.arange(6))
    cb.ax.set_yticklabels(["ignored", *CLASS_NAMES])

    axes[1, 1].imshow(validity.astype(np.uint8), cmap=ListedColormap(["#000000", "#ffffff"]), vmin=0, vmax=1, interpolation="nearest")
    axes[1, 1].set_title("Validity mask: black=ignored, white=valid")

    axes[1, 2].imshow(overlay_on_vv(image[0], display), interpolation="nearest")
    axes[1, 2].set_title("Semantic overlay on VV")

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    report = summarize(sample)
    fig.suptitle(
        f"{sample['tile_id']} | {sample['city_id']} {sample['city_name']} | {sample['split']} — {mode}\n"
        f"supervised={report['supervised_fraction']:.2%}, validity={report['validity_fraction']:.2%}",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return report


def choose_indices(length: int, count: int, seed: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        invalid = [value for value in explicit if value < 0 or value >= length]
        if invalid:
            raise IndexError(f"Dataset indices out of range: {invalid}")
        return explicit
    if count <= 0:
        raise ValueError("--samples must be greater than zero.")
    return sorted(random.Random(seed).sample(range(length), k=min(count, length)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save Dataset V1 visual QA figures.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--enable-sar-intensity", action="store_true")
    parser.add_argument("--enable-speckle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_v1_visual_qa"))
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.augmented and args.split != "train":
        raise ValueError("--augmented is only valid for the training split.")
    if (args.enable_sar_intensity or args.enable_speckle) and not args.augmented:
        raise ValueError("Radiometric perturbations require --augmented.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = (
        build_train_transform(
            enable_sar_intensity=args.enable_sar_intensity,
            enable_speckle=args.enable_speckle,
        )
        if args.augmented
        else build_evaluation_transform()
    )

    dataset = Sentinel1UrbanDataset(
        split=args.split,
        joint_transform=transform,
        exclude_zero_valid=True,
        verify_raster_metadata=True,
    )
    indices = choose_indices(len(dataset), args.samples, args.seed, args.indices)
    mode = "augmented" if args.augmented else "deterministic"
    output_dir = args.output_dir / args.split / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDataset V1 visual QA")
    print("--------------------")
    print(f"Split: {args.split}")
    print(f"Mode: {mode}")
    print(f"Dataset tiles: {len(dataset)}")
    print(f"Selected indices: {indices}")

    reports: list[dict[str, Any]] = []
    for sequence, index in enumerate(indices, start=1):
        sample = dataset[index]
        output_path = output_dir / f"{sequence:02d}_{sample['tile_id']}_{mode}.png"
        report = save_figure(sample, output_path, mode, args.dpi)
        report.update(
            {
                "dataset_index": index,
                "figure_path": str(output_path),
                "augmentation_mode": mode,
                "sar_intensity_enabled": bool(args.enable_sar_intensity),
                "speckle_enabled": bool(args.enable_speckle),
            }
        )
        reports.append(report)
        print(f"  [{sequence}/{len(indices)}] {sample['tile_id']} -> {output_path}")

    qa_report = {
        "dataset_version": dataset.config["dataset_version"],
        "split": args.split,
        "mode": mode,
        "seed": args.seed,
        "dataset_tile_count": len(dataset),
        "selected_indices": indices,
        "sample_count": len(indices),
        "sar_intensity_enabled": bool(args.enable_sar_intensity),
        "speckle_enabled": bool(args.enable_speckle),
        "all_images_finite": all(item["image_all_finite"] for item in reports),
        "all_invalid_pixels_ignored": all(item["invalid_pixels_are_ignored"] for item in reports),
        "samples": reports,
    }
    report_path = output_dir / "visual_qa_report.json"
    report_path.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")

    print(f"\nReport: {report_path}")
    print(f"All images finite: {qa_report['all_images_finite']}")
    print(f"All invalid pixels ignored: {qa_report['all_invalid_pixels_ignored']}")
    print("\nResult: Dataset V1 visual QA figures generated successfully.")


if __name__ == "__main__":
    main()
