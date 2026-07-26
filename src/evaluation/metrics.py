"""Confusion-matrix-based metrics for multiclass semantic segmentation.

Dataset V1 target convention
----------------------------
Model classes:
    0 = buildings
    1 = roads
    2 = vegetation
    3 = bare land
    4 = water

Ignored pixels:
    255 = unlabeled, invalid SAR, or padded area

Metrics are accumulated over all batches using one confusion matrix. This is
the correct basis for epoch-level and dataset-level evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor


DEFAULT_CLASS_NAMES = [
    "buildings",
    "roads",
    "vegetation",
    "bare_land",
    "water",
]


@dataclass(frozen=True)
class MetricsConfig:
    """Serializable segmentation-metrics configuration."""

    num_classes: int = 5
    ignore_index: int = 255
    class_names: tuple[str, ...] = tuple(DEFAULT_CLASS_NAMES)
    include_absent_classes_in_macro: bool = False
    epsilon: float = 1e-12

    def validate(self) -> None:
        """Validate configuration values."""
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        if self.ignore_index != 255:
            raise ValueError("Dataset V1 expects ignore_index=255.")

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names length must equal num_classes."
            )

        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique.")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


def _validate_target(
    target: Tensor,
    *,
    ignore_index: int,
    num_classes: int,
) -> None:
    """Validate target tensor values."""
    if target.dtype != torch.int64:
        raise TypeError(
            f"Target must use torch.int64, found {target.dtype}."
        )

    valid_values = target[target != ignore_index]

    if valid_values.numel() == 0:
        return

    minimum = int(valid_values.min().item())
    maximum = int(valid_values.max().item())

    if minimum < 0 or maximum >= num_classes:
        raise ValueError(
            "Target contains values outside the supported class range."
        )


def _predictions_from_input(
    predictions_or_logits: Tensor,
    *,
    num_classes: int,
) -> Tensor:
    """Convert logits or class predictions to int64 predictions."""
    if predictions_or_logits.ndim == 4:
        if predictions_or_logits.shape[1] != num_classes:
            raise ValueError(
                f"Expected {num_classes} logit channels, "
                f"found {predictions_or_logits.shape[1]}."
            )

        if not predictions_or_logits.is_floating_point():
            raise TypeError(
                "Four-dimensional model output must be floating point."
            )

        if not torch.isfinite(predictions_or_logits).all():
            raise ValueError("Model output contains non-finite values.")

        return torch.argmax(
            predictions_or_logits,
            dim=1,
        ).to(torch.int64)

    if predictions_or_logits.ndim == 3:
        predictions = predictions_or_logits.to(torch.int64)

        if predictions.numel() > 0:
            minimum = int(predictions.min().item())
            maximum = int(predictions.max().item())

            if minimum < 0 or maximum >= num_classes:
                raise ValueError(
                    "Predictions contain values outside the class range."
                )

        return predictions

    raise ValueError(
        "Expected predictions shaped (N,H,W) or logits shaped (N,C,H,W)."
    )


def confusion_matrix_from_tensors(
    predictions_or_logits: Tensor,
    target: Tensor,
    *,
    num_classes: int = 5,
    ignore_index: int = 255,
) -> Tensor:
    """Build a confusion matrix for one batch.

    Rows represent ground-truth classes.
    Columns represent predicted classes.
    """
    predictions = _predictions_from_input(
        predictions_or_logits,
        num_classes=num_classes,
    )

    if target.ndim != 3:
        raise ValueError(
            f"Expected target shape (N,H,W), found {tuple(target.shape)}."
        )

    if predictions.shape != target.shape:
        raise ValueError(
            "Predictions and target spatial dimensions do not match."
        )

    _validate_target(
        target,
        ignore_index=ignore_index,
        num_classes=num_classes,
    )

    valid_mask = target != ignore_index

    if not valid_mask.any():
        return torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
            device=target.device,
        )

    valid_target = target[valid_mask]
    valid_predictions = predictions[valid_mask]

    encoded = (
        valid_target * num_classes
        + valid_predictions
    )

    counts = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )

    return counts.reshape(
        num_classes,
        num_classes,
    ).to(torch.int64)


def _safe_divide(
    numerator: Tensor,
    denominator: Tensor,
) -> Tensor:
    """Element-wise division returning NaN for undefined values."""
    numerator = numerator.to(torch.float64)
    denominator = denominator.to(torch.float64)

    output = torch.full_like(
        numerator,
        fill_value=float("nan"),
        dtype=torch.float64,
    )

    valid = denominator > 0
    output[valid] = numerator[valid] / denominator[valid]

    return output


def metrics_from_confusion_matrix(
    confusion_matrix: Tensor,
    *,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    include_absent_classes_in_macro: bool = False,
) -> dict[str, Any]:
    """Calculate segmentation metrics from a confusion matrix."""
    if confusion_matrix.ndim != 2:
        raise ValueError("Confusion matrix must be two-dimensional.")

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    num_classes = confusion_matrix.shape[0]

    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length must match confusion-matrix size."
        )

    matrix = confusion_matrix.to(torch.float64)

    true_positive = torch.diag(matrix)
    target_support = matrix.sum(dim=1)
    predicted_support = matrix.sum(dim=0)

    false_negative = target_support - true_positive
    false_positive = predicted_support - true_positive

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = _safe_divide(
        true_positive,
        union,
    )

    dice = _safe_divide(
        2.0 * true_positive,
        2.0 * true_positive
        + false_positive
        + false_negative,
    )

    precision = _safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = _safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    total_pixels = matrix.sum()

    pixel_accuracy = (
        float(
            true_positive.sum().item()
            / total_pixels.item()
        )
        if total_pixels.item() > 0
        else float("nan")
    )

    if include_absent_classes_in_macro:
        macro_mask = torch.ones(
            num_classes,
            dtype=torch.bool,
            device=matrix.device,
        )
    else:
        macro_mask = target_support > 0

    def masked_mean(values: Tensor) -> float:
        selected = values[macro_mask]
        selected = selected[torch.isfinite(selected)]

        if selected.numel() == 0:
            return float("nan")

        return float(selected.mean().item())

    frequency = (
        target_support / total_pixels
        if total_pixels.item() > 0
        else torch.zeros_like(target_support)
    )

    finite_iou = torch.nan_to_num(
        iou,
        nan=0.0,
    )
    finite_dice = torch.nan_to_num(
        dice,
        nan=0.0,
    )

    frequency_weighted_iou = (
        float(
            torch.sum(
                frequency * finite_iou
            ).item()
        )
        if total_pixels.item() > 0
        else float("nan")
    )

    frequency_weighted_dice = (
        float(
            torch.sum(
                frequency * finite_dice
            ).item()
        )
        if total_pixels.item() > 0
        else float("nan")
    )

    per_class: dict[str, dict[str, Any]] = {}

    for class_id, class_name in enumerate(class_names):
        per_class[class_name] = {
            "class_id": class_id,
            "target_pixels": int(
                target_support[class_id].item()
            ),
            "predicted_pixels": int(
                predicted_support[class_id].item()
            ),
            "true_positive": int(
                true_positive[class_id].item()
            ),
            "false_positive": int(
                false_positive[class_id].item()
            ),
            "false_negative": int(
                false_negative[class_id].item()
            ),
            "iou": (
                float(iou[class_id].item())
                if torch.isfinite(iou[class_id])
                else None
            ),
            "dice": (
                float(dice[class_id].item())
                if torch.isfinite(dice[class_id])
                else None
            ),
            "precision": (
                float(precision[class_id].item())
                if torch.isfinite(precision[class_id])
                else None
            ),
            "recall": (
                float(recall[class_id].item())
                if torch.isfinite(recall[class_id])
                else None
            ),
            "present_in_target": bool(
                target_support[class_id].item() > 0
            ),
        }

    return {
        "confusion_matrix": (
            confusion_matrix.cpu().tolist()
        ),
        "total_supervised_pixels": int(
            total_pixels.item()
        ),
        "correctly_classified_pixels": int(
            true_positive.sum().item()
        ),
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": masked_mean(iou),
        "mean_dice": masked_mean(dice),
        "macro_precision": masked_mean(precision),
        "macro_recall": masked_mean(recall),
        "frequency_weighted_iou": frequency_weighted_iou,
        "frequency_weighted_dice": frequency_weighted_dice,
        "macro_class_policy": (
            "all_classes"
            if include_absent_classes_in_macro
            else "classes_present_in_target"
        ),
        "classes_present_in_target": [
            class_names[index]
            for index in range(num_classes)
            if target_support[index].item() > 0
        ],
        "per_class": per_class,
    }


class SegmentationMetrics:
    """Running confusion-matrix accumulator."""

    def __init__(
        self,
        config: MetricsConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = (
            MetricsConfig()
            if config is None
            else config
        )
        self.config.validate()

        self.device = torch.device(device)
        self.confusion_matrix = torch.zeros(
            (
                self.config.num_classes,
                self.config.num_classes,
            ),
            dtype=torch.int64,
            device=self.device,
        )
        self.batch_count = 0

    def reset(self) -> None:
        """Clear accumulated metrics."""
        self.confusion_matrix.zero_()
        self.batch_count = 0

    @torch.no_grad()
    def update(
        self,
        predictions_or_logits: Tensor,
        target: Tensor,
    ) -> None:
        """Accumulate one batch."""
        batch_matrix = confusion_matrix_from_tensors(
            predictions_or_logits,
            target,
            num_classes=self.config.num_classes,
            ignore_index=self.config.ignore_index,
        ).to(self.device)

        self.confusion_matrix += batch_matrix
        self.batch_count += 1

    def merge(
        self,
        other: "SegmentationMetrics",
    ) -> None:
        """Merge another compatible accumulator."""
        if self.config != other.config:
            raise ValueError(
                "Cannot merge metrics with different configurations."
            )

        self.confusion_matrix += (
            other.confusion_matrix.to(self.device)
        )
        self.batch_count += other.batch_count

    def compute(self) -> dict[str, Any]:
        """Return accumulated metrics."""
        report = metrics_from_confusion_matrix(
            self.confusion_matrix,
            class_names=self.config.class_names,
            include_absent_classes_in_macro=(
                self.config.include_absent_classes_in_macro
            ),
        )

        report["batch_count"] = self.batch_count
        report["config"] = asdict(self.config)

        return report

    def state_dict(self) -> dict[str, Any]:
        """Return serializable accumulator state."""
        return {
            "config": asdict(self.config),
            "confusion_matrix": (
                self.confusion_matrix.cpu()
            ),
            "batch_count": self.batch_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        """Restore accumulator state."""
        matrix = state["confusion_matrix"]

        if not isinstance(matrix, Tensor):
            matrix = torch.tensor(
                matrix,
                dtype=torch.int64,
            )

        expected_shape = (
            self.config.num_classes,
            self.config.num_classes,
        )

        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                "Stored confusion matrix has an incompatible shape."
            )

        self.confusion_matrix = matrix.to(
            self.device,
            dtype=torch.int64,
        )
        self.batch_count = int(
            state["batch_count"]
        )


def smoke_test(
    seed: int = 20260725,
) -> dict[str, Any]:
    """Run exact synthetic metric checks."""
    torch.manual_seed(seed)

    num_classes = 5
    ignore_index = 255

    target = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 2, 2, 1],
                [3, 3, 4, 4],
                [255, 255, 4, 2],
            ]
        ],
        dtype=torch.int64,
    )

    prediction = torch.tensor(
        [
            [
                [0, 1, 1, 1],
                [0, 2, 0, 1],
                [3, 4, 4, 4],
                [2, 2, 3, 2],
            ]
        ],
        dtype=torch.int64,
    )

    expected_confusion = torch.tensor(
        [
            [2, 1, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [1, 0, 2, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 2],
        ],
        dtype=torch.int64,
    )

    direct_matrix = confusion_matrix_from_tensors(
        prediction,
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    if not torch.equal(
        direct_matrix.cpu(),
        expected_confusion,
    ):
        raise RuntimeError(
            "Confusion-matrix calculation does not match expectation."
        )

    accumulator = SegmentationMetrics(
        MetricsConfig(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    )

    accumulator.update(
        prediction[:, :, :2],
        target[:, :, :2],
    )
    accumulator.update(
        prediction[:, :, 2:],
        target[:, :, 2:],
    )

    accumulated_report = accumulator.compute()

    if accumulator.batch_count != 2:
        raise RuntimeError(
            "Accumulator batch count is incorrect."
        )

    if accumulator.confusion_matrix.sum().item() != 14:
        raise RuntimeError(
            "Ignored pixels were not excluded correctly."
        )

    if not torch.equal(
        accumulator.confusion_matrix.cpu(),
        expected_confusion,
    ):
        raise RuntimeError(
            "Batch accumulation differs from direct calculation."
        )

    perfect_target = torch.randint(
        low=0,
        high=num_classes,
        size=(2, 16, 16),
        dtype=torch.int64,
    )
    perfect_target[:, :2, :] = ignore_index

    perfect_prediction = perfect_target.clone()
    perfect_prediction[
        perfect_prediction == ignore_index
    ] = 0

    perfect_metrics = SegmentationMetrics(
        MetricsConfig(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    )
    perfect_metrics.update(
        perfect_prediction,
        perfect_target,
    )
    perfect_report = perfect_metrics.compute()

    tolerance = 1e-12

    if abs(
        perfect_report["pixel_accuracy"] - 1.0
    ) > tolerance:
        raise RuntimeError(
            "Perfect prediction pixel accuracy is not one."
        )

    if abs(
        perfect_report["mean_iou"] - 1.0
    ) > tolerance:
        raise RuntimeError(
            "Perfect prediction mean IoU is not one."
        )

    if abs(
        perfect_report["mean_dice"] - 1.0
    ) > tolerance:
        raise RuntimeError(
            "Perfect prediction mean Dice is not one."
        )

    logits = torch.full(
        (1, num_classes, 4, 4),
        fill_value=-5.0,
        dtype=torch.float32,
    )
    logits.scatter_(
        1,
        prediction.unsqueeze(1),
        5.0,
    )

    logits_matrix = confusion_matrix_from_tensors(
        logits,
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    if not torch.equal(
        logits_matrix.cpu(),
        expected_confusion,
    ):
        raise RuntimeError(
            "Logit conversion does not match class predictions."
        )

    return {
        "seed": seed,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "expected_supervised_pixels": 14,
        "direct_confusion_matrix": (
            direct_matrix.cpu().tolist()
        ),
        "accumulated_confusion_matrix": (
            accumulator.confusion_matrix.cpu().tolist()
        ),
        "batch_count": accumulator.batch_count,
        "sample_metrics": accumulated_report,
        "perfect_prediction": {
            "pixel_accuracy": (
                perfect_report["pixel_accuracy"]
            ),
            "mean_iou": perfect_report["mean_iou"],
            "mean_dice": perfect_report["mean_dice"],
        },
        "logits_match_class_predictions": True,
        "ignore_pixels_excluded": True,
        "all_checks_passed": True,
    }


def save_metrics_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a metrics report as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run segmentation-metrics exactness checks."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/model_development/"
            "metrics_smoke_test.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run and save the metric smoke test."""
    args = parse_arguments()

    report = smoke_test(
        seed=args.seed,
    )

    save_metrics_report(
        report,
        args.output,
    )

    sample = report["sample_metrics"]

    print("\nSegmentation metrics smoke test")
    print("-------------------------------")
    print(
        "Supervised pixels: "
        f"{sample['total_supervised_pixels']:,}"
    )
    print(
        f"Pixel accuracy: "
        f"{sample['pixel_accuracy']:.6f}"
    )
    print(
        f"Mean IoU: "
        f"{sample['mean_iou']:.6f}"
    )
    print(
        f"Mean Dice: "
        f"{sample['mean_dice']:.6f}"
    )
    print(
        f"Macro precision: "
        f"{sample['macro_precision']:.6f}"
    )
    print(
        f"Macro recall: "
        f"{sample['macro_recall']:.6f}"
    )
    print(
        "Perfect prediction: "
        f"accuracy={report['perfect_prediction']['pixel_accuracy']:.1f}, "
        f"mIoU={report['perfect_prediction']['mean_iou']:.1f}, "
        f"Dice={report['perfect_prediction']['mean_dice']:.1f}"
    )
    print(
        "Logits match class predictions: "
        f"{report['logits_match_class_predictions']}"
    )
    print(
        "Ignore pixels excluded: "
        f"{report['ignore_pixels_excluded']}"
    )
    print(f"Report: {args.output}")
    print(
        "\nResult: segmentation metrics passed "
        "confusion-matrix and exactness validation."
    )


if __name__ == "__main__":
    main()
