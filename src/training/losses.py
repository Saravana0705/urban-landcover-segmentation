"""Reusable multiclass segmentation losses.

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

The recommended baseline loss is CombinedCrossEntropyDiceLoss.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    """Serializable segmentation-loss configuration."""

    name: str = "ce_dice"
    num_classes: int = 5
    ignore_index: int = 255

    ce_weight: float = 0.5

    dice_weight: float = 0.5
    dice_smooth: float = 1.0
    dice_epsilon: float = 1e-7

    tversky_weight: float = 0.7
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_smooth: float = 1.0
    tversky_epsilon: float = 1e-7

    include_absent_classes: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if self.name not in {"cross_entropy", "dice", "ce_dice", "tversky", "ce_tversky"}:
            raise ValueError("name must be one of: cross_entropy, dice, ce_dice, tversky, ce_tversky.")

        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        if self.ignore_index != 255:
            raise ValueError("Dataset V1 expects ignore_index=255.")

        if self.ce_weight < 0 or self.dice_weight < 0:
            raise ValueError("Loss weights cannot be negative.")

        if self.ce_weight + self.dice_weight <= 0:
            raise ValueError("At least one combined-loss weight must be positive.")

        if self.dice_smooth < 0:
            raise ValueError("dice_smooth cannot be negative.")

        if self.dice_epsilon <= 0:
            raise ValueError("dice_epsilon must be positive.")

        if self.tversky_weight < 0:
            raise ValueError("tversky_weight cannot be negative.")

        if self.tversky_alpha < 0 or self.tversky_beta < 0:
            raise ValueError("Tversky alpha and beta cannot be negative.")

        if self.tversky_alpha + self.tversky_beta <= 0:
            raise ValueError("At least one Tversky penalty must be positive.")

        if self.tversky_smooth < 0:
            raise ValueError("tversky_smooth cannot be negative.")

        if self.tversky_epsilon <= 0:
            raise ValueError("tversky_epsilon must be positive.")

        if (self.name == "ce_tversky" and self.ce_weight + self.tversky_weight <= 0):
            raise ValueError("At least one CE-Tversky loss weight must be positive.")


def _validate_inputs(
    logits: Tensor,
    target: Tensor,
    *,
    num_classes: int,
) -> None:
    """Validate segmentation logits and targets."""
    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits shape (N, C, H, W), found {tuple(logits.shape)}."
        )

    if target.ndim != 3:
        raise ValueError(
            f"Expected target shape (N, H, W), found {tuple(target.shape)}."
        )

    if logits.shape[0] != target.shape[0]:
        raise ValueError("Logits and target batch sizes do not match.")

    if logits.shape[2:] != target.shape[1:]:
        raise ValueError("Logits and target spatial dimensions do not match.")

    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} logit channels, found {logits.shape[1]}."
        )

    if not logits.is_floating_point():
        raise TypeError("Logits must use a floating-point dtype.")

    if target.dtype != torch.int64:
        raise TypeError(
            f"Target must use torch.int64, found {target.dtype}."
        )

    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain non-finite values.")


def _prepare_class_weights(
    class_weights: Tensor | Sequence[float] | None,
    *,
    num_classes: int,
) -> Tensor | None:
    """Validate and convert optional class weights."""
    if class_weights is None:
        return None

    weights = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
    )

    if weights.ndim != 1 or weights.numel() != num_classes:
        raise ValueError(
            f"class_weights must contain {num_classes} values."
        )

    if not torch.isfinite(weights).all():
        raise ValueError("class_weights contain non-finite values.")

    if torch.any(weights <= 0):
        raise ValueError("Every class weight must be positive.")

    return weights


class MulticlassDiceLoss(nn.Module):
    """Soft Dice loss for multiclass semantic segmentation.

    Dice is calculated over all valid pixels in the batch. Ignored pixels are
    removed before intersection and denominator calculations.

    By default, classes absent from both target and prediction support are
    excluded from the mean. This avoids rewarding a model for trivially
    predicting no pixels of a class that is absent from the current batch.
    """

    def __init__(
        self,
        *,
        num_classes: int = 5,
        ignore_index: int = 255,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
        class_weights: Tensor | Sequence[float] | None = None,
        include_absent_classes: bool = False,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        if smooth < 0:
            raise ValueError("smooth cannot be negative.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        weights = _prepare_class_weights(
            class_weights,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = float(smooth)
        self.epsilon = float(epsilon)
        self.include_absent_classes = include_absent_classes

        if weights is None:
            self.register_buffer(
                "class_weights",
                None,
            )
        else:
            self.register_buffer(
                "class_weights",
                weights,
            )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Calculate multiclass soft Dice loss."""
        _validate_inputs(
            logits,
            target,
            num_classes=self.num_classes,
        )

        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            raise ValueError(
                "Dice loss received a batch with no supervised pixels."
            )

        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = F.one_hot(
            safe_target,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(
            dtype=probabilities.dtype
        )

        valid_mask_expanded = valid_mask.unsqueeze(1).to(
            dtype=probabilities.dtype
        )

        probabilities = probabilities * valid_mask_expanded
        one_hot = one_hot * valid_mask_expanded

        reduction_dims = (0, 2, 3)

        intersection = torch.sum(
            probabilities * one_hot,
            dim=reduction_dims,
        )

        probability_sum = torch.sum(
            probabilities,
            dim=reduction_dims,
        )

        target_sum = torch.sum(
            one_hot,
            dim=reduction_dims,
        )

        denominator = probability_sum + target_sum

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth + self.epsilon
        )

        class_is_present = target_sum > 0

        if self.include_absent_classes:
            active_classes = torch.ones_like(
                class_is_present,
                dtype=torch.bool,
            )
        else:
            active_classes = class_is_present

        if not active_classes.any():
            raise ValueError(
                "No active classes are available for Dice calculation."
            )

        class_losses = 1.0 - dice_score

        if self.class_weights is None:
            return class_losses[active_classes].mean()

        active_weights = self.class_weights[
            active_classes
        ].to(
            device=logits.device,
            dtype=logits.dtype,
        )

        active_weights = active_weights / active_weights.sum()

        return torch.sum(
            class_losses[active_classes] * active_weights
        )


class CombinedCrossEntropyDiceLoss(nn.Module):
    """Weighted combination of Cross-Entropy and soft Dice loss."""

    def __init__(
        self,
        *,
        num_classes: int = 5,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0,
        dice_epsilon: float = 1e-7,
        class_weights: Tensor | Sequence[float] | None = None,
        include_absent_classes: bool = False,
    ) -> None:
        super().__init__()

        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights cannot be negative.")

        total_weight = ce_weight + dice_weight

        if total_weight <= 0:
            raise ValueError(
                "At least one combined-loss weight must be positive."
            )

        weights = _prepare_class_weights(
            class_weights,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = float(ce_weight / total_weight)
        self.dice_weight = float(dice_weight / total_weight)

        if weights is None:
            self.register_buffer(
                "class_weights",
                None,
            )
        else:
            self.register_buffer(
                "class_weights",
                weights,
            )

        self.dice_loss = MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=dice_smooth,
            epsilon=dice_epsilon,
            class_weights=weights,
            include_absent_classes=include_absent_classes,
        )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Calculate weighted CE + Dice."""
        _validate_inputs(
            logits,
            target,
            num_classes=self.num_classes,
        )

        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            raise ValueError(
                "Combined loss received a batch with no supervised pixels."
            )

        ce = F.cross_entropy(
            logits,
            target,
            weight=(
                self.class_weights.to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
                if self.class_weights is not None
                else None
            ),
            ignore_index=self.ignore_index,
        )

        dice = self.dice_loss(
            logits,
            target,
        )

        return (
            self.ce_weight * ce
            + self.dice_weight * dice
        )


def build_loss(
    config: LossConfig | None = None,
    *,
    class_weights: Tensor | Sequence[float] | None = None,
) -> nn.Module:
    """Build a loss module from configuration."""
    resolved = (
        LossConfig()
        if config is None
        else config
    )
    resolved.validate()

    if resolved.name == "cross_entropy":
        weights = _prepare_class_weights(
            class_weights,
            num_classes=resolved.num_classes,
        )

        return nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=resolved.ignore_index,
        )

    if resolved.name == "dice":
        return MulticlassDiceLoss(
            num_classes=resolved.num_classes,
            ignore_index=resolved.ignore_index,
            smooth=resolved.dice_smooth,
            epsilon=resolved.dice_epsilon,
            class_weights=class_weights,
            include_absent_classes=(
                resolved.include_absent_classes
            ),
        )

    if resolved.name == "ce_dice":
        return CombinedCrossEntropyDiceLoss(
            num_classes=resolved.num_classes,
            ignore_index=resolved.ignore_index,
            ce_weight=resolved.ce_weight,
            dice_weight=resolved.dice_weight,
            dice_smooth=resolved.dice_smooth,
            dice_epsilon=resolved.dice_epsilon,
            class_weights=class_weights,
            include_absent_classes=(
                resolved.include_absent_classes
            ),
        )

    if resolved.name == "tversky":
        return MulticlassTverskyLoss(
            num_classes=resolved.num_classes,
            ignore_index=resolved.ignore_index,
            alpha=resolved.tversky_alpha,
            beta=resolved.tversky_beta,
            smooth=resolved.tversky_smooth,
            epsilon=resolved.tversky_epsilon,
            class_weights=class_weights,
            include_absent_classes=(
                resolved.include_absent_classes
            ),
        )

    if resolved.name == "ce_tversky":
        return CombinedCrossEntropyTverskyLoss(
            num_classes=resolved.num_classes,
            ignore_index=resolved.ignore_index,
            ce_weight=resolved.ce_weight,
            tversky_weight=resolved.tversky_weight,
            tversky_alpha=resolved.tversky_alpha,
            tversky_beta=resolved.tversky_beta,
            tversky_smooth=resolved.tversky_smooth,
            tversky_epsilon=resolved.tversky_epsilon,
            class_weights=class_weights,
            include_absent_classes=(
                resolved.include_absent_classes
            ),
        )

    raise ValueError(f"Unsupported loss configuration: {resolved.name}")

class MulticlassTverskyLoss(nn.Module):
    """Soft Tversky loss for multiclass semantic segmentation."""

    def __init__(
        self,
        *,
        num_classes: int = 5,
        ignore_index: int = 255,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
        class_weights: Tensor | Sequence[float] | None = None,
        include_absent_classes: bool = False,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than one."
            )

        if alpha < 0 or beta < 0:
            raise ValueError(
                "alpha and beta cannot be negative."
            )

        if alpha + beta <= 0:
            raise ValueError(
                "At least one of alpha or beta must be positive."
            )

        if smooth < 0:
            raise ValueError("smooth cannot be negative.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        weights = _prepare_class_weights(
            class_weights,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.epsilon = float(epsilon)
        self.include_absent_classes = include_absent_classes

        if weights is None:
            self.register_buffer(
                "class_weights",
                None,
            )
        else:
            self.register_buffer(
                "class_weights",
                weights,
            )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Calculate multiclass Tversky loss."""
        _validate_inputs(
            logits,
            target,
            num_classes=self.num_classes,
        )

        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            raise ValueError(
                "Tversky loss received a batch with no supervised pixels."
            )

        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = F.one_hot(
            safe_target,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(
            dtype=probabilities.dtype
        )

        valid_mask_expanded = valid_mask.unsqueeze(1).to(
            dtype=probabilities.dtype
        )

        probabilities = probabilities * valid_mask_expanded
        one_hot = one_hot * valid_mask_expanded

        reduction_dims = (0, 2, 3)

        true_positive = torch.sum(
            probabilities * one_hot,
            dim=reduction_dims,
        )

        false_positive = torch.sum(
            probabilities * (1.0 - one_hot),
            dim=reduction_dims,
        )

        false_negative = torch.sum(
            (1.0 - probabilities) * one_hot,
            dim=reduction_dims,
        )

        target_sum = torch.sum(
            one_hot,
            dim=reduction_dims,
        )

        tversky_score = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
            + self.epsilon
        )

        class_is_present = target_sum > 0

        if self.include_absent_classes:
            active_classes = torch.ones_like(
                class_is_present,
                dtype=torch.bool,
            )
        else:
            active_classes = class_is_present

        if not active_classes.any():
            raise ValueError(
                "No active classes are available "
                "for Tversky calculation."
            )

        class_losses = 1.0 - tversky_score

        if self.class_weights is None:
            return class_losses[active_classes].mean()

        weights = self.class_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        active_weights = weights[
            active_classes
        ]

        active_weights = (
            active_weights / active_weights.sum()
        )

        return torch.sum(
            class_losses[active_classes]
            * active_weights
        )

class CombinedCrossEntropyTverskyLoss(nn.Module):
    """Weighted combination of Cross-Entropy and Tversky loss."""

    def __init__(
        self,
        *,
        num_classes: int = 5,
        ignore_index: int = 255,
        ce_weight: float = 0.3,
        tversky_weight: float = 0.7,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_smooth: float = 1.0,
        tversky_epsilon: float = 1e-7,
        class_weights: Tensor | Sequence[float] | None = None,
        include_absent_classes: bool = False,
    ) -> None:
        super().__init__()

        if ce_weight < 0 or tversky_weight < 0:
            raise ValueError(
                "Loss weights cannot be negative."
            )

        total_weight = ce_weight + tversky_weight

        if total_weight <= 0:
            raise ValueError(
                "At least one combined-loss weight "
                "must be positive."
            )

        weights = _prepare_class_weights(
            class_weights,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.ce_weight = float(
            ce_weight / total_weight
        )
        self.tversky_weight = float(
            tversky_weight / total_weight
        )

        if weights is None:
            self.register_buffer(
                "class_weights",
                None,
            )
        else:
            self.register_buffer(
                "class_weights",
                weights,
            )

        self.tversky_loss = MulticlassTverskyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=tversky_smooth,
            epsilon=tversky_epsilon,
            class_weights=weights,
            include_absent_classes=include_absent_classes,
        )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Calculate weighted CE + Tversky."""
        _validate_inputs(
            logits,
            target,
            num_classes=self.num_classes,
        )

        valid_mask = target != self.ignore_index

        if not valid_mask.any():
            raise ValueError(
                "Combined CE-Tversky loss received "
                "a batch with no supervised pixels."
            )

        ce = F.cross_entropy(
            logits,
            target,
            weight=(
                self.class_weights.to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
                if self.class_weights is not None
                else None
            ),
            ignore_index=self.ignore_index,
        )

        tversky = self.tversky_loss(
            logits,
            target,
        )

        return (
            self.ce_weight * ce
            + self.tversky_weight * tversky
        )

def ce_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    num_classes: int = 5,
    ignore_index: int = 255,
    ce_weight: float = 0.30,
    tversky_weight: float = 0.70,
    tversky_alpha: float = 0.30,
    tversky_beta: float = 0.70,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Weighted Cross-Entropy + multiclass Tversky loss.
    """

    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        ignore_index=ignore_index,
    )

    tversky = multiclass_tversky_loss(
        logits,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        alpha=tversky_alpha,
        beta=tversky_beta,
        smooth=smooth,
    )

    return (
        ce_weight * ce
        + tversky_weight * tversky
    )

def smoke_test(
    seed: int = 20260725,
) -> dict[str, Any]:
    """Run numerical and gradient checks."""
    torch.manual_seed(seed)

    batch_size = 2
    num_classes = 5
    height = 32
    width = 32
    ignore_index = 255

    logits = torch.randn(
        batch_size,
        num_classes,
        height,
        width,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size, height, width),
        dtype=torch.int64,
    )

    target[:, :4, :] = ignore_index
    target[:, :, :2] = ignore_index

    valid_pixel_count = int(
        (target != ignore_index).sum().item()
    )

    if valid_pixel_count == 0:
        raise RuntimeError("Synthetic smoke-test target is invalid.")

    class_weights = torch.tensor(
        [1.5, 1.8, 0.5, 2.5, 2.0],
        dtype=torch.float32,
    )

    losses: dict[str, float] = {}
    gradient_checks: dict[str, bool] = {}

    modules = {
        "cross_entropy": build_loss(
            LossConfig(
                name="cross_entropy",
            ),
            class_weights=class_weights,
        ),

        "dice": build_loss(
            LossConfig(
                name="dice",
            ),
            class_weights=class_weights,
        ),

        "ce_dice": build_loss(
            LossConfig(
                name="ce_dice",
            ),
            class_weights=class_weights,
        ),

        "tversky": build_loss(
            LossConfig(
                name="tversky",
                tversky_alpha=0.3,
                tversky_beta=0.7,
            ),
            class_weights=class_weights,
        ),

        "ce_tversky": build_loss(
            LossConfig(
                name="ce_tversky",
                ce_weight=0.3,
                tversky_weight=0.7,
                tversky_alpha=0.3,
                tversky_beta=0.7,
            ),
            class_weights=class_weights,
        ),
    }

    for name, module in modules.items():
        test_logits = logits.detach().clone().requires_grad_(True)

        value = module(
            test_logits,
            target,
        )

        if value.ndim != 0:
            raise RuntimeError(
                f"{name} did not return a scalar."
            )

        if not torch.isfinite(value):
            raise RuntimeError(
                f"{name} returned a non-finite loss."
            )

        value.backward()

        gradient_finite = bool(
            test_logits.grad is not None
            and torch.isfinite(
                test_logits.grad
            ).all().item()
        )

        gradient_nonzero = bool(
            test_logits.grad is not None
            and torch.any(
                test_logits.grad != 0
            ).item()
        )

        if not gradient_finite:
            raise RuntimeError(
                f"{name} produced non-finite gradients."
            )

        if not gradient_nonzero:
            raise RuntimeError(
                f"{name} produced zero gradients."
            )

        losses[name] = float(value.detach().item())
        gradient_checks[name] = (
            gradient_finite and gradient_nonzero
        )

    perfect_target = torch.randint(
        low=0,
        high=num_classes,
        size=(1, height, width),
        dtype=torch.int64,
    )

    perfect_logits = torch.full(
        (1, num_classes, height, width),
        fill_value=-10.0,
        dtype=torch.float32,
    )

    perfect_logits.scatter_(
        1,
        perfect_target.unsqueeze(1),
        10.0,
    )

    perfect_dice_loss = MulticlassDiceLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )(
        perfect_logits,
        perfect_target,
    )

    if float(perfect_dice_loss.item()) > 1e-4:
        raise RuntimeError(
            "Dice loss is unexpectedly high for perfect predictions."
        )

    perfect_tversky_loss = MulticlassTverskyLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        alpha=0.3,
        beta=0.7,
    )(
        perfect_logits,
        perfect_target,
    )

    if float(perfect_tversky_loss.item()) > 1e-4:
        raise RuntimeError(
            "Tversky loss is unexpectedly high "
            "for perfect predictions."
        )

    return {
        "seed": seed,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "valid_pixel_count": valid_pixel_count,
        "class_weights": class_weights.tolist(),
        "loss_values": losses,
        "gradient_checks": gradient_checks,
        "perfect_prediction_dice_loss": float(
            perfect_dice_loss.item()
        ),
        "perfect_prediction_tversky_loss": float(
            perfect_tversky_loss.item()
        ),
        "all_losses_finite": all(
            torch.isfinite(
                torch.tensor(value)
            ).item()
            for value in losses.values()
        ),
        "all_gradient_checks_passed": all(
            gradient_checks.values()
        ),
        "recommended_default": asdict(
            LossConfig()
        ),
        "all_checks_passed": True,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run segmentation-loss numerical and gradient checks."
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
            "loss_smoke_test.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run and save loss smoke-test results."""
    args = parse_arguments()

    report = smoke_test(
        seed=args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSegmentation loss smoke test")
    print("----------------------------")
    print(
        f"Valid pixels: "
        f"{report['valid_pixel_count']:,}"
    )
    print(
        f"Class weights: "
        f"{report['class_weights']}"
    )

    for name, value in report[
        "loss_values"
    ].items():
        print(f"{name}: {value:.6f}")

    print(
        "Perfect-prediction Dice loss: "
        f"{report['perfect_prediction_dice_loss']:.10f}"
    )
    print(
        "Perfect-prediction Tversky loss: "
        f"{report['perfect_prediction_tversky_loss']:.10f}"
    )
    print(
        "All losses finite: "
        f"{report['all_losses_finite']}"
    )
    print(
        "All gradient checks passed: "
        f"{report['all_gradient_checks_passed']}"
    )
    print(f"Report: {args.output}")
    print(
        "\nResult: segmentation losses passed "
        "numerical and gradient validation."
    )


if __name__ == "__main__":
    main()
