"""Common interface and utilities for semantic-segmentation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class BaseSegmentationModel(nn.Module, ABC):
    """Small common contract used by all segmentation architectures.

    Subclasses must return raw logits shaped ``(N, num_classes, H, W)``.
    The trainer deliberately remains architecture-agnostic and does not apply
    softmax before the loss or metric computation.
    """

    model_name: str = "segmentation_model"
    display_name: str = "Segmentation Model"

    def __init__(self, *, input_channels: int, num_classes: int) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)

    def validate_input(self, x: Tensor) -> tuple[int, int]:
        """Validate a model input and return its spatial size."""
        if x.ndim != 4:
            raise ValueError("Expected input shaped (N,C,H,W).")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, found {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError(f"{self.display_name} input must be floating point.")
        if not torch.isfinite(x).all():
            raise ValueError(f"{self.display_name} input contains non-finite values.")
        return int(x.shape[-2]), int(x.shape[-1])

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """Kaiming initialization shared by CNN-based architectures."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def model_metadata(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "model_name": self.model_name,
            "model_display_name": self.display_name,
            "model_class": type(self).__name__,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
        }

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Return raw per-class logits at the input spatial resolution."""
        raise NotImplementedError
