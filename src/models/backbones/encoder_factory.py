"""Factory for segmentation encoder backbones."""

from __future__ import annotations

from torch import nn

from src.models.backbones.resnet_encoder import build_resnet_encoder


def build_encoder(
    *,
    name: str,
    input_channels: int,
    base_channels: int = 64,
    output_stride: int = 16,
    normalization: str = "batchnorm",
    group_norm_groups: int = 32,
) -> nn.Module:
    normalized = name.lower().strip().replace("-", "")
    if normalized.startswith("resnet"):
        return build_resnet_encoder(
            name=name,
            input_channels=input_channels,
            base_channels=base_channels,
            output_stride=output_stride,
            normalization=normalization,
            group_norm_groups=group_norm_groups,
        )
    raise ValueError(f"Unsupported encoder '{name}'.")
