"""Atrous Spatial Pyramid Pooling used by DeepLabV3+."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.normalization import build_norm_layer


class ConvNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        dropout: float = 0.0,
        normalization: str = "batchnorm",
        group_norm_groups: int = 32,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            build_norm_layer(
                out_channels,
                normalization=normalization,
                group_norm_groups=group_norm_groups,
            ),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


# Backward-compatible alias for imports in existing code.
ConvBNReLU = ConvNormReLU


class ASPP(nn.Module):
    """Multi-scale context block with image-level pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        *,
        atrous_rates: tuple[int, int, int] = (6, 12, 18),
        dropout: float = 0.1,
        normalization: str = "batchnorm",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(atrous_rates) != 3 or any(rate <= 0 for rate in atrous_rates):
            raise ValueError("atrous_rates must contain three positive integers.")

        block_kwargs = {
            "normalization": normalization,
            "group_norm_groups": group_norm_groups,
        }
        self.branch1 = ConvNormReLU(
            in_channels, out_channels, kernel_size=1, **block_kwargs
        )
        self.atrous_branches = nn.ModuleList(
            ConvNormReLU(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                **block_kwargs,
            )
            for rate in atrous_rates
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            build_norm_layer(
                out_channels,
                normalization="groupnorm",  # safe for the 1x1 pooled branch
                group_norm_groups=group_norm_groups,
            ),
            nn.ReLU(inplace=True),
        )
        concatenated_channels = out_channels * (2 + len(atrous_rates))
        self.project = ConvNormReLU(
            concatenated_channels,
            out_channels,
            kernel_size=1,
            dropout=dropout,
            **block_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        features = [self.branch1(x)]
        features.extend(branch(x) for branch in self.atrous_branches)
        pooled = F.interpolate(
            self.image_pool(x), size=size, mode="bilinear", align_corners=False
        )
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))
