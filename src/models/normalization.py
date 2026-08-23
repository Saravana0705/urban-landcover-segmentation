"""Normalization helpers for small-batch segmentation models."""

from __future__ import annotations

from torch import nn


def build_norm_layer(
    num_channels: int,
    *,
    normalization: str = "batchnorm",
    group_norm_groups: int = 32,
) -> nn.Module:
    """Construct a 2-D normalization layer.

    GroupNorm is preferred for CPU/small-batch DeepLab experiments because it
    does not depend on batch-level running statistics.
    """
    name = normalization.lower().strip().replace("_", "")
    if name in {"batchnorm", "batchnorm2d", "bn"}:
        return nn.BatchNorm2d(num_channels)
    if name in {"groupnorm", "gn"}:
        groups = min(int(group_norm_groups), int(num_channels))
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(
        f"Unsupported normalization '{normalization}'. "
        "Supported: batchnorm, groupnorm."
    )
