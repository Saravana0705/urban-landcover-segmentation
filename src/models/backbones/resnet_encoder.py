"""Dependency-free ResNet encoders for DeepLabV3+ feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn

from src.models.normalization import build_norm_layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        normalization: str = "batchnorm",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        norm = lambda channels: build_norm_layer(
            channels,
            normalization=normalization,
            group_norm_groups=group_norm_groups,
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.norm1 = norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.norm2 = norm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


@dataclass(frozen=True)
class ResNetEncoderSpec:
    name: str
    layers: tuple[int, int, int, int]


RESNET_SPECS = {
    "resnet18": ResNetEncoderSpec("resnet18", (2, 2, 2, 2)),
    "resnet34": ResNetEncoderSpec("resnet34", (3, 4, 6, 3)),
}


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        layers: Sequence[int],
        base_channels: int = 64,
        output_stride: int = 16,
        normalization: str = "batchnorm",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("input_channels and base_channels must be positive.")
        if output_stride not in {8, 16}:
            raise ValueError("output_stride must be 8 or 16.")
        if len(layers) != 4:
            raise ValueError("layers must contain four stage depths.")
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.output_stride = int(output_stride)
        self.normalization = normalization
        self.group_norm_groups = int(group_norm_groups)
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3, bias=False),
            build_norm_layer(
                base_channels,
                normalization=normalization,
                group_norm_groups=group_norm_groups,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2, dilation=1)
        if output_stride == 16:
            self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=1, dilation=2)
        else:
            self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=1, dilation=4)
        self.low_level_channels = base_channels
        self.high_level_channels = base_channels * 8

    def _make_layer(self, out_channels: int, block_count: int, *, stride: int, dilation: int) -> nn.Sequential:
        kwargs = {
            "normalization": self.normalization,
            "group_norm_groups": self.group_norm_groups,
        }
        blocks = [BasicBlock(self.in_channels, out_channels, stride=stride, dilation=dilation, **kwargs)]
        self.in_channels = out_channels
        blocks.extend(BasicBlock(self.in_channels, out_channels, dilation=dilation, **kwargs) for _ in range(1, block_count))
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        return low_level, self.layer4(x)


def build_resnet_encoder(
    *,
    name: str,
    input_channels: int,
    base_channels: int = 64,
    output_stride: int = 16,
    normalization: str = "batchnorm",
    group_norm_groups: int = 32,
) -> ResNetEncoder:
    normalized = name.lower().strip().replace("-", "")
    aliases = {"resnet18": "resnet18", "resnet_18": "resnet18", "resnet34": "resnet34", "resnet_34": "resnet34"}
    canonical = aliases.get(normalized)
    if canonical is None:
        raise ValueError(f"Unsupported ResNet encoder '{name}'. Supported: resnet18, resnet34.")
    return ResNetEncoder(
        input_channels=input_channels,
        layers=RESNET_SPECS[canonical].layers,
        base_channels=base_channels,
        output_stride=output_stride,
        normalization=normalization,
        group_norm_groups=group_norm_groups,
    )
