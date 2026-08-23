"""DeepLabV3+ for dual-polarization Sentinel-1 semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.aspp import ASPP, ConvBNReLU
from src.models.backbones.encoder_factory import build_encoder
from src.models.base import BaseSegmentationModel


@dataclass(frozen=True)
class DeepLabV3PlusConfig:
    input_channels: int = 2
    num_classes: int = 5
    backbone: str = "resnet18"
    encoder_base_channels: int = 64
    output_stride: int = 16
    aspp_channels: int = 256
    low_level_channels: int = 48
    decoder_channels: int = 256
    dropout: float = 0.1
    atrous_rates: tuple[int, int, int] = (6, 12, 18)
    normalization: str = "groupnorm"
    group_norm_groups: int = 32

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.encoder_base_channels <= 0:
            raise ValueError("encoder_base_channels must be positive.")
        if self.output_stride not in {8, 16}:
            raise ValueError("output_stride must be 8 or 16.")
        for value, name in (
            (self.aspp_channels, "aspp_channels"),
            (self.low_level_channels, "low_level_channels"),
            (self.decoder_channels, "decoder_channels"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class DeepLabV3Plus(BaseSegmentationModel):
    """DeepLabV3+ with a configurable dependency-free ResNet encoder."""

    model_name = "deeplabv3plus"
    display_name = "DeepLabV3+"

    def __init__(self, config: DeepLabV3PlusConfig | None = None) -> None:
        self.config = config or DeepLabV3PlusConfig()
        self.config.validate()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )

        self.encoder = build_encoder(
            name=self.config.backbone,
            input_channels=self.input_channels,
            base_channels=self.config.encoder_base_channels,
            output_stride=self.config.output_stride,
            normalization=self.config.normalization,
            group_norm_groups=self.config.group_norm_groups,
        )
        encoder_low_channels = int(self.encoder.low_level_channels)
        encoder_high_channels = int(self.encoder.high_level_channels)

        self.aspp = ASPP(
            encoder_high_channels,
            self.config.aspp_channels,
            atrous_rates=self.config.atrous_rates,
            dropout=self.config.dropout,
            normalization=self.config.normalization,
            group_norm_groups=self.config.group_norm_groups,
        )
        self.low_level_projection = ConvBNReLU(
            encoder_low_channels,
            self.config.low_level_channels,
            kernel_size=1,
            normalization=self.config.normalization,
            group_norm_groups=self.config.group_norm_groups,
        )
        decoder_input_channels = (
            self.config.aspp_channels + self.config.low_level_channels
        )
        self.decoder = nn.Sequential(
            ConvBNReLU(
                decoder_input_channels,
                self.config.decoder_channels,
                kernel_size=3,
                padding=1,
                dropout=self.config.dropout,
                normalization=self.config.normalization,
                group_norm_groups=self.config.group_norm_groups,
            ),
            ConvBNReLU(
                self.config.decoder_channels,
                self.config.decoder_channels,
                kernel_size=3,
                padding=1,
                dropout=self.config.dropout,
                normalization=self.config.normalization,
                group_norm_groups=self.config.group_norm_groups,
            ),
        )
        self.classifier = nn.Conv2d(
            self.config.decoder_channels,
            self.num_classes,
            kernel_size=1,
        )
        self.apply(self.initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        original_size = self.validate_input(x)
        low_level, high_level = self.encoder(x)
        context = self.aspp(high_level)
        context = F.interpolate(
            context,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_level = self.low_level_projection(low_level)
        decoded = self.decoder(torch.cat([context, low_level], dim=1))
        logits = self.classifier(decoded)
        return F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )


def build_deeplabv3_plus(
    input_channels: int = 2,
    num_classes: int = 5,
    backbone: str = "resnet18",
    encoder_base_channels: int = 64,
    output_stride: int = 16,
    aspp_channels: int = 256,
    low_level_channels: int = 48,
    decoder_channels: int = 256,
    dropout: float = 0.1,
    atrous_rates: tuple[int, int, int] = (6, 12, 18),
    normalization: str = "groupnorm",
    group_norm_groups: int = 32,
) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        DeepLabV3PlusConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            backbone=backbone,
            encoder_base_channels=encoder_base_channels,
            output_stride=output_stride,
            aspp_channels=aspp_channels,
            low_level_channels=low_level_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            atrous_rates=atrous_rates,
            normalization=normalization,
            group_norm_groups=group_norm_groups,
        )
    )
