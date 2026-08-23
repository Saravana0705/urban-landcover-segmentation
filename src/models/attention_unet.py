"""Attention U-Net for five-class Sentinel-1 semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.base import BaseSegmentationModel
from src.models.unet import DoubleConv, DownBlock


@dataclass(frozen=True)
class AttentionUNetConfig:
    input_channels: int = 2
    num_classes: int = 5
    base_channels: int = 32
    dropout: float = 0.1
    use_batch_norm: bool = True
    bilinear_upsampling: bool = False

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.base_channels <= 0 or self.base_channels % 2:
            raise ValueError("base_channels must be a positive even number.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class AttentionGate(nn.Module):
    """Gate an encoder skip feature using the decoder context feature."""

    def __init__(
        self,
        gating_channels: int,
        skip_channels: int,
        intermediate_channels: int,
        *,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        if intermediate_channels <= 0:
            raise ValueError("intermediate_channels must be positive.")

        def projection(in_channels: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    intermediate_channels,
                    kernel_size=1,
                    bias=not use_batch_norm,
                )
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(intermediate_channels))
            return nn.Sequential(*layers)

        self.gating_projection = projection(gating_channels)
        self.skip_projection = projection(skip_channels)
        self.activation = nn.ReLU(inplace=True)
        self.attention = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gating: Tensor, skip: Tensor) -> Tensor:
        gating_projection = self.gating_projection(gating)
        skip_projection = self.skip_projection(skip)

        if gating_projection.shape[-2:] != skip_projection.shape[-2:]:
            gating_projection = F.interpolate(
                gating_projection,
                size=skip_projection.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        coefficients = self.attention(
            self.activation(gating_projection + skip_projection)
        )
        return skip * coefficients


class AttentionUpBlock(nn.Module):
    """Upsample decoder features and fuse an attention-filtered skip."""

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        dropout: float,
        use_batch_norm: bool,
        bilinear_upsampling: bool,
    ) -> None:
        super().__init__()
        if bilinear_upsampling:
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                ),
                nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(
                decoder_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )

        self.attention = AttentionGate(
            gating_channels=out_channels,
            skip_channels=skip_channels,
            intermediate_channels=max(out_channels // 2, 1),
            use_batch_norm=use_batch_norm,
        )
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, decoder: Tensor, skip: Tensor) -> Tensor:
        decoder = self.up(decoder)
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(
                decoder,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attended_skip = self.attention(decoder, skip)
        return self.conv(torch.cat([attended_skip, decoder], dim=1))


class AttentionUNet(BaseSegmentationModel):
    model_name = "attention_unet"
    display_name = "Attention U-Net"
    """Four-level Attention U-Net with the same interface as the baseline."""

    def __init__(self, config: AttentionUNetConfig | None = None) -> None:
        self.config = config or AttentionUNetConfig()
        self.config.validate()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )

        c = self.config.base_channels
        block_kwargs = {
            "dropout": self.config.dropout,
            "use_batch_norm": self.config.use_batch_norm,
        }

        self.enc1 = DoubleConv(
            self.config.input_channels,
            c,
            **block_kwargs,
        )
        self.enc2 = DownBlock(c, c * 2, **block_kwargs)
        self.enc3 = DownBlock(c * 2, c * 4, **block_kwargs)
        self.enc4 = DownBlock(c * 4, c * 8, **block_kwargs)
        self.bottleneck = DownBlock(c * 8, c * 16, **block_kwargs)

        up_kwargs = {
            **block_kwargs,
            "bilinear_upsampling": self.config.bilinear_upsampling,
        }
        self.dec4 = AttentionUpBlock(c * 16, c * 8, c * 8, **up_kwargs)
        self.dec3 = AttentionUpBlock(c * 8, c * 4, c * 4, **up_kwargs)
        self.dec2 = AttentionUpBlock(c * 4, c * 2, c * 2, **up_kwargs)
        self.dec1 = AttentionUpBlock(c * 2, c, c, **up_kwargs)
        self.classifier = nn.Conv2d(c, self.config.num_classes, kernel_size=1)

        self.apply(self.initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        original_size = self.validate_input(x)
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        decoder = self.bottleneck(skip4)

        decoder = self.dec4(decoder, skip4)
        decoder = self.dec3(decoder, skip3)
        decoder = self.dec2(decoder, skip2)
        decoder = self.dec1(decoder, skip1)
        logits = self.classifier(decoder)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_attention_unet(
    input_channels: int = 2,
    num_classes: int = 5,
    base_channels: int = 32,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    bilinear_upsampling: bool = False,
) -> AttentionUNet:
    return AttentionUNet(
        AttentionUNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            bilinear_upsampling=bilinear_upsampling,
        )
    )
