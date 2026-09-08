"""U-Net++ with nested dense skip pathways for Sentinel-1 segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.base import BaseSegmentationModel
from src.models.unet import DoubleConv


@dataclass(frozen=True)
class UNetPlusPlusConfig:
    input_channels: int = 2
    num_classes: int = 5
    base_channels: int = 32
    dropout: float = 0.1
    use_batch_norm: bool = True
    bilinear_upsampling: bool = False
    deep_supervision: bool = False
    deep_supervision_average_inference: bool = True

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.base_channels <= 0 or self.base_channels % 2:
            raise ValueError("base_channels must be a positive even number.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class NestedUp(nn.Module):
    """Upsample one deeper node to the channel width of the target level."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bilinear_upsampling: bool,
    ) -> None:
        super().__init__()
        if bilinear_upsampling:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )

    def forward(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return x


class UNetPlusPlus(BaseSegmentationModel):
    """Four-level U-Net++ using the original nested skip formulation.

    Each node ``X(d,j)`` concatenates all earlier nodes at depth ``d`` with
    the upsampled node from depth ``d+1``. The final prediction is produced
    from ``X(0,4)`` by default. When deep supervision is enabled, auxiliary
    heads supervise ``X(0,1)`` through ``X(0,4)`` during training. Evaluation
    still exposes one tensor, optionally averaging the four prediction heads.
    """

    model_name = "unetpp"
    display_name = "U-Net++"

    def __init__(self, config: UNetPlusPlusConfig | None = None) -> None:
        self.config = config or UNetPlusPlusConfig()
        self.config.validate()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )

        c = self.config.base_channels
        channels = [c, c * 2, c * 4, c * 8, c * 16]
        conv_kwargs = {
            "dropout": self.config.dropout,
            "use_batch_norm": self.config.use_batch_norm,
        }

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.x0_0 = DoubleConv(self.input_channels, channels[0], **conv_kwargs)
        self.x1_0 = DoubleConv(channels[0], channels[1], **conv_kwargs)
        self.x2_0 = DoubleConv(channels[1], channels[2], **conv_kwargs)
        self.x3_0 = DoubleConv(channels[2], channels[3], **conv_kwargs)
        self.x4_0 = DoubleConv(channels[3], channels[4], **conv_kwargs)

        self.up1_0 = NestedUp(channels[1], channels[0], bilinear_upsampling=self.config.bilinear_upsampling)
        self.up2_0 = NestedUp(channels[2], channels[1], bilinear_upsampling=self.config.bilinear_upsampling)
        self.up3_0 = NestedUp(channels[3], channels[2], bilinear_upsampling=self.config.bilinear_upsampling)
        self.up4_0 = NestedUp(channels[4], channels[3], bilinear_upsampling=self.config.bilinear_upsampling)

        self.x0_1 = DoubleConv(channels[0] * 2, channels[0], **conv_kwargs)
        self.x1_1 = DoubleConv(channels[1] * 2, channels[1], **conv_kwargs)
        self.x2_1 = DoubleConv(channels[2] * 2, channels[2], **conv_kwargs)
        self.x3_1 = DoubleConv(channels[3] * 2, channels[3], **conv_kwargs)

        self.up1_1 = NestedUp(channels[1], channels[0], bilinear_upsampling=self.config.bilinear_upsampling)
        self.up2_1 = NestedUp(channels[2], channels[1], bilinear_upsampling=self.config.bilinear_upsampling)
        self.up3_1 = NestedUp(channels[3], channels[2], bilinear_upsampling=self.config.bilinear_upsampling)

        self.x0_2 = DoubleConv(channels[0] * 3, channels[0], **conv_kwargs)
        self.x1_2 = DoubleConv(channels[1] * 3, channels[1], **conv_kwargs)
        self.x2_2 = DoubleConv(channels[2] * 3, channels[2], **conv_kwargs)

        self.up1_2 = NestedUp(channels[1], channels[0], bilinear_upsampling=self.config.bilinear_upsampling)
        self.up2_2 = NestedUp(channels[2], channels[1], bilinear_upsampling=self.config.bilinear_upsampling)

        self.x0_3 = DoubleConv(channels[0] * 4, channels[0], **conv_kwargs)
        self.x1_3 = DoubleConv(channels[1] * 4, channels[1], **conv_kwargs)

        self.up1_3 = NestedUp(channels[1], channels[0], bilinear_upsampling=self.config.bilinear_upsampling)
        self.x0_4 = DoubleConv(channels[0] * 5, channels[0], **conv_kwargs)

        self.classifier = nn.Conv2d(channels[0], self.num_classes, kernel_size=1)
        self.auxiliary_classifiers = nn.ModuleList(
            nn.Conv2d(channels[0], self.num_classes, kernel_size=1)
            for _ in range(3)
        ) if self.config.deep_supervision else nn.ModuleList()
        self.apply(self.initialize_weights)

    @staticmethod
    def _cat(nodes: list[Tensor], upsampled: Tensor) -> Tensor:
        return torch.cat([*nodes, upsampled], dim=1)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        original_size = self.validate_input(x)

        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))

        x0_1 = self.x0_1(self._cat([x0_0], self.up1_0(x1_0, x0_0.shape[-2:])))
        x1_1 = self.x1_1(self._cat([x1_0], self.up2_0(x2_0, x1_0.shape[-2:])))
        x2_1 = self.x2_1(self._cat([x2_0], self.up3_0(x3_0, x2_0.shape[-2:])))
        x3_1 = self.x3_1(self._cat([x3_0], self.up4_0(x4_0, x3_0.shape[-2:])))

        x0_2 = self.x0_2(self._cat([x0_0, x0_1], self.up1_1(x1_1, x0_0.shape[-2:])))
        x1_2 = self.x1_2(self._cat([x1_0, x1_1], self.up2_1(x2_1, x1_0.shape[-2:])))
        x2_2 = self.x2_2(self._cat([x2_0, x2_1], self.up3_1(x3_1, x2_0.shape[-2:])))

        x0_3 = self.x0_3(self._cat([x0_0, x0_1, x0_2], self.up1_2(x1_2, x0_0.shape[-2:])))
        x1_3 = self.x1_3(self._cat([x1_0, x1_1, x1_2], self.up2_2(x2_2, x1_0.shape[-2:])))

        x0_4 = self.x0_4(self._cat([x0_0, x0_1, x0_2, x0_3], self.up1_3(x1_3, x0_0.shape[-2:])))
        final_logits = self.classifier(x0_4)
        if not self.config.deep_supervision:
            return self._resize_logits(final_logits, original_size)

        logits = [
            self.auxiliary_classifiers[0](x0_1),
            self.auxiliary_classifiers[1](x0_2),
            self.auxiliary_classifiers[2](x0_3),
            final_logits,
        ]
        logits = [self._resize_logits(item, original_size) for item in logits]

        # The training loss consumes all four heads. Evaluation remains
        # architecture-independent and receives one fused logit tensor.
        if self.training:
            return tuple(logits)
        if self.config.deep_supervision_average_inference:
            mean_probability = torch.stack(
                [torch.softmax(item, dim=1) for item in logits],
                dim=0,
            ).mean(dim=0)
            # Log-probabilities preserve the common raw-logit interface:
            # softmax(returned_value) exactly recovers the averaged maps.
            return mean_probability.clamp_min(1e-7).log()
        return logits[-1]

    @staticmethod
    def _resize_logits(logits: Tensor, size: tuple[int, int]) -> Tensor:
        if logits.shape[-2:] == size:
            return logits
        return F.interpolate(
            logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


def build_unet_plus_plus(
    input_channels: int = 2,
    num_classes: int = 5,
    base_channels: int = 32,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    bilinear_upsampling: bool = False,
    deep_supervision: bool = False,
    deep_supervision_average_inference: bool = True,
) -> UNetPlusPlus:
    return UNetPlusPlus(
        UNetPlusPlusConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            bilinear_upsampling=bilinear_upsampling,
            deep_supervision=deep_supervision,
            deep_supervision_average_inference=deep_supervision_average_inference,
        )
    )
