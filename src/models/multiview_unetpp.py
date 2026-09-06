"""Shared seasonal/orbit view encoder with attention fusion and U-Net++."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.models.base import BaseSegmentationModel
from src.models.unet_plus_plus import build_unet_plus_plus


@dataclass(frozen=True)
class MultiViewUNetPlusPlusConfig:
    input_channels: int = 16
    num_views: int = 8
    channels_per_view: int = 2
    fused_channels: int = 16
    num_classes: int = 5
    base_channels: int = 32
    dropout: float = 0.1
    use_batch_norm: bool = True
    bilinear_upsampling: bool = False

    def validate(self) -> None:
        if self.input_channels != self.num_views * self.channels_per_view:
            raise ValueError("input_channels must equal num_views * channels_per_view.")
        if self.num_views < 2 or self.channels_per_view < 1 or self.fused_channels < 1:
            raise ValueError("Multi-view dimensions must be positive and include >=2 views.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")


class SharedViewEncoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(output_channels, 1, kernel_size=1)

    def forward(self, view: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(view)
        score = self.score(features.mean(dim=(-2, -1), keepdim=True))
        return features, score


class MultiViewUNetPlusPlus(BaseSegmentationModel):
    """Eight VV/VH seasonal-orbit views, shared encoding, learned fusion."""

    model_name = "multiview_unetpp"
    display_name = "Multi-View U-Net++"

    def __init__(self, config: MultiViewUNetPlusPlusConfig | None = None) -> None:
        self.config = config or MultiViewUNetPlusPlusConfig()
        self.config.validate()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )
        self.view_encoder = SharedViewEncoder(
            self.config.channels_per_view, self.config.fused_channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(self.config.fused_channels * 2, self.config.fused_channels, 1, bias=False),
            nn.BatchNorm2d(self.config.fused_channels),
            nn.ReLU(inplace=True),
        )
        self.segmenter = build_unet_plus_plus(
            input_channels=self.config.fused_channels,
            num_classes=self.config.num_classes,
            base_channels=self.config.base_channels,
            dropout=self.config.dropout,
            use_batch_norm=self.config.use_batch_norm,
            bilinear_upsampling=self.config.bilinear_upsampling,
        )
        self.last_attention: Tensor | None = None

    def split_views(self, x: Tensor) -> tuple[Tensor, ...]:
        self.validate_input(x)
        return tuple(torch.split(x, self.config.channels_per_view, dim=1))

    def forward(self, x: Tensor) -> Tensor:
        encoded = [self.view_encoder(view) for view in self.split_views(x)]
        features = torch.stack([item[0] for item in encoded], dim=1)
        scores = torch.cat([item[1] for item in encoded], dim=1)
        attention = torch.softmax(scores, dim=1)
        self.last_attention = attention.detach()
        weighted = (features * attention.unsqueeze(2)).sum(dim=1)
        maximum = features.max(dim=1).values
        return self.segmenter(self.fusion(torch.cat([weighted, maximum], dim=1)))


def build_multiview_unetpp(**kwargs: Any) -> MultiViewUNetPlusPlus:
    return MultiViewUNetPlusPlus(MultiViewUNetPlusPlusConfig(**kwargs))


def smoke_test() -> dict[str, Any]:
    model = build_multiview_unetpp()
    x = torch.randn(1, 16, 65, 67, requires_grad=True)
    y = model(x)
    y.mean().backward()
    attention = model.last_attention
    result = {
        "config": asdict(model.config),
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "output_finite": bool(torch.isfinite(y).all()),
        "gradients_exist": any(p.grad is not None for p in model.parameters()),
        "attention_shape": list(attention.shape) if attention is not None else None,
        "attention_sums_to_one": bool(
            attention is not None
            and torch.allclose(attention.sum(dim=1), torch.ones_like(attention[:, 0]), atol=1e-6)
        ),
    }
    result["all_checks_passed"] = all(
        (result["output_shape"] == [1, 5, 65, 67], result["output_finite"],
         result["gradients_exist"], result["attention_sums_to_one"])
    )
    return result


if __name__ == "__main__":
    report = smoke_test()
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)
