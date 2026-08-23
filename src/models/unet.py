"""Baseline U-Net for five-class Sentinel-1 segmentation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.base import BaseSegmentationModel


@dataclass(frozen=True)
class UNetConfig:
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


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=not use_batch_norm,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=not use_batch_norm,
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
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
                nn.Conv2d(decoder_channels, out_channels, 1),
            )
        else:
            self.up = nn.ConvTranspose2d(
                decoder_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(BaseSegmentationModel):
    model_name = "unet"
    display_name = "U-Net"
    def __init__(self, config: UNetConfig | None = None) -> None:
        self.config = config or UNetConfig()
        self.config.validate()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )
        c = self.config.base_channels
        kwargs = {
            "dropout": self.config.dropout,
            "use_batch_norm": self.config.use_batch_norm,
        }
        self.enc1 = DoubleConv(self.config.input_channels, c, **kwargs)
        self.enc2 = DownBlock(c, c * 2, **kwargs)
        self.enc3 = DownBlock(c * 2, c * 4, **kwargs)
        self.enc4 = DownBlock(c * 4, c * 8, **kwargs)
        self.bottleneck = DownBlock(c * 8, c * 16, **kwargs)

        up_kwargs = {
            **kwargs,
            "bilinear_upsampling": self.config.bilinear_upsampling,
        }
        self.dec4 = UpBlock(c * 16, c * 8, c * 8, **up_kwargs)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4, **up_kwargs)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2, **up_kwargs)
        self.dec1 = UpBlock(c * 2, c, c, **up_kwargs)
        self.classifier = nn.Conv2d(c, self.config.num_classes, 1)
        self.apply(self.initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        original_size = self.validate_input(x)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x = self.bottleneck(s4)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        logits = self.classifier(x)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_unet(
    input_channels: int = 2,
    num_classes: int = 5,
    base_channels: int = 32,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    bilinear_upsampling: bool = False,
) -> UNet:
    return UNet(
        UNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            bilinear_upsampling=bilinear_upsampling,
        )
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
    }


def smoke_test(seed: int = 20260725) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = UNetConfig()
    model = UNet(config)
    model.train()

    x = torch.randn(2, 2, 256, 256, requires_grad=True)
    y = model(x)
    if tuple(y.shape) != (2, 5, 256, 256):
        raise RuntimeError("Standard output shape is incorrect.")

    y.square().mean().backward()
    gradients_exist = any(
        parameter.grad is not None for parameter in model.parameters()
    )
    gradients_finite = all(
        parameter.grad is None
        or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )

    model.eval()
    with torch.no_grad():
        odd = model(torch.randn(1, 2, 255, 257))
        bilinear = UNet(
            UNetConfig(bilinear_upsampling=True)
        )(torch.randn(1, 2, 128, 128))

    checks = {
        "config_valid": True,
        "standard_output_shape_valid": tuple(y.shape)
        == (2, 5, 256, 256),
        "odd_output_shape_valid": tuple(odd.shape)
        == (1, 5, 255, 257),
        "bilinear_output_shape_valid": tuple(bilinear.shape)
        == (1, 5, 128, 128),
        "standard_output_finite": bool(torch.isfinite(y).all().item()),
        "odd_output_finite": bool(torch.isfinite(odd).all().item()),
        "gradients_exist": gradients_exist,
        "gradients_finite": gradients_finite,
    }
    checks["all_checks_passed"] = all(checks.values())
    if not checks["all_checks_passed"]:
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError("U-Net smoke test failed: " + ", ".join(failed))

    counts = count_parameters(model)
    size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**2)

    return {
        "seed": seed,
        "config": asdict(config),
        "standard_input_shape": list(x.shape),
        "standard_output_shape": list(y.shape),
        "odd_output_shape": list(odd.shape),
        "parameter_counts": counts,
        "estimated_parameter_size_mb": float(size_mb),
        "checks": checks,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline U-Net smoke tests."
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/model_development/unet_smoke_test.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    report = smoke_test(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    counts = report["parameter_counts"]
    print("\nBaseline U-Net smoke test")
    print("-------------------------")
    print(f"Input channels: {report['config']['input_channels']}")
    print(f"Output classes: {report['config']['num_classes']}")
    print(f"Base channels: {report['config']['base_channels']}")
    print(f"Standard input: {report['standard_input_shape']}")
    print(f"Standard output: {report['standard_output_shape']}")
    print(f"Odd output: {report['odd_output_shape']}")
    print(f"Total parameters: {counts['total_parameters']:,}")
    print(f"Trainable parameters: {counts['trainable_parameters']:,}")
    print(
        "Estimated parameter size: "
        f"{report['estimated_parameter_size_mb']:.2f} MB"
    )
    for name, passed in report["checks"].items():
        print(f"{name}: {passed}")
    print(f"Report: {args.output}")
    print(
        "\nResult: baseline U-Net passed forward, backward, "
        "and shape validation."
    )


if __name__ == "__main__":
    main()
