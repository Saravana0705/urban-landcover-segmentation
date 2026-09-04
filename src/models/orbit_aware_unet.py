"""Orbit-aware U-Net for paired ascending/descending Sentinel-1 inputs.

The first half of the input channels must contain the ascending stack and the
second half the descending stack. Separate shallow stems learn pass-specific
features before a lightweight 1x1 fusion and the shared baseline U-Net body.
"""

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
from src.models.unet import DoubleConv, DownBlock, UpBlock


@dataclass(frozen=True)
class OrbitAwareUNetConfig:
    input_channels: int = 16
    orbit_channels: int = 8
    num_classes: int = 5
    base_channels: int = 32
    dropout: float = 0.1
    use_batch_norm: bool = True
    bilinear_upsampling: bool = False

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.orbit_channels <= 0:
            raise ValueError("orbit_channels must be positive.")
        if self.input_channels != 2 * self.orbit_channels:
            raise ValueError(
                "input_channels must equal 2 * orbit_channels for paired "
                "ascending/descending input."
            )
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.base_channels <= 0 or self.base_channels % 2:
            raise ValueError("base_channels must be a positive even number.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class OrbitFusion(nn.Module):
    """Fuse equal-width pass-specific feature maps without spatial mixing."""

    def __init__(
        self,
        channels: int,
        *,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=not use_batch_norm,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, ascending: Tensor, descending: Tensor) -> Tensor:
        if ascending.shape != descending.shape:
            raise ValueError(
                "Ascending and descending stem features must have equal shapes."
            )
        return self.block(torch.cat([ascending, descending], dim=1))


class OrbitAwareUNet(BaseSegmentationModel):
    """U-Net with pass-specific first-level feature extraction."""

    model_name = "orbit_aware_unet"
    display_name = "Orbit-Aware U-Net"

    def __init__(
        self,
        config: OrbitAwareUNetConfig | None = None,
    ) -> None:
        self.config = config or OrbitAwareUNetConfig()
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

        self.ascending_stem = DoubleConv(
            self.config.orbit_channels,
            c,
            **kwargs,
        )
        self.descending_stem = DoubleConv(
            self.config.orbit_channels,
            c,
            **kwargs,
        )
        self.orbit_fusion = OrbitFusion(
            c,
            use_batch_norm=self.config.use_batch_norm,
        )

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

    def split_orbits(self, x: Tensor) -> tuple[Tensor, Tensor]:
        channels = self.config.orbit_channels
        ascending = x[:, :channels, :, :]
        descending = x[:, channels:, :, :]
        if ascending.shape[1] != channels or descending.shape[1] != channels:
            raise ValueError(
                "Input does not contain the configured paired orbit channels."
            )
        return ascending, descending

    def forward(self, x: Tensor) -> Tensor:
        original_size = self.validate_input(x)
        ascending, descending = self.split_orbits(x)
        ascending_features = self.ascending_stem(ascending)
        descending_features = self.descending_stem(descending)
        s1 = self.orbit_fusion(
            ascending_features,
            descending_features,
        )
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


def build_orbit_aware_unet(
    input_channels: int = 16,
    orbit_channels: int = 8,
    num_classes: int = 5,
    base_channels: int = 32,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    bilinear_upsampling: bool = False,
) -> OrbitAwareUNet:
    return OrbitAwareUNet(
        OrbitAwareUNetConfig(
            input_channels=input_channels,
            orbit_channels=orbit_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            bilinear_upsampling=bilinear_upsampling,
        )
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
    }


def smoke_test(seed: int = 20260725) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = OrbitAwareUNetConfig()
    model = OrbitAwareUNet(config)
    model.train()

    x = torch.randn(1, 16, 256, 256, requires_grad=True)
    output = model(x)
    if tuple(output.shape) != (1, 5, 256, 256):
        raise RuntimeError("Standard output shape is incorrect.")

    output.square().mean().backward()
    gradients_exist = any(
        parameter.grad is not None for parameter in model.parameters()
    )
    gradients_finite = all(
        parameter.grad is None
        or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    ascending_stem_gradients_exist = any(
        parameter.grad is not None
        for parameter in model.ascending_stem.parameters()
    )
    descending_stem_gradients_exist = any(
        parameter.grad is not None
        for parameter in model.descending_stem.parameters()
    )
    fusion_gradients_exist = any(
        parameter.grad is not None
        for parameter in model.orbit_fusion.parameters()
    )

    model.eval()
    with torch.no_grad():
        odd = model(torch.randn(1, 16, 255, 257))
        probe = torch.randn(1, 16, 64, 64)
        original = model(probe)
        swapped = model(torch.cat([probe[:, 8:], probe[:, :8]], dim=1))

    invalid_config_rejected = False
    try:
        OrbitAwareUNetConfig(input_channels=16, orbit_channels=7).validate()
    except ValueError:
        invalid_config_rejected = True

    wrong_input_rejected = False
    try:
        model(torch.randn(1, 8, 64, 64))
    except ValueError:
        wrong_input_rejected = True

    checks = {
        "config_valid": True,
        "standard_output_shape_valid": tuple(output.shape)
        == (1, 5, 256, 256),
        "odd_output_shape_valid": tuple(odd.shape)
        == (1, 5, 255, 257),
        "standard_output_finite": bool(torch.isfinite(output).all().item()),
        "odd_output_finite": bool(torch.isfinite(odd).all().item()),
        "gradients_exist": gradients_exist,
        "gradients_finite": gradients_finite,
        "ascending_stem_gradients_exist": ascending_stem_gradients_exist,
        "descending_stem_gradients_exist": descending_stem_gradients_exist,
        "fusion_gradients_exist": fusion_gradients_exist,
        "invalid_config_rejected": invalid_config_rejected,
        "wrong_input_rejected": wrong_input_rejected,
        "orbit_order_affects_output": not bool(
            torch.allclose(original, swapped)
        ),
    }
    checks["all_checks_passed"] = all(checks.values())
    if not checks["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "Orbit-aware U-Net smoke test failed: " + ", ".join(failed)
        )

    counts = count_parameters(model)
    size_mb = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) / (1024**2)
    return {
        "seed": seed,
        "config": asdict(config),
        "standard_input_shape": list(x.shape),
        "standard_output_shape": list(output.shape),
        "odd_output_shape": list(odd.shape),
        "parameter_counts": counts,
        "estimated_parameter_size_mb": float(size_mb),
        "checks": checks,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run orbit-aware U-Net smoke tests."
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/model_development/orbit_aware_unet_smoke_test.json"
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
    print("\nOrbit-aware U-Net smoke test")
    print("----------------------------")
    print(f"Input channels: {report['config']['input_channels']}")
    print(f"Orbit channels: {report['config']['orbit_channels']}")
    print(f"Output classes: {report['config']['num_classes']}")
    print(f"Standard input: {report['standard_input_shape']}")
    print(f"Standard output: {report['standard_output_shape']}")
    print(f"Odd output: {report['odd_output_shape']}")
    print(f"Trainable parameters: {counts['trainable_parameters']:,}")
    print(
        "Estimated parameter size: "
        f"{report['estimated_parameter_size_mb']:.2f} MB"
    )
    for name, passed in report["checks"].items():
        print(f"{name}: {passed}")
    print(f"Report: {args.output}")
    print("\nResult: orbit-aware U-Net smoke test passed.")


if __name__ == "__main__":
    main()
