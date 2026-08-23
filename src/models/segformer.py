"""Dependency-free SegFormer implementation for Sentinel-1 segmentation.

The encoder follows the MiT design principles: overlapping patch embeddings,
efficient spatial-reduction attention, Mix-FFN blocks, and a lightweight MLP
decoder. It intentionally avoids external pretrained weights so experiments are
fully reproducible in the existing offline pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.base import BaseSegmentationModel
from src.models.normalization import build_norm_layer


@dataclass(frozen=True)
class SegFormerConfig:
    input_channels: int = 2
    num_classes: int = 5
    embed_dims: tuple[int, int, int, int] = (32, 64, 160, 256)
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (1, 2, 5, 8)
    sr_ratios: tuple[int, int, int, int] = (8, 4, 2, 1)
    mlp_ratios: tuple[float, float, float, float] = (4.0, 4.0, 4.0, 4.0)
    decoder_channels: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    decoder_normalization: str = "groupnorm"
    group_norm_groups: int = 32

    def validate(self) -> None:
        if self.input_channels <= 0 or self.num_classes <= 1:
            raise ValueError("input_channels must be positive and num_classes > 1.")
        fields = (self.embed_dims, self.depths, self.num_heads, self.sr_ratios, self.mlp_ratios)
        if any(len(values) != 4 for values in fields):
            raise ValueError("SegFormer stage settings must each contain four values.")
        if any(dim <= 0 for dim in self.embed_dims):
            raise ValueError("embed_dims must be positive.")
        for dim, heads in zip(self.embed_dims, self.num_heads):
            if dim % heads != 0:
                raise ValueError(f"Embedding dimension {dim} must be divisible by {heads} heads.")


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.projection(x)
        height, width = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key_value = nn.Linear(embed_dim, embed_dim * 2)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.reduction_norm = nn.LayerNorm(embed_dim)
        else:
            self.spatial_reduction = None
            self.reduction_norm = None

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, channels = x.shape
        query = self.query(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        reduced = x
        if self.spatial_reduction is not None:
            feature = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced_feature = self.spatial_reduction(feature)
            reduced = reduced_feature.flatten(2).transpose(1, 2)
            reduced = self.reduction_norm(reduced)

        kv = self.key_value(reduced)
        kv = kv.reshape(batch, reduced.shape[1], 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection_dropout(self.projection(output))


class MixFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.depthwise(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.dropout(self.activation(x))
        return self.dropout(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = EfficientSelfAttention(
            embed_dim,
            num_heads,
            sr_ratio,
            attention_dropout,
            dropout,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFFN(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class MiTEncoder(nn.Module):
    def __init__(self, config: SegFormerConfig) -> None:
        super().__init__()
        embed_dims = config.embed_dims
        self.patch_embeddings = nn.ModuleList([
            OverlapPatchEmbedding(config.input_channels, embed_dims[0], 7, 4),
            OverlapPatchEmbedding(embed_dims[0], embed_dims[1], 3, 2),
            OverlapPatchEmbedding(embed_dims[1], embed_dims[2], 3, 2),
            OverlapPatchEmbedding(embed_dims[2], embed_dims[3], 3, 2),
        ])

        total_blocks = sum(config.depths)
        rates = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()
        offset = 0
        stages: list[nn.ModuleList] = []
        for stage_index in range(4):
            blocks = nn.ModuleList([
                TransformerBlock(
                    embed_dim=embed_dims[stage_index],
                    num_heads=config.num_heads[stage_index],
                    sr_ratio=config.sr_ratios[stage_index],
                    mlp_ratio=config.mlp_ratios[stage_index],
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    drop_path=rates[offset + block_index],
                )
                for block_index in range(config.depths[stage_index])
            ])
            stages.append(blocks)
            offset += config.depths[stage_index]
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in embed_dims])

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        for patch_embedding, blocks, norm in zip(
            self.patch_embeddings,
            self.stages,
            self.stage_norms,
        ):
            tokens, height, width = patch_embedding(x)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = norm(tokens)
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
            outputs.append(x)
        return outputs


class SegFormerDecoder(nn.Module):
    def __init__(self, config: SegFormerConfig) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, config.decoder_channels, kernel_size=1)
            for dim in config.embed_dims
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(config.decoder_channels * 4, config.decoder_channels, 1, bias=False),
            build_norm_layer(
                config.decoder_channels,
                normalization=config.decoder_normalization,
                group_norm_groups=config.group_norm_groups,
            ),
            nn.GELU(),
            nn.Dropout2d(config.dropout),
        )
        self.classifier = nn.Conv2d(config.decoder_channels, config.num_classes, 1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        return self.classifier(self.fusion(torch.cat(projected, dim=1)))


class SegFormer(BaseSegmentationModel):
    model_name = "segformer"
    display_name = "SegFormer"

    def __init__(self, config: SegFormerConfig | None = None) -> None:
        self.config = config or SegFormerConfig()
        self.config.validate()
        super().__init__(input_channels=self.config.input_channels, num_classes=self.config.num_classes)
        self.encoder = MiTEncoder(self.config)
        self.decoder = SegFormerDecoder(self.config)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        self.validate_input(x)
        input_size = x.shape[-2:]
        logits = self.decoder(self.encoder(x))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_segformer(
    input_channels: int = 2,
    num_classes: int = 5,
    embed_dims: Sequence[int] = (32, 64, 160, 256),
    depths: Sequence[int] = (2, 2, 2, 2),
    num_heads: Sequence[int] = (1, 2, 5, 8),
    sr_ratios: Sequence[int] = (8, 4, 2, 1),
    mlp_ratios: Sequence[float] = (4.0, 4.0, 4.0, 4.0),
    decoder_channels: int = 256,
    dropout: float = 0.1,
    attention_dropout: float = 0.0,
    drop_path_rate: float = 0.1,
    decoder_normalization: str = "groupnorm",
    group_norm_groups: int = 32,
) -> SegFormer:
    return SegFormer(SegFormerConfig(
        input_channels=input_channels,
        num_classes=num_classes,
        embed_dims=tuple(int(v) for v in embed_dims),
        depths=tuple(int(v) for v in depths),
        num_heads=tuple(int(v) for v in num_heads),
        sr_ratios=tuple(int(v) for v in sr_ratios),
        mlp_ratios=tuple(float(v) for v in mlp_ratios),
        decoder_channels=decoder_channels,
        dropout=dropout,
        attention_dropout=attention_dropout,
        drop_path_rate=drop_path_rate,
        decoder_normalization=decoder_normalization,
        group_norm_groups=group_norm_groups,
    ))
