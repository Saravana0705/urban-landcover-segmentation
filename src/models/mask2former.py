"""Compact dependency-free Mask2Former-style semantic segmentation model.

This implementation preserves the central Mask2Former ideas while remaining
practical for CPU pilot experiments:

* a hierarchical multi-scale image encoder;
* a top-down pixel decoder that produces dense mask features;
* learnable object/mask queries;
* a transformer query decoder attending to multi-scale image tokens;
* separate class and mask embeddings combined into dense semantic logits.

The shared project trainer expects one tensor shaped ``(N, C, H, W)``. The
query predictions are therefore converted into normalized per-pixel semantic
log-probabilities, which can be consumed by the existing CE-Dice objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.base import BaseSegmentationModel
from src.models.normalization import build_norm_layer


def _valid_groups(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        normalization: str = "groupnorm",
        group_norm_groups: int = 32,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            build_norm_layer(
                output_channels,
                normalization=normalization,
                group_norm_groups=_valid_groups(
                    output_channels,
                    group_norm_groups,
                ),
            ),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        normalization: str,
        group_norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormActivation(
                channels,
                channels,
                normalization=normalization,
                group_norm_groups=group_norm_groups,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvNormActivation(
                channels,
                channels,
                normalization=normalization,
                group_norm_groups=group_norm_groups,
                activation=False,
            ),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class HierarchicalEncoder(nn.Module):
    """Four-stage convolutional pyramid at 1/4, 1/8, 1/16 and 1/32."""

    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        depths: Sequence[int],
        *,
        normalization: str,
        group_norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("Encoder channels and depths must contain four values.")

        self.stem = nn.Sequential(
            ConvNormActivation(
                input_channels,
                channels[0],
                kernel_size=7,
                stride=2,
                normalization=normalization,
                group_norm_groups=group_norm_groups,
            ),
            ConvNormActivation(
                channels[0],
                channels[0],
                stride=2,
                normalization=normalization,
                group_norm_groups=group_norm_groups,
            ),
        )

        stages: list[nn.Module] = []
        downsamplers: list[nn.Module] = []
        for index, (stage_channels, stage_depth) in enumerate(
            zip(channels, depths, strict=True)
        ):
            stages.append(
                nn.Sequential(
                    *[
                        ResidualConvBlock(
                            stage_channels,
                            normalization=normalization,
                            group_norm_groups=group_norm_groups,
                            dropout=dropout,
                        )
                        for _ in range(int(stage_depth))
                    ]
                )
            )
            if index < 3:
                downsamplers.append(
                    ConvNormActivation(
                        channels[index],
                        channels[index + 1],
                        stride=2,
                        normalization=normalization,
                        group_norm_groups=group_norm_groups,
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if index < len(self.downsamplers):
                x = self.downsamplers[index](x)
        return features


class PixelDecoder(nn.Module):
    """FPN-style pixel decoder producing 1/4-resolution mask features."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        hidden_dim: int,
        mask_dim: int,
        *,
        normalization: str,
        group_norm_groups: int,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(channels, hidden_dim, kernel_size=1)
                for channels in encoder_channels
            ]
        )
        self.output = nn.ModuleList(
            [
                ConvNormActivation(
                    hidden_dim,
                    hidden_dim,
                    normalization=normalization,
                    group_norm_groups=group_norm_groups,
                )
                for _ in encoder_channels
            ]
        )
        self.mask_projection = nn.Conv2d(
            hidden_dim,
            mask_dim,
            kernel_size=1,
        )

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, list[Tensor]]:
        if len(features) != 4:
            raise ValueError("Pixel decoder expects four encoder feature maps.")

        projected = [
            layer(feature)
            for layer, feature in zip(self.lateral, features, strict=True)
        ]
        pyramid: list[Tensor] = [torch.empty(0)] * 4
        current = projected[-1]
        pyramid[-1] = self.output[-1](current)

        for index in range(2, -1, -1):
            current = F.interpolate(
                current,
                size=projected[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            current = current + projected[index]
            pyramid[index] = self.output[index](current)

        mask_features = self.mask_projection(pyramid[0])
        return mask_features, pyramid


class MultiScaleQueryDecoder(nn.Module):
    """Learnable-query transformer decoder with cyclic multi-scale memory."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        num_feature_levels: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_layers = int(num_layers)
        self.num_feature_levels = int(num_feature_levels)
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)
        self.level_embedding = nn.Embedding(num_feature_levels, hidden_dim)

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _memory_tokens(self, feature: Tensor, level: int) -> Tensor:
        tokens = feature.flatten(2).transpose(1, 2)
        return tokens + self.level_embedding.weight[level].view(1, 1, -1)

    def forward(self, pyramid: Sequence[Tensor]) -> Tensor:
        if len(pyramid) != self.num_feature_levels:
            raise ValueError("Unexpected number of pixel-decoder feature levels.")
        batch_size = pyramid[0].shape[0]
        query = self.query_content.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_position = self.query_position.weight.unsqueeze(0)

        # Deepest levels first; cycle across scales for successive layers.
        memories = [
            self._memory_tokens(feature, index)
            for index, feature in enumerate(reversed(pyramid))
        ]
        for index, layer in enumerate(self.layers):
            memory = memories[index % len(memories)]
            query = layer(query + query_position, memory)
        return self.final_norm(query)


@dataclass(frozen=True)
class Mask2FormerConfig:
    input_channels: int = 2
    num_classes: int = 5
    encoder_channels: tuple[int, int, int, int] = (48, 96, 192, 384)
    encoder_depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    hidden_dim: int = 192
    mask_dim: int = 128
    num_queries: int = 50
    decoder_layers: int = 4
    decoder_heads: int = 8
    feedforward_dim: int = 768
    dropout: float = 0.1
    normalization: str = "groupnorm"
    group_norm_groups: int = 32


class Mask2Former(BaseSegmentationModel):
    """Compact Mask2Former-style model returning dense semantic logits."""

    model_name = "mask2former"
    display_name = "Mask2Former"

    def __init__(self, config: Mask2FormerConfig | None = None) -> None:
        self.config = config or Mask2FormerConfig()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )
        if self.config.hidden_dim <= 0 or self.config.mask_dim <= 0:
            raise ValueError("hidden_dim and mask_dim must be positive.")
        if self.config.num_queries < self.config.num_classes:
            raise ValueError("num_queries must be at least num_classes.")

        self.encoder = HierarchicalEncoder(
            self.input_channels,
            self.config.encoder_channels,
            self.config.encoder_depths,
            normalization=self.config.normalization,
            group_norm_groups=self.config.group_norm_groups,
            dropout=self.config.dropout,
        )
        self.pixel_decoder = PixelDecoder(
            self.config.encoder_channels,
            self.config.hidden_dim,
            self.config.mask_dim,
            normalization=self.config.normalization,
            group_norm_groups=self.config.group_norm_groups,
        )
        self.query_decoder = MultiScaleQueryDecoder(
            hidden_dim=self.config.hidden_dim,
            num_queries=self.config.num_queries,
            num_layers=self.config.decoder_layers,
            num_heads=self.config.decoder_heads,
            feedforward_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
            num_feature_levels=4,
        )
        self.class_head = nn.Linear(
            self.config.hidden_dim,
            self.num_classes + 1,
        )
        self.mask_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.mask_dim),
        )

        self.apply(self.initialize_weights)
        nn.init.normal_(self.query_decoder.query_content.weight, std=0.02)
        nn.init.normal_(self.query_decoder.query_position.weight, std=0.02)
        nn.init.normal_(self.query_decoder.level_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        input_height, input_width = self.validate_input(x)
        encoder_features = self.encoder(x)
        mask_features, pyramid = self.pixel_decoder(encoder_features)
        queries = self.query_decoder(pyramid)

        class_logits = self.class_head(queries)
        mask_embeddings = self.mask_head(queries)
        mask_logits = torch.einsum(
            "bqd,bdhw->bqhw",
            mask_embeddings,
            mask_features,
        ) / sqrt(float(self.config.mask_dim))

        class_probabilities = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_probabilities = torch.sigmoid(mask_logits)
        semantic_probabilities = torch.einsum(
            "bqc,bqhw->bchw",
            class_probabilities,
            mask_probabilities,
        )
        semantic_probabilities = semantic_probabilities.clamp_min(1.0e-7)
        semantic_probabilities = semantic_probabilities / (
            semantic_probabilities.sum(dim=1, keepdim=True) + 1.0e-7
        )
        semantic_logits = torch.log(semantic_probabilities)
        return F.interpolate(
            semantic_logits,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )


def build_mask2former(
    *,
    input_channels: int = 2,
    num_classes: int = 5,
    encoder_channels: Sequence[int] = (48, 96, 192, 384),
    encoder_depths: Sequence[int] = (2, 2, 2, 2),
    hidden_dim: int = 192,
    mask_dim: int = 128,
    num_queries: int = 50,
    decoder_layers: int = 4,
    decoder_heads: int = 8,
    feedforward_dim: int = 768,
    dropout: float = 0.1,
    normalization: str = "groupnorm",
    group_norm_groups: int = 32,
) -> Mask2Former:
    if len(encoder_channels) != 4 or len(encoder_depths) != 4:
        raise ValueError("encoder_channels and encoder_depths require four values.")
    config = Mask2FormerConfig(
        input_channels=int(input_channels),
        num_classes=int(num_classes),
        encoder_channels=tuple(int(value) for value in encoder_channels),
        encoder_depths=tuple(int(value) for value in encoder_depths),
        hidden_dim=int(hidden_dim),
        mask_dim=int(mask_dim),
        num_queries=int(num_queries),
        decoder_layers=int(decoder_layers),
        decoder_heads=int(decoder_heads),
        feedforward_dim=int(feedforward_dim),
        dropout=float(dropout),
        normalization=str(normalization),
        group_norm_groups=int(group_norm_groups),
    )
    return Mask2Former(config)
