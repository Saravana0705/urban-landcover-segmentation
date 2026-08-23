"""Dependency-free hierarchical Swin Transformer for semantic segmentation.

The implementation uses shifted-window self-attention as a four-stage encoder
and a lightweight multi-scale decoder. It returns raw logits at the input
spatial resolution and therefore plugs directly into the shared trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.base import BaseSegmentationModel
from src.models.normalization import build_norm_layer


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("Expected a scalar or a two-value sequence.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1).")
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_probability) * random_tensor


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition NHWC features into flattened non-overlapping windows."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(
    windows: Tensor,
    window_size: int,
    height: int,
    width: int,
    batch_size: int,
) -> Tensor:
    """Reverse ``window_partition`` and return NHWC features."""
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, channels)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        table_size = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coordinates_flat = coordinates.flatten(1)
        relative_coordinates = (
            coordinates_flat[:, :, None] - coordinates_flat[:, None, :]
        )
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += window_size - 1
        relative_coordinates[:, :, 1] += window_size - 1
        relative_coordinates[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coordinates.sum(-1)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_bias = relative_bias.view(
            tokens,
            tokens,
            self.num_heads,
        ).permute(2, 0, 1)
        attention = attention + relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(
                -1,
                self.num_heads,
                tokens,
                tokens,
            )

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        x = (attention @ value).transpose(1, 2).reshape(
            batch_windows,
            tokens,
            channels,
        )
        x = self.projection(x)
        return self.projection_dropout(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError("shift_size must be in [0, window_size).")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def _attention_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.shift_size == 0:
            return None
        mask = torch.zeros((1, height, width, 1), device=device)
        slices_h = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        slices_w = slices_h
        counter = 0
        for h_slice in slices_h:
            for w_slice in slices_w:
                mask[:, h_slice, w_slice, :] = counter
                counter += 1
        mask_windows = window_partition(mask, self.window_size).squeeze(-1)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            float(-100.0),
        ).masked_fill(attention_mask == 0, float(0.0))
        return attention_mask.to(dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        if pad_right or pad_bottom:
            x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = x.shape[1:3]

        if self.shift_size > 0:
            shifted = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted = x

        windows = window_partition(shifted, self.window_size)
        mask = self._attention_mask(
            padded_height,
            padded_width,
            x.device,
            x.dtype,
        )
        attended = self.attention(windows, mask=mask)
        shifted = window_reverse(
            attended,
            self.window_size,
            padded_height,
            padded_width,
            batch,
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted
        if pad_right or pad_bottom:
            x = x[:, :height, :width, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, input_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        if height % 2 or width % 2:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
            height, width = x.shape[1:3]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path_values: Sequence[float],
        downsample: bool,
    ) -> None:
        super().__init__()
        if len(drop_path_values) != depth:
            raise ValueError("drop_path_values length must equal depth.")
        blocks = []
        for index in range(depth):
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=float(drop_path_values[index]),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        for block in self.blocks:
            x = block(x)
        feature = x
        next_feature = self.downsample(x) if self.downsample is not None else None
        return feature, next_feature


@dataclass(frozen=True)
class SwinTransformerConfig:
    input_channels: int = 2
    num_classes: int = 5
    patch_size: int = 4
    embed_dim: int = 48
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    window_size: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    decoder_channels: int = 192
    decoder_normalization: str = "groupnorm"
    group_norm_groups: int = 32

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if len(self.depths) != 4 or len(self.num_heads) != 4:
            raise ValueError("depths and num_heads must contain four values.")
        if any(depth <= 0 for depth in self.depths):
            raise ValueError("All stage depths must be positive.")
        dims = [self.embed_dim * (2 ** index) for index in range(4)]
        for dim, heads in zip(dims, self.num_heads):
            if heads <= 0 or dim % heads != 0:
                raise ValueError(
                    f"Stage dim {dim} must be divisible by head count {heads}."
                )
        if self.window_size <= 1:
            raise ValueError("window_size must be greater than one.")
        if self.decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")


class SwinTransformerSegmentation(BaseSegmentationModel):
    """Hierarchical Swin encoder with an FPN-like semantic decoder."""

    model_name = "swin_transformer"
    display_name = "Swin Transformer"

    def __init__(self, config: SwinTransformerConfig | None = None) -> None:
        self.config = config or SwinTransformerConfig()
        self.config.validate()
        super().__init__(
            input_channels=self.config.input_channels,
            num_classes=self.config.num_classes,
        )

        dims = [self.config.embed_dim * (2 ** index) for index in range(4)]
        self.patch_embedding = PatchEmbedding(
            self.config.input_channels,
            dims[0],
            self.config.patch_size,
        )

        total_blocks = sum(self.config.depths)
        drop_path_values = torch.linspace(
            0,
            self.config.drop_path_rate,
            total_blocks,
        ).tolist()
        cursor = 0
        stages: list[SwinStage] = []
        for stage_index in range(4):
            depth = self.config.depths[stage_index]
            stage_drop_paths = drop_path_values[cursor : cursor + depth]
            cursor += depth
            stages.append(
                SwinStage(
                    dim=dims[stage_index],
                    depth=depth,
                    num_heads=self.config.num_heads[stage_index],
                    window_size=self.config.window_size,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                    attention_dropout=self.config.attention_dropout,
                    drop_path_values=stage_drop_paths,
                    downsample=stage_index < 3,
                )
            )
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])

        self.projections = nn.ModuleList(
            [nn.Conv2d(dim, self.config.decoder_channels, 1) for dim in dims]
        )
        fused_channels = self.config.decoder_channels * 4
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_channels, self.config.decoder_channels, 3, padding=1, bias=False),
            build_norm_layer(
                self.config.decoder_channels,
                normalization=self.config.decoder_normalization,
                group_norm_groups=self.config.group_norm_groups,
            ),
            nn.GELU(),
            nn.Dropout2d(self.config.dropout),
            nn.Conv2d(self.config.decoder_channels, self.config.decoder_channels, 3, padding=1, bias=False),
            build_norm_layer(
                self.config.decoder_channels,
                normalization=self.config.decoder_normalization,
                group_norm_groups=self.config.group_norm_groups,
            ),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(
            self.config.decoder_channels,
            self.config.num_classes,
            kernel_size=1,
        )
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

    def forward_features(self, x: Tensor) -> list[Tensor]:
        x = self.patch_embedding(x)
        features: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            feature, next_feature = stage(x)
            feature = self.stage_norms[index](feature)
            feature = feature.permute(0, 3, 1, 2).contiguous()
            features.append(feature)
            if next_feature is not None:
                x = next_feature
        return features

    def forward(self, x: Tensor) -> Tensor:
        input_height, input_width = self.validate_input(x)
        features = self.forward_features(x)
        reference_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != reference_size:
                feature = F.interpolate(
                    feature,
                    size=reference_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)
        x = torch.cat(projected, dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        return F.interpolate(
            x,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )


def build_swin_transformer(
    *,
    input_channels: int = 2,
    num_classes: int = 5,
    patch_size: int = 4,
    embed_dim: int = 48,
    depths: Sequence[int] = (2, 2, 2, 2),
    num_heads: Sequence[int] = (3, 6, 12, 24),
    window_size: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    attention_dropout: float = 0.0,
    drop_path_rate: float = 0.1,
    decoder_channels: int = 192,
    decoder_normalization: str = "groupnorm",
    group_norm_groups: int = 32,
) -> SwinTransformerSegmentation:
    """Construct a Swin Transformer segmentation model."""
    config = SwinTransformerConfig(
        input_channels=int(input_channels),
        num_classes=int(num_classes),
        patch_size=int(patch_size),
        embed_dim=int(embed_dim),
        depths=tuple(int(value) for value in depths),
        num_heads=tuple(int(value) for value in num_heads),
        window_size=int(window_size),
        mlp_ratio=float(mlp_ratio),
        dropout=float(dropout),
        attention_dropout=float(attention_dropout),
        drop_path_rate=float(drop_path_rate),
        decoder_channels=int(decoder_channels),
        decoder_normalization=str(decoder_normalization),
        group_norm_groups=int(group_norm_groups),
    )
    return SwinTransformerSegmentation(config)
