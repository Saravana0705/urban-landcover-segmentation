"""Semantic-segmentation model architectures and factory."""

from src.models.attention_unet import (
    AttentionUNet,
    AttentionUNetConfig,
    build_attention_unet,
)
from src.models.base import BaseSegmentationModel
from src.models.deeplabv3_plus import (
    DeepLabV3Plus,
    DeepLabV3PlusConfig,
    build_deeplabv3_plus,
)
from src.models.mask2former import (
    Mask2Former,
    Mask2FormerConfig,
    build_mask2former,
)
from src.models.model_factory import build_model
from src.models.segformer import SegFormer, SegFormerConfig, build_segformer
from src.models.swin_transformer import (
    SwinTransformerConfig,
    SwinTransformerSegmentation,
    build_swin_transformer,
)
from src.models.unet import UNet, UNetConfig, build_unet
from src.models.unet_plus_plus import (
    UNetPlusPlus,
    UNetPlusPlusConfig,
    build_unet_plus_plus,
)

__all__ = [
    "AttentionUNet",
    "AttentionUNetConfig",
    "BaseSegmentationModel",
    "DeepLabV3Plus",
    "DeepLabV3PlusConfig",
    "Mask2Former",
    "Mask2FormerConfig",
    "SegFormer",
    "SegFormerConfig",
    "SwinTransformerConfig",
    "SwinTransformerSegmentation",
    "UNet",
    "UNetConfig",
    "UNetPlusPlus",
    "UNetPlusPlusConfig",
    "build_attention_unet",
    "build_deeplabv3_plus",
    "build_mask2former",
    "build_model",
    "build_segformer",
    "build_swin_transformer",
    "build_unet",
    "build_unet_plus_plus",
]
