"""Central model construction for reproducible segmentation experiments."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.attention_unet import build_attention_unet
from src.models.deeplabv3_plus import build_deeplabv3_plus
from src.models.unet import build_unet
from src.models.unet_plus_plus import build_unet_plus_plus
from src.models.segformer import build_segformer
from src.models.swin_transformer import build_swin_transformer
from src.models.mask2former import build_mask2former
from src.models.orbit_aware_unet import build_orbit_aware_unet
from src.models.multiview_unetpp import build_multiview_unetpp


MODEL_ALIASES = {
    "unet": "unet",
    "u_net": "unet",
    "orbit_aware_unet": "orbit_aware_unet",
    "orbit-aware-unet": "orbit_aware_unet",
    "dual_stem_unet": "orbit_aware_unet",
    "attention_unet": "attention_unet",
    "attention-u-net": "attention_unet",
    "att_unet": "attention_unet",
    "unetpp": "unetpp",
    "unet++": "unetpp",
    "unet_plus_plus": "unetpp",
    "u_net_plus_plus": "unetpp",
    "deeplabv3plus": "deeplabv3plus",
    "deeplabv3+": "deeplabv3plus",
    "deeplab_v3_plus": "deeplabv3plus",
    "deeplab": "deeplabv3plus",
    "segformer": "segformer",
    "seg_former": "segformer",
    "swin_transformer": "swin_transformer",
    "swin": "swin_transformer",
    "swin_unet": "swin_transformer",
    "mask2former": "mask2former",
    "mask_2_former": "mask2former",
    "mask2_former": "mask2former",
    "multiview_unetpp": "multiview_unetpp",
    "multi_view_unetpp": "multiview_unetpp",
}


def _nested_get(
    payload: Mapping[str, Any],
    dotted_path: str,
    default: Any,
) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build the architecture named in the YAML model section."""
    requested_name = str(
        _nested_get(config, "model.name", "unet")
    ).lower().strip()
    model_name = MODEL_ALIASES.get(requested_name)
    if model_name is None:
        supported = ", ".join(sorted(set(MODEL_ALIASES.values())))
        raise ValueError(
            f"Unsupported model '{requested_name}'. Supported: {supported}."
        )

    common_kwargs = {
        "input_channels": int(
            _nested_get(config, "model.input_channels", 2)
        ),
        "num_classes": int(
            _nested_get(config, "model.num_classes", 5)
        ),
        "base_channels": int(
            _nested_get(config, "model.base_channels", 32)
        ),
        "dropout": float(
            _nested_get(config, "model.dropout", 0.1)
        ),
        "use_batch_norm": bool(
            _nested_get(config, "model.use_batch_norm", True)
        ),
        "bilinear_upsampling": bool(
            _nested_get(config, "model.bilinear_upsampling", False)
        ),
    }

    if model_name == "unet":
        model = build_unet(**common_kwargs)
    elif model_name == "orbit_aware_unet":
        model = build_orbit_aware_unet(
            **common_kwargs,
            orbit_channels=int(
                _nested_get(config, "model.orbit_channels", 8)
            ),
        )
    elif model_name == "attention_unet":
        model = build_attention_unet(**common_kwargs)
    elif model_name == "unetpp":
        model = build_unet_plus_plus(
            **common_kwargs,
            deep_supervision=bool(
                _nested_get(config, "model.deep_supervision", False)
            ),
            deep_supervision_average_inference=bool(
                _nested_get(
                    config,
                    "model.deep_supervision_average_inference",
                    True,
                )
            ),
        )
    elif model_name == "multiview_unetpp":
        model = build_multiview_unetpp(
            **common_kwargs,
            num_views=int(_nested_get(config, "model.num_views", 8)),
            channels_per_view=int(_nested_get(config, "model.channels_per_view", 2)),
            fused_channels=int(_nested_get(config, "model.fused_channels", 16)),
        )
    elif model_name == "deeplabv3plus":
        rates_raw = _nested_get(config, "model.atrous_rates", [6, 12, 18])
        if not isinstance(rates_raw, (list, tuple)) or len(rates_raw) != 3:
            raise ValueError("model.atrous_rates must contain exactly three integers.")
        model = build_deeplabv3_plus(
            input_channels=common_kwargs["input_channels"],
            num_classes=common_kwargs["num_classes"],
            backbone=str(_nested_get(config, "model.backbone", "resnet18")),
            encoder_base_channels=int(_nested_get(config, "model.encoder_base_channels", 64)),
            output_stride=int(_nested_get(config, "model.output_stride", 16)),
            aspp_channels=int(_nested_get(config, "model.aspp_channels", 256)),
            low_level_channels=int(_nested_get(config, "model.low_level_channels", 48)),
            decoder_channels=int(_nested_get(config, "model.decoder_channels", 256)),
            dropout=common_kwargs["dropout"],
            atrous_rates=tuple(int(value) for value in rates_raw),
            normalization=str(_nested_get(config, "model.normalization", "groupnorm")),
            group_norm_groups=int(_nested_get(config, "model.group_norm_groups", 32)),
        )
    elif model_name == "segformer":
        def stage_values(path: str, default: list[Any]) -> list[Any]:
            values = _nested_get(config, path, default)
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError(f"{path} must contain exactly four values.")
            return list(values)

        model = build_segformer(
            input_channels=common_kwargs["input_channels"],
            num_classes=common_kwargs["num_classes"],
            embed_dims=stage_values("model.embed_dims", [32, 64, 160, 256]),
            depths=stage_values("model.depths", [2, 2, 2, 2]),
            num_heads=stage_values("model.num_heads", [1, 2, 5, 8]),
            sr_ratios=stage_values("model.sr_ratios", [8, 4, 2, 1]),
            mlp_ratios=stage_values("model.mlp_ratios", [4.0, 4.0, 4.0, 4.0]),
            decoder_channels=int(_nested_get(config, "model.decoder_channels", 256)),
            dropout=common_kwargs["dropout"],
            attention_dropout=float(_nested_get(config, "model.attention_dropout", 0.0)),
            drop_path_rate=float(_nested_get(config, "model.drop_path_rate", 0.1)),
            decoder_normalization=str(_nested_get(config, "model.decoder_normalization", "groupnorm")),
            group_norm_groups=int(_nested_get(config, "model.group_norm_groups", 32)),
        )
    elif model_name == "mask2former":
        def mask_stage_values(path: str, default: list[Any]) -> list[Any]:
            values = _nested_get(config, path, default)
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError(f"{path} must contain exactly four values.")
            return list(values)

        model = build_mask2former(
            input_channels=common_kwargs["input_channels"],
            num_classes=common_kwargs["num_classes"],
            encoder_channels=mask_stage_values(
                "model.encoder_channels", [48, 96, 192, 384]
            ),
            encoder_depths=mask_stage_values(
                "model.encoder_depths", [2, 2, 2, 2]
            ),
            hidden_dim=int(_nested_get(config, "model.hidden_dim", 192)),
            mask_dim=int(_nested_get(config, "model.mask_dim", 128)),
            num_queries=int(_nested_get(config, "model.num_queries", 50)),
            decoder_layers=int(_nested_get(config, "model.decoder_layers", 4)),
            decoder_heads=int(_nested_get(config, "model.decoder_heads", 8)),
            feedforward_dim=int(
                _nested_get(config, "model.feedforward_dim", 768)
            ),
            dropout=common_kwargs["dropout"],
            normalization=str(
                _nested_get(config, "model.normalization", "groupnorm")
            ),
            group_norm_groups=int(
                _nested_get(config, "model.group_norm_groups", 32)
            ),
        )
    elif model_name == "swin_transformer":
        def swin_stage_values(path: str, default: list[Any]) -> list[Any]:
            values = _nested_get(config, path, default)
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError(f"{path} must contain exactly four values.")
            return list(values)

        model = build_swin_transformer(
            input_channels=common_kwargs["input_channels"],
            num_classes=common_kwargs["num_classes"],
            patch_size=int(_nested_get(config, "model.patch_size", 4)),
            embed_dim=int(_nested_get(config, "model.embed_dim", 48)),
            depths=swin_stage_values("model.depths", [2, 2, 2, 2]),
            num_heads=swin_stage_values("model.num_heads", [3, 6, 12, 24]),
            window_size=int(_nested_get(config, "model.window_size", 8)),
            mlp_ratio=float(_nested_get(config, "model.mlp_ratio", 4.0)),
            dropout=common_kwargs["dropout"],
            attention_dropout=float(_nested_get(config, "model.attention_dropout", 0.0)),
            drop_path_rate=float(_nested_get(config, "model.drop_path_rate", 0.1)),
            decoder_channels=int(_nested_get(config, "model.decoder_channels", 192)),
            decoder_normalization=str(_nested_get(config, "model.decoder_normalization", "groupnorm")),
            group_norm_groups=int(_nested_get(config, "model.group_norm_groups", 32)),
        )
    else:  # pragma: no cover - protected by alias validation
        raise AssertionError(f"Unhandled model: {model_name}")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters <= 0:
        raise RuntimeError(
            f"Constructed model '{model_name}' has no trainable parameters."
        )

    return model
