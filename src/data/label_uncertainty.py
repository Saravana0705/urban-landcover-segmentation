"""Training-only boundary uncertainty for noisy semantic labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LabelUncertaintyConfig:
    enabled: bool = False
    class_ids: tuple[int, ...] = (1, 3)
    radius_pixels: int = 1
    ignore_index: int = 255
    boundary_side: str = "inner"
    preserve_thin_structures: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "LabelUncertaintyConfig":
        values = dict(payload or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            class_ids=tuple(int(value) for value in values.get("class_ids", (1, 3))),
            radius_pixels=int(values.get("radius_pixels", 1)),
            ignore_index=int(values.get("ignore_index", 255)),
            boundary_side=str(values.get("boundary_side", "inner")).strip().lower(),
            preserve_thin_structures=bool(values.get("preserve_thin_structures", True)),
        )

    def validate(self, *, num_classes: int) -> None:
        if self.radius_pixels < 1:
            raise ValueError("label_uncertainty.radius_pixels must be >= 1.")
        if not self.class_ids or len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("label_uncertainty.class_ids must be unique and non-empty.")
        if any(value < 0 or value >= num_classes for value in self.class_ids):
            raise ValueError("label_uncertainty.class_ids contains an invalid model class ID.")
        if self.boundary_side not in {"inner", "symmetric"}:
            raise ValueError("label_uncertainty.boundary_side must be inner or symmetric.")


def apply_boundary_uncertainty(
    target: np.ndarray,
    config: LabelUncertaintyConfig,
    *,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace selected-class boundary bands with the loss ignore index.

    Class IDs are model-facing IDs (0--4). The input array is never modified.
    The symmetric morphological band covers both sides of each selected class
    boundary, where rasterised labels are most likely to be spatially uncertain.
    """
    if target.ndim != 2:
        raise ValueError("target must be a two-dimensional semantic mask.")
    config.validate(num_classes=num_classes)
    output = target.astype(np.int64, copy=True)
    uncertain = np.zeros(target.shape, dtype=bool)
    kernel = 2 * config.radius_pixels + 1

    for class_id in config.class_ids:
        mask = torch.from_numpy((target == class_id).astype(np.float32))[None, None]
        dilated = F.max_pool2d(mask, kernel, stride=1, padding=config.radius_pixels)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=config.radius_pixels)
        if config.boundary_side == "inner":
            class_uncertain = (mask > 0.5) & (eroded < 0.5)
            if config.preserve_thin_structures:
                supported_by_core = F.max_pool2d(
                    eroded,
                    kernel,
                    stride=1,
                    padding=config.radius_pixels,
                ) > 0.5
                class_uncertain &= supported_by_core
        else:
            class_uncertain = (dilated > 0.5) & (eroded < 0.5)
        uncertain |= class_uncertain[0, 0].numpy()

    uncertain &= target != config.ignore_index
    output[uncertain] = config.ignore_index
    return output, uncertain


def smoke_test() -> dict[str, Any]:
    target = np.zeros((13, 13), dtype=np.int64)
    # A wide road-like region has a stable interior core, while the isolated
    # one-pixel bare-land region verifies that tiny structures are preserved.
    target[2:9, 2:9] = 1
    target[11, 11] = 3
    original = target.copy()
    config = LabelUncertaintyConfig(enabled=True)
    transformed, uncertain = apply_boundary_uncertainty(target, config, num_classes=5)
    report = {
        "source_unchanged": bool(np.array_equal(target, original)),
        "uncertain_pixels": int(uncertain.sum()),
        "ignore_pixels_added": int((transformed == 255).sum()),
        "shape_preserved": transformed.shape == target.shape,
        "thin_bare_land_preserved": bool(transformed[11, 11] == 3),
    }
    report["all_checks_passed"] = all(
        (
            report["source_unchanged"],
            report["uncertain_pixels"] > 0,
            report["shape_preserved"],
            report["thin_bare_land_preserved"],
        )
    )
    return report


if __name__ == "__main__":
    import json

    result = smoke_test()
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)
