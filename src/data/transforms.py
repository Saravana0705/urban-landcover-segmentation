"""Joint augmentation transforms for Sentinel-1 semantic segmentation.

The transforms operate on a dictionary with:
    image:    NumPy array shaped (C, H, W), linear Sigma0
    target:   NumPy array shaped (H, W), model IDs 0-4 or ignore_index 255
    validity: NumPy array shaped (H, W), values 0/1

All spatial transforms are applied identically to image, target and validity.

Research policy
---------------
Default training augmentation uses only label-preserving geometric transforms:
- random horizontal flip
- random vertical flip
- random 90-degree rotation

Optional SAR radiometric perturbations are implemented but disabled by
default. They should be evaluated as a separate ablation:
- small multiplicative gain per band
- light multiplicative speckle noise

Validation and test use the identity transform.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ArrayDict = dict[str, np.ndarray]


def _validate_probability(value: float, name: str) -> None:
    """Validate a probability."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _validate_sample(sample: Mapping[str, np.ndarray]) -> None:
    """Validate the augmentation sample structure."""
    required = {"image", "target", "validity"}
    missing = required.difference(sample)

    if missing:
        raise ValueError(
            f"Augmentation sample is missing keys: {sorted(missing)}"
        )

    image = np.asarray(sample["image"])
    target = np.asarray(sample["target"])
    validity = np.asarray(sample["validity"])

    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape (C, H, W), found {image.shape}."
        )

    if target.ndim != 2:
        raise ValueError(
            f"Expected target shape (H, W), found {target.shape}."
        )

    if validity.ndim != 2:
        raise ValueError(
            f"Expected validity shape (H, W), found {validity.shape}."
        )

    if image.shape[1:] != target.shape:
        raise ValueError(
            "Image and target dimensions are not aligned."
        )

    if target.shape != validity.shape:
        raise ValueError(
            "Target and validity dimensions are not aligned."
        )


@dataclass(frozen=True)
class AugmentationConfig:
    """Serializable augmentation settings."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotation_90_probability: float = 0.5

    enable_sar_intensity: bool = False
    sar_intensity_probability: float = 0.25

    gain_db_min: float = -1.0
    gain_db_max: float = 1.0

    enable_speckle: bool = False
    speckle_probability: float = 0.15
    speckle_looks: float = 25.0

    ignore_index: int = 255

    def validate(self) -> None:
        """Validate configuration values."""
        _validate_probability(
            self.horizontal_flip_probability,
            "horizontal_flip_probability",
        )
        _validate_probability(
            self.vertical_flip_probability,
            "vertical_flip_probability",
        )
        _validate_probability(
            self.rotation_90_probability,
            "rotation_90_probability",
        )
        _validate_probability(
            self.sar_intensity_probability,
            "sar_intensity_probability",
        )
        _validate_probability(
            self.speckle_probability,
            "speckle_probability",
        )

        if self.gain_db_min > self.gain_db_max:
            raise ValueError(
                "gain_db_min cannot be greater than gain_db_max."
            )

        if self.speckle_looks <= 0:
            raise ValueError(
                "speckle_looks must be greater than zero."
            )

        if self.ignore_index != 255:
            raise ValueError(
                "Dataset V1 expects ignore_index 255."
            )


class IdentityTransform:
    """Return an unchanged contiguous copy."""

    def __call__(
        self,
        sample: Mapping[str, np.ndarray],
    ) -> ArrayDict:
        _validate_sample(sample)

        return {
            "image": np.ascontiguousarray(
                np.asarray(sample["image"], dtype=np.float32)
            ),
            "target": np.ascontiguousarray(
                np.asarray(sample["target"], dtype=np.int64)
            ),
            "validity": np.ascontiguousarray(
                np.asarray(sample["validity"], dtype=np.uint8)
            ),
        }


class Sentinel1TrainTransform:
    """Joint geometric augmentation with optional SAR perturbation.

    The transform uses NumPy's process-local random generator. DataLoader
    workers are seeded by ``src.data.dataloader.seed_worker``, so runs are
    reproducible when the same global seed and worker settings are used.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
    ) -> None:
        self.config = (
            AugmentationConfig()
            if config is None
            else config
        )
        self.config.validate()

    def _apply_spatial(
        self,
        image: np.ndarray,
        target: np.ndarray,
        validity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply synchronized flips and right-angle rotations."""
        if (
            np.random.random()
            < self.config.horizontal_flip_probability
        ):
            image = np.flip(image, axis=2)
            target = np.flip(target, axis=1)
            validity = np.flip(validity, axis=1)

        if (
            np.random.random()
            < self.config.vertical_flip_probability
        ):
            image = np.flip(image, axis=1)
            target = np.flip(target, axis=0)
            validity = np.flip(validity, axis=0)

        if (
            np.random.random()
            < self.config.rotation_90_probability
        ):
            quarter_turns = int(
                np.random.randint(1, 4)
            )
            image = np.rot90(
                image,
                k=quarter_turns,
                axes=(1, 2),
            )
            target = np.rot90(
                target,
                k=quarter_turns,
                axes=(0, 1),
            )
            validity = np.rot90(
                validity,
                k=quarter_turns,
                axes=(0, 1),
            )

        return image, target, validity

    def _apply_gain(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply a small independent multiplicative gain per SAR band."""
        channel_count = image.shape[0]

        gain_db = np.random.uniform(
            low=self.config.gain_db_min,
            high=self.config.gain_db_max,
            size=(channel_count, 1, 1),
        ).astype(np.float32)

        linear_gain = np.power(
            10.0,
            gain_db / 10.0,
        ).astype(np.float32)

        return image * linear_gain

    def _apply_speckle(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply light multiplicative Gamma speckle noise.

        For L looks, Gamma(shape=L, scale=1/L) has mean 1 and variance 1/L.
        """
        looks = float(self.config.speckle_looks)

        noise = np.random.gamma(
            shape=looks,
            scale=1.0 / looks,
            size=image.shape,
        ).astype(np.float32)

        return image * noise

    def __call__(
        self,
        sample: Mapping[str, np.ndarray],
    ) -> ArrayDict:
        """Apply augmentation and preserve mask semantics."""
        _validate_sample(sample)

        image = np.asarray(
            sample["image"],
            dtype=np.float32,
        ).copy()

        target = np.asarray(
            sample["target"],
            dtype=np.int64,
        ).copy()

        validity = np.asarray(
            sample["validity"],
            dtype=np.uint8,
        ).copy()

        original_target_values = set(
            int(value)
            for value in np.unique(target)
        )
        original_valid_pixel_count = int(
            (validity == 1).sum()
        )
        original_ignore_pixel_count = int(
            (target == self.config.ignore_index).sum()
        )

        image, target, validity = self._apply_spatial(
            image,
            target,
            validity,
        )

        if (
            self.config.enable_sar_intensity
            and np.random.random()
            < self.config.sar_intensity_probability
        ):
            image = self._apply_gain(image)

        if (
            self.config.enable_speckle
            and np.random.random()
            < self.config.speckle_probability
        ):
            image = self._apply_speckle(image)

        if not np.isfinite(image).all():
            raise ValueError(
                "Augmentation produced non-finite image values."
            )

        if np.any(image < 0):
            raise ValueError(
                "Augmentation produced negative linear Sigma0 values."
            )

        transformed_target_values = set(
            int(value)
            for value in np.unique(target)
        )

        if transformed_target_values != original_target_values:
            raise RuntimeError(
                "Spatial augmentation changed target class values."
            )

        if int((validity == 1).sum()) != original_valid_pixel_count:
            raise RuntimeError(
                "Spatial augmentation changed valid-pixel count."
            )

        if (
            int((target == self.config.ignore_index).sum())
            != original_ignore_pixel_count
        ):
            raise RuntimeError(
                "Spatial augmentation changed ignore-pixel count."
            )

        invalid_not_ignored = (
            (validity != 1)
            & (target != self.config.ignore_index)
        )

        if invalid_not_ignored.any():
            raise RuntimeError(
                "Augmentation broke target/validity consistency."
            )

        return {
            "image": np.ascontiguousarray(
                image,
                dtype=np.float32,
            ),
            "target": np.ascontiguousarray(
                target,
                dtype=np.int64,
            ),
            "validity": np.ascontiguousarray(
                validity,
                dtype=np.uint8,
            ),
        }


def build_train_transform(
    *,
    enable_sar_intensity: bool = False,
    enable_speckle: bool = False,
) -> Sentinel1TrainTransform:
    """Build the recommended training transform."""
    config = AugmentationConfig(
        enable_sar_intensity=enable_sar_intensity,
        enable_speckle=enable_speckle,
    )
    return Sentinel1TrainTransform(config=config)


def build_evaluation_transform() -> IdentityTransform:
    """Build the deterministic validation/test transform."""
    return IdentityTransform()


def smoke_test(
    iterations: int = 25,
    seed: int = 20260725,
) -> dict[str, Any]:
    """Run a synthetic augmentation integrity test."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero.")

    np.random.seed(seed)

    height = 256
    width = 256

    image = np.stack(
        [
            np.linspace(
                0.001,
                1.0,
                height * width,
                dtype=np.float32,
            ).reshape(height, width),
            np.linspace(
                0.002,
                0.5,
                height * width,
                dtype=np.float32,
            ).reshape(height, width),
        ],
        axis=0,
    )

    target = np.zeros(
        (height, width),
        dtype=np.int64,
    )
    target[:64, :] = 0
    target[64:128, :] = 1
    target[128:192, :] = 2
    target[192:, :128] = 3
    target[192:, 128:] = 4
    target[:, :16] = 255

    validity = (
        target != 255
    ).astype(np.uint8)

    transform = Sentinel1TrainTransform(
        AugmentationConfig(
            enable_sar_intensity=False,
            enable_speckle=False,
        )
    )

    expected_values = sorted(
        int(value)
        for value in np.unique(target)
    )
    expected_valid_count = int(
        validity.sum()
    )
    expected_ignore_count = int(
        (target == 255).sum()
    )

    for _ in range(iterations):
        transformed = transform(
            {
                "image": image,
                "target": target,
                "validity": validity,
            }
        )

        transformed_image = transformed["image"]
        transformed_target = transformed["target"]
        transformed_validity = transformed["validity"]

        if transformed_image.shape != image.shape:
            raise RuntimeError(
                "Image shape changed during augmentation."
            )

        if transformed_target.shape != target.shape:
            raise RuntimeError(
                "Target shape changed during augmentation."
            )

        if transformed_validity.shape != validity.shape:
            raise RuntimeError(
                "Validity shape changed during augmentation."
            )

        if sorted(
            int(value)
            for value in np.unique(transformed_target)
        ) != expected_values:
            raise RuntimeError(
                "Target values changed during augmentation."
            )

        if int(transformed_validity.sum()) != expected_valid_count:
            raise RuntimeError(
                "Validity count changed during augmentation."
            )

        if int(
            (transformed_target == 255).sum()
        ) != expected_ignore_count:
            raise RuntimeError(
                "Ignore count changed during augmentation."
            )

    return {
        "iterations": iterations,
        "seed": seed,
        "image_shape": list(image.shape),
        "target_shape": list(target.shape),
        "target_values": expected_values,
        "valid_pixel_count": expected_valid_count,
        "ignore_pixel_count": expected_ignore_count,
        "default_policy": asdict(AugmentationConfig()),
        "all_checks_passed": True,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Sentinel-1 augmentation smoke test."
        )
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "metadata/dataset_v1/"
            "augmentation_smoke_test.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run and save augmentation smoke-test results."""
    args = parse_arguments()

    report = smoke_test(
        iterations=args.iterations,
        seed=args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSentinel-1 augmentation smoke test")
    print("----------------------------------")
    print(f"Iterations: {report['iterations']}")
    print(f"Image shape: {report['image_shape']}")
    print(f"Target shape: {report['target_shape']}")
    print(f"Target values: {report['target_values']}")
    print(
        f"Valid pixels: {report['valid_pixel_count']:,}"
    )
    print(
        f"Ignore pixels: {report['ignore_pixel_count']:,}"
    )
    print(
        "SAR intensity augmentation enabled by default: "
        f"{report['default_policy']['enable_sar_intensity']}"
    )
    print(
        "Speckle augmentation enabled by default: "
        f"{report['default_policy']['enable_speckle']}"
    )
    print(f"Report: {args.output}")
    print(
        "\nResult: augmentation integrity checks passed."
    )


if __name__ == "__main__":
    main()
