"""PyTorch Dataset for frozen Sentinel-1 segmentation datasets.

The loader preserves the geospatial source files and performs all model-facing
transformations in memory.

Model target mapping
--------------------
Source semantic mask:
    0   = unlabeled
    1   = buildings
    2   = roads
    3   = vegetation
    4   = bare_land
    5   = water
    255 = semantic NoData

PyTorch target:
    0   = buildings
    1   = roads
    2   = vegetation
    3   = bare_land
    4   = water
    255 = ignore_index

Invalid, padded, unlabeled and semantic-NoData pixels are assigned ignore_index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_IGNORE_INDEX = 255

PATH_COLUMN_CANDIDATES = {
    "image": (
        "image_path",
        "image_tile_path",
        "sar_path",
        "sar_tile_path",
        "image",
    ),
    "semantic": (
        "semantic_mask_path",
        "semantic_tile_path",
        "label_path",
        "mask_path",
        "semantic_mask",
    ),
    "validity": (
        "validity_mask_path",
        "validity_tile_path",
        "valid_mask_path",
        "validity_mask",
    ),
}


def _normalization_key(channel_name: str) -> str:
    """Map legacy channel labels to their two-band normalization keys."""
    normalized = str(channel_name).strip()
    aliases = {
        "Sigma0_VV": "VV",
        "Sigma0_VH": "VH",
        "sigma0_vv": "VV",
        "sigma0_vh": "VH",
    }
    return aliases.get(normalized, normalized)


def _normalization_arrays(
    config: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ordered frozen normalization arrays.

    Legacy two-band freezes store ``normalization.VV`` and
    ``normalization.VH``. Dataset V3-MT stores an explicit ordered channel
    list and ``normalization.bands.<band>``. Supporting both layouts keeps
    all earlier frozen datasets reproducible.
    """
    input_config = config.get("input")
    if not isinstance(input_config, Mapping):
        raise ValueError("Dataset configuration is missing input settings.")

    channel_count = int(input_config.get("channel_count", 0))
    channels_raw = input_config.get("channels")
    if channels_raw is None and channel_count == 2:
        # Earliest V1 freezes predate the explicit ordered channel list.
        channels_raw = ["Sigma0_VV", "Sigma0_VH"]
    if not isinstance(channels_raw, (list, tuple)):
        raise ValueError("input.channels must be an ordered list.")
    channels = [str(value).strip() for value in channels_raw]
    if channel_count <= 0 or len(channels) != channel_count:
        raise ValueError(
            "input.channel_count must equal the length of input.channels."
        )
    if any(not value for value in channels) or len(set(channels)) != len(channels):
        raise ValueError("input.channels must contain unique non-empty names.")

    normalization = config.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Dataset configuration is missing normalization settings.")
    bands_raw = normalization.get("bands")
    bands = bands_raw if isinstance(bands_raw, Mapping) else normalization

    constants: list[tuple[float, float, float, float]] = []
    missing: list[str] = []
    for channel in channels:
        key = _normalization_key(channel)
        entry = bands.get(key) if isinstance(bands, Mapping) else None
        if not isinstance(entry, Mapping):
            missing.append(key)
            continue
        constants.append(
            (
                float(entry["clip_lower_db"]),
                float(entry["clip_upper_db"]),
                float(entry["mean_db"]),
                float(entry["std_db"]),
            )
        )
    if missing:
        raise ValueError(
            "Frozen normalization is missing channels: " + ", ".join(missing)
        )

    values = np.asarray(constants, dtype=np.float32)
    if values.shape != (channel_count, 4) or not np.isfinite(values).all():
        raise ValueError("Frozen normalization constants must be finite per band.")
    if np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("Each normalization clip lower bound must be below its upper bound.")
    if np.any(values[:, 3] <= 0):
        raise ValueError("Frozen normalization standard deviations must be positive.")

    epsilon = float(normalization.get("epsilon", 1e-10))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization.epsilon must be finite and positive.")

    shaped = [values[:, index, None, None] for index in range(4)]
    return channels, shaped[0], shaped[1], shaped[2], shaped[3], epsilon


def _require_file(path: Path, label: str) -> None:
    """Require a non-empty file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    _require_file(path, "dataset configuration")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def _resolve_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    role: str,
) -> str:
    """Resolve a manifest column from supported candidate names."""
    available = set(columns)

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"Could not resolve the {role} path column. "
        f"Expected one of {list(candidates)}; "
        f"available columns are {list(columns)}."
    )


def _as_bool(value: Any) -> bool:
    """Convert common CSV boolean values to bool."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    raise ValueError(f"Cannot interpret boolean value: {value!r}")


class Sentinel1UrbanDataset(Dataset[dict[str, Any]]):
    """Frozen Sentinel-1 dataset loader for semantic segmentation.

    Parameters
    ----------
    split:
        One of ``train``, ``val`` or ``test``.
    manifest_path:
        Global Dataset V1 manifest.
    dataset_config_path:
        Frozen model-facing dataset configuration.
    project_root:
        Base directory used to resolve relative paths. Defaults to the current
        working directory.
    joint_transform:
        Optional callable receiving and returning a dictionary containing
        ``image``, ``target`` and ``validity`` NumPy arrays. It must apply the
        same spatial transformation to all three arrays.
    exclude_zero_valid:
        Exclude tiles whose manifest valid-pixel count is zero. Recommended for
        all splits.
    verify_raster_metadata:
        Validate dimensions, band count and alignment while loading each tile.
        Disable only after the dataset pipeline is well established.
    return_raw_semantic:
        Include the original 0/1/2/3/4/5/255 semantic array in each sample.
    """

    def __init__(
        self,
        split: str,
        manifest_path: str | Path = (
            "metadata/dataset_v1/dataset_manifest.csv"
        ),
        dataset_config_path: str | Path = (
            "metadata/dataset_v1/freeze/dataset_config.json"
        ),
        project_root: str | Path | None = None,
        joint_transform: Callable[
            [dict[str, np.ndarray]],
            Mapping[str, np.ndarray],
        ]
        | None = None,
        exclude_zero_valid: bool = True,
        verify_raster_metadata: bool = True,
        return_raw_semantic: bool = False,
    ) -> None:
        super().__init__()

        normalized_split = str(split).strip().lower()
        split_aliases = {
            "training": "train",
            "validation": "val",
            "valid": "val",
            "testing": "test",
        }
        normalized_split = split_aliases.get(
            normalized_split,
            normalized_split,
        )

        if normalized_split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: train, val, test."
            )

        self.split = normalized_split
        self.project_root = (
            Path.cwd()
            if project_root is None
            else Path(project_root)
        ).resolve()

        self.manifest_path = self._resolve_project_path(
            Path(manifest_path)
        )
        self.dataset_config_path = self._resolve_project_path(
            Path(dataset_config_path)
        )

        _require_file(self.manifest_path, "dataset manifest")
        _require_file(
            self.dataset_config_path,
            "dataset configuration",
        )

        self.config = _load_json(self.dataset_config_path)
        self.joint_transform = joint_transform
        self.exclude_zero_valid = exclude_zero_valid
        self.verify_raster_metadata = verify_raster_metadata
        self.return_raw_semantic = return_raw_semantic

        self.ignore_index = int(
            self.config["labels"].get(
                "semantic_nodata",
                DEFAULT_IGNORE_INDEX,
            )
        )

        if self.ignore_index != DEFAULT_IGNORE_INDEX:
            raise ValueError(
                "Frozen Sentinel-1 datasets expect semantic ignore index 255."
            )

        self.tile_size = int(
            self.config["input"]["tile_size"]
        )
        self.channel_count = int(
            self.config["input"]["channel_count"]
        )

        (
            self.channel_names,
            self.clip_lower,
            self.clip_upper,
            self.mean,
            self.std,
            self.epsilon,
        ) = _normalization_arrays(self.config)
        if len(self.channel_names) != self.channel_count:
            raise ValueError("Frozen channel metadata is internally inconsistent.")
        self.enforce_band_descriptions = bool(
            self.config["input"].get("enforce_band_descriptions", False)
        )

        classes = self.config["labels"]["classes"]
        self.class_names = [
            str(item["class_name"])
            for item in sorted(
                classes,
                key=lambda item: int(item["class_id"]),
            )
        ]

        self.num_classes = len(self.class_names)

        if self.num_classes != 5:
            raise ValueError(
                "The dataset must contain exactly five semantic classes."
            )

        manifest = pd.read_csv(self.manifest_path)

        required_columns = {
            "tile_id",
            "city_id",
            "city_name",
            "split",
            "valid_pixel_count",
        }
        missing = required_columns.difference(manifest.columns)

        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        if manifest["tile_id"].duplicated().any():
            raise ValueError(
                "Manifest contains duplicate tile IDs."
            )

        self.image_column = _resolve_column(
            manifest.columns,
            PATH_COLUMN_CANDIDATES["image"],
            "image",
        )
        self.semantic_column = _resolve_column(
            manifest.columns,
            PATH_COLUMN_CANDIDATES["semantic"],
            "semantic-mask",
        )
        self.validity_column = _resolve_column(
            manifest.columns,
            PATH_COLUMN_CANDIDATES["validity"],
            "validity-mask",
        )

        if "tile_qa_status" in manifest.columns:
            failed_qa = (
                manifest["tile_qa_status"].astype(str) != "PASS"
            )
            if failed_qa.any():
                raise ValueError(
                    "Manifest contains tiles that did not pass QA."
                )

        split_rows = manifest.loc[
            manifest["split"].astype(str).str.lower()
            == self.split
        ].copy()

        expected_city_ids = self._expected_city_ids(self.split)
        actual_city_ids = set(
            split_rows["city_id"].astype(str).unique()
        )

        if actual_city_ids != set(expected_city_ids):
            raise ValueError(
                f"{self.split} manifest cities do not match "
                "the frozen dataset configuration."
            )

        if self.exclude_zero_valid:
            split_rows = split_rows.loc[
                split_rows["valid_pixel_count"].astype(np.int64) > 0
            ].copy()

        split_rows = split_rows.sort_values(
            ["city_id", "tile_id"]
        ).reset_index(drop=True)

        if split_rows.empty:
            raise ValueError(
                f"No usable rows found for split {self.split!r}."
            )

        self.records = split_rows.to_dict(orient="records")

    def _resolve_project_path(self, path: Path) -> Path:
        """Resolve relative paths against project root."""
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _record_path(self, value: Any) -> Path:
        """Resolve a manifest path across Windows and POSIX systems."""
        normalized = str(value).replace("\\", "/")
        path = Path(normalized)
        return self._resolve_project_path(path)

    def _expected_city_ids(self, split: str) -> list[str]:
        """Return frozen city IDs for one split."""
        split_key = {
            "train": "train_city_ids",
            "val": "validation_city_ids",
            "test": "test_city_ids",
        }[split]

        return [
            str(value)
            for value in self.config["splits"][split_key]
        ]

    def __len__(self) -> int:
        """Return usable tile count."""
        return len(self.records)

    def _read_tile_triplet(
        self,
        record: Mapping[str, Any],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """Read image, semantic mask and validity mask."""
        image_path = self._record_path(
            record[self.image_column]
        )
        semantic_path = self._record_path(
            record[self.semantic_column]
        )
        validity_path = self._record_path(
            record[self.validity_column]
        )

        _require_file(image_path, "image tile")
        _require_file(semantic_path, "semantic tile")
        _require_file(validity_path, "validity tile")

        with rasterio.open(image_path) as image_dataset:
            image = image_dataset.read().astype(
                np.float32,
                copy=False,
            )
            image_transform = image_dataset.transform
            image_crs = image_dataset.crs
            image_shape = (
                image_dataset.height,
                image_dataset.width,
            )
            image_descriptions = tuple(
                str(value).strip() if value is not None else ""
                for value in image_dataset.descriptions
            )

        with rasterio.open(semantic_path) as semantic_dataset:
            semantic = semantic_dataset.read(1)
            semantic_transform = semantic_dataset.transform
            semantic_crs = semantic_dataset.crs
            semantic_shape = (
                semantic_dataset.height,
                semantic_dataset.width,
            )

        with rasterio.open(validity_path) as validity_dataset:
            validity = validity_dataset.read(1)
            validity_transform = validity_dataset.transform
            validity_crs = validity_dataset.crs
            validity_shape = (
                validity_dataset.height,
                validity_dataset.width,
            )

        if self.verify_raster_metadata:
            expected_shape = (
                self.tile_size,
                self.tile_size,
            )

            if image.shape != (
                self.channel_count,
                *expected_shape,
            ):
                raise ValueError(
                    f"Unexpected image shape {image.shape} "
                    f"for tile {record['tile_id']}."
                )

            if self.enforce_band_descriptions and image_descriptions != tuple(
                self.channel_names
            ):
                raise ValueError(
                    f"Unexpected image band order {image_descriptions} "
                    f"for tile {record['tile_id']}; expected "
                    f"{tuple(self.channel_names)}."
                )

            if semantic_shape != expected_shape:
                raise ValueError(
                    f"Unexpected semantic shape {semantic_shape} "
                    f"for tile {record['tile_id']}."
                )

            if validity_shape != expected_shape:
                raise ValueError(
                    f"Unexpected validity shape {validity_shape} "
                    f"for tile {record['tile_id']}."
                )

            if image_shape != semantic_shape or image_shape != validity_shape:
                raise ValueError(
                    f"Raster dimensions are not aligned for "
                    f"tile {record['tile_id']}."
                )

            if not (
                image_transform == semantic_transform
                and image_transform == validity_transform
            ):
                raise ValueError(
                    f"Raster transforms are not aligned for "
                    f"tile {record['tile_id']}."
                )

            if not (
                image_crs == semantic_crs
                and image_crs == validity_crs
            ):
                raise ValueError(
                    f"Raster CRSs are not aligned for "
                    f"tile {record['tile_id']}."
                )

        metadata = {
            "image_path": str(image_path),
            "semantic_path": str(semantic_path),
            "validity_path": str(validity_path),
            "transform": image_transform,
            "crs": str(image_crs),
            "band_descriptions": image_descriptions,
        }

        return image, semantic, validity, metadata

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the frozen training-only normalization policy."""
        finite = np.isfinite(image)
        positive = image > self.epsilon
        usable = finite & positive

        db = np.full_like(
            image,
            fill_value=np.nan,
            dtype=np.float32,
        )

        db[usable] = (
            10.0
            * np.log10(
                np.maximum(
                    image[usable],
                    self.epsilon,
                )
            )
        ).astype(np.float32)

        db = np.clip(
            db,
            self.clip_lower,
            self.clip_upper,
        )

        normalized = (
            (db - self.mean)
            / self.std
        ).astype(np.float32)

        normalized[~np.isfinite(normalized)] = 0.0

        return normalized

    def _prepare_target(
        self,
        semantic: np.ndarray,
        validity: np.ndarray,
    ) -> np.ndarray:
        """Convert geospatial IDs 1–5 to model IDs 0–4."""
        semantic_int = semantic.astype(
            np.int64,
            copy=False,
        )
        validity_bool = validity.astype(
            np.uint8,
            copy=False,
        ) == 1

        valid_class = (
            (semantic_int >= 1)
            & (semantic_int <= self.num_classes)
            & validity_bool
        )

        target = np.full(
            semantic_int.shape,
            fill_value=self.ignore_index,
            dtype=np.int64,
        )

        target[valid_class] = (
            semantic_int[valid_class] - 1
        )

        return target

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return one normalized segmentation sample."""
        record = self.records[index]

        image, raw_semantic, validity, raster_metadata = (
            self._read_tile_triplet(record)
        )

        target = self._prepare_target(
            raw_semantic,
            validity,
        )

        if self.joint_transform is not None:
            transformed = self.joint_transform(
                {
                    "image": image,
                    "target": target,
                    "validity": validity,
                }
            )

            required = {"image", "target", "validity"}
            missing = required.difference(transformed)

            if missing:
                raise ValueError(
                    f"joint_transform did not return: {sorted(missing)}"
                )

            image = np.asarray(
                transformed["image"],
                dtype=np.float32,
            )
            target = np.asarray(
                transformed["target"],
                dtype=np.int64,
            )
            validity = np.asarray(
                transformed["validity"],
                dtype=np.uint8,
            )

        normalized_image = self._normalize_image(image)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(normalized_image)
        ).to(dtype=torch.float32)

        target_tensor = torch.from_numpy(
            np.ascontiguousarray(target)
        ).to(dtype=torch.long)

        validity_tensor = torch.from_numpy(
            np.ascontiguousarray(validity == 1)
        ).to(dtype=torch.bool)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "target": target_tensor,
            "validity": validity_tensor,
            "tile_id": str(record["tile_id"]),
            "city_id": str(record["city_id"]),
            "city_name": str(record["city_name"]),
            "split": self.split,
            "ignore_index": self.ignore_index,
            "paths": {
                "image": raster_metadata["image_path"],
                "semantic": raster_metadata["semantic_path"],
                "validity": raster_metadata["validity_path"],
            },
        }

        if self.return_raw_semantic:
            sample["raw_semantic"] = torch.from_numpy(
                np.ascontiguousarray(
                    raw_semantic.astype(np.uint8)
                )
            )

        return sample

    def describe(self) -> dict[str, Any]:
        """Return a compact dataset description."""
        city_ids = sorted(
            {
                str(record["city_id"])
                for record in self.records
            }
        )

        return {
            "dataset_version": self.config["dataset_version"],
            "split": self.split,
            "tile_count": len(self),
            "city_count": len(city_ids),
            "city_ids": city_ids,
            "tile_size": self.tile_size,
            "channel_count": self.channel_count,
            "channel_names": list(self.channel_names),
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "exclude_zero_valid": self.exclude_zero_valid,
            "manifest_path": str(self.manifest_path),
            "dataset_config_path": str(
                self.dataset_config_path
            ),
        }


def inspect_dataset(
    split: str = "train",
    sample_index: int = 0,
) -> dict[str, Any]:
    """Convenience smoke test callable from a Python shell."""
    dataset = Sentinel1UrbanDataset(split=split)
    sample = dataset[sample_index]

    target = sample["target"]
    valid_target = target != dataset.ignore_index

    return {
        **dataset.describe(),
        "sample_index": sample_index,
        "sample_tile_id": sample["tile_id"],
        "image_shape": tuple(sample["image"].shape),
        "image_dtype": str(sample["image"].dtype),
        "image_all_finite": bool(
            torch.isfinite(sample["image"]).all().item()
        ),
        "target_shape": tuple(target.shape),
        "target_dtype": str(target.dtype),
        "valid_target_pixels": int(
            valid_target.sum().item()
        ),
        "target_values": sorted(
            int(value)
            for value in torch.unique(target).tolist()
        ),
    }
