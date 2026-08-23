"""Validate and record Dataset V2.2 normalization inheritance.

Dataset V2.2 uses the same 14 training cities as Dataset V1/V2.1.
The established normalization procedure computes statistics from the
full-resolution SAR rasters of those training cities, not from individual
selected tiles.

Therefore Dataset V2.2 legitimately inherits the existing training-only
normalization without recomputation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


V22_MANIFEST = Path(
    "metadata/dataset_v22/freeze/dataset_manifest.csv"
)

V22_CONFIG = Path(
    "metadata/dataset_v22/freeze/dataset_config.json"
)

SOURCE_NORMALIZATION_JSON = Path(
    "metadata/dataset_v1/normalization/training_normalization.json"
)

SOURCE_NORMALIZATION_CSV = Path(
    "metadata/dataset_v1/normalization/training_normalization.csv"
)

OUTPUT_DIR = Path(
    "metadata/dataset_v22/normalization"
)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}"
        )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:

    for path, label in (
        (
            V22_MANIFEST,
            "Frozen V2.2 manifest",
        ),
        (
            V22_CONFIG,
            "Frozen V2.2 config",
        ),
        (
            SOURCE_NORMALIZATION_JSON,
            "V1 training normalization JSON",
        ),
    ):
        require_file(path, label)

    manifest = pd.read_csv(
        V22_MANIFEST
    )

    if manifest.empty:
        raise RuntimeError(
            "Frozen V2.2 manifest is empty."
        )

    splits = set(
        manifest["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if splits != {"train"}:
        raise RuntimeError(
            "Frozen V2.2 training manifest must contain "
            f"train rows only. Found: {sorted(splits)}"
        )

    v22_training_city_ids = (
        manifest["city_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    with SOURCE_NORMALIZATION_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        normalization = json.load(file)

    source_training_city_ids = sorted(
        str(city_id)
        for city_id in normalization[
            "training_city_ids"
        ]
    )

    training_city_count = int(
        normalization[
            "training_city_count"
        ]
    )

    validation_and_test_used = bool(
        normalization[
            "validation_and_test_used"
        ]
    )

    city_ids_match = (
        v22_training_city_ids
        == source_training_city_ids
    )

    city_count_match = (
        len(v22_training_city_ids)
        == training_city_count
        == 14
    )

    leakage_check = (
        validation_and_test_used is False
    )

    mandatory_checks = {
        "v22_training_city_count_is_14":
            len(v22_training_city_ids) == 14,

        "normalization_training_city_count_is_14":
            training_city_count == 14,

        "training_city_ids_match":
            city_ids_match,

        "validation_and_test_not_used":
            leakage_check,
    }

    print()
    print("Dataset V2.2 normalization inheritance")
    print("======================================")
    print()
    print(
        f"V2.2 training cities: "
        f"{len(v22_training_city_ids)}"
    )
    print(
        f"Normalization training cities: "
        f"{training_city_count}"
    )

    print()
    print("Mandatory checks")
    print("----------------")

    for name, passed in mandatory_checks.items():
        print(
            f"{name}: {passed}"
        )

    if not all(
        mandatory_checks.values()
    ):
        raise RuntimeError(
            "Dataset V2.2 normalization inheritance "
            "validation FAILED."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inherited_json = (
        OUTPUT_DIR
        / "training_normalization.json"
    )

    shutil.copy2(
        SOURCE_NORMALIZATION_JSON,
        inherited_json,
    )

    inherited_csv = None

    if SOURCE_NORMALIZATION_CSV.exists():

        inherited_csv = (
            OUTPUT_DIR
            / "training_normalization.csv"
        )

        shutil.copy2(
            SOURCE_NORMALIZATION_CSV,
            inherited_csv,
        )

    provenance = {
        "dataset_version": "v2.2",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "normalization_policy": (
            "inherited_from_dataset_v1"
        ),
        "reason": (
            "Existing normalization is computed from "
            "full-resolution SAR rasters of the 14 training "
            "cities. Dataset V2.2 uses the same 14 training "
            "cities; only training-tile membership changed."
        ),
        "source_normalization": str(
            SOURCE_NORMALIZATION_JSON
        ),
        "source_sha256": sha256_file(
            SOURCE_NORMALIZATION_JSON
        ),
        "v22_copy": str(
            inherited_json
        ),
        "v22_copy_sha256": sha256_file(
            inherited_json
        ),
        "training_city_ids": (
            v22_training_city_ids
        ),
        "validation_and_test_used": False,
        "mandatory_checks": (
            mandatory_checks
        ),
    }

    provenance_path = (
        OUTPUT_DIR
        / "normalization_provenance.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            provenance,
            file,
            indent=2,
            ensure_ascii=False,
        )

    checksum_match = (
        provenance["source_sha256"]
        == provenance["v22_copy_sha256"]
    )

    print()
    print("Normalization freeze")
    print("--------------------")
    print(
        f"Source: {SOURCE_NORMALIZATION_JSON}"
    )
    print(
        f"V2.2 copy: {inherited_json}"
    )
    print(
        f"Checksum identical: {checksum_match}"
    )

    if not checksum_match:
        raise RuntimeError(
            "Copied normalization checksum differs "
            "from Dataset V1 source."
        )

    print()
    print(
        "Result: Dataset V2.2 normalization "
        "inheritance validated successfully."
    )


if __name__ == "__main__":
    main()