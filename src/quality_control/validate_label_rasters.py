"""Validate Sentinel-1 imagery, semantic labels and validity masks.

The script independently verifies that:

1. The Sentinel-1 raster, semantic mask and validity mask use the
   exact same grid.
2. Semantic and validity values belong to the permitted sets.
3. Semantic and validity masks are logically consistent.
4. Semantic NoData occurs only where Sentinel-1 data are invalid.
5. Raster statistics agree with the rasterization report.
6. Checksums and detailed QA results are recorded.

The script exits with an error when any mandatory QA check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox


FLOAT_TOLERANCE = 1e-9

SEMANTIC_CLASS_VALUES = {0, 1, 2, 3, 4, 5}
SEMANTIC_NODATA = 255
VALIDITY_VALUES = {0, 1}

CLASS_NAMES = {
    0: "unlabeled",
    1: "buildings",
    2: "roads",
    3: "vegetation",
    4: "bare_land",
    5: "water",
}


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return data


def approximately_equal(
    value_a: float,
    value_b: float,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    """Compare floating-point values."""
    return math.isclose(
        value_a,
        value_b,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def affine_values(transform: Affine) -> tuple[float, ...]:
    """Return the six affine coefficients."""
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def transforms_equal(
    transform_a: Affine,
    transform_b: Affine,
) -> bool:
    """Compare affine transforms coefficient by coefficient."""
    return all(
        approximately_equal(value_a, value_b)
        for value_a, value_b in zip(
            affine_values(transform_a),
            affine_values(transform_b),
            strict=True,
        )
    )


def bounds_values(
    bounds: BoundingBox,
) -> tuple[float, ...]:
    """Return raster bounds in a stable order."""
    return (
        float(bounds.left),
        float(bounds.bottom),
        float(bounds.right),
        float(bounds.top),
    )


def bounds_equal(
    bounds_a: BoundingBox,
    bounds_b: BoundingBox,
) -> bool:
    """Compare raster bounds."""
    return all(
        approximately_equal(value_a, value_b)
        for value_a, value_b in zip(
            bounds_values(bounds_a),
            bounds_values(bounds_b),
            strict=True,
        )
    )


def raster_metadata(
    dataset: rasterio.io.DatasetReader,
) -> dict[str, Any]:
    """Extract raster-grid metadata."""
    return {
        "driver": dataset.driver,
        "width": int(dataset.width),
        "height": int(dataset.height),
        "band_count": int(dataset.count),
        "crs": str(dataset.crs),
        "epsg": (
            dataset.crs.to_epsg()
            if dataset.crs is not None
            else None
        ),
        "transform": list(affine_values(dataset.transform)),
        "bounds": {
            "left": float(dataset.bounds.left),
            "bottom": float(dataset.bounds.bottom),
            "right": float(dataset.bounds.right),
            "top": float(dataset.bounds.top),
        },
        "pixel_width": float(dataset.transform.a),
        "pixel_height": float(abs(dataset.transform.e)),
        "dtypes": list(dataset.dtypes),
        "nodatavals": list(dataset.nodatavals),
        "descriptions": list(dataset.descriptions),
    }


def value_counts(
    array: np.ndarray,
) -> dict[int, int]:
    """Calculate integer raster-value counts."""
    values, counts = np.unique(
        array,
        return_counts=True,
    )

    return {
        int(value): int(count)
        for value, count in zip(
            values,
            counts,
            strict=True,
        )
    }


def report_count_for_class(
    rasterization_report: dict[str, Any],
    class_name: str,
) -> int | None:
    """Read a final class pixel count from the rasterization report."""
    statistics = rasterization_report.get(
        "final_class_statistics",
        {},
    )

    class_statistics = statistics.get(class_name)

    if not isinstance(class_statistics, dict):
        return None

    pixel_count = class_statistics.get("pixel_count")

    return (
        int(pixel_count)
        if pixel_count is not None
        else None
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate aligned Sentinel-1 imagery and label rasters."
        )
    )

    parser.add_argument(
        "--city-id",
        required=True,
    )

    parser.add_argument(
        "--reference-raster",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--semantic-mask",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--validity-mask",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--rasterization-report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/raster_qa"),
    )

    return parser.parse_args()


def main() -> None:
    """Run all label-raster QA checks."""
    args = parse_arguments()

    input_paths = {
        "reference_raster": args.reference_raster,
        "semantic_mask": args.semantic_mask,
        "validity_mask": args.validity_mask,
        "rasterization_report": args.rasterization_report,
    }

    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rasterization_report = load_json(
        args.rasterization_report
    )

    print("\nLabel Raster QA")
    print("---------------")
    print(f"City: {args.city_id}")
    print(f"Reference: {args.reference_raster}")
    print(f"Semantic mask: {args.semantic_mask}")
    print(f"Validity mask: {args.validity_mask}")

    with rasterio.open(
        args.reference_raster
    ) as reference, rasterio.open(
        args.semantic_mask
    ) as semantic_dataset, rasterio.open(
        args.validity_mask
    ) as validity_dataset:
        reference_metadata = raster_metadata(reference)
        semantic_metadata = raster_metadata(semantic_dataset)
        validity_metadata = raster_metadata(validity_dataset)

        if semantic_dataset.count != 1:
            raise ValueError(
                "Semantic mask must contain exactly one band."
            )

        if validity_dataset.count != 1:
            raise ValueError(
                "Validity mask must contain exactly one band."
            )

        semantic = semantic_dataset.read(1)
        validity = validity_dataset.read(1)

        sar_valid = reference.dataset_mask() > 0

        grid_checks = {
            "semantic_crs_matches_reference":
                semantic_dataset.crs == reference.crs,
            "validity_crs_matches_reference":
                validity_dataset.crs == reference.crs,
            "semantic_dimensions_match_reference": (
                semantic_dataset.width == reference.width
                and semantic_dataset.height == reference.height
            ),
            "validity_dimensions_match_reference": (
                validity_dataset.width == reference.width
                and validity_dataset.height == reference.height
            ),
            "semantic_transform_matches_reference":
                transforms_equal(
                    semantic_dataset.transform,
                    reference.transform,
                ),
            "validity_transform_matches_reference":
                transforms_equal(
                    validity_dataset.transform,
                    reference.transform,
                ),
            "semantic_bounds_match_reference":
                bounds_equal(
                    semantic_dataset.bounds,
                    reference.bounds,
                ),
            "validity_bounds_match_reference":
                bounds_equal(
                    validity_dataset.bounds,
                    reference.bounds,
                ),
            "semantic_pixel_size_matches_reference": (
                approximately_equal(
                    semantic_dataset.transform.a,
                    reference.transform.a,
                )
                and approximately_equal(
                    semantic_dataset.transform.e,
                    reference.transform.e,
                )
            ),
            "validity_pixel_size_matches_reference": (
                approximately_equal(
                    validity_dataset.transform.a,
                    reference.transform.a,
                )
                and approximately_equal(
                    validity_dataset.transform.e,
                    reference.transform.e,
                )
            ),
        }

        grid_checks["all_grid_checks_passed"] = all(
            grid_checks.values()
        )

        semantic_counts = value_counts(semantic)
        validity_counts = value_counts(validity)

        semantic_values = set(semantic_counts)
        validity_values = set(validity_counts)

        allowed_semantic_storage_values = (
            SEMANTIC_CLASS_VALUES | {SEMANTIC_NODATA}
        )

        value_checks = {
            "semantic_values_allowed":
                semantic_values.issubset(
                    allowed_semantic_storage_values
                ),
            "validity_values_allowed":
                validity_values.issubset(
                    VALIDITY_VALUES
                ),
            "semantic_dtype_uint8":
                semantic_dataset.dtypes[0] == "uint8",
            "validity_dtype_uint8":
                validity_dataset.dtypes[0] == "uint8",
            "semantic_nodata_is_255":
                semantic_dataset.nodata == SEMANTIC_NODATA,
            "validity_nodata_is_0":
                validity_dataset.nodata == 0,
        }

        value_checks["all_value_checks_passed"] = all(
            value_checks.values()
        )

        semantic_is_class = np.isin(
            semantic,
            [1, 2, 3, 4, 5],
        )
        semantic_is_unlabeled = semantic == 0
        semantic_is_nodata = (
            semantic == SEMANTIC_NODATA
        )
        validity_is_valid = validity == 1
        validity_is_ignore = validity == 0

        class_without_validity = int(
            np.count_nonzero(
                semantic_is_class
                & ~validity_is_valid
            )
        )

        valid_without_class = int(
            np.count_nonzero(
                validity_is_valid
                & ~semantic_is_class
            )
        )

        unlabeled_marked_valid = int(
            np.count_nonzero(
                semantic_is_unlabeled
                & validity_is_valid
            )
        )

        nodata_marked_valid = int(
            np.count_nonzero(
                semantic_is_nodata
                & validity_is_valid
            )
        )

        semantic_nodata_on_valid_sar = int(
            np.count_nonzero(
                semantic_is_nodata
                & sar_valid
            )
        )

        invalid_sar_without_semantic_nodata = int(
            np.count_nonzero(
                ~sar_valid
                & ~semantic_is_nodata
            )
        )

        invalid_sar_marked_valid = int(
            np.count_nonzero(
                ~sar_valid
                & validity_is_valid
            )
        )

        consistency_checks = {
            "all_class_pixels_have_validity_1":
                class_without_validity == 0,
            "all_validity_1_pixels_have_classes_1_to_5":
                valid_without_class == 0,
            "all_unlabeled_pixels_have_validity_0":
                unlabeled_marked_valid == 0,
            "all_semantic_nodata_pixels_have_validity_0":
                nodata_marked_valid == 0,
            "semantic_nodata_only_on_invalid_sar":
                semantic_nodata_on_valid_sar == 0,
            "all_invalid_sar_pixels_are_semantic_nodata":
                invalid_sar_without_semantic_nodata == 0,
            "all_invalid_sar_pixels_have_validity_0":
                invalid_sar_marked_valid == 0,
        }

        consistency_checks[
            "all_consistency_checks_passed"
        ] = all(consistency_checks.values())

        total_pixels = int(
            reference.width * reference.height
        )

        class_pixel_counts = {
            CLASS_NAMES[class_id]:
                semantic_counts.get(class_id, 0)
            for class_id in CLASS_NAMES
        }

        semantic_nodata_count = (
            semantic_counts.get(
                SEMANTIC_NODATA,
                0,
            )
        )

        validity_valid_count = (
            validity_counts.get(1, 0)
        )
        validity_ignore_count = (
            validity_counts.get(0, 0)
        )

        report_checks: dict[str, bool] = {}

        for class_id, class_name in CLASS_NAMES.items():
            expected_count = report_count_for_class(
                rasterization_report,
                class_name,
            )

            report_checks[
                f"{class_name}_count_matches_report"
            ] = (
                expected_count is not None
                and class_pixel_counts[class_name]
                == expected_count
            )

        report_checks[
            "labelled_count_matches_report"
        ] = (
            validity_valid_count
            == int(
                rasterization_report.get(
                    "labelled_pixels",
                    -1,
                )
            )
        )

        report_checks[
            "unlabelled_count_matches_report"
        ] = (
            class_pixel_counts["unlabeled"]
            == int(
                rasterization_report.get(
                    "unlabelled_pixels",
                    -1,
                )
            )
        )

        report_checks[
            "semantic_nodata_count_matches_report"
        ] = (
            semantic_nodata_count
            == int(
                rasterization_report.get(
                    "semantic_nodata_pixels",
                    -1,
                )
            )
        )

        report_checks[
            "total_pixel_count_is_consistent"
        ] = (
            sum(semantic_counts.values())
            == total_pixels
            and sum(validity_counts.values())
            == total_pixels
        )

        report_checks[
            "all_report_checks_passed"
        ] = all(report_checks.values())

        class_presence_checks = {
            f"{class_name}_present":
                class_pixel_counts[class_name] > 0
            for class_name in (
                "buildings",
                "roads",
                "vegetation",
                "bare_land",
                "water",
            )
        }

        class_presence_checks[
            "all_trainable_classes_present"
        ] = all(class_presence_checks.values())

    checksums = {
        "reference_raster_sha256":
            sha256_file(args.reference_raster),
        "semantic_mask_sha256":
            sha256_file(args.semantic_mask),
        "validity_mask_sha256":
            sha256_file(args.validity_mask),
        "rasterization_report_sha256":
            sha256_file(args.rasterization_report),
    }

    checksum_checks = {
        "reference_checksum_matches_rasterization_report": (
            checksums["reference_raster_sha256"]
            == rasterization_report.get(
                "reference_raster_sha256"
            )
        ),
        "semantic_checksum_matches_rasterization_report": (
            checksums["semantic_mask_sha256"]
            == rasterization_report.get(
                "semantic_mask_sha256"
            )
        ),
        "validity_checksum_matches_rasterization_report": (
            checksums["validity_mask_sha256"]
            == rasterization_report.get(
                "validity_mask_sha256"
            )
        ),
    }

    checksum_checks["all_checksum_checks_passed"] = all(
        checksum_checks.values()
    )

    mandatory_group_checks = {
        "grid": grid_checks["all_grid_checks_passed"],
        "values": value_checks["all_value_checks_passed"],
        "consistency": consistency_checks[
            "all_consistency_checks_passed"
        ],
        "rasterization_report": report_checks[
            "all_report_checks_passed"
        ],
        "class_presence": class_presence_checks[
            "all_trainable_classes_present"
        ],
        "checksums": checksum_checks[
            "all_checksum_checks_passed"
        ],
    }

    overall_pass = all(
        mandatory_group_checks.values()
    )

    qa_report = {
        "city_id": args.city_id,
        "qa_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "input_files": {
            "reference_raster": str(
                args.reference_raster
            ),
            "semantic_mask": str(
                args.semantic_mask
            ),
            "validity_mask": str(
                args.validity_mask
            ),
            "rasterization_report": str(
                args.rasterization_report
            ),
        },
        "reference_metadata": reference_metadata,
        "semantic_metadata": semantic_metadata,
        "validity_metadata": validity_metadata,
        "grid_checks": grid_checks,
        "value_checks": value_checks,
        "consistency_checks": consistency_checks,
        "consistency_error_counts": {
            "class_pixels_without_validity_1":
                class_without_validity,
            "validity_1_pixels_without_class":
                valid_without_class,
            "unlabeled_pixels_marked_valid":
                unlabeled_marked_valid,
            "semantic_nodata_pixels_marked_valid":
                nodata_marked_valid,
            "semantic_nodata_on_valid_sar":
                semantic_nodata_on_valid_sar,
            "invalid_sar_without_semantic_nodata":
                invalid_sar_without_semantic_nodata,
            "invalid_sar_marked_valid":
                invalid_sar_marked_valid,
        },
        "pixel_statistics": {
            "total_pixels": total_pixels,
            "semantic_value_counts": {
                str(value): count
                for value, count
                in sorted(semantic_counts.items())
            },
            "validity_value_counts": {
                str(value): count
                for value, count
                in sorted(validity_counts.items())
            },
            "class_pixel_counts":
                class_pixel_counts,
            "semantic_nodata_pixels":
                semantic_nodata_count,
            "valid_training_pixels":
                validity_valid_count,
            "ignored_pixels":
                validity_ignore_count,
        },
        "rasterization_report_checks":
            report_checks,
        "class_presence_checks":
            class_presence_checks,
        "checksums": checksums,
        "checksum_checks": checksum_checks,
        "mandatory_group_checks":
            mandatory_group_checks,
    }

    output_path = (
        args.output_dir
        / f"{args.city_id}_label_raster_qa.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            qa_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nGrid checks")
    for check_name, passed in grid_checks.items():
        print(f"  {check_name}: {passed}")

    print("\nValue checks")
    for check_name, passed in value_checks.items():
        print(f"  {check_name}: {passed}")

    print("\nCross-mask consistency")
    for check_name, passed in consistency_checks.items():
        print(f"  {check_name}: {passed}")

    print("\nClass pixel counts")
    for class_name, pixel_count in class_pixel_counts.items():
        print(
            f"  {class_name:<12}: "
            f"{pixel_count:,}"
        )

    print(
        f"  semantic_nodata: "
        f"{semantic_nodata_count:,}"
    )
    print(
        f"  valid_training: "
        f"{validity_valid_count:,}"
    )

    print("\nRasterization-report checks")
    for check_name, passed in report_checks.items():
        print(f"  {check_name}: {passed}")

    print("\nChecksum checks")
    for check_name, passed in checksum_checks.items():
        print(f"  {check_name}: {passed}")

    print("\nOverall QA result")
    print("-----------------")
    print("PASS" if overall_pass else "FAIL")
    print(f"QA report: {output_path}")

    if not overall_pass:
        failed_groups = [
            name
            for name, passed
            in mandatory_group_checks.items()
            if not passed
        ]

        raise RuntimeError(
            "Label-raster QA failed. "
            f"Failed groups: {failed_groups}"
        )


if __name__ == "__main__":
    main()