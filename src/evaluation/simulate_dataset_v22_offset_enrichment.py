"""Simulate Dataset V2.2 minority-class enrichment using an offset tile grid.

This script DOES NOT create raster tiles.

It evaluates new 256x256 candidate windows positioned halfway between the
existing Dataset V1/V2.1 grid cells. Candidate class composition is measured
directly from the full-city semantic and validity rasters.

The objective is to enrich minority classes while keeping the frozen
validation/test datasets unchanged.

Offset grid:
    existing grid: 0, 256, 512, ...
    V2.2 grid:     128, 384, 640, ...

Outputs are analysis CSV files only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


MANIFEST_PATH = Path(
    "metadata/dataset_v21/dataset_manifest.csv"
)

OUTPUT_DIR = Path(
    "reports/dataset_v22_offset_enrichment"
)

TILE_SIZE = 256
OFFSET = TILE_SIZE // 2
STRIDE = 256

# Candidate must contain enough usable labelled pixels.
MIN_VALID_FRACTION = 0.20

# Minority-class thresholds measured as fraction of VALID pixels.
BUILDING_THRESHOLD = 0.10
ROAD_THRESHOLD = 0.05

BARE_LAND_MODERATE_THRESHOLD = 0.01
BARE_LAND_RICH_THRESHOLD = 0.05

WATER_THRESHOLD = 0.05

# Maximum enrichment contribution from any one city.
MAX_SELECTED_PER_CITY = 80

CLASS_IDS = {
    "buildings": 1,
    "roads": 2,
    "vegetation": 3,
    "bare_land": 4,
    "water": 5,
}


def offset_starts(
    size: int,
    offset: int = OFFSET,
    stride: int = STRIDE,
) -> list[int]:
    """Generate the non-overlapping offset-grid start coordinates."""
    if size <= offset:
        return []

    return list(range(offset, size, stride))


def calculate_candidate_statistics(
    semantic: np.ndarray,
    validity: np.ndarray,
) -> dict[str, float | int | bool]:
    """Calculate valid-pixel class composition for one candidate."""

    tile_pixels = semantic.size

    valid = validity == 1
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / tile_pixels

    result: dict[str, float | int | bool] = {
        "valid_pixel_count": valid_count,
        "valid_fraction": valid_fraction,
    }

    for class_name, class_id in CLASS_IDS.items():
        count = int(
            np.count_nonzero(
                valid & (semantic == class_id)
            )
        )

        fraction = (
            count / valid_count
            if valid_count > 0
            else 0.0
        )

        result[f"{class_name}_pixel_count"] = count
        result[f"{class_name}_fraction_valid"] = fraction

    result["building_rich"] = (
        result["buildings_fraction_valid"]
        >= BUILDING_THRESHOLD
    )

    result["road_rich"] = (
        result["roads_fraction_valid"]
        >= ROAD_THRESHOLD
    )

    result["bare_land_moderate"] = (
        result["bare_land_fraction_valid"]
        >= BARE_LAND_MODERATE_THRESHOLD
    )

    result["bare_land_rich"] = (
        result["bare_land_fraction_valid"]
        >= BARE_LAND_RICH_THRESHOLD
    )

    result["water_rich"] = (
        result["water_fraction_valid"]
        >= WATER_THRESHOLD
    )

    minority_classes_ge_5pct = sum(
        [
            result["buildings_fraction_valid"] >= 0.05,
            result["roads_fraction_valid"] >= 0.05,
            result["bare_land_fraction_valid"] >= 0.05,
            result["water_fraction_valid"] >= 0.05,
        ]
    )

    result["minority_classes_ge_5pct"] = (
        minority_classes_ge_5pct
    )

    result["mixed_minority"] = (
        minority_classes_ge_5pct >= 2
    )

    return result


def enrichment_score(
    row: pd.Series,
) -> float:
    """Assign priority to useful minority-class enrichment candidates."""

    score = 0.0

    # Bare land receives the strongest emphasis because it is currently
    # the most difficult/underrepresented class.
    score += min(
        float(row["bare_land_fraction_valid"]) / 0.05,
        3.0,
    ) * 4.0

    # Roads are the next major minority-class target.
    score += min(
        float(row["roads_fraction_valid"]) / 0.05,
        3.0,
    ) * 3.0

    # Water is useful but geographically concentrated.
    score += min(
        float(row["water_fraction_valid"]) / 0.05,
        3.0,
    ) * 2.0

    # Buildings already perform better but remain useful urban context.
    score += min(
        float(row["buildings_fraction_valid"]) / 0.10,
        3.0,
    ) * 1.5

    # Reward genuinely mixed urban/minority tiles.
    if bool(row["mixed_minority"]):
        score += 2.0

    # Reward good labelled coverage.
    score += float(row["valid_fraction"])

    # Slightly reduce vegetation-dominated candidates.
    vegetation_fraction = float(
        row["vegetation_fraction_valid"]
    )

    if vegetation_fraction >= 0.90:
        score *= 0.40
    elif vegetation_fraction >= 0.80:
        score *= 0.70

    return float(score)


def useful_candidate(row: pd.Series) -> bool:
    """Return whether candidate contributes to V2.2 enrichment."""

    if float(row["valid_fraction"]) < MIN_VALID_FRACTION:
        return False

    return bool(
        row["bare_land_moderate"]
        or row["road_rich"]
        or row["water_rich"]
        or row["building_rich"]
        or row["mixed_minority"]
    )


def main() -> None:
    """Run V2.2 offset-grid enrichment simulation."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"V2.1 manifest not found: {MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "city_id",
        "city_name",
        "split",
    }

    missing = required_columns.difference(
        manifest.columns
    )

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    # V2.2 modifies training data only.
    train_manifest = manifest.loc[
        manifest["split"].astype(str) == "train"
    ].copy()

    cities = (
        train_manifest[
            ["city_id", "city_name"]
        ]
        .drop_duplicates()
        .sort_values("city_id")
        .reset_index(drop=True)
    )

    print("\nDataset V2.2 offset enrichment simulation")
    print("-----------------------------------------")
    print(f"Training cities: {len(cities)}")
    print(f"V2.1 training rows: {len(train_manifest)}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Offset: {OFFSET}")
    print(f"Candidate stride: {STRIDE}")
    print(
        f"Minimum valid fraction: "
        f"{MIN_VALID_FRACTION:.2f}"
    )

    candidate_rows: list[dict] = []

    for _, city in cities.iterrows():

        city_id = str(city["city_id"])
        city_name = str(city["city_name"])

        folder = f"{city_id}_{city_name}"

        semantic_path = (
            Path("data/processed/masks")
            / folder
            / f"{city_id}_label_mask.tif"
        )

        validity_path = (
            Path("data/processed/masks")
            / folder
            / f"{city_id}_validity_mask.tif"
        )

        if not semantic_path.exists():
            raise FileNotFoundError(
                f"Semantic mask missing: {semantic_path}"
            )

        if not validity_path.exists():
            raise FileNotFoundError(
                f"Validity mask missing: {validity_path}"
            )

        print(
            f"\nScanning {city_id} {city_name}"
        )

        with rasterio.open(
            semantic_path
        ) as semantic_ds, rasterio.open(
            validity_path
        ) as validity_ds:

            if (
                semantic_ds.width != validity_ds.width
                or semantic_ds.height != validity_ds.height
                or semantic_ds.transform
                != validity_ds.transform
                or semantic_ds.crs != validity_ds.crs
            ):
                raise RuntimeError(
                    f"Mask alignment mismatch for {city_id}"
                )

            row_starts = offset_starts(
                semantic_ds.height
            )

            col_starts = offset_starts(
                semantic_ds.width
            )

            city_candidate_count = 0

            for grid_row, row_off in enumerate(
                row_starts
            ):
                for grid_col, col_off in enumerate(
                    col_starts
                ):

                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=TILE_SIZE,
                        height=TILE_SIZE,
                    )

                    semantic = semantic_ds.read(
                        1,
                        window=window,
                        out_shape=(
                            TILE_SIZE,
                            TILE_SIZE,
                        ),
                        boundless=True,
                        fill_value=255,
                    )

                    validity = validity_ds.read(
                        1,
                        window=window,
                        out_shape=(
                            TILE_SIZE,
                            TILE_SIZE,
                        ),
                        boundless=True,
                        fill_value=0,
                    )

                    statistics = (
                        calculate_candidate_statistics(
                            semantic,
                            validity,
                        )
                    )

                    candidate_id = (
                        f"{city_id}_"
                        f"off128_"
                        f"r{grid_row:03d}_"
                        f"c{grid_col:03d}"
                    )

                    row = {
                        "candidate_id": candidate_id,
                        "city_id": city_id,
                        "city_name": city_name,
                        "split": "train",
                        "grid_row": grid_row,
                        "grid_col": grid_col,
                        "row_offset": int(row_off),
                        "col_offset": int(col_off),
                        "tile_size": TILE_SIZE,
                        "grid_offset": OFFSET,
                        "semantic_source": str(
                            semantic_path
                        ),
                        "validity_source": str(
                            validity_path
                        ),
                        **statistics,
                    }

                    candidate_rows.append(row)
                    city_candidate_count += 1

            print(
                f"  Offset candidates: "
                f"{city_candidate_count}"
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    if candidates.empty:
        raise RuntimeError(
            "No V2.2 offset candidates generated."
        )

    candidates["useful_candidate"] = (
        candidates.apply(
            useful_candidate,
            axis=1,
        )
    )

    candidates["enrichment_score"] = (
        candidates.apply(
            enrichment_score,
            axis=1,
        )
    )

    useful = candidates.loc[
        candidates["useful_candidate"]
    ].copy()

    useful = useful.sort_values(
        [
            "enrichment_score",
            "bare_land_fraction_valid",
            "roads_fraction_valid",
            "water_fraction_valid",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    # Per-city cap prevents a few large/rich cities from dominating V2.2.
    selected = (
        useful
        .groupby(
            "city_id",
            group_keys=False,
        )
        .head(MAX_SELECTED_PER_CITY)
        .copy()
    )

    selected["selected_v22"] = True

    selected_ids = set(
        selected["candidate_id"]
    )

    candidates["selected_v22"] = (
        candidates["candidate_id"].isin(
            selected_ids
        )
    )

    # ---------------------------------------------------------
    # City summary
    # ---------------------------------------------------------

    city_summary = (
        candidates
        .groupby(
            ["city_id", "city_name"],
            as_index=False,
        )
        .agg(
            candidate_tiles=(
                "candidate_id",
                "count",
            ),
            useful_candidates=(
                "useful_candidate",
                "sum",
            ),
            selected_tiles=(
                "selected_v22",
                "sum",
            ),
            road_rich_tiles=(
                "road_rich",
                "sum",
            ),
            bare_land_moderate_tiles=(
                "bare_land_moderate",
                "sum",
            ),
            bare_land_rich_tiles=(
                "bare_land_rich",
                "sum",
            ),
            building_rich_tiles=(
                "building_rich",
                "sum",
            ),
            water_rich_tiles=(
                "water_rich",
                "sum",
            ),
            mixed_minority_tiles=(
                "mixed_minority",
                "sum",
            ),
        )
    )

    # ---------------------------------------------------------
    # Projected pixel distribution
    # ---------------------------------------------------------

    class_columns = {
        "buildings": "buildings_pixel_count",
        "roads": "roads_pixel_count",
        "vegetation": "vegetation_pixel_count",
        "bare_land": "bare_land_pixel_count",
        "water": "water_pixel_count",
    }

    baseline_counts = {}

    for class_name, column in class_columns.items():

        if column not in train_manifest.columns:
            raise ValueError(
                f"V2.1 manifest missing {column}"
            )

        baseline_counts[class_name] = int(
            train_manifest[column].sum()
        )

    added_counts = {
        class_name: int(
            selected[column].sum()
        )
        for class_name, column
        in class_columns.items()
    }

    projected_rows = []

    baseline_total = sum(
        baseline_counts.values()
    )

    added_total = sum(
        added_counts.values()
    )

    projected_total = (
        baseline_total + added_total
    )

    for class_name in CLASS_IDS:

        before = baseline_counts[class_name]
        added = added_counts[class_name]
        after = before + added

        projected_rows.append(
            {
                "class_name": class_name,
                "v21_pixels": before,
                "v22_added_pixels": added,
                "projected_v22_pixels": after,
                "v21_fraction": (
                    before / baseline_total
                    if baseline_total
                    else 0.0
                ),
                "projected_v22_fraction": (
                    after / projected_total
                    if projected_total
                    else 0.0
                ),
                "relative_pixel_increase": (
                    added / before
                    if before
                    else np.nan
                ),
            }
        )

    projected_distribution = pd.DataFrame(
        projected_rows
    )

    # ---------------------------------------------------------
    # Save analysis
    # ---------------------------------------------------------

    candidates_path = (
        OUTPUT_DIR
        / "offset_candidates.csv"
    )

    selected_path = (
        OUTPUT_DIR
        / "selected_v22_candidates.csv"
    )

    city_summary_path = (
        OUTPUT_DIR
        / "city_enrichment_summary.csv"
    )

    distribution_path = (
        OUTPUT_DIR
        / "projected_class_distribution.csv"
    )

    candidates.to_csv(
        candidates_path,
        index=False,
    )

    selected.to_csv(
        selected_path,
        index=False,
    )

    city_summary.to_csv(
        city_summary_path,
        index=False,
    )

    projected_distribution.to_csv(
        distribution_path,
        index=False,
    )

    print("\nV2.2 simulation result")
    print("----------------------")
    print(
        f"Total offset candidates: "
        f"{len(candidates)}"
    )
    print(
        f"Useful candidates: "
        f"{len(useful)}"
    )
    print(
        f"Selected after city cap: "
        f"{len(selected)}"
    )

    print("\nSelected candidate composition")
    print("------------------------------")
    print(
        "Bare-land rich: "
        f"{int(selected['bare_land_rich'].sum())}"
    )
    print(
        "Bare-land moderate: "
        f"{int(selected['bare_land_moderate'].sum())}"
    )
    print(
        "Road rich: "
        f"{int(selected['road_rich'].sum())}"
    )
    print(
        "Water rich: "
        f"{int(selected['water_rich'].sum())}"
    )
    print(
        "Building rich: "
        f"{int(selected['building_rich'].sum())}"
    )
    print(
        "Mixed minority: "
        f"{int(selected['mixed_minority'].sum())}"
    )

    print("\nProjected class distribution")
    print("----------------------------")

    display = projected_distribution.copy()

    display["v21_fraction"] *= 100
    display["projected_v22_fraction"] *= 100
    display["relative_pixel_increase"] *= 100

    print(
        display[
            [
                "class_name",
                "v21_fraction",
                "projected_v22_fraction",
                "relative_pixel_increase",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nReports written")
    print("---------------")
    print(candidates_path)
    print(selected_path)
    print(city_summary_path)
    print(distribution_path)

    print(
        "\nNOTE: Simulation only. "
        "No Dataset V2.2 raster tiles were created."
    )


if __name__ == "__main__":
    main()