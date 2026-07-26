"""Build the authoritative Dataset Version 1 global manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

TRAINABLE_CLASSES = {
    1: "buildings",
    2: "roads",
    3: "vegetation",
    4: "bare_land",
    5: "water",
}

REQUIRED_CITY_COLUMNS = {"city_id", "city_name", "split"}
REQUIRED_TILE_COLUMNS = {
    "tile_id", "city_id", "city_name", "split", "tile_row", "tile_col",
    "tile_size", "stride", "padding_pixel_count", "padding_fraction",
    "valid_pixel_count", "valid_fraction", "unlabeled_pixel_count",
    "semantic_nodata_pixel_count", "buildings_pixel_count",
    "roads_pixel_count", "vegetation_pixel_count", "bare_land_pixel_count",
    "water_pixel_count", "image_path", "semantic_mask_path",
    "validity_mask_path",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def validate_city_registry(cities: pd.DataFrame) -> None:
    missing = REQUIRED_CITY_COLUMNS.difference(cities.columns)
    if missing:
        raise ValueError(f"City registry is missing columns: {sorted(missing)}")
    if cities.empty:
        raise ValueError("City registry is empty.")
    if cities["city_id"].duplicated().any():
        duplicates = cities.loc[cities["city_id"].duplicated(), "city_id"].tolist()
        raise ValueError(f"Duplicate city IDs in registry: {duplicates}")
    allowed_splits = {"train", "val", "test"}
    found_splits = set(cities["split"].astype(str))
    if not found_splits.issubset(allowed_splits):
        raise ValueError(
            f"Unsupported split values: {sorted(found_splits.difference(allowed_splits))}"
        )
    missing_splits = allowed_splits.difference(found_splits)
    if missing_splits:
        raise ValueError(f"Registry is missing required splits: {sorted(missing_splits)}")


def read_city_manifest(path: Path) -> pd.DataFrame:
    require_file(path, "city tile manifest")
    dataframe = pd.read_csv(path)
    missing = REQUIRED_TILE_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"City tile manifest is empty: {path}")
    if dataframe["tile_id"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["tile_id"].duplicated(), "tile_id"
        ].tolist()
        raise ValueError(f"Duplicate tile IDs in {path}: {duplicates[:10]}")
    return dataframe


def qa_checksum_lookup(qa_report: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    tiles = qa_report.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError("Tile-QA report does not contain a tile list.")
    for tile in tiles:
        tile_id = str(tile["tile_id"])
        if tile.get("status") != "PASS":
            raise ValueError(f"Tile-QA report contains non-PASS tile: {tile_id}")
        lookup[tile_id] = {
            "image_sha256": str(tile["image_sha256"]),
            "semantic_sha256": str(tile["semantic_sha256"]),
            "validity_sha256": str(tile["validity_sha256"]),
        }
    return lookup


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the global Dataset Version 1 manifest."
    )
    parser.add_argument("--cities-file", type=Path, default=Path("config/cities.csv"))
    parser.add_argument(
        "--all-city-summary",
        type=Path,
        default=Path(
            "metadata/all_city_tile_pipeline/all_city_tiles_latest.json"
        ),
    )
    parser.add_argument(
        "--tile-manifest-dir", type=Path, default=Path("metadata/tile_manifests")
    )
    parser.add_argument(
        "--tile-qa-dir", type=Path, default=Path("metadata/tile_qa")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("metadata/dataset_v1")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    require_file(args.cities_file, "city registry")
    require_file(args.all_city_summary, "all-city tile summary")

    cities = pd.read_csv(args.cities_file)
    validate_city_registry(cities)
    cities = cities.sort_values("city_id").reset_index(drop=True)

    all_city_summary = load_json(args.all_city_summary)
    summary_results = all_city_summary.get("results")
    if not isinstance(summary_results, list):
        raise ValueError("All-city tile summary does not contain a results list.")

    summary_status = {
        str(row["city_id"]): str(row["status"]) for row in summary_results
    }
    approved_statuses = {"PASS", "SKIPPED_PASS"}

    global_rows: list[pd.DataFrame] = []
    city_summary_rows: list[dict[str, Any]] = []

    print("\nBuilding Dataset Version 1 manifest")
    print("-----------------------------------")
    print(f"Cities in registry: {len(cities)}")

    for _, city in cities.iterrows():
        city_id = str(city["city_id"])
        city_name = str(city["city_name"])
        split = str(city["split"])

        if summary_status.get(city_id) not in approved_statuses:
            raise RuntimeError(
                f"All-city tile pipeline has no approved status for {city_id}: "
                f"{summary_status.get(city_id)}"
            )

        manifest_path = args.tile_manifest_dir / f"{city_id}_tile_manifest.csv"
        qa_path = args.tile_qa_dir / f"{city_id}_tile_qa.json"

        city_manifest = read_city_manifest(manifest_path)
        qa_report = load_json(qa_path)

        if qa_report.get("qa_status") != "PASS":
            raise RuntimeError(f"Tile QA is not PASS for {city_id}.")
        if int(qa_report.get("failed_tile_count", -1)) != 0:
            raise RuntimeError(f"Tile QA reports failed tiles for {city_id}.")
        if int(qa_report.get("tile_count", -1)) != len(city_manifest):
            raise RuntimeError(
                f"Tile-QA count mismatch for {city_id}: "
                f"{qa_report.get('tile_count')} != {len(city_manifest)}"
            )

        if set(city_manifest["city_id"].astype(str)) != {city_id}:
            raise ValueError(f"Unexpected city IDs in {manifest_path}")
        if set(city_manifest["city_name"].astype(str)) != {city_name}:
            raise ValueError(f"Unexpected city names in {manifest_path}")
        if set(city_manifest["split"].astype(str)) != {split}:
            raise ValueError(f"Split mismatch for {city_id}")

        checksum_lookup = qa_checksum_lookup(qa_report)
        if set(checksum_lookup) != set(city_manifest["tile_id"].astype(str)):
            raise RuntimeError(
                f"Tile checksum lookup does not match manifest for {city_id}."
            )

        city_manifest = city_manifest.copy()
        city_manifest["image_sha256"] = city_manifest["tile_id"].map(
            lambda tile_id: checksum_lookup[str(tile_id)]["image_sha256"]
        )
        city_manifest["semantic_sha256"] = city_manifest["tile_id"].map(
            lambda tile_id: checksum_lookup[str(tile_id)]["semantic_sha256"]
        )
        city_manifest["validity_sha256"] = city_manifest["tile_id"].map(
            lambda tile_id: checksum_lookup[str(tile_id)]["validity_sha256"]
        )
        city_manifest["tile_qa_status"] = "PASS"
        city_manifest["city_tile_qa_report"] = str(qa_path)
        city_manifest["city_tile_manifest"] = str(manifest_path)

        for _, tile in city_manifest.iterrows():
            for path_column in (
                "image_path", "semantic_mask_path", "validity_mask_path"
            ):
                require_file(
                    Path(str(tile[path_column])),
                    f"{tile['tile_id']} {path_column}",
                )

        city_manifest = city_manifest.sort_values(
            ["tile_row", "tile_col", "tile_id"]
        )
        global_rows.append(city_manifest)

        city_summary_rows.append(
            {
                "city_id": city_id,
                "city_name": city_name,
                "split": split,
                "tile_count": len(city_manifest),
                "valid_pixel_count": int(city_manifest["valid_pixel_count"].sum()),
                "padding_pixel_count": int(
                    city_manifest["padding_pixel_count"].sum()
                ),
                "zero_valid_tile_count": int(
                    (city_manifest["valid_pixel_count"] == 0).sum()
                ),
                "buildings_pixel_count": int(
                    city_manifest["buildings_pixel_count"].sum()
                ),
                "roads_pixel_count": int(city_manifest["roads_pixel_count"].sum()),
                "vegetation_pixel_count": int(
                    city_manifest["vegetation_pixel_count"].sum()
                ),
                "bare_land_pixel_count": int(
                    city_manifest["bare_land_pixel_count"].sum()
                ),
                "water_pixel_count": int(city_manifest["water_pixel_count"].sum()),
                "tile_qa_status": "PASS",
            }
        )
        print(f"  {city_id} {city_name}: {len(city_manifest)} approved tiles")

    dataset = pd.concat(global_rows, ignore_index=True)

    if dataset["tile_id"].duplicated().any():
        duplicates = dataset.loc[
            dataset["tile_id"].duplicated(), "tile_id"
        ].tolist()
        raise RuntimeError(f"Duplicate global tile IDs: {duplicates[:20]}")

    expected_total_tiles = sum(
        int(row.get("tile_count", 0))
        for row in summary_results
        if row.get("status") in approved_statuses
    )
    if len(dataset) != expected_total_tiles:
        raise RuntimeError(
            f"Global tile count mismatch: {len(dataset)} != {expected_total_tiles}"
        )

    registry_split_map = {
        str(row["city_id"]): str(row["split"]) for _, row in cities.iterrows()
    }
    dataset_split_map = (
        dataset[["city_id", "split"]]
        .drop_duplicates()
        .set_index("city_id")["split"]
        .to_dict()
    )
    if registry_split_map != dataset_split_map:
        raise RuntimeError("Global dataset split assignments do not match registry.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = args.output_dir / "dataset_manifest.csv"
    city_summary_output = args.output_dir / "dataset_city_summary.csv"
    split_summary_output = args.output_dir / "dataset_split_summary.csv"
    class_distribution_output = (
        args.output_dir / "dataset_class_distribution.csv"
    )
    summary_output = args.output_dir / "dataset_manifest_summary.json"

    ordered_columns = [
        "tile_id", "city_id", "city_name", "split", "tile_row", "tile_col",
        "row_offset", "col_offset", "tile_size", "stride",
        "source_width_pixels", "source_height_pixels", "padding_pixel_count",
        "padding_fraction", "valid_pixel_count", "valid_fraction",
        "unlabeled_pixel_count", "semantic_nodata_pixel_count",
        "buildings_pixel_count", "roads_pixel_count",
        "vegetation_pixel_count", "bare_land_pixel_count", "water_pixel_count",
        "contains_buildings", "contains_roads", "contains_vegetation",
        "contains_bare_land", "contains_water", "bounds_left", "bounds_bottom",
        "bounds_right", "bounds_top", "crs", "image_path",
        "semantic_mask_path", "validity_mask_path", "image_sha256",
        "semantic_sha256", "validity_sha256", "tile_qa_status",
        "city_tile_manifest", "city_tile_qa_report",
    ]
    missing_ordered = [
        column for column in ordered_columns if column not in dataset.columns
    ]
    if missing_ordered:
        raise RuntimeError(
            f"Global dataset lacks expected columns: {missing_ordered}"
        )

    dataset = dataset[ordered_columns].sort_values(
        ["split", "city_id", "tile_row", "tile_col"]
    )
    dataset.to_csv(manifest_output, index=False)

    city_summary = pd.DataFrame(city_summary_rows).sort_values("city_id")
    city_summary.to_csv(city_summary_output, index=False)

    split_summary_rows: list[dict[str, Any]] = []
    for split, split_data in dataset.groupby("split", sort=True):
        split_summary_rows.append(
            {
                "split": split,
                "city_count": int(split_data["city_id"].nunique()),
                "tile_count": len(split_data),
                "zero_valid_tile_count": int(
                    (split_data["valid_pixel_count"] == 0).sum()
                ),
                "valid_pixel_count": int(split_data["valid_pixel_count"].sum()),
                "padding_pixel_count": int(
                    split_data["padding_pixel_count"].sum()
                ),
                "buildings_pixel_count": int(
                    split_data["buildings_pixel_count"].sum()
                ),
                "roads_pixel_count": int(split_data["roads_pixel_count"].sum()),
                "vegetation_pixel_count": int(
                    split_data["vegetation_pixel_count"].sum()
                ),
                "bare_land_pixel_count": int(
                    split_data["bare_land_pixel_count"].sum()
                ),
                "water_pixel_count": int(split_data["water_pixel_count"].sum()),
            }
        )
    split_summary = pd.DataFrame(split_summary_rows)
    split_summary.to_csv(split_summary_output, index=False)

    class_distribution_rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test", "all"]:
        split_data = dataset if split_name == "all" else dataset.loc[
            dataset["split"] == split_name
        ]
        total_valid = int(split_data["valid_pixel_count"].sum())
        for class_name in TRAINABLE_CLASSES.values():
            pixel_column = f"{class_name}_pixel_count"
            pixels = int(split_data[pixel_column].sum())
            class_distribution_rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "pixel_count": pixels,
                    "percentage_of_valid_pixels": (
                        pixels / total_valid * 100.0 if total_valid > 0 else 0.0
                    ),
                    "area_km2_at_10m": pixels * 100.0 / 1_000_000.0,
                    "tiles_containing_class": int(
                        split_data[f"contains_{class_name}"].astype(bool).sum()
                    ),
                }
            )
    class_distribution = pd.DataFrame(class_distribution_rows)
    class_distribution.to_csv(class_distribution_output, index=False)

    split_city_sets = {
        split: sorted(
            dataset.loc[dataset["split"] == split, "city_id"].unique().tolist()
        )
        for split in ("train", "val", "test")
    }
    no_city_leakage = (
        set(split_city_sets["train"]).isdisjoint(split_city_sets["val"])
        and set(split_city_sets["train"]).isdisjoint(split_city_sets["test"])
        and set(split_city_sets["val"]).isdisjoint(split_city_sets["test"])
    )

    summary = {
        "dataset_version": "1.0-pre-freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_registry": str(args.cities_file),
        "all_city_tile_summary": str(args.all_city_summary),
        "city_count": int(dataset["city_id"].nunique()),
        "tile_count": len(dataset),
        "split_city_ids": split_city_sets,
        "split_city_counts": {
            split: len(city_ids) for split, city_ids in split_city_sets.items()
        },
        "split_tile_counts": {
            split: int((dataset["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "zero_valid_tile_count": int(
            (dataset["valid_pixel_count"] == 0).sum()
        ),
        "all_tile_qa_passed": bool(
            (dataset["tile_qa_status"] == "PASS").all()
        ),
        "all_paths_exist": True,
        "tile_ids_unique": bool(dataset["tile_id"].is_unique),
        "city_splits_disjoint": no_city_leakage,
        "manifest_sha256": sha256_file(manifest_output),
        "outputs": {
            "dataset_manifest": str(manifest_output),
            "dataset_city_summary": str(city_summary_output),
            "dataset_split_summary": str(split_summary_output),
            "dataset_class_distribution": str(class_distribution_output),
        },
    }

    with summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nDataset V1 manifest summary")
    print("---------------------------")
    print(f"Cities: {summary['city_count']}")
    print(f"Tiles: {summary['tile_count']}")
    print(f"Split city counts: {summary['split_city_counts']}")
    print(f"Split tile counts: {summary['split_tile_counts']}")
    print(f"Zero-valid tiles: {summary['zero_valid_tile_count']}")
    print(f"All tile QA passed: {summary['all_tile_qa_passed']}")
    print(f"City splits disjoint: {summary['city_splits_disjoint']}")
    print(f"Manifest: {manifest_output}")
    print(f"Summary: {summary_output}")

    mandatory_checks = {
        "city_count_is_20": summary["city_count"] == 20,
        "tile_count_is_2880": summary["tile_count"] == 2880,
        "all_tile_qa_passed": summary["all_tile_qa_passed"],
        "tile_ids_unique": summary["tile_ids_unique"],
        "city_splits_disjoint": summary["city_splits_disjoint"],
    }

    print("\nMandatory checks")
    for name, passed in mandatory_checks.items():
        print(f"  {name}: {passed}")

    if not all(mandatory_checks.values()):
        raise RuntimeError(
            "Dataset V1 global manifest failed mandatory checks."
        )

    print("\nResult: Dataset V1 global manifest built successfully.")


if __name__ == "__main__":
    main()
