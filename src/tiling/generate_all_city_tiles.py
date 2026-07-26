"""Generate and validate tiled datasets for all configured cities.

The controller is resumable:

- A city is skipped when its existing tile-QA report has status PASS and
  its manifest and tiling summary still exist.
- A city with a manifest and tiling summary but no PASS report runs QA only.
- A city without completed tiling metadata runs tiling followed by QA.
- Partial tile outputs are not overwritten silently. Use --force for a clean
  regeneration of selected cities.

This preserves city-level train/validation/test splits from config/cities.csv.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PASS_STATUSES = {"PASS", "SKIPPED_PASS"}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return data


def require_file(path: Path, label: str) -> None:
    """Require a non-empty file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def run_command(
    command: list[str],
    stage: str,
) -> float:
    """Run a subprocess stage and return its duration."""
    print("\n" + "=" * 76)
    print(f"Stage: {stage}")
    print("=" * 76)
    print(" ".join(command))

    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def save_progress(
    results: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Persist current all-city progress."""
    dataframe = pd.DataFrame(results)
    dataframe.to_csv(csv_path, index=False)

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_city_count": len(results),
        "passed_city_count": sum(
            row["status"] in PASS_STATUSES
            for row in results
        ),
        "failed_city_count": sum(
            row["status"] == "FAIL"
            for row in results
        ),
        "generated_city_count": sum(
            row["status"] == "PASS"
            for row in results
        ),
        "skipped_pass_city_count": sum(
            row["status"] == "SKIPPED_PASS"
            for row in results
        ),
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def remove_existing_city_outputs(
    tile_root: Path,
    manifest: Path,
    tiling_summary: Path,
    qa_report: Path,
) -> None:
    """Delete one city's derived tile outputs for forced regeneration."""
    if tile_root.exists():
        shutil.rmtree(tile_root)

    for path in (
        manifest,
        tiling_summary,
        qa_report,
    ):
        if path.exists():
            path.unlink()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate tiled SAR datasets for configured cities."
        )
    )

    parser.add_argument(
        "--cities-file",
        type=Path,
        default=Path("config/cities.csv"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/tiling.yaml"),
    )

    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help=(
            "Optional city ID. Repeat this option to process multiple "
            "selected cities. Without it, all registry cities are processed."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete and regenerate tile outputs for the selected cities."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later cities after a city failure.",
    )

    return parser.parse_args()


def main() -> None:
    """Run all-city tiling and tile QA."""
    args = parse_arguments()

    require_file(args.cities_file, "city registry")
    require_file(args.config, "tiling configuration")

    cities = pd.read_csv(args.cities_file)

    required_columns = {
        "city_id",
        "city_name",
        "split",
    }
    missing = required_columns.difference(cities.columns)

    if missing:
        raise ValueError(
            f"City registry is missing columns: {sorted(missing)}"
        )

    cities = cities.sort_values("city_id").reset_index(drop=True)

    if args.cities:
        requested = set(args.cities)
        available = set(cities["city_id"].astype(str))
        unknown = sorted(requested.difference(available))

        if unknown:
            raise ValueError(f"Unknown city IDs: {unknown}")

        cities = cities.loc[
            cities["city_id"].astype(str).isin(requested)
        ].copy()
        cities = cities.sort_values("city_id").reset_index(drop=True)

    output_dir = Path("metadata/all_city_tile_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    latest_csv = output_dir / "all_city_tiles_latest.csv"
    latest_json = output_dir / "all_city_tiles_latest.json"

    timestamped_csv = (
        output_dir / f"all_city_tiles_{timestamp}.csv"
    )
    timestamped_json = (
        output_dir / f"all_city_tiles_{timestamp}.json"
    )

    results: list[dict[str, Any]] = []
    overall_started = time.perf_counter()

    print("\nAll-city tiling and QA")
    print("----------------------")
    print(f"Cities requested: {len(cities)}")
    print(f"Force regeneration: {args.force}")
    print(
        "Failure policy: "
        + (
            "continue"
            if args.continue_on_error
            else "stop immediately"
        )
    )

    for position, (_, row) in enumerate(
        cities.iterrows(),
        start=1,
    ):
        city_id = str(row["city_id"])
        city_name = str(row["city_name"])
        split = str(row["split"])
        folder_name = f"{city_id}_{city_name}"

        image = (
            Path("data/processed/sar")
            / folder_name
            / f"{folder_name}_S1_VV_VH.tif"
        )
        semantic_mask = (
            Path("data/processed/masks")
            / folder_name
            / f"{city_id}_label_mask.tif"
        )
        validity_mask = (
            Path("data/processed/masks")
            / folder_name
            / f"{city_id}_validity_mask.tif"
        )

        tile_root = (
            Path("data/processed/tiles")
            / split
            / folder_name
        )
        image_tiles = tile_root / "images"
        semantic_tiles = tile_root / "semantic_masks"
        validity_tiles = tile_root / "validity_masks"

        manifest = (
            Path("metadata/tile_manifests")
            / f"{city_id}_tile_manifest.csv"
        )
        tiling_summary = (
            Path("metadata/tile_manifests")
            / f"{city_id}_tiling_summary.json"
        )
        qa_report = (
            Path("metadata/tile_qa")
            / f"{city_id}_tile_qa.json"
        )

        print("\n" + "#" * 80)
        print(
            f"City {position}/{len(cities)}: "
            f"{city_id} {city_name} [{split}]"
        )
        print("#" * 80)

        city_started = time.perf_counter()
        stage_durations: dict[str, float] = {
            "tiling": 0.0,
            "tile_qa": 0.0,
        }

        try:
            for path, label in (
                (image, "SAR image"),
                (semantic_mask, "semantic mask"),
                (validity_mask, "validity mask"),
            ):
                require_file(path, label)

            if args.force:
                remove_existing_city_outputs(
                    tile_root=tile_root,
                    manifest=manifest,
                    tiling_summary=tiling_summary,
                    qa_report=qa_report,
                )

            existing_pass = False

            if (
                qa_report.exists()
                and manifest.exists()
                and tiling_summary.exists()
                and not args.force
            ):
                qa_data = load_json(qa_report)

                existing_pass = (
                    qa_data.get("qa_status") == "PASS"
                    and int(qa_data.get("failed_tile_count", -1)) == 0
                )

            if existing_pass:
                require_file(manifest, "tile manifest")
                require_file(tiling_summary, "tiling summary")
                require_file(qa_report, "tile-QA report")

                status = "SKIPPED_PASS"
                action = "existing PASS reused"
                error = ""

                print(
                    f"Skipping {city_id}: existing tile QA is PASS."
                )

            else:
                completed_tiling_metadata = (
                    manifest.exists()
                    and tiling_summary.exists()
                )

                if completed_tiling_metadata:
                    print(
                        "Existing tiling metadata found; running tile QA only."
                    )
                    action = "QA_ONLY"
                else:
                    has_partial_outputs = (
                        tile_root.exists()
                        or manifest.exists()
                        or tiling_summary.exists()
                    )

                    if has_partial_outputs and not args.force:
                        raise RuntimeError(
                            "Partial tile outputs detected for "
                            f"{city_id}. Re-run this city with --force."
                        )

                    stage_durations["tiling"] = run_command(
                        [
                            sys.executable,
                            "-m",
                            "src.tiling.tile_city_dataset",
                            "--city-id",
                            city_id,
                            "--city-name",
                            city_name,
                            "--split",
                            split,
                            "--image",
                            str(image),
                            "--semantic-mask",
                            str(semantic_mask),
                            "--validity-mask",
                            str(validity_mask),
                            "--config",
                            str(args.config),
                        ],
                        "Tile generation",
                    )

                    action = "TILED_AND_QA"

                require_file(manifest, "tile manifest")
                require_file(tiling_summary, "tiling summary")

                stage_durations["tile_qa"] = run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.quality_control.validate_tile_dataset",
                        "--city-id",
                        city_id,
                        "--manifest",
                        str(manifest),
                        "--tiling-summary",
                        str(tiling_summary),
                    ],
                    "Tile dataset QA",
                )

                require_file(qa_report, "tile-QA report")

                qa_data = load_json(qa_report)

                if qa_data.get("qa_status") != "PASS":
                    raise RuntimeError(
                        f"Tile QA did not pass for {city_id}."
                    )

                if int(qa_data.get("failed_tile_count", -1)) != 0:
                    raise RuntimeError(
                        f"Tile QA reports failed tiles for {city_id}."
                    )

                status = "PASS"
                error = ""

            tiling_data = load_json(tiling_summary)
            qa_data = load_json(qa_report)

            result = {
                "city_id": city_id,
                "city_name": city_name,
                "split": split,
                "status": status,
                "action": action,
                "error": error,
                "tile_count": int(
                    tiling_data["tile_count"]
                ),
                "tiles_with_padding": int(
                    tiling_data["tiles_with_padding"]
                ),
                "zero_valid_tiles": int(
                    tiling_data[
                        "tiles_with_zero_valid_pixels"
                    ]
                ),
                "passed_tile_count": int(
                    qa_data["passed_tile_count"]
                ),
                "failed_tile_count": int(
                    qa_data["failed_tile_count"]
                ),
                "valid_pixel_count": int(
                    qa_data[
                        "aggregate_statistics"
                    ]["total_valid_pixels"]
                ),
                "tiling_seconds": (
                    stage_durations["tiling"]
                ),
                "tile_qa_seconds": (
                    stage_durations["tile_qa"]
                ),
                "duration_seconds": (
                    time.perf_counter() - city_started
                ),
                "manifest": str(manifest),
                "tiling_summary": str(tiling_summary),
                "tile_qa_report": str(qa_report),
            }

        except Exception as exc:
            result = {
                "city_id": city_id,
                "city_name": city_name,
                "split": split,
                "status": "FAIL",
                "action": "FAILED",
                "error": str(exc),
                "tile_count": None,
                "tiles_with_padding": None,
                "zero_valid_tiles": None,
                "passed_tile_count": None,
                "failed_tile_count": None,
                "valid_pixel_count": None,
                "tiling_seconds": (
                    stage_durations["tiling"]
                ),
                "tile_qa_seconds": (
                    stage_durations["tile_qa"]
                ),
                "duration_seconds": (
                    time.perf_counter() - city_started
                ),
                "manifest": str(manifest),
                "tiling_summary": str(tiling_summary),
                "tile_qa_report": str(qa_report),
            }

            print(
                f"\nCity failed: {city_id}\n"
                f"Reason: {result['error']}"
            )

        results.append(result)

        save_progress(
            results,
            latest_csv,
            latest_json,
        )

        if (
            result["status"] == "FAIL"
            and not args.continue_on_error
        ):
            print(
                "\nStopping after failure. Fix the city and rerun. "
                "Completed PASS cities will be skipped automatically."
            )
            break

    save_progress(
        results,
        timestamped_csv,
        timestamped_json,
    )
    save_progress(
        results,
        latest_csv,
        latest_json,
    )

    passed = sum(
        row["status"] in PASS_STATUSES
        for row in results
    )
    failed = sum(
        row["status"] == "FAIL"
        for row in results
    )
    generated = sum(
        row["status"] == "PASS"
        for row in results
    )
    skipped = sum(
        row["status"] == "SKIPPED_PASS"
        for row in results
    )
    total_tiles = sum(
        int(row["tile_count"])
        for row in results
        if row["status"] in PASS_STATUSES
    )

    print("\nAll-city tiling summary")
    print("-----------------------")
    print(f"Requested: {len(cities)}")
    print(f"Processed: {len(results)}")
    print(f"Passed or reused: {passed}")
    print(f"Newly generated/validated: {generated}")
    print(f"Existing PASS skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total approved tiles: {total_tiles}")
    print(
        f"Duration: "
        f"{time.perf_counter() - overall_started:.2f} seconds"
    )
    print(f"Latest CSV: {latest_csv}")
    print(f"Latest JSON: {latest_json}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
