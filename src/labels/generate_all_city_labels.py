"""Generate and validate labels for all configured cities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the city-label pipeline over the registry."
    )
    parser.add_argument(
        "--cities-file",
        type=Path,
        default=Path("config/cities.csv"),
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="Optional city ID; repeat to process several cities.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    return parser.parse_args()


def save_progress(
    results: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    pd.DataFrame(results).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main() -> None:
    args = parse_args()

    if not args.cities_file.exists():
        raise FileNotFoundError(
            f"City registry not found: {args.cities_file}"
        )

    cities = pd.read_csv(args.cities_file)
    required = {"city_id", "city_name"}
    missing = required.difference(cities.columns)
    if missing:
        raise ValueError(f"Missing city columns: {sorted(missing)}")

    cities = cities.sort_values("city_id").reset_index(drop=True)

    if args.cities:
        unknown = sorted(
            set(args.cities).difference(set(cities["city_id"]))
        )
        if unknown:
            raise ValueError(f"Unknown city IDs: {unknown}")

        cities = cities.loc[
            cities["city_id"].isin(args.cities)
        ].copy()

    output_dir = Path("metadata/all_city_label_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_csv = output_dir / "all_city_labels_latest.csv"
    latest_json = output_dir / "all_city_labels_latest.json"
    timestamped_csv = output_dir / f"all_city_labels_{timestamp}.csv"
    timestamped_json = output_dir / f"all_city_labels_{timestamp}.json"

    results: list[dict[str, Any]] = []
    overall_started = time.perf_counter()

    for position, (_, row) in enumerate(cities.iterrows(), start=1):
        city_id = str(row["city_id"])
        city_name = str(row["city_name"])

        print("\n" + "#" * 80)
        print(
            f"City {position}/{len(cities)}: "
            f"{city_id} {city_name}"
        )
        print("#" * 80)

        command = [
            sys.executable,
            "-m",
            "src.labels.generate_city_labels",
            "--city",
            city_id,
            "--cities-file",
            str(args.cities_file),
        ]
        if args.force:
            command.append("--force")

        started = time.perf_counter()

        try:
            subprocess.run(command, check=True)
            status = "PASS"
            error = ""
        except subprocess.CalledProcessError as exc:
            status = "FAIL"
            error = f"Exit code {exc.returncode}"

        results.append(
            {
                "city_id": city_id,
                "city_name": city_name,
                "status": status,
                "error": error,
                "duration_seconds": time.perf_counter() - started,
            }
        )
        save_progress(results, latest_csv, latest_json)

        if status == "FAIL" and not args.continue_on_error:
            print(
                "\nStopping after failure. Correct the issue and rerun, "
                "or use --continue-on-error."
            )
            break

    save_progress(results, timestamped_csv, timestamped_json)
    save_progress(results, latest_csv, latest_json)

    passed = sum(item["status"] == "PASS" for item in results)
    failed = sum(item["status"] == "FAIL" for item in results)

    print("\nAll-city label-generation summary")
    print("---------------------------------")
    print(f"Requested: {len(cities)}")
    print(f"Processed: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(
        f"Duration: {time.perf_counter() - overall_started:.2f} seconds"
    )
    print(f"Latest CSV: {latest_csv}")
    print(f"Latest JSON: {latest_json}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
