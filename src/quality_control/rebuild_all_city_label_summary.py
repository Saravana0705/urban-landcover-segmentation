"""Rebuild the all-city label summary from city-level pipeline reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    cities = pd.read_csv("config/cities.csv").sort_values("city_id")

    summary_dir = Path("metadata/city_label_pipeline")
    output_dir = Path("metadata/all_city_label_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, city in cities.iterrows():
        city_id = str(city["city_id"])
        city_name = str(city["city_name"])

        report_path = (
            summary_dir
            / f"{city_id}_pipeline_summary.json"
        )

        if not report_path.exists():
            rows.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                    "status": "MISSING",
                    "error": "Pipeline summary not found",
                    "duration_seconds": None,
                }
            )
            continue

        with report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        rows.append(
            {
                "city_id": city_id,
                "city_name": city_name,
                "status": report.get(
                    "status",
                    "UNKNOWN",
                ),
                "error": report.get(
                    "error",
                    "",
                ),
                "duration_seconds": report.get(
                    "duration_seconds",
                ),
            }
        )

    dataframe = pd.DataFrame(rows)

    latest_csv = (
        output_dir
        / "all_city_labels_latest.csv"
    )
    latest_json = (
        output_dir
        / "all_city_labels_latest.json"
    )

    dataframe.to_csv(
        latest_csv,
        index=False,
    )

    summary = {
        "requested_city_count": len(dataframe),
        "processed_city_count": int(
            dataframe["status"].isin(
                ["PASS", "FAIL"]
            ).sum()
        ),
        "passed_city_count": int(
            (dataframe["status"] == "PASS").sum()
        ),
        "failed_city_count": int(
            (dataframe["status"] == "FAIL").sum()
        ),
        "missing_city_count": int(
            (dataframe["status"] == "MISSING").sum()
        ),
        "results": rows,
    }

    with latest_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nRebuilt all-city summary")
    print("------------------------")
    print(
        f"Requested: "
        f"{summary['requested_city_count']}"
    )
    print(
        f"Passed: "
        f"{summary['passed_city_count']}"
    )
    print(
        f"Failed: "
        f"{summary['failed_city_count']}"
    )
    print(
        f"Missing: "
        f"{summary['missing_city_count']}"
    )
    print(f"CSV: {latest_csv}")
    print(f"JSON: {latest_json}")

    if (
        summary["failed_city_count"] > 0
        or summary["missing_city_count"] > 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()