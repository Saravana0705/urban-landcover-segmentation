"""Record the completed targeted Dataset V3 visual-QA gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_REASONS = {
    "evaluation_city_representative": 6,
    "high_buildings": 2,
    "high_roads": 2,
    "high_bare_land": 2,
    "high_water": 2,
    "high_source_conflict": 2,
}


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_image_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(
            name.startswith("word/media/") and not name.endswith("/")
            for name in archive.namelist()
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(
            "metadata/dataset_v3/qa/dataset_v3_visual_qa_sample.csv"
        ),
    )
    parser.add_argument(
        "--evidence-docx",
        type=Path,
        default=Path(
            "metadata/dataset_v3/qa/"
            "Dataset_V3_Visual_QA_27Aug2026.docx"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/dataset_v3/qa/manual_visual_qa"),
    )
    parser.add_argument("--confirm-reviewed-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_reviewed_all:
        raise RuntimeError(
            "Explicit confirmation required: pass --confirm-reviewed-all only "
            "after all 16 samples and both conflict overlays were reviewed."
        )
    require_file(args.sample, "visual-QA sample manifest")
    require_file(args.evidence_docx, "visual-QA evidence document")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite manual-QA output: {args.output_dir}"
        )

    with args.sample.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tile_ids = [row["tile_id"] for row in rows]
    reason_counts = {
        reason: sum(row["visual_qa_reason"] == reason for row in rows)
        for reason in EXPECTED_REASONS
    }
    image_count = embedded_image_count(args.evidence_docx)
    checks = {
        "sample_tile_count_is_16": len(rows) == 16,
        "sample_tile_ids_unique": len(set(tile_ids)) == 16,
        "reason_counts_match": reason_counts == EXPECTED_REASONS,
        "evidence_contains_at_least_18_images": image_count >= 18,
        "reviewer_confirmed_all_samples": args.confirm_reviewed_all,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Manual visual-QA record failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    review_csv = args.output_dir / "dataset_v3_manual_visual_qa_review.csv"
    fields = ["tile_id", "city_id", "city_name", "split", "visual_qa_reason", "decision"]
    with review_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields[:-1]} | {"decision": "PASS"})

    audit = {
        "schema_version": "dataset-v3-manual-visual-qa-0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_version": "v3",
        "decision": "PASS",
        "review_scope": "16 targeted tiles plus separate conflict overlays for two high-conflict tiles",
        "reviewed_tile_count": 16,
        "reviewed_tile_ids": tile_ids,
        "reason_counts": reason_counts,
        "findings": {
            "spatial_alignment": "PASS",
            "buildings_and_roads": "PASS",
            "vegetation": "PASS",
            "bare_land": "PASS",
            "water": "PASS",
            "ignore_regions": "PASS",
            "source_conflict_patterns": "plausible and concentrated at complex boundaries",
        },
        "evidence_document": str(args.evidence_docx),
        "evidence_document_sha256": sha256_file(args.evidence_docx),
        "evidence_embedded_image_count": image_count,
        "sample_manifest": str(args.sample),
        "sample_manifest_sha256": sha256_file(args.sample),
        "review_csv": str(review_csv),
        "review_csv_sha256": sha256_file(review_csv),
        "mandatory_checks": checks,
    }
    audit_path = args.output_dir / "dataset_v3_manual_visual_qa.json"
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PASS",
        "reviewed_tile_count": 16,
        "evidence_embedded_image_count": image_count,
        "next": "python -m src.quality_control.freeze_dataset_v3",
    }, indent=2))


if __name__ == "__main__":
    main()
