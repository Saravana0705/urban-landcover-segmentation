"""Validate and archive the outputs of one segmentation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


CORE_ARTIFACTS = {
    "best_checkpoint": ("checkpoints/best.pt",),
    "latest_checkpoint": ("checkpoints/latest.pt",),
    "training_history": (
        "training_history.json",
        "history/training_history.json",
        "logs/training_history.json",
    ),
    "experiment_state": ("experiment_state.json", "reports/experiment_state.json"),
    "overall_metrics": ("metrics/overall_metrics.csv", "overall_metrics.csv"),
    "per_class_metrics": ("metrics/per_class_metrics.csv", "per_class_metrics.csv"),
    "confusion_matrix": ("metrics/confusion_matrix.csv", "confusion_matrix.csv"),
    "evaluation_metrics": ("metrics/evaluation_metrics.json", "evaluation_metrics.json"),
}

OPTIONAL_ARTIFACTS = {
    "benchmark_summary": (
        "benchmark_summary.json",
        "metrics/benchmark_summary.json",
        "logs/benchmark_summary.json",
    ),
    "latest_epoch_metrics": ("latest.json", "metrics/latest.json", "logs/latest.json"),
    "training_history_csv": ("training_history.csv", "logs/training_history.csv"),
}

SENSITIVE_PARTS = {
    ".env", "credential", "credentials", "secret", "secrets", "token",
    "kaggle.json", "id_rsa", "id_ed25519", "github_pat",
}

FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensitive(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    return any(marker in lowered or any(marker in part for part in lowered)
               for marker in SENSITIVE_PARTS)


def first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return None


def read_state(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment state must be a JSON object: {path}")
    return payload


def archive_name(path: Path, experiment_root: Path, category: str) -> str:
    try:
        relative = path.resolve().relative_to(experiment_root.resolve())
        return f"experiment/{relative.as_posix()}"
    except ValueError:
        return f"external/{category}/{path.name}"


def write_deterministic_zip(zip_path: Path, files: list[dict], inventory: dict) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for item in sorted(files, key=lambda value: value["archive_path"]):
            info = ZipInfo(item["archive_path"], FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, Path(item["source_path"]).read_bytes())
        info = ZipInfo("artifact_inventory.json", FIXED_ZIP_TIME)
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--include", type=Path, action="append", default=[],
        help="Additional non-sensitive file, for example a training summary or registry snapshot.",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.experiment_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {root}")

    located: dict[str, Path] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for category, candidates in CORE_ARTIFACTS.items():
        path = first_existing(root, candidates)
        if path is None:
            missing_required.append(category)
        else:
            located[category] = path
    for category, candidates in OPTIONAL_ARTIFACTS.items():
        path = first_existing(root, candidates)
        if path is None:
            missing_optional.append(category)
        else:
            located[category] = path

    state = read_state(located.get("experiment_state"))
    experiment_name = str(state.get("experiment_name") or root.name)
    completed = state.get("completed") is True
    if args.require_complete and not completed:
        missing_required.append("completed_experiment_state")

    external: list[tuple[str, Path]] = []
    if args.runtime_config:
        external.append(("runtime_config", args.runtime_config.resolve()))
    external.extend(("additional", path.resolve()) for path in args.include)
    for category, path in external:
        if not path.is_file():
            missing_required.append(f"{category}:{path}")
            continue
        if sensitive(path):
            raise ValueError(f"Refusing to archive a potentially sensitive file: {path}")
        key = category if category not in located else f"{category}_{len(located)}"
        located[key] = path

    records: list[dict] = []
    for category, path in located.items():
        if sensitive(path):
            raise ValueError(f"Refusing to archive a potentially sensitive file: {path}")
        records.append({
            "category": category,
            "source_path": str(path),
            "archive_path": archive_name(path, root, category),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    status = "PASS" if not missing_required else "FAIL"
    inventory = {
        "status": status,
        "experiment_name": experiment_name,
        "experiment_directory": str(root),
        "experiment_completed": completed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "missing_required": sorted(set(missing_required)),
        "missing_optional": sorted(set(missing_optional)),
    }
    print(json.dumps(inventory, indent=2))

    if args.require_complete and missing_required:
        raise RuntimeError(
            "Artifact packaging failed; required files are missing: "
            + ", ".join(sorted(set(missing_required)))
        )
    if args.dry_run:
        return

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in experiment_name)
    zip_path = args.output_dir.resolve() / f"{safe_name}_artifacts.zip"
    write_deterministic_zip(zip_path, records, inventory)
    print(json.dumps({
        "archive": str(zip_path),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_sha256": sha256(zip_path),
        "status": status,
    }, indent=2))


if __name__ == "__main__":
    main()