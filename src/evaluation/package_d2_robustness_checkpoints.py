"""Build a minimal E1-E6 checkpoint folder for the Kaggle robustness run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE = Path("outputs/model_experiments/benchmark_analysis_runs_latest")
DEFAULT_OUTPUT = Path("outputs/kaggle_upload/d2_final_robustness_checkpoints")
MEMBERS = (
    ("E1", "E1_attention_unet_v3_mt_d2_s0_cross_orbit_50ep", "attention_unet_v3_mt_d2_s0_cross_orbit_50ep"),
    ("E2", "E2_unetpp_v3_mt_d2_s0_cross_orbit_50ep", "unetpp_v3_mt_d2_s0_cross_orbit_50ep"),
    ("E3", "E3_swin_transformer_v3_mt_d2_s0_cross_orbit_50ep", "swin_transformer_v3_mt_d2_s0_cross_orbit_50ep"),
    ("E4", "E4_deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep", "deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep"),
    ("E5", "E5_segformer_v3_mt_d2_s0_cross_orbit_50ep", "segformer_v3_mt_d2_s0_cross_orbit_50ep"),
    ("E6", "E6_mask2former_v3_mt_d2_s0_cross_orbit_50ep", "mask2former_v3_mt_d2_s0_cross_orbit_50ep"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output = args.output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output already contains files: {output}. Use --overwrite explicitly.")
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for code, folder, experiment in MEMBERS:
        source_folder = source_root / folder
        matches = [
            path for path in source_folder.rglob("best.pt")
            if "integration_check" not in path.as_posix().lower()
        ]
        by_hash = {sha256(path): path for path in matches}
        if len(by_hash) != 1:
            raise RuntimeError(f"{code}: expected one distinct best.pt, found {len(by_hash)}")
        checksum, source = next(iter(by_hash.items()))
        destination = output / experiment / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and args.overwrite:
            destination.unlink()
        shutil.copy2(source, destination)
        copied_hash = sha256(destination)
        if copied_hash != checksum:
            raise RuntimeError(f"Checksum mismatch after copying {code}")
        records.append({
            "code": code,
            "experiment_name": experiment,
            "source": str(source),
            "packaged_path": str(destination.relative_to(output)),
            "size_bytes": destination.stat().st_size,
            "sha256": copied_hash,
        })
        print(f"{code}: {destination.name} ({destination.stat().st_size / 1024**2:.1f} MiB) PASS")

    manifest = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "member_count": len(records),
        "e0_included": False,
        "purpose": "Kaggle input for D2 validation-only robustness benchmark",
        "members": records,
    }
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "member_count": len(records),
        "e0_included": False, "output_directory": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()

