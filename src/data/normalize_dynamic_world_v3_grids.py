"""Normalize GEE Dynamic World exports to the exact registered Sentinel-1 grids.

Only a 0/1 pixel bottom/right excess is accepted. No reprojection or resampling
is performed. Every changed source is retained beside the corrected raster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.windows import Window, transform as window_transform


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_dw(city_id: str) -> Path:
    patterns = [
        f"data/raw/labels_v3/dynamic_world/{city_id}_*/{city_id}_dynamic_world_v3_2025.tif",
        f"data/raw/labels_v3/dynamic_world/{city_id}_*/{city_id}_dynamic_world_v3_pilot_2025.tif",
    ]
    hits = sorted({p.resolve() for pattern in patterns for p in Path().glob(pattern) if p.is_file()})
    if len(hits) != 1:
        raise FileNotFoundError(f"{city_id}: expected one active Dynamic World TIFF, found {hits}")
    return hits[0]


def inspect(city: dict) -> dict:
    city_id = city["city_id"]
    source = find_dw(city_id)
    reference = Path(city["grid"]["osm_semantic"])
    if not reference.is_file():
        raise FileNotFoundError(f"{city_id}: reference mask missing: {reference}")
    with rasterio.open(source) as src, rasterio.open(reference) as ref:
        same_crs = src.crs == ref.crs
        width_excess = src.width - ref.width
        height_excess = src.height - ref.height
        same_pixel_geometry = (
            src.transform.a == ref.transform.a and src.transform.b == ref.transform.b
            and src.transform.d == ref.transform.d and src.transform.e == ref.transform.e
        )
        col_offset_float = (ref.transform.c - src.transform.c) / src.transform.a
        row_offset_float = (ref.transform.f - src.transform.f) / src.transform.e
        col_offset, row_offset = round(col_offset_float), round(row_offset_float)
        integer_offset = (
            abs(col_offset_float - col_offset) < 1e-8
            and abs(row_offset_float - row_offset) < 1e-8
        )
        candidate_window = Window(col_offset, row_offset, ref.width, ref.height)
        candidate_transform = window_transform(candidate_window, src.transform)
        matches_reference = candidate_transform.almost_equals(ref.transform, precision=1e-8)
        contained = (
            col_offset >= 0 and row_offset >= 0
            and col_offset + ref.width <= src.width
            and row_offset + ref.height <= src.height
        )
        if not same_crs or not same_pixel_geometry or not integer_offset or not matches_reference:
            status = "REFUSE_SHIFTED_GRID"
        elif width_excess == 0 and height_excess == 0 and col_offset == 0 and row_offset == 0:
            status = "ALIGNED"
        elif width_excess in (0, 1) and height_excess in (0, 1) and contained:
            status = "CROP_TO_REFERENCE"
        else:
            status = "REFUSE_UNEXPECTED_SIZE"
        return {
            "city_id": city_id, "source": str(source), "reference": str(reference),
            "source_size": [src.width, src.height], "reference_size": [ref.width, ref.height],
            "width_excess": width_excess, "height_excess": height_excess,
            "same_crs": same_crs, "same_pixel_geometry": same_pixel_geometry,
            "crop_col_offset": col_offset, "crop_row_offset": row_offset, "status": status,
        }


def crop(item: dict) -> dict:
    source = Path(item["source"])
    reference = Path(item["reference"])
    backup = source.with_name(source.stem + f"_original_{item['source_size'][0]}x{item['source_size'][1]}" + source.suffix)
    temporary = source.with_name(source.stem + ".grid_normalization_tmp" + source.suffix)
    if backup.exists() or temporary.exists():
        raise FileExistsError(f"Refusing collision for {item['city_id']}: {backup} or {temporary}")
    original_hash = sha256(source)
    with rasterio.open(source) as src, rasterio.open(reference) as ref:
        profile = src.profile.copy()
        profile.update(width=ref.width, height=ref.height, transform=ref.transform)
        with rasterio.open(temporary, "w", **profile) as dst:
            for _, window in ref.block_windows(1):
                source_window = Window(
                    window.col_off + item["crop_col_offset"],
                    window.row_off + item["crop_row_offset"],
                    window.width, window.height,
                )
                dst.write(src.read(window=source_window), window=window)
            dst.update_tags(**src.tags())
            for band in range(1, src.count + 1):
                dst.update_tags(band, **src.tags(band))
                if src.descriptions[band - 1]:
                    dst.set_band_description(band, src.descriptions[band - 1])
    with rasterio.open(temporary) as corrected, rasterio.open(reference) as ref:
        if (corrected.width, corrected.height, corrected.crs, corrected.transform) != (
            ref.width, ref.height, ref.crs, ref.transform
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Corrected grid verification failed for {item['city_id']}")
    os.replace(source, backup)
    os.replace(temporary, source)
    item.update({"status": "NORMALIZED", "backup": str(backup),
                 "original_sha256": original_hash, "corrected_sha256": sha256(source)})
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--city", action="append")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--registry", type=Path, default=Path("config/dataset_v3_city_registry.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("metadata/dataset_v3/grid_normalization"))
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    by_id = {city["city_id"]: city for city in registry["cities"]}
    selected = sorted(by_id) if args.all else args.city
    unknown = sorted(set(selected) - set(by_id))
    if unknown: parser.error(f"Unknown city IDs: {unknown}")

    results = [inspect(by_id[city_id]) for city_id in selected]
    refused = [x for x in results if x["status"].startswith("REFUSE")]
    if refused:
        print(json.dumps({"status": "REFUSED", "results": results}, indent=2))
        raise SystemExit(2)
    if not args.dry_run:
        results = [crop(x) if x["status"] == "CROP_TO_REFERENCE" else x for x in results]
        args.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = args.report_dir / f"dynamic_world_grid_normalization_{stamp}.json"
        if report.exists(): raise FileExistsError(f"Refusing overwrite: {report}")
        report.write_text(json.dumps({"schema_version": "v3-dw-grid-normalization-0.1",
                                      "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "method": "integer-pixel window crop to reference bounds; no reprojection or resampling",
                                      "results": results}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "PASS", "report": str(report), "results": results}, indent=2))
    else:
        print(json.dumps({"status": "DRY_RUN", "results": results}, indent=2))


if __name__ == "__main__":
    main()
