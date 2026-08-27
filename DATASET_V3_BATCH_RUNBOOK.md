# Dataset V3 — guarded 20-city batch

This module is additive. It writes only under `data/raw/labels_v3`,
`data/interim/labels_v3`, `metadata/dataset_v3`, and the two new V3 config/script
paths. It never writes Dataset V1, V2.1, or V2.2 paths and refuses to overwrite
existing outputs.

## 1. Install

Extract the ZIP directly into the repository root, the folder containing
`config`, `data`, `metadata`, `scripts`, and `src`. Merge folders when Windows
asks; do not replace or delete those folders. The ZIP contains only new files.

Keep the already validated pilot files for DE03 and DE14 where they are.

## 2. Registry

The supplied `config/dataset_v3_city_registry.json` is already generated from
the audited 20-city files. To regenerate it later, first rename/remove only the
two generated V3 files, then run:

```powershell
python -m src.data.prepare_dataset_v3_batch `
  --cities config/cities.csv `
  --selected-scenes metadata/selected_scenes.csv `
  --rasterization-reports metadata/rasterization
```

Expected split counts are exactly 14 train / 3 validation / 3 test. Duplicate
rasterization reports are accepted only when their grids agree exactly.

## 3. Dynamic World evidence

Paste `scripts/gee/export_dynamic_world_v3_all_cities.js` into Google Earth
Engine and run it. It creates 18 tasks; DE03 and DE14 are skipped because their
pilot exports already passed QA. Download each result to:

```text
data/raw/labels_v3/dynamic_world/<CITY_ID>_<CityName>/<CITY_ID>_dynamic_world_v3_2025.tif
```

Do not rename the existing DE03/DE14 pilot TIFFs.

## 4. Urban Atlas 2021

Download/extract one official Urban Atlas 2021 `.fgb` per remaining city to:

```text
data/raw/urban_atlas_2021/<CITY_ID>_<CityName>/extracted/<official_file>.fgb
```

The runner rejects zero or multiple `.fgb` candidates for a city.

## 5. Validate paths before processing

Start with one city, then the full dry run:

```powershell
python -m src.data.run_dataset_v3_batch --city DE01 --dry-run
python -m src.data.run_dataset_v3_batch --all --dry-run
```

Dry-run performs discovery and safety checks but creates no rasters.

## 6. Build evidence and fused masks

Run one city and inspect its QA before scaling:

```powershell
python -m src.data.run_dataset_v3_batch --city DE01 --stage all
```

Then run all cities. Complete Urban Atlas evidence already present for DE03 and
DE14 is reused, not overwritten:

```powershell
python -m src.data.run_dataset_v3_batch --all --stage all
```

Outputs:

```text
data/interim/labels_v3/urban_atlas/<CITY_ID>/...
data/interim/labels_v3/fused/<CITY_ID>/...
metadata/dataset_v3/three_source_audit/...
metadata/dataset_v3/fusion/...
metadata/dataset_v3/batch_runs/...
```

## 7. Decision gate before tiling

Do not tile yet. First aggregate and inspect valid/Ignore/conflict fractions and
per-class representation by city and by the preserved split. Reject any city
with grid mismatch, missing source, implausible class geography, or unexplained
distribution outlier. Only after that QA passes should V3 candidate-tile
statistics and class-aware selection thresholds be derived.
