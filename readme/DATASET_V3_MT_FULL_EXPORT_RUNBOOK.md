# Dataset V3-MT full source-export runbook

This stage acquires quarterly VV/VH rasters for all existing cities. It does
not change labels, tile selection, splits, normalization, or model code.

## 1. Install and regenerate the exporter

Extract this patch into the project root, then run:

```powershell
python -m src.data.generate_dataset_v3_mt_gee_export --overwrite
```

Expected: 20 audited cities, 18 image exports, with DE01 and DE15 skipped.

## 2. Run the generated script in Earth Engine

Open:

```text
scripts/gee/export_sentinel1_multitemporal_v3_mt_all_cities.js
```

Click Run. Do not immediately start the 18 image tasks.

First start only:

```text
dataset_v3_mt_acquisition_audit_2025
```

Download its CSV to:

```text
metadata/dataset_v3_mt/acquisition/dataset_v3_mt_acquisition_audit_2025.csv
```

Validate it:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_acquisition_audit
```

Only a PASS authorizes the image exports.

## 3. Export the remaining 18 cities

After acquisition-audit PASS, start the DE02-DE14 and DE16-DE20 image tasks.
DE01 and DE15 are already complete and must not be rerun.

Place each download at:

```text
data/raw/sar_multitemporal_v3_mt/<CITY_ID>_<CITY_NAME>/<CITY_ID>_<CITY_NAME>_S1_MT_Q1Q4_2025.tif
```

## 4. Validate all full-resolution sources

After all 20 files are present:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_sources
```

Upload these two reports for the next gate:

```text
metadata/dataset_v3_mt/acquisition/dataset_v3_mt_acquisition_audit_qa.json
metadata/dataset_v3_mt/source_qa/dataset_v3_mt_source_qa.json
```

Do not materialize tiles or train models until both reports pass.
