# Dataset V3-MT-D1 full monthly source export

## Purpose

Export the remaining 18 frozen cities using the 24-channel monthly contract
that passed the acquisition audit and DE01/DE15 pilot. This stage changes no
labels, tile selection, splits, normalization or model code, and consumes no
Kaggle GPU time.

## 1. Install the patch

Extract the patch into the repository root. Retain the already completed DE01
and DE15 pilot rasters.

## 2. Run the Earth Engine exporter

Open the following file and paste it into a new Earth Engine Code Editor
script:

```text
scripts/gee/export_sentinel1_monthly_v3_mt_d1_all_cities.js
```

Click **Run** and verify that exactly 18 image tasks are created:

```text
DE02-DE14, excluding the already completed DE15,
and DE16-DE20
```

DE01 and DE15 must not appear as new image tasks. The Drive folder is:

```text
S1_MT_D1_MONTHLY_ALL_CITIES_2025
```

Start the 18 tasks. Earth Engine may queue some tasks; this is normal.

## 3. Place all source rasters

Place each downloaded GeoTIFF at:

```text
data/raw/sar_multitemporal_v3_mt_d1/<CITY_ID>_<CITY_NAME>/
    <CITY_ID>_<CITY_NAME>_S1_MT_D1_MONTHLY_2025.tif
```

Afterwards, this directory must contain all 20 cities, including the DE01 and
DE15 pilot rasters.

## 4. Run automated full-source QA

From the repository root in PowerShell:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d1_sources `
    --overwrite
```

Expected output:

```text
"status": "PASS"
"passed": 20
"failed": 0
```

The validator checks the 24-band order, float32 type, `-9999` NoData, frozen
grid identity and a minimum 98% positive valid fraction for every band.

The report is written to:

```text
metadata/dataset_v3_mt_d1/source_qa/dataset_v3_mt_d1_source_qa.json
```

## 5. Stop gate

Do not materialize tiles or compute normalization until the 20-city source QA
report has been reviewed. Do not commit the new metadata yet.
