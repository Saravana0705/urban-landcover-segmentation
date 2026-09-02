# Dataset V3-MT pilot runbook

This patch adds a two-city multitemporal Sentinel-1 pilot. It does **not**
modify Dataset V3/V3.1 labels, manifests, splits, or frozen files.

## 1. Install the patch

Extract the ZIP into the project root. Confirm these new files exist:

```text
scripts/gee/export_sentinel1_multitemporal_v3_mt_pilot.js
src/quality_control/validate_dataset_v3_mt_pilot.py
```

## 2. Export two Earth Engine rasters

Open the JavaScript file in the Earth Engine Code Editor and click **Run**.
The Console must show one or more anchor candidates for both DE01 and DE15,
the selected acquisition time close to the registry time, and a positive
observation count for every quarter. Stop if either city has zero candidates
or any quarter has zero observations.

Start only these two tasks:

```text
DE01_Mannheim_S1_MT_Q1Q4_2025
DE15_Leipzig_S1_MT_Q1Q4_2025
```

The export contains eight positive linear-sigma0 bands in this order:

```text
VV_Q1, VH_Q1, VV_Q2, VH_Q2, VV_Q3, VH_Q3, VV_Q4, VH_Q4
```

## 3. Place the downloaded files

```text
data/raw/sar_multitemporal_v3_mt/DE01_Mannheim/DE01_Mannheim_S1_MT_Q1Q4_2025.tif
data/raw/sar_multitemporal_v3_mt/DE15_Leipzig/DE15_Leipzig_S1_MT_Q1Q4_2025.tif
```

Do not rename or overwrite the existing two-band SAR rasters.

## 4. Run automated pilot QA

```powershell
python -m src.quality_control.validate_dataset_v3_mt_pilot
```

Expected result:

```json
{
  "status": "PASS",
  "cities": {"DE01": "PASS", "DE15": "PASS"}
}
```

Upload this generated report for the gate review:

```text
metadata/dataset_v3_mt/pilot/dataset_v3_mt_pilot_qa.json
```

Also share a screenshot of the Earth Engine Console containing the anchor
match, orbit, pass, and four quarterly observation counts for each city.

Do not yet export the remaining 18 cities, compute normalization, materialize
tiles, or start GPU training. Those stages follow only after this pilot passes.
