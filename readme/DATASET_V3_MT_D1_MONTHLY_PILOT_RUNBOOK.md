# Dataset V3-MT-D1 monthly pilot

## Purpose

Export and automatically validate two full-resolution 24-band Sentinel-1
monthly stacks before creating the remaining 18 Earth Engine tasks. The pilot
uses DE01 Mannheim (`train`) and DE15 Leipzig (`val`). It does not modify labels
or split assignments and consumes no Kaggle GPU time.

Expected band order:

```text
VV_M01, VH_M01, VV_M02, VH_M02, ..., VV_M12, VH_M12
```

Each band is a monthly median calculated after converting the matched
Sentinel-1 GRD scenes from dB to positive linear sigma-zero. The frozen V3-MT
anchor pass, relative orbit, CRS and 10 m grid are retained for every city.

## 1. Run the Earth Engine pilot

1. Open `scripts/gee/export_sentinel1_monthly_v3_mt_d1_pilot.js`.
2. Paste it into a new Google Earth Engine Code Editor script.
3. Click **Run**.
4. Confirm that exactly two image tasks appear:
   - `DE01_Mannheim_S1_MT_D1_MONTHLY_2025`
   - `DE15_Leipzig_S1_MT_D1_MONTHLY_2025`
5. Start both tasks.

The Drive folder is `S1_MT_D1_MONTHLY_PILOT_2025`.

## 2. Place the exported rasters

Create these directories in the repository:

```text
data/raw/sar_multitemporal_v3_mt_d1/DE01_Mannheim/
data/raw/sar_multitemporal_v3_mt_d1/DE15_Leipzig/
```

Place each GeoTIFF in its matching directory without renaming it:

```text
DE01_Mannheim_S1_MT_D1_MONTHLY_2025.tif
DE15_Leipzig_S1_MT_D1_MONTHLY_2025.tif
```

## 3. Run automated QA

From the repository root in PowerShell:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d1_pilot `
    --overwrite
```

The validator checks:

- 24-band count and monthly band order;
- float32 type and `-9999` NoData;
- frozen city CRS, dimensions and affine transform;
- positive linear SAR values;
- at least 98% valid pixels per band;
- file checksums and unchanged split assignments.

Expected terminal status:

```text
"status": "PASS"
"DE01": "PASS"
"DE15": "PASS"
```

The report is written to:

```text
metadata/dataset_v3_mt_d1/pilot/dataset_v3_mt_d1_pilot_qa.json
```

## 4. Stop gate

Do not create the remaining 18 exports until the pilot QA report has been
reviewed. A full exporter using the same 24-band contract will follow after the
pilot passes.
