# Dataset V3-MT-D2 cross-orbit pilot

## Purpose

Export and validate the DE01 Mannheim (`train`) and DE15 Leipzig (`val`)
16-band cross-orbit stacks before creating the other 18 exports. The frozen
labels, split assignments and 10 m grids remain unchanged.

Expected band order:

```text
VV_ASC_Q1, VH_ASC_Q1, ..., VV_ASC_Q4, VH_ASC_Q4,
VV_DESC_Q1, VH_DESC_Q1, ..., VV_DESC_Q4, VH_DESC_Q4
```

For each selected pass and relative orbit, GRD slices acquired on the same UTC
date are mosaicked in positive linear sigma-zero. Daily mosaics with less than
98% city-footprint coverage are excluded, and quarterly medians are then
calculated. This matches the mosaic-aware D2 acquisition audit.

## 1. Generate the pilot exporter

From the repository root:

```powershell
python -m src.data.generate_dataset_v3_mt_d2_pilot_export
```

Expected selections:

```text
DE01: ASCENDING 15, DESCENDING 139
DE15: ASCENDING 44, DESCENDING 168
```

The generator must report exactly two image exports.

## 2. Run the Earth Engine pilot

1. Open `scripts/gee/export_sentinel1_cross_orbit_v3_mt_d2_pilot.js`.
2. Paste it into a new Earth Engine Code Editor script and click **Run**.
3. Confirm that every displayed pass/quarter count is at least 2.
4. Confirm that each city reports the expected 16 output bands.
5. Start exactly these two image tasks:
   - `DE01_Mannheim_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025`
   - `DE15_Leipzig_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025`

The Drive folder is `S1_MT_D2_CROSS_ORBIT_PILOT_2025`.

## 3. Place the exported rasters

```text
data/raw/sar_multitemporal_v3_mt_d2/DE01_Mannheim/DE01_Mannheim_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025.tif
data/raw/sar_multitemporal_v3_mt_d2/DE15_Leipzig/DE15_Leipzig_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025.tif
```

## 4. Run automated QA

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d2_pilot `
    --overwrite
```

The validator checks the 16-band order, float32 type, `-9999` NoData, frozen
CRS/dimensions/affine transform, finite positive linear values, at least 98%
valid pixels per band, checksums and unchanged split assignments.

Expected result:

```text
"status": "PASS"
"DE01": "PASS"
"DE15": "PASS"
```

## Stop gate

Do not generate or start the remaining 18 D2 image exports until the pilot QA
report has passed and been reviewed.
