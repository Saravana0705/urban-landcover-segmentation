# Dataset V3-MT-D1 monthly acquisition audit

## Purpose

This gate tests whether the frozen V3-MT orbit for every city supplies enough
Sentinel-1 VV/VH observations for twelve reliable 2025 monthly composites.
It does **not** export imagery.

The validator selects one cadence for every city:

- `monthly`: 12 periods, 24 channels, when every city-month has at least two
  observations;
- `bimonthly`: 6 periods, 12 channels, when monthly fails but every paired
  two-month period has at least two observations;
- failure: image export remains blocked.

Using one cadence across all cities keeps the model input contract fixed.

## Files

- `scripts/gee/audit_sentinel1_monthly_v3_mt_d1_all_cities.js`
- `src/quality_control/validate_dataset_v3_mt_d1_acquisition_audit.py`

## 1. Add the files to the repository

Extract the patch into the repository root. Confirm that the two paths above
exist. Do not replace the frozen V3-MT files.

## 2. Run the Earth Engine audit

1. Open the Google Earth Engine Code Editor.
2. Create a script named `audit_sentinel1_monthly_v3_mt_d1_all_cities`.
3. Paste the complete JavaScript file into it and click **Run**.
4. Inspect the Console entry. It must describe 20 features.
5. Open **Tasks**. There must be exactly one new table-export task and no image
   export tasks.
6. Start `dataset_v3_mt_d1_monthly_acquisition_audit_2025`.

The Drive folder is `S1_MT_D1_MONTHLY_AUDIT_2025`.

## 3. Place the CSV

Download the completed CSV and save it as:

```text
metadata/dataset_v3_mt_d1/acquisition/dataset_v3_mt_d1_monthly_acquisition_audit_2025.csv
```

## 4. Validate and select the cadence

From the repository root in PowerShell:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d1_acquisition_audit `
    --overwrite
```

Expected output when monthly data is adequate:

```text
"status": "PASS"
"selected_cadence": "monthly"
"selected_channel_count": 24
"image_exports_authorized": true
```

If the output selects `bimonthly`, do not override it merely to retain 24
channels. The next exporter must use the selected 12-channel cadence.

## 5. Stop gate

Do not start image acquisition yet. Review these two generated files first:

- the acquisition CSV;
- `metadata/dataset_v3_mt_d1/acquisition/dataset_v3_mt_d1_acquisition_audit_qa.json`.

The full exporter will be generated only after this gate determines the safe
cadence.
