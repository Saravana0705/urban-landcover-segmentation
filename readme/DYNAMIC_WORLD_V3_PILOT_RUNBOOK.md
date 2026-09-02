# Dataset V3 Dynamic World pilot

This module is additive. It does not overwrite or modify Dataset V1, V2.1,
V2.2, existing masks, manifests, normalization files, or model experiments.

## 1. Copy the module into the repository

Copy the contents while preserving paths:

```text
config/dataset_v3_pilot.yaml
scripts/gee/export_dynamic_world_v3_pilot.js
src/data/validate_dynamic_world_v3_pilot.py
```

## 2. Export from Google Earth Engine

1. Open <https://code.earthengine.google.com/>.
2. Create a new script.
3. Paste `scripts/gee/export_dynamic_world_v3_pilot.js`.
4. Click **Run**.
5. Confirm the Console reports a non-zero source image count for DE03 and DE14.
6. Open **Tasks** and run both exports.

The script creates two 21-band Cloud-Optimized GeoTIFFs in the Google Drive
folder `urban_landcover_v3_pilot`:

```text
DE03_dynamic_world_v3_pilot_2025.tif
DE14_dynamic_world_v3_pilot_2025.tif
```

The end dates are exclusive in Earth Engine. Thus DE03 covers 20 June through
19 August 2025, and DE14 covers 13 June through 12 August 2025.

## 3. Place the exports locally

```text
data/raw/labels_v3/dynamic_world/DE03_Karlsruhe/
data/raw/labels_v3/dynamic_world/DE14_Berlin/
```

Do not place these files under any V1 or V2 directory.

## 4. Validate without altering the exports

From the project root with the existing virtual environment active:

```powershell
python -m src.data.validate_dynamic_world_v3_pilot `
  --city-id DE03 `
  --raster data/raw/labels_v3/dynamic_world/DE03_Karlsruhe/DE03_dynamic_world_v3_pilot_2025.tif

python -m src.data.validate_dynamic_world_v3_pilot `
  --city-id DE14 `
  --raster data/raw/labels_v3/dynamic_world/DE14_Berlin/DE14_dynamic_world_v3_pilot_2025.tif
```

Expected result for both commands: `qa_status: PASS`.

Reports are written only to:

```text
metadata/dataset_v3/pilot/dynamic_world/
```

## 5. Stop after validation

Do not threshold probabilities or create V3 semantic masks yet. The next step
is to inspect observation counts and probability distributions, determine
whether either city needs the documented ±45-day fallback window, and derive
confidence thresholds from the pilot evidence.
