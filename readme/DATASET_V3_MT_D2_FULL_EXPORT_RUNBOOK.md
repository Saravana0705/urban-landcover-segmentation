# Dataset V3-MT-D2 full cross-orbit source export

## Purpose

Export the remaining 18 cities using the exact 16-channel cross-orbit method
that passed the acquisition and DE01/DE15 pilot gates. This stage changes no
labels, splits, tile selection, normalization or model code.

## 1. Generate the remaining-city exporter

Keep the completed DE01 and DE15 pilot rasters, then run:

```powershell
python -m src.data.generate_dataset_v3_mt_d2_full_export
```

Expected result:

```text
"cities_embedded": 18
"image_exports_created": 18
"skipped_completed_pilot_cities": ["DE01", "DE15"]
```

## 2. Run the Earth Engine exports

Open this generated file in the Earth Engine Code Editor:

```text
scripts/gee/export_sentinel1_cross_orbit_v3_mt_d2_all_cities.js
```

Click **Run** and confirm exactly 18 image tasks are created. DE01 and DE15
must not appear. Start the tasks; queued tasks are normal. The Drive folder is:

```text
S1_MT_D2_CROSS_ORBIT_ALL_CITIES_2025
```

## 3. Place the GeoTIFF files

Place every downloaded file at:

```text
data/raw/sar_multitemporal_v3_mt_d2/<CITY_ID>_<CITY_NAME>/
    <CITY_ID>_<CITY_NAME>_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025.tif
```

The root must then contain all 20 cities, including the retained DE01 and DE15
pilot files.

## 4. Run full-source QA

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d2_sources `
    --overwrite
```

Expected result:

```text
"status": "PASS"
"passed": 20
"failed": 0
```

The validator checks all 16 band names/order, float32 type, `-9999` NoData,
frozen CRS/dimensions/affine transform, finite positive values, at least 98%
valid coverage per band, checksums and unchanged splits.

## Stop gate

Do not compute D2 normalization or materialize tiles until the 20-city source
QA report has passed and been reviewed. Do not commit the new D2 metadata yet.
