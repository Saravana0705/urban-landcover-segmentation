# Dataset V3-MT-D2 Cross-Orbit Acquisition Audit

## Purpose

This is the mandatory first gate for Dataset V3-MT-D2. It checks whether every
frozen Dataset V3 city has adequate full-coverage Sentinel-1 observations from
both ascending and descending passes in all four 2025 quarters. Audit v0.2
measures coverage from the merged footprint of all GRD slices acquired on the
same UTC date and relative orbit; it does not reject a usable acquisition just
because one constituent slice covers less than 98% of the city.

The controlled D2 contract is:

- unchanged 20 cities and 14/3/3 city split;
- unchanged labels and tile grid;
- preserved frozen V3-MT primary pass and relative orbit;
- one added opposite-pass relative orbit selected deterministically;
- four quarterly composites per pass;
- VV and VH for every pass/quarter;
- 16 input channels;
- no image exports before this audit passes.

## 1. Generate the Earth Engine audit script

From the repository root:

```powershell
python -m src.data.generate_dataset_v3_mt_d2_acquisition_audit
```

Expected result:

```text
"status": "GENERATED"
"city_count": 20
"image_exports_created": 0
"table_exports_created": 1
```

## 2. Run the audit in Google Earth Engine

1. Open `scripts/gee/audit_sentinel1_cross_orbit_v3_mt_d2_all_cities.js`.
2. Copy it into a new Earth Engine Code Editor script.
3. Click **Run**.
4. Inspect the console for a non-zero candidate row count.
5. Open **Tasks**.
6. Start only the table task named
   `dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025`.

The script must create exactly one CSV task and no image tasks.

## 3. Place the downloaded CSV

Create this directory:

```powershell
New-Item -ItemType Directory -Force `
  metadata/dataset_v3_mt_d2/acquisition
```

Copy the downloaded file to:

```text
metadata/dataset_v3_mt_d2/acquisition/dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025.csv
```

## 4. Validate and select the two orbits

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d2_acquisition_audit `
  --overwrite
```

A safe result is:

```text
"status": "PASS"
"city_count": 20
"selected_channel_count": 16
"image_exports_authorized": true
```

The validator requires at least two full-coverage observations in every quarter
for each selected pass. It keeps the original V3-MT relative orbit for the
primary pass and selects the best adequate relative orbit from the opposite
pass using, in order:

1. highest minimum quarterly observation count;
2. highest annual observation count;
3. highest mean coverage fraction;
4. lowest relative-orbit number as deterministic tie-breaker.

Do not lower the observation threshold to bypass missing quarters. If this
mosaic-aware audit still fails, the 16-channel 2025 D2 contract is not uniformly
available for all frozen cities and must be redesigned before image export.

## Stop gate

Do not create or start D2 image exports unless the QA report says both:

```text
"status": "PASS"
"image_exports_authorized": true
```

After the PASS report is reviewed, generate the DE01/DE15 16-channel pilot
export from its selected-orbit records.
