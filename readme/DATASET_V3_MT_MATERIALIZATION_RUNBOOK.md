# Dataset V3-MT materialization and independent QA

This stage binds the accepted temporal sources and training-only normalization
to the frozen Dataset V3 manifest. Dataset V3/V3.1 files are never modified.

## 1. Materialize

Extract the patch into the project root and run:

```powershell
python -m src.data.materialize_dataset_v3_mt
```

Expected: 1,664 tile triplets (4,992 GeoTIFFs), split 974/351/339.

If interrupted, first confirm there are no `.partial.tif` files or incomplete
three-file tile groups, then resume with:

```powershell
python -m src.data.materialize_dataset_v3_mt --resume
```

## 2. Independent tile QA

Only after materialization reports PASS, run:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_tiles
```

## 3. Upload the gate evidence

```text
metadata/dataset_v3_mt/materialization/dataset_v3_mt_materialization_audit.json
metadata/dataset_v3_mt/materialization/dataset_v3_mt_materialization_city_summary.csv
metadata/dataset_v3_mt/qa/dataset_v3_mt_tile_qa.json
metadata/dataset_v3_mt/qa/dataset_v3_mt_city_qa_summary.csv
metadata/dataset_v3_mt/dataset_manifest.csv
```

Compress the manifest if necessary. Do not start model training yet. Loader
integration, frozen configuration and a one-batch/one-epoch integration check
follow after this gate passes.
