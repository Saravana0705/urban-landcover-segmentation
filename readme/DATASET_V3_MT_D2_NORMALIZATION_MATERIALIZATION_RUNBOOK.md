# Dataset V3-MT-D2 normalization and materialization

## Purpose

Create leakage-free normalization constants and materialize the frozen 1,664
V3-MT tile locations with the validated 16-channel D2 imagery. Labels, city
splits, tile selection and semantic masks are inherited unchanged.

## 1. Compute training-only normalization

```powershell
python -m src.data.compute_dataset_v3_mt_d2_normalization
```

Expected result:

```text
"status": "PASS"
"training_cities": 14
"bands": 16
"validation_and_test_used": false
```

Only DE01-DE14 are used. For each band, positive linear sigma-zero is converted
to dB, clipped using training-only sampled p01/p99 values, and standardized
using the exact clipped training-pixel mean and population standard deviation.

Outputs:

```text
metadata/dataset_v3_mt_d2/normalization/training_normalization.json
metadata/dataset_v3_mt_d2/normalization/training_normalization.csv
metadata/dataset_v3_mt_d2/normalization/normalization_provenance.json
```

Use `--overwrite` only when intentionally recomputing these outputs.

## 2. Materialize D2 tiles

```powershell
python -m src.data.materialize_dataset_v3_mt_d2
```

Expected result:

```text
"status": "PASS"
"tile_count": 1664
"written": 1664
```

If an interrupted run left only complete tile triplets, resume with:

```powershell
python -m src.data.materialize_dataset_v3_mt_d2 --resume
```

Do not use `--resume` to conceal a partial triplet; the materializer will stop
when only some files for a tile exist.

## 3. Validate all tile triplets

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d2_tiles `
    --overwrite
```

Expected result:

```text
"status": "PASS"
"passed": 1664
"failed": 0
```

The validator independently checks 16-band image shape/order/type, masks,
grids, valid-pixel semantics, class counts and SHA-256 hashes.

## Stop gate

Do not freeze D2, create its Kaggle dataset or start training until the tile QA
report passes and is reviewed. Do not commit the new D2 metadata yet.
