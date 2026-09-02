# Dataset V3-MT-D1 normalization

## Purpose

Compute the 24 monthly VV/VH normalization records using only the 14 training
cities (DE01-DE14). DE15-DE20 are explicitly excluded to prevent validation or
test leakage.

For every band, the process:

1. reads valid positive linear sigma-zero values;
2. converts them to dB;
3. estimates training-only p01 and p99 clipping limits deterministically;
4. computes the exact mean and population standard deviation after clipping.

This stage consumes no Kaggle GPU time.

## 1. Install the patch

Extract the patch into the repository root. Confirm that the accepted source QA
report exists at:

```text
metadata/dataset_v3_mt_d1/source_qa/dataset_v3_mt_d1_source_qa.json
```

## 2. Compute normalization

From the repository root in PowerShell:

```powershell
python -m src.data.compute_dataset_v3_mt_d1_normalization `
    --overwrite
```

The script performs two passes over DE01-DE14. Processing 24 full-resolution
bands can take noticeably longer than the earlier eight-band quarterly run.

Expected final output:

```text
"status": "PASS"
"training_cities": 14
"bands": 24
"validation_and_test_used": false
```

## 3. Outputs

```text
metadata/dataset_v3_mt_d1/normalization/training_normalization.json
metadata/dataset_v3_mt_d1/normalization/training_normalization.csv
metadata/dataset_v3_mt_d1/normalization/normalization_provenance.json
```

## 4. Stop gate

Upload all three outputs for review. Do not materialize tiles, create a frozen
dataset or commit the new pipeline until the normalization records pass review.
