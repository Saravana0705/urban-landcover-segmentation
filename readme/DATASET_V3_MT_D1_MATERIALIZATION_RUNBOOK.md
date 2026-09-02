# Dataset V3-MT-D1 materialization and tile QA

## Purpose

Create 24-band monthly tiles for the exact 1,664 frozen V3-MT tile identities.
Semantic labels and city splits are reused unchanged. Each output validity mask
is the frozen V3-MT validity intersected with finite positive validity across
all 24 monthly channels.

This stage consumes no Kaggle GPU time.

## Preconditions

- Full 20-city V3-MT-D1 source QA is `PASS`.
- Training-only normalization contains 24 bands and excludes DE15-DE20.
- The frozen parent manifest is:
  `metadata/dataset_v3_mt/dataset_manifest.csv`.
- Its accepted SHA-256 is either the repository LF serialization
  (`8a6c3266609fbb6e060f754c9f53dd55a3b4e3681e8adafe1a88b72bda9a38b0`)
  or the semantically identical Windows CRLF serialization
  (`6f342ae5341b0c59ca6e2eb88e12192ecad1ee676eef47fe4536d8d20937f3de`).

Keep substantial free disk space available. Twenty-four-band image tiles can
require several times the storage used by the eight-band V3-MT images.

## 1. Materialize

Extract the patch into the repository root, then run:

```powershell
python -m src.data.materialize_dataset_v3_mt_d1
```

Expected terminal summary:

```text
"status": "PASS"
"tile_count": 1664
"written": 1664
```

If a genuine interruption occurs after complete tile groups were written,
inspect the output and resume with:

```powershell
python -m src.data.materialize_dataset_v3_mt_d1 --resume
```

Do not use `--resume` to bypass an unexplained validation or grid error.

## 2. Validate every tile

After materialization succeeds, run:

```powershell
python -m src.quality_control.validate_dataset_v3_mt_d1_tiles `
    --overwrite
```

Expected result:

```text
"status": "PASS"
"passed": 1664
"failed": 0
```

The QA independently checks every 24-band image, semantic mask and validity
mask, including shapes, grids, band order, pixel values, class counts and
SHA-256 hashes.

## Outputs for review

```text
metadata/dataset_v3_mt_d1/dataset_manifest.csv
metadata/dataset_v3_mt_d1/materialization/dataset_v3_mt_d1_materialization_audit.json
metadata/dataset_v3_mt_d1/materialization/dataset_v3_mt_d1_materialization_city_summary.csv
metadata/dataset_v3_mt_d1/qa/dataset_v3_mt_d1_tile_qa.json
metadata/dataset_v3_mt_d1/qa/dataset_v3_mt_d1_city_qa_summary.csv
```

Do not freeze, package, commit or train on V3-MT-D1 until these files have been
reviewed.
