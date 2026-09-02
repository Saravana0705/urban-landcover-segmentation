# Dataset V3-MT normalization gate

This stage calculates eight independent normalization channels using only the
14 training cities. It does not read DE15-DE20 and does not modify rasters,
labels, manifests, tiles or frozen Dataset V3/V3.1 files.

## Run

Extract the patch into the project root and execute:

```powershell
python -m src.data.compute_dataset_v3_mt_normalization
```

The calculation performs two passes over DE01-DE14 and may take several
minutes. Expected final output:

```json
{
  "status": "PASS",
  "training_cities": 14,
  "bands": 8,
  "validation_and_test_used": false
}
```

## Review gate

Upload these three files:

```text
metadata/dataset_v3_mt/normalization/training_normalization.json
metadata/dataset_v3_mt/normalization/training_normalization.csv
metadata/dataset_v3_mt/normalization/normalization_provenance.json
```

Do not materialize temporal tiles until these outputs are reviewed. The next
patch will bind the constants to the frozen V3 manifest and create the
eight-band temporal tiles with temporal NoData intersected into validity.
