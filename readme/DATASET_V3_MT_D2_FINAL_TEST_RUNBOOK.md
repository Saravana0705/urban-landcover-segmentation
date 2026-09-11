# Dataset V3-MT-D2 Locked Final Test Runbook

## Locked decision

The controlled architecture benchmark selected U-Net++ (`0.6860177249`
validation mIoU). Later validation-only optimization and E11 checkpoint-soup
selection locked the stronger final method before test access:

- E9B U-Net++ checkpoint SHA-256:
  `9ea0b5220b96acc26cce52f289e2f5b35a085dab0b0927d0794490907f70c799`;
- four-way TTA: identity, horizontal flip, vertical flip, 180-degree rotation;
- E10C/E11 class-probability multipliers in ontology order;
- float16-CPU probability replay matching E10C/E11;
- locked validation mIoU: `0.6982309105`;
- validation lock SHA-256:
  `06c5d6cba9774bc0c0ad70ddddcaba8e9d90f3ea5e2cf2f068c3f3ea0946233c`.

E2 is the controlled architecture-comparison checkpoint. E9B with the locked
inference policy is the final test method. The test result must never be used
to select another model, checkpoint, threshold, multiplier, TTA policy or
dataset version.

## Local VS Code verification

Run only static and synthetic checks locally:

```powershell
python -m compileall -q src

python -m src.evaluation.evaluate_d2_final_test `
    --smoke-test

python -m src.quality_control.validate_d2_final_test_groundwork
```

Both commands must report `PASS`, `test_split_loaded: false`,
`test_rasters_opened: false`, and `training_started: false`.

Commit and push these four files. Do not run `--authorize-test` locally.

## Kaggle inputs

Use a fresh or retained session with:

1. the frozen Dataset V3-MT-D2 paths linked into the repository;
2. the current Git commit;
3. the existing private `E11 Checkpoint Soup Input` dataset.

The required checkpoint is `e9b/best.pt`. The evaluator searches recursively,
so one additional Kaggle directory level is supported.

## Metadata/checkpoint preflight

This reads the manifest/config metadata and verifies the checkpoint, model and
hashes. It does not instantiate the test dataset or open a test raster:

```bash
python -m src.evaluation.evaluate_d2_final_test \
  --config config/evaluation_d2_final_locked_test.yaml \
  --checkpoint-root /kaggle/input/datasets/sarav07/e11-checkpoint-soup-input \
  --device cuda \
  --preflight-only
```

Require `status=PASS`, `test_split_loaded=false`, and
`test_rasters_opened=false` before authorization.

## One authorized test evaluation

Run exactly once:

```bash
python -u -m src.evaluation.evaluate_d2_final_test \
  --config config/evaluation_d2_final_locked_test.yaml \
  --checkpoint-root /kaggle/input/datasets/sarav07/e11-checkpoint-soup-input \
  --device cuda \
  --authorize-test
```

The evaluator creates the output directory before loading the test dataset and
writes `test_access_ledger.json`. Any existing output directory blocks another
run. A failure after authorization leaves a failure ledger and requires manual
review rather than an automatic rerun.

Expected test split: 339 tiles from DE18 Freiburg, DE19 Kiel and DE20 Rostock.

## Outputs

```text
outputs/final_evaluation/d2_locked_test_e9b_tta_calibrated/
  test_access_ledger.json
  metrics/
    confusion_matrix.csv
    evaluation_metrics.json
    overall_metrics.csv
    per_class_metrics.csv
  reports/
    per_city_metrics.csv
    per_city_class_metrics.csv
    final_evaluation_summary.json
    artifact_inventory.json
```

Package and download this output directory immediately after the run. Report
validation and test results separately. The test mIoU is the final unbiased
generalization estimate even if it is higher or lower than validation.
