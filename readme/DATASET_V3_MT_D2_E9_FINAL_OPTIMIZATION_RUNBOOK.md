# Dataset V3-MT-D2 E9 final optimization runbook

## Scope

These are the final two low-cost experiments before report preparation. They
reuse the frozen D2 data and existing checkpoints. No Sentinel-1 acquisition or
label export is performed.

| ID | Experiment | Purpose | GPU work |
|---|---|---|---|
| E9A | Calibrated ensemble + 4-way TTA | Tune ensemble weights and small class-probability multipliers directly against validation mIoU | Inference only |
| E9B | U-Net++ CE + Lovasz continuation | Fine-tune the best U-Net++ weights with an IoU-oriented surrogate loss | Up to 15 epochs |

E9A is run first. E9B can then be included as another E9A member only as a
separate, post-hoc comparison; do not silently change the locked E9A result.

## Local checks

```powershell
python -m compileall -q src
python -m src.training.losses
python -m src.evaluation.calibrate_d2_ensemble --smoke-test
python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_e9b_lovasz_finetune_15ep.yaml `
    --integration-check
```

The E9B integration check deliberately uses random initialization. The real GPU
run must pass `--initialize-from` and the best U-Net++ checkpoint.

## E9A Kaggle sequence

First edit only the four checkpoint paths in
`config/evaluation_d2_e9a_calibrated_ensemble_tta.yaml` to match the attached
Kaggle artifact dataset. Keep member order, search grid, data split, and seed
unchanged.

```bash
python -m src.evaluation.calibrate_d2_ensemble \
  --config config/evaluation_d2_e9a_calibrated_ensemble_tta.yaml \
  --cache --search --split val --device cuda
```

Required evidence:

- `cache/val/cache_manifest.json`
- `reports/locked_calibration.json`
- the runtime YAML and console log

The probability cache is reproducible intermediate data and does not need to be
downloaded. Preserve only the report/config/log files in the small artifact ZIP.

Do not cache or evaluate the test split while choosing weights or class
multipliers. If a final test evaluation is authorized after locking, use the
unchanged lock:

```bash
python -m src.evaluation.calibrate_d2_ensemble \
  --config config/evaluation_d2_e9a_calibrated_ensemble_tta.yaml \
  --cache --split test --authorize-test --device cuda

python -m src.evaluation.calibrate_d2_ensemble \
  --config config/evaluation_d2_e9a_calibrated_ensemble_tta.yaml \
  --evaluate-locked --split test --authorize-test \
  --locked-config outputs/model_experiments/d2_e9a_calibrated_ensemble_tta/reports/locked_calibration.json
```

## E9B Kaggle sequence

Run weights-only initialization from the best checkpoint of the original D2
U-Net++ experiment. This is intentionally not `--resume`: optimizer, scheduler,
epoch, and early-stopping state must start fresh for the new loss and learning
rate.

```bash
python -m src.training.train_segmentation \
  --config config/training_unetpp_v3_mt_d2_e9b_lovasz_finetune_15ep.yaml \
  --initialize-from /kaggle/path/to/unetpp_v3_mt_d2_s0_cross_orbit_50ep/checkpoints/best.pt \
  2>&1 | tee /kaggle/working/e9b_training_console.log
```

Before accepting the run, confirm the console contains:

- `Loss: ce_lovasz`
- `Lovasz weight: 0.7`
- `Optimizer/scheduler state restored: False`
- the expected source checkpoint
- `training_completed: True` and `all_checks_passed: True`

Preserve only the important artifacts: best checkpoint, latest checkpoint,
training histories, metrics, confusion matrix, experiment state, benchmark or
training summary, runtime YAML, and console log. Do not create a multi-gigabyte
archive of the complete output directory.

## Decision rule

Compare validation mIoU and every class IoU with E6A (mIoU 0.69023). Treat
0.70 as a target, not a guaranteed outcome. Do not report improvement from an
integration-check batch or select a configuration using test labels.
