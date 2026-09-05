# Dataset V3-MT-D2 S0 Attention U-Net Runbook

## Purpose

This is a controlled architecture comparison on the frozen Dataset V3-MT-D2
cross-orbit input. It changes only the model from the D2-E0 U-Net reference to
Attention U-Net. The dataset, split, input order, loss, class weights, optimizer,
augmentation and stopping rules remain fixed.

Reference validation result:

- Experiment: `unet_v3_mt_d2_e0_cross_orbit_50ep`
- Best validation mIoU: approximately `0.6632`
- Frozen input: 16 quarterly ascending/descending VV/VH channels
- Split counts: train 974, validation 351, test 339

The integration check is a wiring test only. Its one-batch mIoU is not an
experimental result and must not be compared with the D2-E0 result.

## Files

- Configuration:
  `config/training_attention_unet_v3_mt_d2_s0_cross_orbit_50ep.yaml`
- Frozen manifest:
  `metadata/dataset_v3_mt_d2/dataset_manifest.csv`
- Frozen dataset contract:
  `metadata/dataset_v3_mt_d2/freeze/dataset_config.json`

## Local integration gate

Run from the repository root in the activated virtual environment:

```powershell
python -m src.training.train_segmentation `
    --config config/training_attention_unet_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check
```

Accept the gate only when all of the following are visible:

- Dataset/model contract: `PASS`
- Input channels: `16`
- Semantic classes: `5`
- Model identifier: `attention_unet`
- Train and validation batches: `1`
- `all_checks_passed: True`
- The best and latest checkpoints exist

Do not interpret the integration-check validation mIoU.

## Full GPU command

Run this later in the dedicated GPU workflow, after adapting only the runtime
paths and device settings for Kaggle:

```powershell
python -m src.training.train_segmentation `
    --config config/training_attention_unet_v3_mt_d2_s0_cross_orbit_50ep.yaml
```

Preserve the complete experiment directory before turning off the accelerator.
Use `src.evaluation.package_experiment_artifacts` as described in the
consolidated groundwork runbook.

## Decision rule

Compare the best-checkpoint validation mIoU and per-class IoU with D2-E0. Do not
open or use the test split for experiment selection. Retain Attention U-Net as a
candidate only if it gives a meaningful validation improvement or a useful
minority-class improvement without materially damaging the remaining classes.

