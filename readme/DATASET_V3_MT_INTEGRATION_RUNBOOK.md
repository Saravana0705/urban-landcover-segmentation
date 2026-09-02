# Dataset V3-MT: Freeze and Local Integration Check

## Purpose

This is a controlled input ablation against `unet_v3_e0_paired_50ep`.
Labels, selected tiles, city splits, seed, loss, optimizer, scheduler and U-Net
capacity remain unchanged. The intervention is only:

- V3 baseline: two Sentinel-1 channels.
- V3-MT: eight quarterly Sentinel-1 channels in the frozen order
  `VV_Q1, VH_Q1, VV_Q2, VH_Q2, VV_Q3, VH_Q3, VV_Q4, VH_Q4`.

## Files changed

- `src/data/segmentation_dataset.py`
- `src/training/train_segmentation.py`
- `src/quality_control/freeze_dataset_v3_mt.py`
- `config/training_unet_v3_mt_e0_paired_50ep.yaml`

The U-Net and model factory already support configurable input channels and do
not need modification.

## 1. Apply the patch

Extract the patch ZIP into the repository root, preserving directories. Review:

```powershell
git status --short
git diff --check
```

## 2. Freeze the validated dataset

```powershell
python -m src.quality_control.freeze_dataset_v3_mt
```

Expected summary:

- status `FROZEN`
- 1,664 tiles
- 974/351/339 train/validation/test
- 8 input channels
- next gate `local_unet_v3_mt_integration_check`

Do not delete and regenerate the freeze directory merely to change metadata.
Any dataset change requires a new dataset version.

## 3. Run the real-data integration check

```powershell
python -m src.training.train_segmentation `
  --config config/training_unet_v3_mt_e0_paired_50ep.yaml `
  --integration-check
```

Required evidence:

- `Dataset/model contract: PASS`
- `Input channels: 8`
- one real training batch and one real validation batch complete
- finite loss and validation mean IoU
- optimizer/global steps are positive
- latest and best checkpoints and history exist
- `all_checks_passed: True`

Do not start the 50-epoch Kaggle run if this gate fails.

## 4. Send back for review

Upload:

- terminal output from the freeze command
- terminal output from the integration check
- `metadata/dataset_v3_mt/freeze/dataset_config.json`
- `metadata/dataset_v3_mt/freeze/dataset_v3_mt_freeze.json`
- `metadata/model_development/unet_integration_check.json`

After review, commit the code and frozen metadata, build the Kaggle input, and
run the paired 50-epoch GPU experiment.
