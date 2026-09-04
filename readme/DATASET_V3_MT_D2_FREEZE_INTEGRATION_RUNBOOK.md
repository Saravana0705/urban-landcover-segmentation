# Dataset V3-MT-D2 freeze and integration gate

## Purpose

Freeze the validated 16-channel cross-orbit dataset and run one real training and
validation batch through the existing U-Net pipeline. This confirms the
dataset/loader/model contract without spending Kaggle GPU quota.

V3-MT-D2 is a controlled data-only intervention:

- parent: frozen V3-MT;
- tiles: 1,664, unchanged;
- split: 974 train / 351 validation / 339 locked test;
- labels and validity population: unchanged;
- input: 16 quarterly ascending/descending VV/VH channels instead of the
  single-pass eight-channel stack;
- sampling: standard;
- loss and all other training settings: identical to V3-MT E0.

## 1. Freeze the dataset

Extract the patch into the repository root, then run:

```powershell
python -m src.quality_control.freeze_dataset_v3_mt_d2
```

Expected result:

```text
"status": "FROZEN"
"dataset_version": "v3-mt-d2"
"tile_count": 1664
"input_channels": 16
"next_gate": "local_unet_v3_mt_d2_integration_check"
```

The command intentionally refuses to overwrite an existing freeze directory.

## 2. Run the local integration check

```powershell
python -m src.training.train_segmentation `
    --config config/training_unet_v3_mt_d2_e0_cross_orbit_50ep.yaml `
    --integration-check
```

Expected contract:

```text
Training samples: 974
Validation samples: 351
Input channels: 16
Semantic classes: 5
Train batches: 1
Validation batches: 1
Trainable parameters: 7,766,917
all_checks_passed: True
```

The one-batch mIoU is not an experiment result and must not be compared with
the trained V3-MT metrics.

## 3. Review gate

Review the generated freeze directory and integration report before committing
or packaging the dataset for Kaggle:

```text
metadata/dataset_v3_mt_d2/freeze/
metadata/model_development/unet_integration_check.json
```

Do not start the full 50-epoch run yet.
