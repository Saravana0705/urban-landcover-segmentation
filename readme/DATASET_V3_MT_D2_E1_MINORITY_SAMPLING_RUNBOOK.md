# Dataset V3-MT-D2 E1 minority-sampling experiment

## Purpose

Test whether the previously validated moderate fraction-aware sampler adds
value after the cross-orbit data improvement. Dataset V3-MT-D2 E0 remains the
paired control.

Only the training-sample order changes. The frozen 16-channel dataset, city
split, labels, normalization, U-Net, loss, class weights, optimizer,
augmentations, scheduler, seed and validation loader remain unchanged.

## Configuration

```text
config/training_unet_v3_mt_d2_e1_minority_sampling_50ep.yaml
```

The sampling policy is copied exactly from the completed V3-MT E1 experiment:

```yaml
strategy: fraction_aware
bare_land_low_threshold: 0.01
bare_land_high_threshold: 0.05
road_threshold: 0.1
water_threshold: 0.05
bare_land_low_boost: 0.5
bare_land_high_boost: 0.5
road_boost: 0.25
water_boost: 0.0
max_weight: 2.25
replacement: true
```

## Local integration gate

Run from the repository root:

```powershell
python -m src.training.train_segmentation `
    --config config/training_unet_v3_mt_d2_e1_minority_sampling_50ep.yaml `
    --integration-check
```

Expected contract:

```text
Training samples: 974
Validation samples: 351
Strategy: fraction_aware
Replacement: True
Seed: 20260725
Mean weight: 1.2179
Maximum weight: 2.2500
Tiles with weight > 1: 586 (60.16%)
Dataset/model contract: PASS
Input channels: 16
Semantic classes: 5
Train batches: 1
Validation batches: 1
Trainable parameters: 7,766,917
all_checks_passed: True
```

The one-batch loss and mIoU are not experimental results.

## GPU decision rule

Compare primarily with the paired D2 E0 validation mIoU of `0.663156`.
Prefer E1 only if it produces a useful overall or minority-class improvement
without materially degrading roads or buildings. Keep the test split locked.

Do not commit a Kaggle runtime YAML. Create the runtime copy inside
`/kaggle/working` after this local gate and review are complete.
