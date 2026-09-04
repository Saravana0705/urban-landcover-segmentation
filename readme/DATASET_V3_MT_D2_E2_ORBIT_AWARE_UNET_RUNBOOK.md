# Dataset V3-MT-D2 E2 orbit-aware U-Net experiment

## Purpose

Test whether explicitly separating ascending and descending Sentinel-1 inputs
before feature fusion improves on the D2 E0 early-concatenation result.

The frozen D2 dataset is not modified. The model consumes the existing ordered
16-channel tensor:

```text
channels 0-7:  VV_ASC_Q1, VH_ASC_Q1, ..., VV_ASC_Q4, VH_ASC_Q4
channels 8-15: VV_DESC_Q1, VH_DESC_Q1, ..., VV_DESC_Q4, VH_DESC_Q4
```

Each eight-channel group passes through an independent `DoubleConv` stem. The
two 32-channel feature maps are concatenated and projected back to 32 channels
with a learned 1x1 fusion. The remaining encoder, decoder and skip topology are
the same as the baseline U-Net.

## Experimental control

- Dataset, labels, normalization and city splits: frozen V3-MT-D2.
- Sampling: standard.
- Loss, class weights, optimizer and scheduler: unchanged from D2 E0.
- Augmentation and seed: unchanged from D2 E0.
- Only intervention: pass-specific shallow stems and learned fusion.
- Paired control: `unet_v3_mt_d2_e0_cross_orbit_50ep` (`mIoU 0.663156`).

## 1. Model smoke test

```powershell
python -m src.models.orbit_aware_unet
```

Expected checks include:

```text
Input channels: 16
Orbit channels: 8
Output classes: 5
Standard output: [1, 5, 256, 256]
Odd output: [1, 5, 255, 257]
Trainable parameters: 7,778,373
all_checks_passed: True
```

## 2. Real-data CPU integration gate

```powershell
python -m src.training.train_segmentation `
    --config config/training_orbit_aware_unet_v3_mt_d2_e2_50ep.yaml `
    --integration-check
```

Expected contract:

```text
Training samples: 974
Validation samples: 351
Dataset/model contract: PASS
Input channels: 16
Semantic classes: 5
Model identifier: orbit_aware_unet
Model class: OrbitAwareUNet
Train batches: 1
Validation batches: 1
Trainable parameters: 7,778,373
all_checks_passed: True
```

The one-batch loss and mIoU are not experimental results. Do not create or
commit a runtime YAML until the smoke and integration gates both pass.
