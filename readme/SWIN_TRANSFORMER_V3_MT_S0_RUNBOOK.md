# Swin Transformer Dataset V3-MT S0 Runbook

## Architecture identity

The repository model named `swin_transformer` is a hierarchical four-stage
shifted-window Swin encoder with an FPN-like multi-scale convolutional decoder.
It is not a classical symmetric Swin-Unet and must not be reported as one.

## Controlled comparison

This experiment changes only the architecture relative to U-Net MT-E0:

- parent: `unet_v3_mt_e0_paired_50ep`
- intervention: U-Net to hierarchical Swin + FPN-like decoder
- input: frozen eight-channel Dataset V3-MT
- sampling: standard
- seed, loss, class weights, optimizer, scheduler and augmentation: unchanged
- validation: frozen validation split
- test: locked and not evaluated

Reference U-Net MT-E0 validation mIoU: approximately `0.604705`.
A Swin validation mIoU of approximately `0.615` or higher is the predeclared
threshold for a meaningful architecture improvement.

## Local integration gate

```powershell
python -m src.training.train_segmentation `
  --config config/training_swin_transformer_v3_mt_s0_paired_50ep.yaml `
  --integration-check
```

Required output:

```text
Dataset/model contract: PASS
Input channels: 8
Semantic classes: 5
Training samples: 974
Validation samples: 351
all_checks_passed: True
```

Do not run the full training locally. After the gate passes, commit the YAML
and this runbook, then use a Kaggle runtime copy of the Git YAML for GPU
training. Do not evaluate the locked test split.

