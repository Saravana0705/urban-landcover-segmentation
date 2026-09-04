# Dataset V3-MT-D2-F1 cross-orbit ratio experiment

## Purpose

This is a single-variable ablation against `unet_v3_mt_d2_e0_cross_orbit_50ep`.
It preserves the frozen D2 images, labels, city splits, seed, loss, optimizer,
schedule, standard sampler, and U-Net. The only intervention is eight derived
channels:

- `CR_ASC_Q1` through `CR_ASC_Q4`
- `CR_DESC_Q1` through `CR_DESC_Q4`

Each channel is computed in memory as
`VH_dB - VV_dB = 10*log10(VH/VV)`. No GeoTIFF is modified or duplicated.

## 1. Generate training-only normalization

Run from the repository root after applying the patch:

```powershell
python -m src.data.compute_dataset_v3_mt_d2_ratio_normalization
```

If the three outputs already exist and are intentionally being regenerated:

```powershell
python -m src.data.compute_dataset_v3_mt_d2_ratio_normalization --overwrite
```

Expected result:

- dataset variant `v3-mt-d2-f1`
- 16 source channels
- 8 derived ratio channels
- 24 model channels
- 974 training tiles
- `validation_and_test_used: false`

Generated files:

- `metadata/dataset_v3_mt_d2_features/cross_orbit_ratio/dataset_config.json`
- `metadata/dataset_v3_mt_d2_features/cross_orbit_ratio/training_ratio_normalization.json`
- `metadata/dataset_v3_mt_d2_features/cross_orbit_ratio/normalization_provenance.json`

## 2. Run the local integration gate

```powershell
python -m src.training.train_segmentation `
    --config config/training_unet_v3_mt_d2_f1_cross_orbit_ratio_50ep.yaml `
    --integration-check
```

Require `Dataset/model contract: PASS` and `all_checks_passed: True`.
The one-batch integration mIoU is not an experiment result.

## 3. Review before commit

```powershell
git diff --check
git status --short
git diff --stat
```

Commit the two modified loader/feature files, the new normalization utility,
YAML, runbook, and the three generated metadata JSON files. Do not commit local
tiles, integration outputs, checkpoints, runtime YAML files, or `data/kaggle/`.

## 4. Full GPU command

After cloning/pulling the commit and preparing Kaggle absolute paths in a
runtime copy of the YAML:

```bash
python -m src.training.train_segmentation \
  --config /kaggle/working/training_unet_v3_mt_d2_f1_cross_orbit_ratio_50ep_runtime.yaml
```

Keep the test split locked. Select the best checkpoint only by validation
mean IoU and preserve the complete output directory before ending the session.

## Decision rule

Compare D2-F1 with D2-E0 (`0.6631549967` validation mean IoU), including all
five validation class IoUs. Treat gains smaller than about `0.003` as likely
run-level noise unless they are supported by useful road or bare-land gains.
Do not combine D2-F1 with minority sampling or the orbit-aware model until this
controlled result is known.
