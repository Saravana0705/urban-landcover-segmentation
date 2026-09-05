# Dataset V3-MT-D2 S0 Swin Transformer Runbook

## Purpose

This is a controlled architecture experiment on the frozen V3-MT-D2 dataset.
It tests whether the repository's hierarchical Swin Transformer encoder with
an FPN-like semantic decoder benefits from the 16-channel cross-orbit input.

This model is **not a classical Swin-Unet**. The run and report must use the
name **Swin Transformer with FPN-like decoder**.

## Research question

With the labels, tiles, city split, loss and training protocol held fixed, does
the existing Swin/FPN-like architecture outperform the D2 U-Net reference?

Reference experiment:

- `unet_v3_mt_d2_e0_cross_orbit_50ep`
- best validation mIoU: approximately `0.6632`

## Controlled dataset

- Manifest: `metadata/dataset_v3_mt_d2/dataset_manifest.csv`
- Dataset configuration:
  `metadata/dataset_v3_mt_d2/freeze/dataset_config.json`
- Normalization:
  `metadata/dataset_v3_mt_d2/normalization/training_normalization.json`
- Tiles: 1,664
- Split: 974 train / 351 validation / 339 locked test
- Inputs: 16 quarterly ascending/descending VV/VH channels
- Classes: buildings, roads, vegetation, bare land and water

The test split must not be evaluated during model selection.

## Experimental change

The model architecture is the only intended intervention. The experiment keeps
the frozen D2 data, standard sampler, CE-Tversky loss, class weights, seed and
validation protocol fixed. Swin-specific architecture and optimizer settings
are inherited from the earlier controlled Swin experiment.

## Local integration check

Run from the repository root in the activated environment:

```powershell
python -m src.training.train_segmentation `
    --config config/training_swin_transformer_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check
```

The check must report:

- dataset/model contract: PASS
- 16 input channels and 5 semantic classes
- one real training batch and one real validation batch
- positive global and optimizer steps
- recorded best metric and epoch
- latest checkpoint, best checkpoint and history present
- all checks passed

The one-batch integration-check mIoU is not an experimental result.

## Full GPU command

Run only in the GPU notebook after adapting the dataset paths in a runtime copy
of the configuration:

```bash
python -m src.training.train_segmentation \
  --config /kaggle/working/training_swin_transformer_v3_mt_d2_s0_cross_orbit_50ep_runtime.yaml
```

Do not modify the committed configuration merely to insert Kaggle paths.

## Expected outputs

Experiment directory:

`outputs/model_experiments/swin_transformer_v3_mt_d2_s0_cross_orbit_50ep`

Before ending the GPU session, preserve at least:

- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- training history
- experiment state
- overall metrics
- per-class metrics
- confusion matrix
- evaluation metrics
- benchmark summary, when generated
- runtime YAML
- training summary and relevant experiment-registry row

Download or archive these files before disabling the Kaggle GPU.

## Interpretation

Compare the best validation checkpoint with the D2 U-Net reference using mIoU
as the primary metric. Also inspect buildings, roads and bare-land IoU.

- An absolute mIoU change smaller than about `0.005` is marginal.
- An improvement of about `0.010` or more is a meaningful candidate.
- A small overall gain must not hide a material loss in minority-class IoU.
- Pixel accuracy alone is not a model-selection criterion.

Do not combine this architecture with other interventions until their
individual effects have been measured.
