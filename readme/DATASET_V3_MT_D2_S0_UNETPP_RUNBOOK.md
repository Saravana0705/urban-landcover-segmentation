# Dataset V3-MT-D2 S0 U-Net++ Runbook

## Purpose

This controlled experiment applies the repository's existing U-Net++ model to
the frozen 16-channel V3-MT-D2 cross-orbit dataset. It tests whether nested
dense skip connections extract more value from the cross-orbit inputs than the
standard U-Net reference.

## Research question

With the dataset, labels, sampling, loss and evaluation protocol fixed, does
U-Net++ improve validation performance over D2-E0 U-Net?

Reference:

- experiment: `unet_v3_mt_d2_e0_cross_orbit_50ep`
- best validation mIoU: approximately `0.6632`

The earlier U-Net++ result on Dataset V3 was approximately `0.5002`, below the
V3 U-Net result of approximately `0.5133`. D2 therefore tests the architecture
again only after the input information has materially improved.

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

Do not load or evaluate the test split during model selection.

## Experimental control

The architecture is the only intended intervention. The experiment retains the
frozen D2 inputs, standard sampler, seed, class weights, CE-Tversky loss,
augmentation policy, optimizer, scheduler and validation protocol. U-Net++
architecture settings are inherited from its earlier controlled V3 run.

## Local integration check

From the activated repository environment, run:

```powershell
python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check
```

Required outcome:

- dataset/model contract: PASS
- 16 input channels and 5 semantic classes
- one real training and validation batch
- positive global and optimizer steps
- finite loss and recorded validation metric
- best and latest checkpoints present
- history present
- all checks passed

The integration-check mIoU is only a wiring check and must not be compared with
full-run results.

## Future GPU command

Use a runtime copy containing Kaggle dataset paths:

```bash
python -m src.training.train_segmentation \
  --config /kaggle/working/training_unetpp_v3_mt_d2_s0_cross_orbit_50ep_runtime.yaml
```

Expected repository-relative output directory:

`outputs/model_experiments/unetpp_v3_mt_d2_s0_cross_orbit_50ep`

The GPU notebook may override the output directory in the runtime copy to a
direct child of `/kaggle/working` for easier preservation.

## Required artifacts

Before ending the GPU session, preserve:

- best and latest checkpoints
- training history
- experiment state
- overall and per-class metrics
- confusion matrix
- evaluation metrics
- benchmark summary, when generated
- runtime configuration
- training summary and experiment-registry record

Do not disable the GPU until the archive has been created and downloaded or
saved as notebook output.

## Interpretation

Use validation mIoU as the primary metric and examine buildings, roads and bare
land separately.

- Absolute mIoU change below about `0.005`: marginal.
- Improvement of about `0.010` or more: meaningful candidate.
- A small overall gain is not sufficient if minority-class IoU deteriorates.
- Pixel accuracy alone must not determine model selection.

Do not combine U-Net++ with sampling or feature interventions until its
individual effect on frozen D2 has been measured.
