# Remaining Controlled D2 Architecture Runs

## Objective

Complete the primary architecture benchmark on one frozen dataset and one
training protocol. These configurations add the three architectures that have
not yet been trained on Dataset V3-MT-D2:

1. DeepLabV3+;
2. SegFormer;
3. Mask2Former.

The only intended change relative to D2 E0 U-Net is the model architecture.
The dataset, normalization, geographic split, seed, loss, class weights,
optimizer, scheduler, batch size, augmentation, early stopping, epoch budget,
sampling and metric monitor are identical.

The test split must not be evaluated during this phase.

## Local verification

Run from the repository root in the existing Windows virtual environment:

```powershell
python -m compileall -q src

python -m src.quality_control.validate_d2_s0_remaining_architectures
```

Expected: `status: PASS`, `test_split_loaded: false`, and
`all_checks_passed: true`.

Then run one-real-batch integration checks:

```powershell
python -m src.training.train_segmentation `
    --config config/training_deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check

python -m src.training.train_segmentation `
    --config config/training_segformer_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check

python -m src.training.train_segmentation `
    --config config/training_mask2former_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check
```

Each run must report:

- Dataset/model contract: `PASS`;
- input channels: `16`;
- semantic classes: `5`;
- train batches: `1` and validation batches: `1`;
- positive global and optimizer steps;
- latest and best checkpoints present;
- `all_checks_passed: True`.

The one-batch integration mIoU is not a model result and must not be reported.

## Full Kaggle commands

After committing and pushing the groundwork, run the experiments separately in
this order:

```bash
python -m src.training.train_segmentation \
  --config config/training_deeplabv3plus_v3_mt_d2_s0_cross_orbit_50ep.yaml
```

```bash
python -m src.training.train_segmentation \
  --config config/training_segformer_v3_mt_d2_s0_cross_orbit_50ep.yaml
```

```bash
python -m src.training.train_segmentation \
  --config config/training_mask2former_v3_mt_d2_s0_cross_orbit_50ep.yaml
```

Do not alter the seed, loss, batch size or early-stopping settings between
architectures. If a genuine CUDA out-of-memory error occurs, stop and report it
before changing batch size because an uncontrolled change would weaken the
comparison.

## Important artifacts per experiment

Retain only:

```text
checkpoints/best.pt
checkpoints/latest.pt
logs/benchmark_summary.json
logs/latest.json
logs/training_history.csv
logs/training_history.json
metrics/confusion_matrix.csv
metrics/evaluation_metrics.json
metrics/overall_metrics.csv
metrics/per_class_metrics.csv
reports/experiment_state.json
runtime training YAML
training console log
```

Do not create a multi-GB complete-output ZIP. Package the important files with
`src.evaluation.package_experiment_artifacts --require-complete`, plus a small
supplementary ZIP only if the console/runtime files are outside that package.

## Final benchmark boundary

The primary controlled table will contain exactly these seven architectures:

- U-Net;
- Attention U-Net;
- U-Net++;
- DeepLabV3+;
- SegFormer;
- Swin Transformer;
- Mask2Former.

E9B fine-tuning, TTA/calibration, orbit-aware inputs, sampling interventions and
other optimization experiments belong in separate ablation/final-system
tables. They must not be used as architecture-only rows.
