# Final D2 robustness and inference benchmark

This stage evaluates the seven controlled architecture checkpoints on the
frozen validation split only. It produces cross-city consistency, additional
Gamma-speckle sensitivity and GPU inference profiles. It does not train,
update checkpoints, select using test data, or open test rasters.

## Local groundwork

```powershell
python -m compileall -q src
python -m src.quality_control.validate_d2_final_robustness_groundwork
```

Expected: `status: PASS`, seven model contracts and parameter counts matched,
`test_split_loaded: false`, and `training_started: false`.

## Checkpoint package

Create one Kaggle dataset containing the six retained `best.pt` files. Preserve
the existing benchmark folder names so automatic discovery can identify them.
Do not include `latest.pt`; it doubles storage without helping this evaluation.

Build the minimal upload folder locally:

```powershell
python -m src.evaluation.package_d2_robustness_checkpoints
```

Expected output:

`outputs/kaggle_upload/d2_final_robustness_checkpoints`

Create a Kaggle dataset from the contents of this directory. Each checkpoint is
stored below its full experiment name, and `checkpoint_manifest.json` records
its SHA-256 checksum.

The E0 checkpoint will be produced by the controlled reproduction run. Pass it
as an explicit override while using the Kaggle input dataset as the checkpoint
root for E1-E6.

## Kaggle preflight

```bash
python -m src.evaluation.evaluate_d2_final_robustness \
  --checkpoint-root /kaggle/input/datasets/USER/DATASET \
  --checkpoint E0=/kaggle/working/urban-landcover-segmentation/outputs/model_experiments/unet_v3_mt_d2_e0_cross_orbit_50ep/checkpoints/best.pt \
  --preflight-only
```

## Kaggle evaluation

```bash
python -u -m src.evaluation.evaluate_d2_final_robustness \
  --checkpoint-root /kaggle/input/datasets/USER/DATASET \
  --checkpoint E0=/kaggle/working/urban-landcover-segmentation/outputs/model_experiments/unet_v3_mt_d2_e0_cross_orbit_50ep/checkpoints/best.pt
```

The live display contains one dynamic progress bar per model/condition and one
persistent summary line after each condition. The output directory is:

`outputs/model_comparisons/d2_final_robustness`

Zip and download the entire directory after a PASS result.
