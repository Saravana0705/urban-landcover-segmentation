# Dataset V3-MT-F2 Local Spatial Feature Experiment

## Controlled intervention

Dataset V3-MT-F2 keeps the frozen V3-MT GeoTIFFs, labels, manifest and city
splits unchanged. It retains the four F1 cross-ratio channels and adds four
5x5 local statistics for each quarterly VH dB band:

- median
- minimum
- maximum
- range (`maximum - minimum`)

The model input therefore contains 28 channels:

- 8 original quarterly VV/VH channels
- 4 quarterly cross-ratio channels
- 16 quarterly VH local-spatial channels

Only the local-spatial channels are new relative to MT-F1. Standard sampling,
the U-Net, seed, loss, class weights, optimizer, scheduler and augmentation
settings remain unchanged.

## Local preparation

Copy the patch files to the matching repository paths, then run:

```powershell
python -m src.data.compute_dataset_v3_mt_spatial_normalization
```

Expected gate:

```text
status: PASS
dataset_variant: v3-mt-f2
source_channels: 8
cross_ratio_channels: 4
local_spatial_channels: 16
model_channels: 28
training_tiles: 974
validation_and_test_used: false
```

Run the real-data integration check:

```powershell
python -m src.training.train_segmentation `
  --config config/training_unet_v3_mt_f2_local_spatial_50ep.yaml `
  --integration-check
```

The integration output must report 28 input channels, five classes, 974
training samples, 351 validation samples and `all_checks_passed: True`.

Do not run the full experiment locally and do not evaluate the locked test
split. Commit the generated metadata with the code/config after the local gate.

## Boundary and invalid-pixel policy

Filters use a fixed 5x5 window with reflected tile boundaries. Invalid
neighbours are filled using the corresponding frozen training-only VH mean.
Outputs whose centre pixel is invalid are set to NaN and subsequently to zero
after feature normalization. Supervision continues to use the frozen validity
mask, so invalid pixels do not contribute to the loss or metrics.

