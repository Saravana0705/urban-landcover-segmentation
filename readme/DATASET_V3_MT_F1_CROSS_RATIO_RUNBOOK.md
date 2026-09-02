# Dataset V3-MT-F1: quarterly cross-ratio features

This additive experiment keeps the frozen Dataset V3-MT GeoTIFFs, labels,
manifest and city splits unchanged. Four deterministic log-domain
cross-polarisation channels are calculated in memory:

`CR_Qn = VH_Qn_dB - VV_Qn_dB = 10 log10(VH_Qn / VV_Qn)`

The resulting model input has 12 channels: the original eight quarterly
VV/VH channels followed by CR_Q1--CR_Q4.

Copy the files in this patch to the matching repository paths, then run:

```powershell
python -m src.data.compute_dataset_v3_mt_ratio_normalization
python -m src.training.train_segmentation `
  --config config/training_unet_v3_mt_f1_cross_ratio_50ep.yaml `
  --integration-check
```

The normalization command reads only training tiles and writes a new feature
configuration under `metadata/dataset_v3_mt_features/cross_ratio`. It never
overwrites the frozen V3-MT configuration.

