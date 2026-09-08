# Final D2/D3 experiment groundwork

This batch contains three GPU-ready D2 experiments and one gated Sentinel-1
SLC coherence pilot. Keep the frozen D2 labels, manifests, splits and
normalisation unchanged.

## 1. Static and model smoke checks

Run from the repository root in the project venv:

```powershell
python -m compileall -q src
python -m src.data.label_uncertainty
python -m src.models.multiview_unetpp
python -m src.evaluation.evaluate_d2_ensemble_tta `
    --config config/evaluation_d2_e6a_ensemble_tta.yaml `
    --smoke-test
```

E6A intentionally evaluates `val` only. Before its GPU evaluation, update the
three checkpoint paths in `evaluation_d2_e6a_ensemble_tta.yaml` if downloaded
artifacts use a different local/Kaggle directory. It creates no new trained
checkpoint.

## 2. E7 audit and integration check

The core-supported inner uncertainty band is constructed in memory after
spatial augmentation and only by the training dataset. It masks selected-class
boundary pixels only where an eroded interior core survives. Thin roads and
small bare-land regions without a core remain supervised, and neighboring
classes are not masked. Source GeoTIFF labels and validation/test targets are
never rewritten.

```powershell
python -m src.quality_control.validate_d2_label_uncertainty `
    --config config/training_unetpp_v3_mt_d2_e7_uncertainty_labels_50ep.yaml `
    --max-tiles 25 `
    --overwrite

python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_e7_uncertainty_labels_50ep.yaml `
    --integration-check
```

Review the ignored fraction in
`metadata/model_development/d2_e7_label_uncertainty_audit.json`. If it exceeds
0.15, stop and inspect class retention before authorising the full GPU run.

## 3. E8 integration check

The 16 D2 channels are split into eight ordered two-channel VV/VH views:
ASC Q1--Q4 followed by DESC Q1--Q4. A shared encoder processes every view;
learned attention and max fusion produce the input to U-Net++.

```powershell
python -m src.training.train_segmentation `
    --config config/training_multiview_unetpp_v3_mt_d2_e8_50ep.yaml `
    --integration-check
```

## 4. D3 SLC coherence pilot (not yet GPU-authorised)

SLC coherence cannot be computed from the existing GRD/D2 GeoTIFFs. First
search SLC scenes for only Mannheim (DE01) and Leipzig (DE15):

```powershell
python -m src.acquisition.search_sentinel1_catalog `
    --cities config/dataset_v3_mt_d3_slc_pilot_cities.csv `
    --aois data/aoi/all_city_aois.gpkg `
    --config config/acquisition_v3_mt_d3_slc_pilot.yaml `
    --candidates-output metadata/dataset_v3_mt_d3/pilot/slc_catalog.csv `
    --selected-output metadata/dataset_v3_mt_d3/pilot/slc_single_scene_selection_unused.csv `
    --raw-output-dir metadata/dataset_v3_mt_d3/pilot/raw_stac

python -m src.data.select_dataset_v3_mt_d3_slc_pairs `
    --catalog metadata/dataset_v3_mt_d3/pilot/slc_catalog.csv `
    --overwrite

python -m src.quality_control.validate_dataset_v3_mt_d3_slc_pilot
```

The selector intentionally chooses at most one same-platform S1A pair per city,
preferably with a 12-day baseline near 15 July 2025. Empty STAC product URLs do
not fail pair QA; authenticated download identifiers are resolved through CDSE
OData in the next gated stage.

Resolve the four selected product names and inspect the total transfer size.
This step queries only catalogue metadata and downloads no SLC bytes:

```powershell
python -m src.acquisition.resolve_dataset_v3_mt_d3_slc_products `
    --overwrite

python -m src.quality_control.validate_dataset_v3_mt_d3_slc_download_plan

python -m src.quality_control.validate_dataset_v3_mt_d3_slc_burst_coverage `
    --overwrite
```

Stop if either command fails, if the manifest contains anything other than two
products per pilot city, if no common IW burst covers an AOI, or if the reported storage requirement is unsuitable.
Share the resolution and QA reports before downloading. The downloader remains
blocked unless `--authorize-download` is explicitly supplied:

```powershell
python -m src.acquisition.download_dataset_v3_mt_d3_slc_pilot `
    --authorize-download `
    --overwrite-audit
```

The downloader requires `CDSE_USERNAME` and `CDSE_PASSWORD` in `.env`, checks
free disk space, supports `.part` resume files, and verifies size, available
catalogue checksum, and ZIP integrity. It refuses any manifest outside the
fixed two-city/four-product pilot scope. Do not run this command until the
download-size report has been reviewed.

Only after pair QA passes should the selected master/slave products be
downloaded and processed with SNAP GPT. For each pair, call the graph with
parameters equivalent to:

```powershell
& "D:/Program Files/esa-snap/bin/gpt.exe" `
  snap_graphs/sentinel1_slc_coherence_d3_pilot.xml `
  "-Pmaster=<MASTER_SAFE_OR_ZIP>" `
  "-Pslave=<SLAVE_SAFE_OR_ZIP>" `
  "-Psubswath=IW2" `
  "-Ppolarisations=VV,VH" `
  "-PcohWinAz=3" `
  "-PcohWinRg=10" `
  "-PpixelSpacing=10.0" `
  "-Poutput=<OUTPUT_TIF>"
```

The fixed `IW2` choice and target CRS must be checked against each AOI and
burst footprint before batch processing. This pilot does not authorise a full
20-city export or a GPU training run. A D3 dataset configuration and training
YAML must be created only after processed-band QA, training-only
normalisation, materialisation and freeze gates pass.

## 5. Final repository checks (one commit/push)

After all local checks pass:

```powershell
git diff --check
git status --short
git add config src snap_graphs readme/DATASET_V3_MT_D2_FINAL_EXPERIMENT_GROUNDWORK_RUNBOOK.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add final D2 experiments and D3 coherence pilot groundwork"
git fetch origin
git rebase origin/main
git push origin main
git status
```

Do not stage generated integration-check outputs or source SLC products.
