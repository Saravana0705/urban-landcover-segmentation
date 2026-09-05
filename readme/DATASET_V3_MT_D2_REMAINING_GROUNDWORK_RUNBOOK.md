# Dataset V3-MT-D2 Remaining Groundwork Runbook

## Scope

This bundle completes the remaining local preparation before the next GPU
batch. It contains:

1. the controlled D2 Attention U-Net configuration;
2. a pre-shutdown experiment artifact packager;
3. a validation-only comparison utility;
4. the runbooks for checking and using them.

The Swin Transformer and U-Net++ D2 configurations were prepared and committed
separately before this bundle. The already prepared D2-E1 minority sampling,
D2-E2 orbit-aware U-Net and D2-F1 ratio experiments remain part of the same GPU
queue.

## Copy the bundle into the repository

Extract the ZIP and copy its `config`, `readme` and `src` directories over the
matching directories in the repository. No dataset tiles, checkpoints or GPU
outputs are included.

Before running anything:

```powershell
git status --short
git diff --check
```

Keep all changes uncommitted until every gate below has passed. Then make one
final commit and push.

## Gate 1: Attention U-Net integration check

```powershell
python -m src.training.train_segmentation `
    --config config/training_attention_unet_v3_mt_d2_s0_cross_orbit_50ep.yaml `
    --integration-check
```

Expected result: the real frozen D2 dataset contract passes, the model receives
16 channels, one optimizer step is completed, both checkpoints are written and
`all_checks_passed` is true. The one-batch mIoU is not evidence of performance.

## Gate 2: Artifact-packager dry run

Use the complete saved D2-E0 GPU experiment directory. A one-batch integration
check does not normally produce the full evaluation evidence required by this
gate:

```powershell
python -m src.evaluation.package_experiment_artifacts `
    --experiment-dir outputs/model_experiments/unet_v3_mt_d2_e0_cross_orbit_50ep `
    --runtime-config config/training_unet_v3_mt_d2_e0_cross_orbit_50ep.yaml `
    --output-dir outputs/artifact_packages `
    --require-complete `
    --dry-run
```

Expected result: `status` is `PASS`, `experiment_completed` is true and
`missing_required` is empty. If the saved D2-E0 directory is elsewhere, replace
only the `--experiment-dir` value with that directory.

The packager rejects filenames that appear to contain credentials, secrets or
tokens. It records SHA-256 hashes and packages only a fixed evidence whitelist,
plus explicitly supplied safe files.

## Gate 3: Create a disposable local package

Remove only `--dry-run` from the previous command:

```powershell
python -m src.evaluation.package_experiment_artifacts `
    --experiment-dir outputs/model_experiments/unet_v3_mt_d2_e0_cross_orbit_50ep `
    --runtime-config config/training_unet_v3_mt_d2_e0_cross_orbit_50ep.yaml `
    --output-dir outputs/artifact_packages `
    --require-complete
```

Open the resulting ZIP and confirm that it contains `artifact_inventory.json`,
both checkpoints, training history, experiment state, overall metrics,
per-class metrics, confusion matrix, evaluation metrics and the supplied
configuration. The generated ZIP is an output artifact and should not be added
to Git.

For every full Kaggle run, execute the same command against that run's output
directory before disabling the GPU or ending the session. Download the ZIP
first, verify it is present locally, and only then stop the Kaggle session.

## Gate 4: Validation comparison utility

If the complete D2-E0 experiment directory is already stored at its canonical
location, run:

```powershell
python -m src.evaluation.compare_d2_experiments
```

If it is stored elsewhere, pass an explicit override. PowerShell paths may be
quoted:

```powershell
python -m src.evaluation.compare_d2_experiments `
    --experiment "unet_v3_mt_d2_e0_cross_orbit_50ep=D:\path\to\unet_v3_mt_d2_e0_cross_orbit_50ep"
```

Expected result while only D2-E0 has a complete saved run:

- report status: `PASS`;
- completed experiments: at least `1`;
- unrun experiments: `pending`;
- `test_split_loaded: false`;
- D2-E0 delta from itself: `0.0`.

Outputs are written to `outputs/model_comparisons/v3_mt_d2/`:

- `d2_experiment_summary.csv`
- `d2_per_class_comparison.csv`
- `d2_comparison_provenance.json`

These generated comparison outputs need not be committed during groundwork.
After GPU runs, preserve them with the final experimental evidence.

## GPU experiment queue

Use this order to protect GPU quota and obtain the highest-value evidence
first:

1. `unet_v3_mt_d2_e1_minority_sampling_50ep`
2. `orbit_aware_unet_v3_mt_d2_e2_50ep`
3. `unet_v3_mt_d2_f1_cross_orbit_ratio_50ep`
4. `unetpp_v3_mt_d2_s0_cross_orbit_50ep`
5. `attention_unet_v3_mt_d2_s0_cross_orbit_50ep`
6. `swin_transformer_v3_mt_d2_s0_cross_orbit_50ep`

The first three isolate sampling, orbit-specific fusion and derived-feature
effects. U-Net++ and Attention U-Net are controlled CNN architecture checks.
Swin is last because it is a broader architecture comparison and is less
directly tied to the data improvement that produced the current gain.

Run one experiment at a time. Package and download its evidence before starting
the next one. Use validation mIoU and per-class IoU for model selection; reserve
the test split until the final model and protocol have been frozen.

## One final Git commit

After all four local gates pass:

```powershell
git status --short
git diff --check
```

Stage only the source, configuration and runbook files from this bundle. Do not
stage `outputs/`, datasets, checkpoints, generated ZIP files, secrets or runtime
path overrides.

Then inspect the staged change:

```powershell
git diff --cached --check
git diff --cached --stat
git status --short
```

Commit and push once only after the staged list is correct. The exact `git add`
command should be based on the final `git status --short` output so unrelated
local files are not included accidentally.
