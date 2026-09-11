# Dataset V3-MT-D2 final controlled benchmark

This stage compiles retained validation evidence for the seven controlled
architecture runs. It does not load imagery, perform inference, or access the
test split.

## Input folder

Place the seven extracted folders under:

`outputs/model_experiments/benchmark_analysis_runs_latest`

Folder-internal layouts may differ. Artifact discovery is recursive. The E0
checkpoint may be absent; all other checkpoints and all seven metrics packages
are required.

## Commands

```powershell
python -m compileall -q src

python -m src.evaluation.compile_d2_final_benchmark `
    --validate-only

python -m src.evaluation.compile_d2_final_benchmark
```

Expected validation result:

- `status: PASS`
- `experiment_count: 7`
- `core_metrics_complete: true`
- `missing_checkpoint_allowed: ["E0"]`
- `selection_split: validation`
- `test_split_loaded: false`
- best controlled architecture: U-Net++

Generated evidence is written to:

`reports/final_benchmark/d2_controlled_architectures`

Do not run test evaluation at this stage. Geographic robustness and controlled
SAR-noise sensitivity are separate validation-only stages and must remain
reported as unavailable until those evaluations are run.
