# D2 E11 Checkpoint Model Soup Runbook

## Purpose

E11 is one final, bounded validation experiment. It averages compatible
U-Net++ checkpoints from E9B, E10B and E10D and evaluates five weights fixed
before looking at the result. It performs no training and does not search TTA
settings, class multipliers or thresholds.

E11 preserves the final E10C evaluation policy:

- validation split only;
- identity, horizontal flip, vertical flip and 180-degree rotation TTA;
- fixed class multipliers `[1.15, 1.15, 1.025, 0.70, 1.075]`;
- float16 probability quantization before scoring;
- E10C reference mIoU `0.6982309104696373`.

The program cannot access the test split. Test evaluation remains a separate,
explicitly authorized step after the E11 winner is locked.

## Files

- `config/evaluation_d2_e11_checkpoint_soup.yaml`
- `src/evaluation/evaluate_d2_checkpoint_soup.py`
- `src/quality_control/validate_d2_e11_groundwork.py`

## Local groundwork checks

These checks require neither the image dataset nor the three checkpoints:

```powershell
python -m compileall -q src

python -m src.evaluation.evaluate_d2_checkpoint_soup `
    --smoke-test

python -m src.quality_control.validate_d2_e11_groundwork

git diff --check
git status --short
```

Do not run `--select` locally unless the complete Dataset V3-MT-D2 validation
data and all three best checkpoints are present. The full run belongs on the
Kaggle GPU.

## Required private Kaggle checkpoint dataset

The attached dataset must contain exactly one selected best checkpoint for
each experiment:

```text
e11-checkpoint-soup-input/
├── e9b/best.pt
├── e10b/best.pt
└── e10d/best.pt
```

Kaggle may add one extra directory level. The evaluator recursively resolves
the three exact relative suffixes, so either of these roots is acceptable:

```text
/kaggle/input/e11-checkpoint-soup-input
/kaggle/input/e11-checkpoint-soup-input/e11-checkpoint-soup-input
```

## Fixed candidates

| Candidate | E9B | E10B | E10D |
|---|---:|---:|---:|
| S0 | 1.000 | 0.000 | 0.000 |
| S1 | 0.750 | 0.250 | 0.000 |
| S2 | 0.750 | 0.000 | 0.250 |
| S3 | 0.750 | 0.125 | 0.125 |
| S4 | 0.500 | 0.250 | 0.250 |

Floating-point tensors, including Batch Normalization running statistics, are
averaged in float32 and converted back to their original dtype. Integer
buffers are copied from E9B. Batch Normalization is not recalibrated because
S0 must remain an exact E9B replay under the E10C evaluation policy.

## Kaggle command

After attaching the code repository, Dataset V3-MT-D2 and the private E11
checkpoint dataset, use the resolved dataset root:

```bash
python -m src.evaluation.evaluate_d2_checkpoint_soup \
  --config config/evaluation_d2_e11_checkpoint_soup.yaml \
  --checkpoint-root /kaggle/input/e11-checkpoint-soup-input \
  --device cuda \
  --select 2>&1 | tee e11_execution_console.log
```

The command fails if S0 does not reproduce the E10C score within `0.0001`.
This prevents selection under a silently changed dataset or numerical policy.

## Essential output artifacts

Preserve only:

```text
outputs/model_experiments/d2_e11_checkpoint_model_soup/
├── checkpoints/selected_soup.pt
├── metrics/
│   ├── confusion_matrix.csv
│   ├── evaluation_metrics.json
│   ├── overall_metrics.csv
│   └── per_class_metrics.csv
├── reports/
│   ├── candidate_metrics.csv
│   ├── candidate_results.json
│   ├── checkpoint_compatibility_audit.json
│   ├── experiment_summary.json
│   └── locked_soup.json
└── checksums.json
```

Also preserve:

- `e11_execution_console.log`;
- the runtime E11 YAML if Kaggle paths were changed;
- the checkpoint-dataset inventory/checksums.

Do not copy the three source checkpoints into the output package because they
already exist in the private Kaggle input dataset.

## Stopping rule

- If no candidate exceeds `0.6982309104696373`, retain refined E10C.
- If a candidate improves the reference, lock it immediately.
- Do not add candidates or tune multipliers after reading E11 results.
- The strict target is crossed only when mIoU is greater than `0.70`.
- Perform the test evaluation once, only after final selection is closed.

