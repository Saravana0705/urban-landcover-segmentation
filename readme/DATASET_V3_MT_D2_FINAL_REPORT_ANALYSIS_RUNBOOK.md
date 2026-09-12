# Dataset V3-MT-D2 Final Report Analysis Runbook

## Purpose

This compiler combines three evidence layers without rerunning any model:

1. the controlled seven-architecture validation benchmark;
2. validation-only robustness analysis;
3. the single validation-locked final test evaluation.

It searches recursively through extracted folders and ZIP archives. Identical
copies are deduplicated by SHA-256. Conflicting copies stop the compilation.
Folder depth and individual experiment folder layouts therefore do not affect
the result.

## Required inputs

Keep the following evidence somewhere below `reports/final_benchmark`:

- `d2_controlled_architectures` folder or ZIP;
- `d2_final_robustness_results` folder or ZIP;
- `d2_final_locked_test_results` folder or ZIP.

The external package-manifest JSON files may remain beside the ZIPs. The
compiler reads the evidence inside the result packages.

## Local validation

From the repository root:

```powershell
python -m compileall -q src

python -m src.evaluation.compile_d2_final_report_analysis `
    --input-root reports/final_benchmark `
    --validate-only
```

Require `status: PASS`, `test_evaluation_performed: false` and
`all_checks_passed: true`.

## Generate report artifacts

```powershell
python -m src.evaluation.compile_d2_final_report_analysis `
    --input-root reports/final_benchmark
```

The output is written to:

```text
reports/final_benchmark/d2_final_report_analysis/
  tables/
  figures/
  reports/
```

Five figures are exported as both 300-dpi PNG and editable SVG. Report-ready
tables are exported as CSV and Markdown. `source_validation.json` records all
input hashes and duplicate locations. `analysis_manifest.json` records all
generated artifact hashes.

If the analysis output already exists, inspect it first. Regeneration requires
the explicit `--overwrite` flag and replaces only
`d2_final_report_analysis`; it never changes source evidence.

## Interpretation rules

- E2 U-Net++ is the controlled architecture-comparison winner.
- E9B U-Net++ with locked four-way TTA and class multipliers is the final
  selected method.
- Validation, robustness and final-test metrics remain separately labeled.
- The test result cannot be used for additional selection or tuning.
- Pixel accuracy must be discussed with class imbalance; mean IoU remains the
  primary metric.
