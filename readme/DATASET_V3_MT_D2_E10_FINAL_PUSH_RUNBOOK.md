# Dataset V3-MT-D2: E10 final optimization groundwork

This patch adds four bounded experiments. It does not acquire data, alter the
frozen D2 split, or access the test split.

## Experiments

| ID | Experiment | Controlled intervention |
|---|---|---|
| E10A | `unetpp_v3_mt_d2_e10a_deep_supervision_15ep` | Four U-Net++ prediction heads; CE-Lovasz applied at every depth |
| E10B | `unetpp_v3_mt_d2_e10b_road_cldice_12ep` | Road-only soft-clDice auxiliary loss |
| E10C | `d2_e10c_final_calibrated_ensemble` | Validation-only calibration over previous and E10 checkpoints |
| E10D | `unetpp_v3_mt_d2_e10d_boundary_12ep` | Building/road differentiable boundary-F1 auxiliary loss |

E10A, E10B, and E10D are independent weights-only continuations from the E9B
best checkpoint. This keeps their comparison controlled. E10C must be run last.

## Local validation

Run from the repository root in PowerShell:

```powershell
python -m compileall -q src
python -m src.training.losses --output outputs/smoke_tests/e10_loss_smoke.json
python -m src.quality_control.validate_d2_e10_groundwork
python -m src.evaluation.calibrate_d2_ensemble --smoke-test
```

Run the three real-data integration checks separately:

```powershell
python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_e10a_deep_supervision_15ep.yaml `
    --integration-check

python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_e10b_road_cldice_12ep.yaml `
    --integration-check

python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_e10d_boundary_12ep.yaml `
    --integration-check
```

Integration checks deliberately do not load the E9B checkpoint. The actual GPU
runs must use weights-only initialization:

```powershell
python -m src.training.train_segmentation `
    --config config/training_unetpp_v3_mt_d2_e10a_deep_supervision_15ep.yaml `
    --initialize-from outputs/model_experiments/unetpp_v3_mt_d2_e9b_lovasz_finetune_15ep/checkpoints/best.pt
```

Use the same command pattern for E10B and E10D with their corresponding YAML.

E10A permits missing checkpoint keys only for its three new auxiliary heads.
E10B and E10D use strict checkpoint loading because their architecture is
unchanged from E9B.

## E10C execution order

After all required checkpoints are present:

```powershell
python -m src.evaluation.calibrate_d2_ensemble `
    --config config/evaluation_d2_e10c_final_calibrated_ensemble.yaml `
    --cache `
    --search `
    --split val
```

Do not cache or evaluate the test split while selecting weights. A single test
evaluation is authorized only after the validation configuration is reviewed
and locked.

## Git verification

```powershell
git diff --check
git status --short
git add `
    config/training_unetpp_v3_mt_d2_e10a_deep_supervision_15ep.yaml `
    config/training_unetpp_v3_mt_d2_e10b_road_cldice_12ep.yaml `
    config/training_unetpp_v3_mt_d2_e10d_boundary_12ep.yaml `
    config/evaluation_d2_e10c_final_calibrated_ensemble.yaml `
    readme/DATASET_V3_MT_D2_E10_FINAL_PUSH_RUNBOOK.md `
    src/models/unet_plus_plus.py `
    src/models/model_factory.py `
    src/training/losses.py `
    src/training/train_segmentation.py `
    src/evaluation/benchmarking.py `
    src/quality_control/validate_d2_e10_groundwork.py
git diff --cached --check
git diff --cached --stat
```

Commit and push only after all three integration checks report
`all_checks_passed: True`.
