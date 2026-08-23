"""Diagnose Trainer-vs-final-evaluator validation mIoU consistency."""

from __future__ import annotations

from pathlib import Path

import torch

from src.evaluation.benchmarking import load_best_checkpoint
from src.evaluation.evaluate_segmentation import evaluate_segmentation_model
from src.training.train_segmentation import (
    build_project_dataloaders,
    build_segmentation_model,
    read_yaml,
    set_seed,
)

CONFIG_PATH = Path(
    "config/training_unet_e6_p1_tversky.yaml"
)

CHECKPOINT_PATH = Path(
    "outputs/model_experiments/"
    "unet_e6_p1_v21_ce30_tversky70/"
    "checkpoints/best.pt"
)

NUM_CLASSES = 5
IGNORE_INDEX = 255

def unpack_batch(batch):
    """Extract inputs, targets and optional validity mask from a batch."""

    if isinstance(batch, dict):

        def first_present(keys):
            for key in keys:
                if key in batch and batch[key] is not None:
                    return batch[key]
            return None

        inputs = first_present(
            ("image", "images", "input", "inputs")
        )

        targets = first_present(
            ("mask", "target", "targets", "label", "labels")
        )

        validity = first_present(
            ("validity", "validity_mask", "valid_mask")
        )

        if inputs is None or targets is None:
            raise KeyError(
                "Could not resolve input/target tensors from batch dictionary. "
                f"Available keys: {list(batch.keys())}"
            )

        return inputs, targets, validity

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            return batch[0], batch[1], None

        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]

    raise TypeError(
        f"Unsupported validation batch type: {type(batch)}"
    )


def extract_logits(output):
    """Return segmentation logits from common model-output formats."""

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for key in ("logits", "out", "prediction", "predictions"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]

    logits = getattr(output, "logits", None)

    if isinstance(logits, torch.Tensor):
        return logits

    raise TypeError(
        f"Could not extract logits from model output type: {type(output)}"
    )

def build_confusion_matrix(
    model,
    loader,
    device,
) -> torch.Tensor:
    """Accumulate one confusion matrix over the complete validation set."""

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
    )

    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs, targets, validity = unpack_batch(batch)

            inputs = inputs.to(device)
            targets = targets.to(device)

            if validity is not None:
                validity = validity.to(device)

            logits = extract_logits(model(inputs))
            predictions = logits.argmax(dim=1)

            valid = targets != IGNORE_INDEX

            if validity is not None:
                valid = valid & validity.bool()

            targets_valid = targets[valid].to(torch.int64)
            predictions_valid = predictions[valid].to(torch.int64)

            indices = (
                targets_valid * NUM_CLASSES
                + predictions_valid
            )

            confusion += torch.bincount(
                indices.cpu(),
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion


def metrics_from_confusion(
    confusion: torch.Tensor,
) -> dict[str, object]:
    """Calculate dataset-level IoU directly from accumulated confusion."""

    matrix = confusion.to(torch.float64)

    true_positive = torch.diag(matrix)
    target_count = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)

    union = (
        target_count
        + predicted_count
        - true_positive
    )

    iou = torch.where(
        union > 0,
        true_positive / union,
        torch.zeros_like(union),
    )

    return {
        "per_class_iou": iou.tolist(),
        "mean_iou": float(iou.mean().item()),
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    config = read_yaml(CONFIG_PATH)

    seed = int(config.get("seed", 20260725))
    set_seed(seed)

    loaders = build_project_dataloaders(
        config,
        integration_check=False,
    )

    validation_loader = loaders["val"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nValidation metric diagnostic")
    print("----------------------------")
    print(f"Config: {CONFIG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {device}")
    print(
        "Validation samples: "
        f"{len(validation_loader.dataset)}"
    )
    print(
        "Validation batches: "
        f"{len(validation_loader)}"
    )

    model = build_segmentation_model(config)
    model.to(device)

    checkpoint = load_best_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
    )

    print(
        "Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        "Checkpoint best metric: "
        f"{checkpoint.get('best_metric', 'unknown')}"
    )

    # ---------------------------------------------------------
    # Path A:
    # Dataset-level confusion matrix, equivalent to corrected
    # Trainer validation aggregation.
    # ---------------------------------------------------------

    confusion = build_confusion_matrix(
        model,
        validation_loader,
        device,
    )

    trainer_style = metrics_from_confusion(confusion)

    # ---------------------------------------------------------
    # Path B:
    # Existing final evaluation implementation.
    # ---------------------------------------------------------

    evaluator_result = evaluate_segmentation_model(
        model=model,
        loader=validation_loader,
        device=device,
    )

    trainer_miou = float(
        trainer_style["mean_iou"]
    )

    evaluator_miou = float(
        evaluator_result["mean_iou"]
    )

    difference = abs(
        trainer_miou - evaluator_miou
    )

    print("\nDataset-level confusion-matrix result")
    print("-------------------------------------")
    print(
        f"mean_iou: {trainer_miou:.10f}"
    )
    print(
        "per_class_iou: "
        f"{trainer_style['per_class_iou']}"
    )

    print("\nFinal evaluator result")
    print("----------------------")
    print(
        f"mean_iou: {evaluator_miou:.10f}"
    )

    print("\nComparison")
    print("----------")
    print(
        f"Absolute difference: {difference:.12f}"
    )

    if difference < 1e-8:
        print(
            "PASS: Trainer-style dataset-level mIoU and "
            "final evaluator mIoU are equivalent."
        )
    elif difference < 1e-6:
        print(
            "PASS: Results agree within floating-point tolerance."
        )
    else:
        print(
            "FAIL: Metric implementations still disagree."
        )


if __name__ == "__main__":
    main()