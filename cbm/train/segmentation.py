"""Segmentation-first comparator: epoch loop, Dice-checkpoint policy, mask prediction.

The frozen augmentation wiring, the AdamW + warmup-cosine epoch loop with
gradient accumulation, and image-only mask prediction (the ground-truth
mask is read solely to compute the QC Dice column and never repairs a
prediction). Scripts stay thin CLIs over these primitives.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cbm.data.thyroidxl import ThyroidXLFullFrameDataset, inference_records
from cbm.data.thyroidxl.segmentation import ThyroidXLSegmentationAugment
from cbm.train.stage1 import build_warmup_cosine_scheduler

__all__ = [
    "binary_dice",
    "build_augmentation",
    "half_bce_half_dice_loss",
    "predict_masks",
    "segmentation_loader",
    "soft_dice_loss",
    "soft_dice_score",
    "train_epochs",
]


# ------------------------------------------- segmentation-first comparator


def build_augmentation(config: dict[str, Any]) -> ThyroidXLSegmentationAugment | None:
    """Build the frozen conservative augmentation from a run configuration."""
    settings = config["augmentation"]
    if not settings["enabled"]:
        return None
    return ThyroidXLSegmentationAugment(
        horizontal_flip_prob=settings["horizontal_flip_prob"],
        max_rotation_degrees=settings["max_rotation_degrees"],
        max_translate_fraction=settings["max_translate_fraction"],
        scale_range=tuple(settings["scale_range"]),
        intensity_scale_range=tuple(settings["intensity_scale_range"]),
        intensity_shift=settings["intensity_shift"],
        noise_std=settings["noise_std"],
    )


def segmentation_loader(dataset, settings: dict[str, Any], *, shuffle: bool):
    """Build the frozen DataLoader settings shared by both comparators."""
    workers = int(settings["workers"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def soft_dice_score(
    logits: torch.Tensor, target: torch.Tensor, *, smooth: float = 1e-5
) -> torch.Tensor:
    """Mean per-sample soft Dice."""
    probability = torch.sigmoid(logits)
    axes = tuple(range(1, probability.ndim))
    numerator = 2.0 * (probability * target).sum(dim=axes) + smooth
    denominator = probability.sum(dim=axes) + target.sum(dim=axes) + smooth
    return (numerator / denominator).mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pure soft-Dice loss (1 − Dice)."""
    return 1.0 - soft_dice_score(logits, target)


def half_bce_half_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The frozen 50/50 BCE/soft-Dice loss."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    axes = tuple(range(1, probability.ndim))
    dice = ((2 * probability * target).sum(axes) + 1e-5) / (
        probability.sum(axes) + target.sum(axes) + 1e-5
    )
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def binary_dice(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-sample binary Dice; empty pairs count as one, never zero."""
    axes = tuple(range(1, prediction.ndim))
    numerator = 2 * (prediction * truth).sum(axes)
    denominator = prediction.sum(axes) + truth.sum(axes)
    return torch.where(denominator > 0, numerator / denominator, torch.ones_like(denominator))


def _accumulation_divisors(batch_count: int, accumulation: int) -> list[int]:
    """Per-minibatch loss divisor for gradient accumulation.

    Each optimizer step must average over its own accumulation window, so the
    final partial window (``batch_count % accumulation`` minibatches when
    nonzero) divides by its own size; dividing everything by ``accumulation``
    under-scales that step's gradient.
    """
    if accumulation < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {accumulation}")
    divisors = [accumulation] * batch_count
    last_window = batch_count % accumulation or accumulation
    for index in range(batch_count - last_window, batch_count):
        divisors[index] = last_window
    return divisors


def train_epochs(
    model,
    loader,
    *,
    settings: dict[str, Any],
    device: torch.device,
    epochs: int,
    validation_loader=None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    schedule_epochs: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Train with AdamW + warmup-cosine, gradient accumulation, and AMP.

    Checkpoints are selected only by validation frame-mean Dice when a
    validation loader is given; otherwise the final epoch state is returned.
    """
    import json

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    # Segmentation loss is config-mandatory: every live segmentation config
    # sets ``loss: dice`` (pure soft Dice); the legacy mixture
    # "0.5_bce_plus_0.5_soft_dice" is accepted only as an explicit value.
    loss_name = str(settings["loss"])
    if loss_name == "dice":
        criterion = soft_dice_loss
    elif loss_name == "0.5_bce_plus_0.5_soft_dice":
        criterion = half_bce_half_dice_loss
    else:
        raise ValueError(f"unknown segmentation loss {loss_name!r}")
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=epochs if schedule_epochs is None else schedule_epochs,
        warmup_epochs=int(settings["warmup_epochs"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(settings["amp"]))
    accumulation = int(settings["gradient_accumulation_steps"])
    divisors = _accumulation_divisors(len(loader), accumulation)
    history = []
    best: dict[str, Any] = {"dice": -1.0, "state": None, "epoch": None}
    for epoch in range(epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = criterion(model(image), mask) / divisors[batch_index]
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * divisors[batch_index])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": learning_rate,
        }
        if validation_loader is not None:
            model.eval()
            dices = []
            with torch.no_grad():
                for batch in validation_loader:
                    image = batch["image"].to(device, non_blocking=True)
                    truth = batch["mask"].to(device, non_blocking=True)
                    prediction = (torch.sigmoid(model(image)) >= 0.5).float()
                    dices.extend(binary_dice(prediction, truth).cpu().tolist())
            row["validation_frame_mean_dice"] = float(np.mean(dices))
            if row["validation_frame_mean_dice"] > best["dice"]:
                best = {
                    "dice": row["validation_frame_mean_dice"],
                    "state": {key: value.cpu() for key, value in model.state_dict().items()},
                    "epoch": epoch,
                }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, time.monotonic())
    if validation_loader is None:
        best = {
            "dice": None,
            "state": {key: value.cpu() for key, value in model.state_dict().items()},
            "epoch": epochs - 1,
        }
    return history, best


def predict_masks(
    model,
    frame: pd.DataFrame,
    *,
    settings: dict[str, Any],
    device: torch.device,
    destination: Path,
    source: str,
    expected_preprocessing_version: str | None = None,
) -> pd.DataFrame:
    """Predict masks for complete frames and audit Dice against training truth.

    Inference receives only image records; the ground-truth mask is read
    solely to compute the QC Dice column and is never used to repair a
    prediction.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if expected_preprocessing_version is None:
        records = inference_records(frame)
    else:
        records = inference_records(
            frame, expected_preprocessing_version=expected_preprocessing_version
        )
    if any(any("mask" in key or "radiomics" in key for key in record) for record in records):
        raise ValueError("Segmentation prediction record includes prohibited fields")
    loader = segmentation_loader(ThyroidXLFullFrameDataset(records), settings, shuffle=False)
    truth_paths = frame.set_index("sample_id")["training_only_mask_path"]
    metadata = frame.set_index("sample_id")[["patient_id", "label"]]
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(image))
            predictions = probabilities >= 0.5
            for index, sample_id in enumerate(batch["sample_id"]):
                predicted = predictions[index, 0].byte().cpu().numpy()
                truth = np.asarray(Image.open(truth_paths.loc[sample_id]).convert("L")) > 0
                denominator = int(predicted.sum() + truth.sum())
                dice = 2 * int((predicted & truth).sum()) / denominator if denominator else 1.0
                path = destination / f"{sample_id}.png"
                Image.fromarray(predicted * 255).save(path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "patient_id": str(metadata.loc[sample_id, "patient_id"]),
                        "label": int(metadata.loc[sample_id, "label"]),
                        "prediction_source": source,
                        "predicted_mask_path": str(path),
                        "predicted_mask_voxels": int(predicted.sum()),
                        "empty_predicted_mask": not bool(predicted.any()),
                        "frame_dice_qc_only": float(dice),
                        "mean_probability": float(probabilities[index].mean().cpu()),
                        # Constant since the ground-truth-mask fallback route was removed;
                        # kept so downstream schemas stay stable.
                        "ground_truth_fallback_used": False,
                    }
                )
    return pd.DataFrame(rows)
