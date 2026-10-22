"""Direct image-only comparator: pooled endpoint metrics and the training loop.

The performance-reference direct classifier of the study under its fixed
selection rule. Scripts stay thin CLIs over these primitives.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cbm.data.thyroidxl import INFERENCE_FORBIDDEN_BATCH_KEYS
from cbm.train.stage1 import build_warmup_cosine_scheduler
from cbm.utils import update_estimated_process

__all__ = [
    "patient_log_loss",
    "predict_endpoint",
    "train_direct_classifier",
    "weighted_bce",
]


# ------------------------------------------------- pooled endpoint metrics


def patient_log_loss(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    pos_weight: float,
) -> float:
    """Patient-mean pooled weighted log loss (frames pooled per patient first)."""
    frame = pd.DataFrame(
        {"patient_id": patient_ids, "label": labels, "probability": probabilities}
    )
    pooled = frame.groupby("patient_id", sort=True).agg(
        label=("label", "first"), probability=("probability", "mean")
    )
    return weighted_bce(
        pooled["label"].to_numpy(np.float64),
        pooled["probability"].to_numpy(np.float64),
        pos_weight,
    )


def weighted_bce(labels: np.ndarray, probabilities: np.ndarray, pos_weight: float) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    losses = -(
        pos_weight * labels * np.log(probabilities)
        + (1.0 - labels) * np.log(1.0 - probabilities)
    )
    return float(losses.mean())


# ------------------------------------------------------ direct classifier

#: Checkpoint metrics for the direct baseline. ``validation_loss`` is the
#: default: picking the maximum validation AUC over 100 epochs on a 202-sample
#: split is a selection over ~100 noisy estimates and inflates the reported
#: test AUC, while the validation loss is a smoother, less gameable signal.
#: ``validation_patient_log_loss`` is the unified 2D-ultrasound selection
#: metric (patient-mean pooled, mirrors the E2E core).
_DIRECT_CHECKPOINT_METRICS = (
    "validation_loss",
    "validation_auc",
    "validation_patient_log_loss",
)


def _set_feature_phase(model: object, *, frozen: bool) -> None:
    """Toggle a ConvNeXt-style ``network.features`` stack for a warm phase.

    No-op when unfrozen on models without ``network.features``; a frozen
    request on such a model is an error so ``freeze_epochs`` can never
    silently train the whole stack.
    """
    features = getattr(getattr(model, "network", None), "features", None)
    if features is None:
        if frozen:
            raise TypeError("freeze_epochs requires a model exposing network.features")
        return
    for parameter in features.parameters():
        parameter.requires_grad = not frozen
    if frozen:
        features.eval()
    else:
        features.train()


def train_direct_classifier(
    model: object,
    train_loader: object,
    validation_loader: object,
    *,
    output_path: Path | str,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 5,
    device: str = "cuda",
    seed: int = 42,
    amp: bool = False,
    checkpoint_metric: str = "validation_loss",
    freeze_epochs: int = 0,
    validation_patient_ids: object = None,
    eta_extra_epochs: int = 0,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Train the required direct-image baseline under a fixed selection rule.

    ``freeze_epochs`` (head-only warm-up) freezes a ``network.features``
    stack for the first epochs — the ConvNeXt trunk; 0 trains everything
    from epoch 0. ``validation_patient_ids`` is
    required with the patient-level checkpoint metric and must align with the
    shuffle-free validation loader order.
    """
    import torch

    if checkpoint_metric not in _DIRECT_CHECKPOINT_METRICS:
        raise ValueError(
            f"unknown checkpoint_metric {checkpoint_metric!r}; options are "
            f"{list(_DIRECT_CHECKPOINT_METRICS)}"
        )
    if checkpoint_metric == "validation_patient_log_loss" and validation_patient_ids is None:
        raise ValueError(
            "checkpoint_metric='validation_patient_log_loss' requires validation_patient_ids "
            "aligned with the (shuffle-free) validation loader order"
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs)
    labels = [float(record["label"]) for record in train_loader.dataset.records]
    positive = max(float(np.sum(labels)), 1.0)
    negative = max(float(len(labels) - np.sum(labels)), 1.0)
    train_pos_weight = float(negative / positive)
    pos_weight = torch.tensor(train_pos_weight, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = bool(amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -float("inf")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    history = []
    started_at = time.monotonic()
    for epoch in range(epochs):
        model.train()
        phase = "head_only" if epoch < freeze_epochs else "full_finetune"
        if freeze_epochs:
            _set_feature_phase(model, frozen=epoch < freeze_epochs)
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = criterion(model(image), label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_labels, validation_probabilities = predict_endpoint(
            model, validation_loader, device=device, amp=use_amp
        )[1:]
        validation_auc = float(roc_auc_score(validation_labels, validation_probabilities))
        validation_loss = weighted_bce(
            validation_labels, validation_probabilities, float(pos_weight.item())
        )
        if checkpoint_metric == "validation_patient_log_loss":
            validation_score = patient_log_loss(
                np.asarray(validation_patient_ids, dtype=str),
                validation_labels,
                validation_probabilities,
                float(pos_weight.item()),
            )
        else:
            validation_score = validation_loss
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_bce": float(np.mean(losses)),
                "validation_auc": validation_auc,
                "validation_loss": validation_loss,
                "validation_patient_log_loss": validation_score
                if checkpoint_metric == "validation_patient_log_loss"
                else float("nan"),
                "learning_rate": current_lr,
            }
        )
        print(
            f"{log_prefix}epoch={epoch + 1}/{epochs} phase={phase} "
            f"train_bce={history[-1]['train_bce']:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_auc={validation_auc:.6f} "
            + (
                f"patient_log_loss={validation_score:.6f} "
                if checkpoint_metric == "validation_patient_log_loss"
                else ""
            )
            + f"lr={current_lr:.2e}",
            flush=True,
        )
        scheduler.step()
        elapsed = time.monotonic() - started_at
        remaining_epochs = epochs - epoch - 1 + eta_extra_epochs
        update_estimated_process(elapsed / (epoch + 1) * remaining_epochs)
        score = (
            validation_auc
            if checkpoint_metric == "validation_auc"
            else -validation_score
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_auc": validation_auc,
                    "validation_loss": validation_loss,
                    "checkpoint_metric": checkpoint_metric,
                    "seed": seed,
                },
                destination,
            )

    def _row_score(item: dict[str, Any]) -> float:
        if checkpoint_metric == "validation_auc":
            return item["validation_auc"]
        if checkpoint_metric == "validation_patient_log_loss":
            return -item["validation_patient_log_loss"]
        return -item["validation_loss"]

    best_row = max(history, key=_row_score)
    return {
        "checkpoint_metric": checkpoint_metric,
        "best_epoch": int(best_row["epoch"]),
        "best_validation_auc": float(best_row["validation_auc"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "best_validation_patient_log_loss": float(best_row["validation_patient_log_loss"]),
        "train_pos_weight": train_pos_weight,
        "train_positive_frames": int(np.sum(labels)),
        "train_negative_frames": int(len(labels) - np.sum(labels)),
        "history": history,
    }


def predict_endpoint(
    model: object,
    loader: object,
    *,
    device: str = "cuda",
    amp: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    import torch

    model = model.to(device).eval()
    use_amp = bool(amp) and str(device).startswith("cuda")
    sample_ids, labels, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            forbidden = INFERENCE_FORBIDDEN_BATCH_KEYS.intersection(batch)
            if forbidden:
                raise ValueError(f"Inference batch contains forbidden fields: {sorted(forbidden)}")
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(batch["image"].to(device))
            probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())
            sample_ids.extend(str(value) for value in batch["sample_id"])
    return sample_ids, np.asarray(labels), np.asarray(probabilities)
