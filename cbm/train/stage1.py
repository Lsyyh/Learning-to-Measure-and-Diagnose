"""Stage-1 named-radiomics regression training and its shared plumbing.

Owns the deterministic-seeding / DataLoader-RNG plumbing every training
entry point funnels through, the shared warmup+cosine and plateau LR
schedules, the trainable-scope masks and head/backbone parameter split,
the offline per-feature loss-weight table loader, and the Stage-1 training
loop itself (SmoothL1 by default; every search knob defaults to the frozen
recipe of record so existing configurations replay bit-for-bit).
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from cbm.data.thyroidxl import INFERENCE_FORBIDDEN_BATCH_KEYS
from cbm.eval.stage1 import stage1_quality
from cbm.utils import update_estimated_process

# --------------------------------------------------------- determinism

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def seed_everything(
    seed: int,
    *,
    warn_only: bool = False,
    deterministic_kernels: bool = True,
) -> None:
    """Seed python/NumPy/torch RNG and request deterministic kernels.

    ``deterministic_kernels=True`` (the default) additionally enables
    ``torch.use_deterministic_algorithms``, so CUDA kernels without a
    deterministic implementation raise instead of silently diverging; if a
    required kernel has no deterministic path, callers may pass
    ``warn_only=True`` and must record the offending op plus the observed max
    absolute error in the replay verdict. ``deterministic_kernels=False``
    keeps only the cuDNN flags — use it to reproduce runs trained before
    deterministic-kernel selection was enforced (the frozen segmentation
    comparator runs). ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first
    cuBLAS handle is created; the entry scripts set it at module import,
    ahead of any CUDA op.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic_kernels:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def determinism_record(seed: int) -> dict[str, object]:
    """Snapshot the process determinism state for run metadata."""
    import torch

    return {
        "seed": int(seed),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_warn_always_enabled()),
    }


def loader_generator(seed: int):
    """Fresh torch generator fixing a DataLoader's shuffle and worker base seed."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seeded_worker_init(worker_id: int) -> None:
    """Derive per-worker NumPy/python seeds from the loader's torch base seed."""
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------ LR schedules


def build_warmup_cosine_scheduler(
    optimizer: object,
    *,
    epochs: int,
    warmup_epochs: int = 5,
    eta_min: float = 0.0,
) -> object:
    """Return a per-epoch LambdaLR scheduler: warmup then cosine anneal.

    The scheduler is stepped once per epoch (``scheduler.step()`` at the end of
    each epoch). During warmup the multiplier rises linearly from
    ``1/(warmup_epochs)`` to 1.0; afterwards it follows cosine annealing from
    1.0 down to ``eta_min / base_lr``.

    The optimizer is expected to use a single learning rate across all param
    groups; the lambda returns a scalar multiplier that torch broadcasts to
    every group.
    """
    import math

    import torch

    warmup_epochs = max(0, int(warmup_epochs))
    if warmup_epochs >= epochs:
        warmup_epochs = max(0, epochs - 1)
    base_lr = float(optimizer.param_groups[0]["lr"])

    def _linear_warmup_then_cosine(current_epoch: int) -> float:
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        # Cosine from the end of warmup to the last epoch.
        progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min / base_lr, cosine) if base_lr > 0 else 0.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _linear_warmup_then_cosine)


def build_plateau_scheduler(
    optimizer: object,
    *,
    factor: float = 0.5,
    patience: int = 8,
) -> object:
    """Validation-plateau LR decay (the alternative frozen schedule).

    Stepped once per epoch with the checkpoint metric
    (``scheduler.step(validation_mean_r2)``); halving the LR after ``patience``
    epochs without improvement. ReduceLROnPlateau has no warmup phase, so the
    ``warmup_epochs`` setting is ignored under this schedule.
    """
    import torch

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=factor, patience=patience
    )


# ----------------------------------------------------- trainable-scope masks

TRAIN_SCOPES = ("full", "last_stage", "head")


def _parameters_of(modules: list) -> list:
    seen: set[int] = set()
    parameters = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def head_modules(model: object) -> list:
    """Return the radiomics-head modules of the frozen ConvNeXt regressors."""
    classifier = getattr(getattr(model, "network", None), "classifier", None)
    if classifier is not None:
        return [classifier]
    raise ValueError(
        "Cannot locate the radiomics head; expected a ThyroidXL ConvNeXt "
        "regressor (``.network.classifier``)"
    )


def last_stage_modules(model: object) -> list:
    """Return the final backbone-stage modules of the frozen ConvNeXt regressors."""
    features = getattr(getattr(model, "network", None), "features", None)
    if features is not None and len(features) > 6:
        return list(features[6:])
    raise ValueError(
        "Cannot locate the last backbone stage; expected a torchvision "
        "ConvNeXt trunk"
    )


def apply_train_scope(model: object, scope: str) -> dict[str, int]:
    """Freeze parameters per ``scope`` and return parameter counts.

    ``full`` leaves the constructor's trainable set untouched; ``head`` keeps
    only the head trainable; ``last_stage`` keeps the head plus the final
    backbone stage. Constructor-frozen parameters stay frozen in every scope.

    Freezing only clears ``requires_grad``. Running statistics of frozen
    normalisation layers still update during ``.train()`` forward passes — a
    deliberate simplification shared identically by every scope arm so the scope
    comparison is not confounded by an extra eval-mode switch.
    """
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"unknown train_scope {scope!r}; options are {list(TRAIN_SCOPES)}")
    head_ids = {id(parameter) for parameter in _parameters_of(head_modules(model))}
    stage_ids = (
        {id(parameter) for parameter in _parameters_of(last_stage_modules(model))}
        if scope == "last_stage"
        else set()
    )
    # The constructor-frozen baseline is snapshotted on the first call:
    # re-applying a different scope later must never mistake scope-frozen
    # parameters for constructor-frozen ones.
    baseline = getattr(model, "_cbm_train_scope_baseline", None)
    if baseline is None:
        baseline = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        setattr(model, "_cbm_train_scope_baseline", baseline)
    for parameter in model.parameters():
        if id(parameter) not in baseline:
            parameter.requires_grad = False
        elif scope == "full":
            parameter.requires_grad = True
        elif scope == "head":
            parameter.requires_grad = id(parameter) in head_ids
        else:
            parameter.requires_grad = id(parameter) in head_ids or id(parameter) in stage_ids
    trainable = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": sum(
            1 for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def parameter_groups(
    model: object,
    *,
    learning_rate: float,
    weight_decay: float,
    head_lr_multiplier: float = 1.0,
) -> list[dict]:
    """Optimizer param groups: one group at ``head_lr_multiplier == 1``.

    With a multiplier the head forms its own group at
    ``learning_rate * head_lr_multiplier`` (group 0 stays the backbone group so
    per-epoch LR logging keeps reporting the base learning rate).
    """
    if head_lr_multiplier <= 0:
        raise ValueError("head_lr_multiplier must be positive")
    head_ids = {id(parameter) for parameter in _parameters_of(head_modules(model))}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone = [parameter for parameter in trainable if id(parameter) not in head_ids]
    head = [parameter for parameter in trainable if id(parameter) in head_ids]
    if head_lr_multiplier == 1.0 or not backbone:
        return [
            {
                "params": backbone + head,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
    return [
        {"params": backbone, "lr": learning_rate, "weight_decay": weight_decay},
        {
            "params": head,
            "lr": learning_rate * head_lr_multiplier,
            "weight_decay": weight_decay,
        },
    ]


# ---------------------------------------------------- per-feature loss table


def load_weight_table(path, feature_names: list[str]) -> np.ndarray:
    """Load and validate a driver-written weight table against the inventory.

    The per-feature weightings (dictionary-category-balanced, reliability-weighted) are
    computed offline by the search driver from train-side artifacts and handed
    to training as a ``feature,weight`` CSV; the trainer only validates
    alignment (names, order, positivity) so no dictionary or reliability
    computation ever runs inside the training process.
    """
    import pandas as pd

    table = pd.read_csv(path)
    if list(table.columns) != ["feature", "weight"]:
        raise ValueError(f"Weight table {path} must have columns feature,weight")
    if table["feature"].astype(str).tolist() != [str(name) for name in feature_names]:
        raise ValueError(f"Weight table {path} is not aligned with the feature inventory")
    weights = table["weight"].to_numpy(dtype=np.float32)
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError(f"Weight table {path} has non-positive or non-finite weights")
    return weights


# --------------------------------------------------------- Stage-1 training

#: Checkpoint metrics. ``validation_mean_r2`` is the default because the
#: aggregated MSE is dominated by a handful of heavy-tailed features and cannot
#: distinguish a model that learned something from one that predicts the mean.
CHECKPOINT_METRICS = ("validation_mean_r2", "validation_mse")

#: Search knobs accepted by :func:`train_radiomics_regressor`. Every default
#: reproduces the pre-search training path bit-for-bit (single param group,
#: SmoothL1, warmup+cosine, no clipping, no accumulation).
LOSS_KINDS = ("smooth_l1", "mse")
SCHEDULER_KINDS = ("warmup_cosine", "plateau")
OPTIMIZER_KINDS = ("adamw", "radam", "nadam")


def fit_radiomics_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only target z-score statistics."""
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean.astype(np.float32), np.where(std < 1e-8, 1.0, std).astype(np.float32)


def train_radiomics_regressor(
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
    checkpoint_metric: str = "validation_mean_r2",
    eta_extra_epochs: int = 0,
    log_prefix: str = "",
    loss: str = "smooth_l1",
    feature_weights: np.ndarray | None = None,
    grad_clip_norm: float | None = None,
    head_lr_multiplier: float = 1.0,
    train_scope: str = "full",
    scheduler_kind: str = "warmup_cosine",
    accumulation_steps: int = 1,
    optimizer: str = "adamw",
    early_stop_patience: int | None = None,
) -> dict[str, Any]:
    """Train with SmoothL1 (default) and retain the best checkpoint under ``checkpoint_metric``.

    Every epoch reports the validation MSE, the MSE of the mean predictor on
    the same split, and the mean/median per-feature R^2, so a run that never
    beats the constant predictor is visible in the log instead of hiding behind
    a slowly falling loss curve.

    The search knobs (loss kind, per-feature weights, gradient clipping,
    head/backbone LR multiplier, trainable scope, plateau schedule, gradient
    accumulation, optimizer family, early stopping) all default to the
    frozen recipe of record: with default values the optimizer, scheduler, criterion,
    and step order are exactly the pre-search construction, so existing
    configurations replay bit-for-bit.

    ``early_stop_patience`` ends training when ``checkpoint_metric`` sees no
    new best for that many epochs. It is a compute saver only — the retained
    checkpoint is still the best epoch, so a stopped run reports the same
    model an uninterrupted run would have kept ("earlier stopping" is never
    reported as a method advantage).
    """
    import torch

    if checkpoint_metric not in CHECKPOINT_METRICS:
        raise ValueError(
            f"unknown checkpoint_metric {checkpoint_metric!r}; options are "
            f"{list(CHECKPOINT_METRICS)}"
        )
    if loss not in LOSS_KINDS:
        raise ValueError(f"unknown loss {loss!r}; options are {list(LOSS_KINDS)}")
    if optimizer not in OPTIMIZER_KINDS:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; options are {list(OPTIMIZER_KINDS)}"
        )
    if early_stop_patience is not None and early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1 when set")
    if scheduler_kind not in SCHEDULER_KINDS:
        raise ValueError(
            f"unknown scheduler_kind {scheduler_kind!r}; options are {list(SCHEDULER_KINDS)}"
        )
    if train_scope not in TRAIN_SCOPES:
        raise ValueError(f"unknown train_scope {train_scope!r}; options are {list(TRAIN_SCOPES)}")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive when set")
    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=np.float32).reshape(-1)
        if (weights <= 0).any() or not np.isfinite(weights).all():
            raise ValueError("feature_weights must be finite and strictly positive")
        weights = weights / weights.mean()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    scope_counts = apply_train_scope(model, train_scope)
    groups = parameter_groups(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        head_lr_multiplier=head_lr_multiplier,
    )
    optimizer_class = {
        "adamw": torch.optim.AdamW,
        "radam": torch.optim.RAdam,
        "nadam": torch.optim.NAdam,
    }[optimizer]
    optimizer_ = optimizer_class(groups, lr=learning_rate, weight_decay=weight_decay)
    if scheduler_kind == "plateau":
        scheduler = build_plateau_scheduler(optimizer_)
    else:
        scheduler = build_warmup_cosine_scheduler(
            optimizer_, epochs=epochs, warmup_epochs=warmup_epochs
        )
    if feature_weights is None:
        criterion = (
            torch.nn.SmoothL1Loss() if loss == "smooth_l1" else torch.nn.MSELoss()
        )

        def compute_loss(prediction: object, target: object) -> object:
            return criterion(prediction, target)

    else:
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)

        def compute_loss(prediction: object, target: object) -> object:
            elementwise = (
                torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
                if loss == "smooth_l1"
                else torch.nn.functional.mse_loss(prediction, target, reduction="none")
            )
            return (elementwise * weight_tensor).mean()

    use_amp = bool(amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    n_batches = len(train_loader)
    best_score = -float("inf")
    best_epoch_seen = -1
    history = []
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        epoch_grad_norms = []
        for batch_index, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if batch_index % accumulation_steps == 0:
                optimizer_.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                prediction = model(image)
                batch_loss = compute_loss(prediction, target)
            scaler.scale(batch_loss / accumulation_steps).backward()
            step_now = (batch_index + 1) % accumulation_steps == 0 or (
                batch_index + 1
            ) == n_batches
            if step_now:
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer_)
                    epoch_grad_norms.append(
                        float(
                            torch.nn.utils.clip_grad_norm_(
                                trainable_parameters, grad_clip_norm
                            )
                        )
                    )
                scaler.step(optimizer_)
                scaler.update()
            train_losses.append(float(batch_loss.detach().cpu()))

        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in validation_loader:
                image = batch["image"].to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=use_amp):
                    prediction = model(image)
                predictions.append(prediction.float().cpu().numpy())
                targets.append(batch["target"].numpy())
        prediction_matrix = np.concatenate(predictions)
        target_matrix = np.concatenate(targets)
        quality = stage1_quality(
            target_matrix,
            prediction_matrix,
            reference_mean=np.zeros(target_matrix.shape[1], dtype=np.float64),
        )
        current_lr = float(optimizer_.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_smooth_l1": float(np.mean(train_losses)),
            "validation_mse": float(quality["mse"]),
            "validation_mean_predictor_mse": float(quality["mean_predictor_mse"]),
            "validation_mean_r2": float(quality["mean_r2"]),
            "validation_median_r2": float(quality["median_r2"]),
            "validation_features_above_r2": int(quality["n_features_above_r2"]),
            "learning_rate": current_lr,
        }
        if len(optimizer_.param_groups) > 1:
            row["head_learning_rate"] = float(optimizer_.param_groups[1]["lr"])
        if epoch_grad_norms:
            row["grad_norm"] = float(np.mean(epoch_grad_norms))
        history.append(row)
        print(
            f"{log_prefix}epoch={epoch + 1}/{epochs} "
            f"train_smooth_l1={row['train_smooth_l1']:.6f} "
            f"validation_mse={row['validation_mse']:.6f} "
            f"mean_predictor_mse={row['validation_mean_predictor_mse']:.6f} "
            f"val_meanR2={row['validation_mean_r2']:.4f} "
            f"val_medianR2={row['validation_median_r2']:.4f} "
            f"n_R2_above={row['validation_features_above_r2']}/{quality['n_features']} "
            f"lr={current_lr:.2e}",
            flush=True,
        )
        if scheduler_kind == "plateau":
            scheduler.step(row["validation_mean_r2"])
        else:
            scheduler.step()
        elapsed = time.monotonic() - started_at
        remaining_epochs = epochs - epoch - 1 + eta_extra_epochs
        update_estimated_process(elapsed / (epoch + 1) * remaining_epochs)

        # Both metrics are compared as "higher is better" after negating MSE.
        score = (
            row["validation_mean_r2"]
            if checkpoint_metric == "validation_mean_r2"
            else -row["validation_mse"]
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch_seen = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_mse": row["validation_mse"],
                    "validation_mean_r2": row["validation_mean_r2"],
                    "checkpoint_metric": checkpoint_metric,
                    "seed": seed,
                },
                destination,
            )
        if (
            early_stop_patience is not None
            and (epoch - best_epoch_seen) >= early_stop_patience
        ):
            print(
                f"{log_prefix}early stopping at epoch {epoch + 1}: "
                f"no checkpoint-metric improvement for {early_stop_patience} epochs",
                flush=True,
            )
            break
    best_row = max(
        history,
        key=lambda item: (
            item["validation_mean_r2"]
            if checkpoint_metric == "validation_mean_r2"
            else -item["validation_mse"]
        ),
    )
    return {
        "checkpoint_metric": checkpoint_metric,
        "best_epoch": int(best_row["epoch"]),
        "best_validation_mse": float(best_row["validation_mse"]),
        "best_validation_mean_r2": float(best_row["validation_mean_r2"]),
        "mean_predictor_mse": float(best_row["validation_mean_predictor_mse"]),
        "beats_mean_predictor": bool(
            best_row["validation_mse"] < best_row["validation_mean_predictor_mse"]
        ),
        "epochs_run": len(history),
        "early_stopped": len(history) < epochs,
        # Key name kept for schema compatibility with frozen run metadata.
        "search_knobs": {
            "optimizer": optimizer,
            "loss": loss,
            "feature_weighted": feature_weights is not None,
            "grad_clip_norm": grad_clip_norm,
            "head_lr_multiplier": head_lr_multiplier,
            "train_scope": train_scope,
            "scheduler_kind": scheduler_kind,
            "accumulation_steps": accumulation_steps,
            "early_stop_patience": early_stop_patience,
        },
        **scope_counts,
        "history": history,
    }


def predict_radiomics(
    model: object,
    loader: object,
    *,
    device: str = "cuda",
    amp: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Predict radiomics from full images; loader batches cannot carry masks."""
    import torch

    model = model.to(device).eval()
    use_amp = bool(amp) and str(device).startswith("cuda")
    sample_ids: list[str] = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            forbidden = INFERENCE_FORBIDDEN_BATCH_KEYS.intersection(batch)
            if forbidden:
                raise ValueError(f"Inference batch contains forbidden fields: {sorted(forbidden)}")
            with torch.autocast("cuda", enabled=use_amp):
                prediction = model(batch["image"].to(device))
            predictions.append(prediction.float().cpu().numpy())
            sample_ids.extend(str(value) for value in batch["sample_id"])
    return sample_ids, np.concatenate(predictions)
