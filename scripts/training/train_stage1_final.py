#!/usr/bin/env python3
"""Final-fit Stage-1 trainer for the unified paper protocol.

One dataset/arm/seed per invocation. The model trains on the frozen final
train split with the validation split used only for checkpoint monitoring
(best epoch by the 102-target mean transformed R²); neither cohort's test
partition is selected or read. Arms share the formal recipe exactly —
ConvNeXt-T generic ImageNet start, 102 targets, online augmentation, AdamW,
warmup+cosine, AMP, per-split rank-gauss refit on train rows only — and
differ only in the loss:

- ``J``: mean SmoothL1 over all 102 targets (conventional regression);
- ``W``: the equally weighted mean of the three per-family SmoothL1 means
  (geometry=shape2D, intensity=firstorder, texture=the five texture
  classes).

The two datasets select the same frozen partitions as every other R-series
step: ThyroidXL uses the development train/validation frames (patient-safe,
official test excluded); TN3K uses the source-respecting final split
(``final_partition`` train/validation from source trainval, source test
excluded).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

from cbm.data.selection import select_train_validation
from cbm.data.ultrasound_augment import UltrasoundAugment
from cbm.eval.stage1 import stage1_quality, training_patient_sd
from cbm.models.zoo import ThyroidXLConvNeXtRadiomicsRegressor, radiomics_feature_family
from cbm.train.stage1 import (
    CUBLAS_WORKSPACE_CONFIG,
    build_warmup_cosine_scheduler,
    loader_generator,
    seed_everything,
    seeded_worker_init,
)
from cbm.train.target_transform import fit_target_transform
from cbm.utils import atomic_csv, atomic_json, canonical_config_hash, sha256_file

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

DEFAULT_CONFIG = ROOT / "configs/thyroidxl/stage1_final.yaml"
ARMS = ("J", "W")
IMAGENET_FILE = "convnext_tiny-983f1562.pth"


def family_groups(feature_names: list[str], config: dict) -> dict[str, list[int]]:
    """Computation-defined family groups; full coverage, no overlap."""
    families_per_group = {
        group: set(families) for group, families in config["model"]["groups"].items()
    }
    groups: dict[str, list[int]] = {group: [] for group in families_per_group}
    for index, name in enumerate(feature_names):
        family = radiomics_feature_family(name)
        members = [group for group, families in families_per_group.items() if family in families]
        if len(members) != 1:
            raise ValueError(f"feature {name!r} (family {family!r}) matches {members}")
        groups[members[0]].append(index)
    sizes = {group: len(indices) for group, indices in groups.items()}
    if sum(sizes.values()) != len(feature_names):
        raise ValueError(f"group sizes {sizes} do not cover {len(feature_names)} features")
    return groups


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        targets: np.ndarray,
        augment: UltrasoundAugment,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.targets = np.asarray(targets, np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image = torch.from_numpy(
            np.load(row.network_input_path, allow_pickle=False).astype(np.float32)
        )
        return {
            "sample_id": str(row.sample_id),
            "image": self.augment(image.clone()),
            "target": torch.from_numpy(self.targets[index]),
        }


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, targets: np.ndarray) -> None:
        self.rows = rows.reset_index(drop=True)
        self.targets = np.asarray(targets, np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image = np.load(row.network_input_path, allow_pickle=False).astype(np.float32)
        return {
            "sample_id": str(row.sample_id),
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(self.targets[index]),
        }


def family_loss(
    prediction: torch.Tensor, target: torch.Tensor, groups: dict[str, list[int]]
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    return torch.stack([elementwise[:, indices].mean() for indices in groups.values()]).mean()


def predict(model: torch.nn.Module, loader: Any, device: str, amp: bool) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", enabled=amp):
                prediction = model(batch["image"].to(device))
            ids.extend(map(str, batch["sample_id"]))
            predictions.append(prediction.float().cpu().numpy())
    return ids, np.concatenate(predictions)


def _select_radiomics(
    root: Path, data: dict, selected: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    radiomics_settings = data["radiomics"]
    oracle = pd.read_csv(root / radiomics_settings["features"], dtype={"patient_id": str})
    if oracle["sample_id"].duplicated().any():
        raise ValueError("Stage-1 final radiomics table has duplicate sample IDs")
    for column, expected in (radiomics_settings.get("required_columns") or {}).items():
        if column not in oracle or not oracle[column].eq(str(expected)).all():
            raise ValueError(f"Stage-1 radiomics contains rows outside {column}={expected!r}")
    split_column = radiomics_settings.get("split_column")
    if split_column is not None:
        oracle = oracle.loc[oracle[split_column].eq(str(radiomics_settings["split"]))].copy()
    oracle = oracle.set_index("sample_id")
    missing = sorted(set(selected["sample_id"]) - set(oracle.index))
    if missing:
        raise ValueError(f"Stage-1 radiomics misses {len(missing)} selected rows")
    return oracle.loc[selected["sample_id"], ["patient_id", *features]].reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="smoke override only")
    parser.add_argument("--device", default=None, help="smoke override only")
    args = parser.parse_args(argv)
    from setproctitle import setproctitle

    config = yaml.safe_load(args.config.read_text())
    dataset = str(config["dataset"])
    setproctitle(f"cbm_r2_{dataset}_{args.arm.lower()}_seed{args.seed}")
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    data, settings, model_settings = config["data"], config["training"], config["model"]
    if bool(settings.get("oof", False)):
        raise ValueError("train_stage1_final.py is the final-fit trainer; OOF runs use train_ours_oof.py")

    arm = args.arm
    seed = int(args.seed)
    epochs = int(args.epochs or settings["epochs"])
    device = str(args.device or settings["device"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    resolved_config = json.loads(json.dumps(config))
    resolved_config["training"]["seed"] = seed
    resolved_config["training"]["device"] = device
    resolved_config["training"]["arm"] = arm
    atomic_json(
        output / "config.json",
        {
            "config": resolved_config,
            "config_sha256": canonical_config_hash(resolved_config),
            "requested_config_sha256": sha256_file(args.config),
        },
    )

    root = resolve(data["root"])
    train, validation = select_train_validation(dataset, root, data)

    features = pd.read_csv(root / data["inventory"]).pyradiomics_name.astype(str).tolist()
    groups = family_groups(features, config)
    oracle = _select_radiomics(root, data, pd.concat([train, validation]), features)
    train_raw = oracle.loc[oracle["sample_id"].isin(set(train["sample_id"])), features].to_numpy(
        np.float64
    )
    val_raw = oracle.loc[oracle["sample_id"].isin(set(validation["sample_id"])), features].to_numpy(
        np.float64
    )
    transform = fit_target_transform(train_raw, str(config["target_transform"]["method"]))
    joblib.dump(transform, output / "target_transform.joblib")
    train_y = transform.transform(train_raw).astype(np.float32)
    val_y = transform.transform(val_raw).astype(np.float32)

    seed_everything(seed)
    model = ThyroidXLConvNeXtRadiomicsRegressor(
        n_features=len(features),
        dropout=float(model_settings["dropout"]),
        pretrained=bool(model_settings["pretrained"]),
    )
    model.to(device)
    imagenet_weights = Path(torch.hub.get_dir(), "checkpoints", IMAGENET_FILE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs=epochs, warmup_epochs=int(settings["warmup_epochs"])
    )
    amp = bool(settings["amp"]) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    augment = UltrasoundAugment(**{
        key: (tuple(value) if isinstance(value, list) else value)
        for key, value in config["augmentation"].items()
    })
    common = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": int(settings["workers"]),
        "pin_memory": True,
        "persistent_workers": int(settings["workers"]) > 0,
        "worker_init_fn": seeded_worker_init,
    }
    train_rows = train.reset_index(drop=True)
    train_dataset = TrainDataset(train_rows, train_y, augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, generator=loader_generator(seed), **common
    )
    val_loader = torch.utils.data.DataLoader(
        EvalDataset(validation.reset_index(drop=True), val_y), shuffle=False, **common
    )

    def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if arm == "J":
            return F.smooth_l1_loss(prediction, target)
        return family_loss(prediction, target, groups)

    best = {"score": -float("inf"), "epoch": -1}
    history, started = [], time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                prediction = model(batch["image"].to(device))
                loss = compute_loss(prediction, batch["target"].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        _, val_pred = predict(model, val_loader, device, amp)
        quality = stage1_quality(
            val_y, val_pred, reference_mean=np.zeros(val_y.shape[1], dtype=np.float64)
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_mean_r2": float(quality["mean_r2"]),
            "validation_median_r2": float(quality["median_r2"]),
            "validation_mse": float(quality["mse"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        atomic_csv(output / "history.csv", pd.DataFrame(history))
        marker = ""
        if row["validation_mean_r2"] > best["score"]:
            best = {"score": row["validation_mean_r2"], "epoch": epoch + 1}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arm": arm,
                    "epoch": epoch + 1,
                    "validation_mean_r2": row["validation_mean_r2"],
                    "seed": seed,
                },
                output / "best_checkpoint.pt",
            )
            marker = " *"
        print(
            f"stage1-final {dataset} {arm} seed{seed} epoch={epoch + 1}/{epochs} "
            f"loss={row['train_loss']:.6f} "
            f"val_meanR2={row['validation_mean_r2']:.4f}{marker}",
            flush=True,
        )
        scheduler.step()

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    ids, transformed = predict(model, val_loader, device, amp)
    if ids != validation["sample_id"].astype(str).tolist():
        raise AssertionError("Stage-1 final validation prediction order changed")
    raw = pd.DataFrame(transform.inverse_transform(transformed), columns=features)
    raw.insert(0, "label", validation.label.to_numpy())
    raw.insert(0, "patient_id", validation.patient_id.astype(str).to_numpy())
    raw.insert(0, "sample_id", ids)
    atomic_csv(output / "validation_predictions_raw.csv", raw)

    training_sd = training_patient_sd(oracle, train, features)
    truth = pd.DataFrame(val_raw, columns=features)
    truth.insert(0, "patient_id", validation["patient_id"].astype(str).to_numpy())
    pred_patient = raw.groupby("patient_id", sort=True)[features].mean()
    true_patient = truth.groupby("patient_id", sort=True)[features].mean()
    metric_rows = []
    for index, name in enumerate(features):
        y = true_patient[name].to_numpy()
        p = pred_patient[name].to_numpy()
        mae = float(np.mean(np.abs(y - p)))
        metric_rows.append(
            {
                "feature": name,
                "patient_rho": float(spearmanr(y, p).statistic),
                "patient_mae": mae,
                "training_patient_sd": float(training_sd[name]),
                "patient_mae_over_training_sd": mae / float(training_sd[name]),
            }
        )
    atomic_csv(output / "validation_feature_metrics.csv", pd.DataFrame(metric_rows))

    metadata = {
        "status": "complete",
        "protocol": str(config.get("protocol", "r2_stage1_final")),
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "groups": {g: len(i) for g, i in groups.items()},
        "loss": "family-equal-weighted group means" if arm == "W" else "mean over 102",
        "init": {
            "source": "torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1 (generic pretraining)",
            "imagenet_weights_sha256": sha256_file(imagenet_weights)
            if imagenet_weights.is_file()
            else "cache-file-not-found",
            "project_checkpoint_used": False,
        },
        "target_transform": {
            "method": str(config["target_transform"]["method"]),
            "fit": "final train-split oracle labels only (all 102)",
        },
        "augmentation": config["augmentation"],
        "checkpoint_rule": {
            "metric": str(settings["checkpoint_metric"]),
            "rule": "best epoch by validation mean transformed R2 over all 102 targets",
            "best_epoch": best["epoch"],
            "best_validation_mean_r2": best["score"],
            "monitor_slice": str(settings.get("monitor_slice", "final validation split")),
        },
        "train_frames": len(train),
        "train_patients": int(train["patient_id"].nunique()),
        "validation_frames": len(validation),
        "validation_patients": int(validation["patient_id"].nunique()),
        "test_split_selected": False,
        "official_test_used": False,
        "config_sha256": canonical_config_hash(resolved_config),
        "requested_config_sha256": sha256_file(args.config),
        "trainer_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
