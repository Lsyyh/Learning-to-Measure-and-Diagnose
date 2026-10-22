#!/usr/bin/env python3
"""Per-fold Stage-1 OOF trainer for the from-scratch Ours campaign.

One (cohort, arm, seed, fold) cell of Section 3.1 in
``docs/ours_experiments.md``: train the radiomics regressor from scratch on
the fold's training patients with an 80-epoch warm-up and a separate 20-epoch
regression continuation. Each phase
runs its full budget and selects its checkpoint by the independent validation
split's 102-target mean transformed R². The outer held-out fold is then
predicted once from the selected checkpoint to produce ``oof80.csv`` and
``oof100.csv``; it is never used for checkpoint selection.

The two arms share the frozen Stage-1 recipe of record exactly (ConvNeXt-T
generic ImageNet start, 102 targets, online augmentation, AdamW, warmup+
cosine, AMP, per-fold rank-gauss fitted on the fold's training patients
only) and differ only in the loss:

- ``conventional``: mean SmoothL1 over all 102 targets;
- ``family_balanced``: the equally weighted mean of the three per-family
  SmoothL1 means (geometry=shape2D, intensity=firstorder, texture=the five
  texture classes).

Fold assignments are frozen per cohort before any run. ThyroidXL derives
deterministically from the frozen development split (StratifiedGroupKFold,
patient-grouped and label-stratified, crossfit seed 42) and is written once
to ``oof_assignment.csv`` via ``--freeze-assignment``; every later run
rebuilds it and requires bit-equality with the frozen file. TN3K reads the
frozen ``train_oof_inputs_v1`` assignment unchanged. Training seeds vary
model init and data order only — the fold structure is fixed.

Never reads either cohort's test partition; the development/final validation
splits are used only through the frozen train-partition row selector shared
with ``train_stage1_final.py``.
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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from cbm.data.selection import select_train_validation
from cbm.data.ultrasound_augment import UltrasoundAugment
from cbm.eval.stage1 import build_stage1_oof_assignment, stage1_quality
from cbm.models.zoo import ThyroidXLConvNeXtRadiomicsRegressor, radiomics_feature_family
from cbm.train.stage1 import (
    CUBLAS_WORKSPACE_CONFIG,
    build_warmup_cosine_scheduler,
    determinism_record,
    loader_generator,
    seed_everything,
    seeded_worker_init,
)
from cbm.train.target_transform import fit_target_transform
from cbm.utils import atomic_csv, atomic_json, canonical_config_hash, sha256_file

# Pin the cuBLAS workspace before the first CUDA op so identical seeds give
# bit-identical GEMM results.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

ARMS = ("conventional", "family_balanced")


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


def family_loss(
    prediction: torch.Tensor, target: torch.Tensor, groups: dict[str, list[int]]
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    return torch.stack([elementwise[:, indices].mean() for indices in groups.values()]).mean()


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


def predict(
    model: torch.nn.Module, loader: Any, device: str, amp: bool
) -> tuple[list[str], np.ndarray]:
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
        raise ValueError("Ours OOF radiomics table has duplicate sample IDs")
    for column, expected in (radiomics_settings.get("required_columns") or {}).items():
        if column not in oracle or not oracle[column].eq(str(expected)).all():
            raise ValueError(f"Ours OOF radiomics contains rows outside {column}={expected!r}")
    oracle = oracle.set_index("sample_id")
    missing = sorted(set(selected["sample_id"]) - set(oracle.index))
    if missing:
        raise ValueError(f"Ours OOF radiomics misses {len(missing)} selected rows")
    return oracle.loc[selected["sample_id"], ["patient_id", *features]].reset_index()


def _aligned_feature_values(
    oracle: pd.DataFrame, rows: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """Return feature values in exact Dataset row order."""
    indexed = oracle.set_index("sample_id", verify_integrity=True)
    sample_ids = rows["sample_id"].astype(str).tolist()
    aligned = indexed.loc[sample_ids]
    if aligned.index.astype(str).tolist() != sample_ids:
        raise AssertionError("Ours OOF radiomics order changed")
    return aligned[features].to_numpy(np.float64)


def _load_assignment(config: dict, root: Path, train: pd.DataFrame) -> pd.DataFrame:
    """Load the frozen per-cohort OOF assignment over exactly the train rows."""
    assignment_config = config["oof"]["assignment"]
    assignment_path = ROOT / str(assignment_config["path"])
    source = str(assignment_config["source"])
    if source == "deterministic_stratified_group_kfold_frozen_once":
        manifest = pd.read_csv(root / config["data"]["manifest"], dtype={"patient_id": str})
        rebuilt = build_stage1_oof_assignment(
            manifest,
            n_splits=int(assignment_config["folds"]),
            seed=int(assignment_config["seed"]),
        )
        if not assignment_path.is_file():
            raise FileNotFoundError(
                f"ThyroidXL OOF assignment is not frozen yet: {assignment_path} "
                "(run with --freeze-assignment once before any training run)"
            )
        frozen = pd.read_csv(assignment_path, dtype={"patient_id": str})
        if not frozen.equals(rebuilt):
            raise ValueError("Frozen ThyroidXL OOF assignment disagrees with the deterministic rebuild")
        assignment = frozen
    elif source == "frozen_train_oof_inputs_v1":
        assignment = pd.read_csv(assignment_path, dtype={"patient_id": str})
    else:
        raise ValueError(f"unknown OOF assignment source {source!r}")
    if assignment["sample_id"].duplicated().any():
        raise ValueError("Ours OOF assignment has duplicate sample IDs")
    if set(assignment["sample_id"]) != set(train["sample_id"]):
        raise ValueError("Ours OOF assignment does not cover exactly the frozen train partition")
    if assignment.groupby("sample_id")["oof_fold"].nunique().max() != 1:
        raise ValueError("Ours OOF assignment gives a sample multiple folds")
    if set(assignment["oof_fold"]) != set(range(int(assignment_config["folds"]))):
        raise ValueError("Ours OOF assignment does not contain exactly the configured folds")
    indexed = train.set_index("sample_id")
    aligned = indexed.loc[assignment["sample_id"]].reset_index()
    if aligned["patient_id"].astype(str).tolist() != assignment["patient_id"].astype(str).tolist():
        raise ValueError("Ours OOF assignment patient IDs disagree with the train partition")
    if aligned["label"].astype(int).tolist() != assignment["label"].astype(int).tolist():
        raise ValueError("Ours OOF assignment labels disagree with the train partition")
    return assignment


def _freeze_assignment(config: dict, root: Path, train: pd.DataFrame) -> None:
    """Write the ThyroidXL OOF assignment once; refuse to overwrite."""
    assignment_config = config["oof"]["assignment"]
    if str(assignment_config["source"]) != "deterministic_stratified_group_kfold_frozen_once":
        raise ValueError("--freeze-assignment applies to the ThyroidXL deterministic source only")
    assignment_path = ROOT / str(assignment_config["path"])
    if assignment_path.exists():
        raise FileExistsError(f"OOF assignment is already frozen: {assignment_path}")
    manifest = pd.read_csv(root / config["data"]["manifest"], dtype={"patient_id": str})
    assignment = build_stage1_oof_assignment(
        manifest,
        n_splits=int(assignment_config["folds"]),
        seed=int(assignment_config["seed"]),
    )
    if set(assignment["sample_id"]) != set(train["sample_id"]):
        raise ValueError("deterministic OOF assignment does not cover exactly the train partition")
    atomic_csv(assignment_path, assignment)
    print(json.dumps({
        "status": "assignment_frozen",
        "path": str(assignment_path),
        "sha256": sha256_file(assignment_path),
        "frames": len(assignment),
        "patients": int(assignment["patient_id"].nunique()),
        "fold_frames": {
            str(fold): int((assignment["oof_fold"] == fold).sum())
            for fold in sorted(assignment["oof_fold"].unique())
        },
    }, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default=None, help="override config training.device")
    parser.add_argument("--execute", action="store_true", help="authorize this training run")
    parser.add_argument(
        "--freeze-assignment",
        action="store_true",
        help="write the ThyroidXL oof_assignment.csv once; incompatible with training",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help="smoke override only: total epochs, snapshots at N-1 and N, outputs under smoke/",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = str(config["dataset"])
    root = ROOT / str(config["data"]["root"])
    data, settings = config["data"], config["training"]
    train, validation = select_train_validation(dataset, root, data)

    if args.freeze_assignment:
        if args.execute or args.smoke_epochs:
            raise ValueError("--freeze-assignment runs standalone")
        _freeze_assignment(config, root, train)
        return

    assignment = _load_assignment(config, root, train)
    arm, seed, fold = args.arm, int(args.seed), int(args.fold)
    snapshots = [int(epoch) for epoch in config["oof"]["snapshot_epochs"]]
    epochs_total = int(config["oof"]["epochs_total"])
    smoke = args.smoke_epochs is not None
    if smoke:
        if args.smoke_epochs < 2:
            raise ValueError("--smoke-epochs requires at least 2 epochs")
        epochs_total = int(args.smoke_epochs)
        snapshots = [epochs_total - 1, epochs_total]
    if fold not in set(assignment["oof_fold"]):
        raise ValueError(f"fold {fold} is not one of the frozen OOF folds")
    if seed not in [int(candidate) for candidate in config["oof"]["seeds"]]:
        raise ValueError(f"seed {seed} is not one of the frozen campaign seeds")
    device = str(args.device or settings["device"])

    output_root = ROOT / str(config["oof"]["output_root"])
    fold_output = (
        output_root / "smoke" / f"{arm}_seed{seed}_fold{fold}"
        if smoke
        else output_root / arm / f"seed{seed}" / f"fold{fold}"
    )
    assignment_path = ROOT / str(config["oof"]["assignment"]["path"])

    selection_assignment = assignment.loc[assignment["oof_fold"].eq(fold)].copy()
    training_assignment = assignment.loc[~assignment["oof_fold"].eq(fold)].copy()
    if set(training_assignment["patient_id"]) & set(selection_assignment["patient_id"]):
        raise AssertionError("Ours OOF fold crosses patients")
    indexed_inputs = train.set_index("sample_id")
    training = indexed_inputs.loc[training_assignment["sample_id"]].reset_index()
    selection = indexed_inputs.loc[selection_assignment["sample_id"]].reset_index()

    features = pd.read_csv(root / data["inventory"])["pyradiomics_name"].astype(str).tolist()
    groups = family_groups(features, config)
    oracle = _select_radiomics(root, data, pd.concat([train, validation]), features)
    training_values = _aligned_feature_values(oracle, training, features)
    selection_values = _aligned_feature_values(oracle, selection, features)
    validation_values = _aligned_feature_values(oracle, validation, features)

    resolved_config = json.loads(json.dumps(config))
    resolved_config["training"]["device"] = device
    resolved_config["oof"]["arm"] = arm
    resolved_config["oof"]["seed"] = seed
    resolved_config["oof"]["fold"] = fold
    resolved_config["oof"]["epochs_total"] = epochs_total
    resolved_config["oof"]["snapshot_epochs"] = snapshots
    resolved_config["oof"]["smoke"] = smoke
    config_sha256 = canonical_config_hash(resolved_config)

    plan = {
        "status": "ready_requires_explicit_execute",
        "protocol": config["protocol"],
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "epochs_total": epochs_total,
        "snapshot_epochs": snapshots,
        "training_frames": len(training),
        "training_patients": int(training["patient_id"].nunique()),
        "selection_frames": len(selection),
        "selection_patients": int(selection["patient_id"].nunique()),
        "validation_frames": len(validation),
        "validation_patients": int(validation["patient_id"].nunique()),
        "patient_overlap": 0,
        "output": str(fold_output),
        "assignment": str(assignment_path),
        "config_sha256": config_sha256,
        "official_test_used": False,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return

    from setproctitle import setproctitle

    setproctitle(f"cbm_ours_oof_{dataset}_{arm}_s{seed}_f{fold}")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"Ours OOF execution requires a CUDA device (requested {device!r})")
    metadata_path = fold_output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") in {"complete", "smoke_complete"}
            and metadata.get("config_sha256") == config_sha256
        ):
            print(json.dumps({"status": "already_complete", "output": str(fold_output)}, indent=2))
            return
        raise ValueError("Ours OOF fold metadata present with a different configuration")
    if fold_output.exists():
        raise FileExistsError(f"Incomplete Ours OOF fold cannot be resumed safely: {fold_output}")
    fold_output.mkdir(parents=True)
    atomic_json(
        fold_output / "config.json",
        {
            "config": resolved_config,
            "config_sha256": config_sha256,
            "requested_config_sha256": sha256_file(args.config),
        },
    )

    transform = fit_target_transform(training_values, str(config["target_transform"]["method"]))
    transform.save(fold_output / "target_transform.joblib")
    training_y = transform.transform(training_values).astype(np.float32)
    selection_y = transform.transform(selection_values).astype(np.float32)
    validation_y = transform.transform(validation_values).astype(np.float32)

    seed_everything(seed)
    model = ThyroidXLConvNeXtRadiomicsRegressor(
        n_features=len(features),
        dropout=float(config["model"]["dropout"]),
        pretrained=bool(config["model"]["pretrained"]),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    warmup_epochs = snapshots[0] if smoke else int(config["oof"]["warmup_epochs"])
    continuation_epochs = epochs_total - warmup_epochs
    if snapshots != [warmup_epochs, epochs_total] or continuation_epochs <= 0:
        raise ValueError("OOF snapshots must delimit a positive warm-up and continuation")
    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs=warmup_epochs, warmup_epochs=int(settings["warmup_epochs"])
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
    train_loader = torch.utils.data.DataLoader(
        TrainDataset(training.reset_index(drop=True), training_y, augment),
        shuffle=True,
        generator=loader_generator(seed + fold),
        **common,
    )
    selection_loader = torch.utils.data.DataLoader(
        EvalDataset(selection.reset_index(drop=True), selection_y), shuffle=False, **common
    )
    validation_loader = torch.utils.data.DataLoader(
        EvalDataset(validation.reset_index(drop=True), validation_y), shuffle=False, **common
    )

    def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if arm == "conventional":
            return F.smooth_l1_loss(prediction, target)
        return family_loss(prediction, target, groups)

    history: list[dict[str, Any]] = []
    snapshots_record: list[dict[str, Any]] = []
    started = time.time()

    def run_phase(
        *, phase: str, phase_epochs: int, epoch_offset: int, endpoint_epoch: int
    ) -> dict[str, Any]:
        nonlocal scheduler
        best_score = -float("inf")
        best_phase_epoch = -1
        checkpoint_path = fold_output / f"checkpoint_oof{endpoint_epoch}.pt"
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch = epoch_offset + phase_epoch
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

            validation_ids, validation_prediction = predict(
                model, validation_loader, device, amp
            )
            if validation_ids != validation["sample_id"].astype(str).tolist():
                raise AssertionError("Ours OOF validation prediction order changed")
            validation_quality = stage1_quality(
                validation_y,
                validation_prediction,
                reference_mean=np.zeros(validation_y.shape[1]),
            )
            score = float(validation_quality["mean_r2"])
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mean_r2": score,
                "validation_median_r2": float(validation_quality["median_r2"]),
                "validation_mse": float(validation_quality["mse"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            marker = ""
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_phase_epoch = phase_epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "fold": fold,
                        "phase": phase,
                        "phase_epoch": phase_epoch,
                        "epoch": global_epoch,
                        "validation_mean_r2": score,
                    },
                    checkpoint_path,
                )
                marker = " *"
            history.append(row)
            atomic_csv(fold_output / "history.csv", pd.DataFrame(history))
            print(
                f"ours-oof {dataset} {arm} seed{seed} fold{fold} "
                f"{phase}={phase_epoch}/{phase_epochs} loss={row['train_loss']:.6f} "
                f"val_meanR2={score:.4f}{marker}",
                flush=True,
            )
            scheduler.step()

        if best_phase_epoch < 0:
            raise RuntimeError(f"{phase} produced no finite validation R2 checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        ids, transformed = predict(model, selection_loader, device, amp)
        if ids != selection["sample_id"].astype(str).tolist():
            raise AssertionError("Ours OOF held-out prediction order changed")
        raw = transform.inverse_transform(transformed)
        frame = selection_assignment[["sample_id", "patient_id", "label", "oof_fold"]].copy()
        frame = pd.concat(
            [frame.reset_index(drop=True), pd.DataFrame(raw, columns=features)], axis=1
        )
        prediction_path = fold_output / f"oof{endpoint_epoch}.csv"
        atomic_csv(prediction_path, frame)
        heldout_quality = stage1_quality(
            selection_y, transformed, reference_mean=np.zeros(selection_y.shape[1])
        )
        record = {
            "endpoint": f"oof{endpoint_epoch}",
            "phase": phase,
            "phase_budget": phase_epochs,
            "selected_phase_epoch": best_phase_epoch,
            "selected_global_epoch": int(checkpoint["epoch"]),
            "selection_metric": "validation_mean_r2_over_102",
            "best_validation_mean_r2": best_score,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions": prediction_path.name,
            "predictions_sha256": sha256_file(prediction_path),
            "heldout_used_for_selection": False,
            "heldout_mean_transformed_r2_diagnostic": float(heldout_quality["mean_r2"]),
            "heldout_median_transformed_r2_diagnostic": float(heldout_quality["median_r2"]),
        }
        snapshots_record.append(record)
        return record

    warmup_record = run_phase(
        phase="warmup", phase_epochs=warmup_epochs, epoch_offset=0, endpoint_epoch=warmup_epochs
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs=continuation_epochs, warmup_epochs=0
    )
    continuation_record = run_phase(
        phase="continuation",
        phase_epochs=continuation_epochs,
        epoch_offset=warmup_epochs,
        endpoint_epoch=epochs_total,
    )

    metadata = {
        **plan,
        "status": "smoke_complete" if smoke else "complete",
        "loss": "family-equal-weighted group means" if arm == "family_balanced" else "mean over 102",
        "groups": {group: len(indices) for group, indices in groups.items()},
        "assignment": {
            "path": str(assignment_path),
            "sha256": sha256_file(assignment_path),
            "source": str(config["oof"]["assignment"]["source"]),
        },
        "target_transform": {
            "method": str(config["target_transform"]["method"]),
            "fit": "fold training patients only (all 102)",
            "path": "target_transform.joblib",
        },
        "endpoints": snapshots_record,
        "checkpoint_selection": {
            "metric": "validation_mean_r2_over_102",
            "early_stopping": False,
            "outer_heldout_role": "prediction_only",
            "warmup_selected_epoch": warmup_record["selected_phase_epoch"],
            "continuation_selected_epoch": continuation_record["selected_phase_epoch"],
        },
        "scheduler": {
            "warmup_phase": f"warmup({settings['warmup_epochs']})+cosine over {warmup_epochs} epochs",
            "continuation_phase": f"fresh cosine over {continuation_epochs} epochs",
            "optimizer_state_reset": False,
            "amp_scaler_state_reset": False,
        },
        "determinism": {
            **determinism_record(seed),
            "loader_generator_seed": seed + fold,
        },
        "requested_config_sha256": sha256_file(args.config),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
