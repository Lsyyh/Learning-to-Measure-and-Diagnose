"""Train--inference-consistent Stage-2 anchored joint refinement.

The Stage-2 method of record (PI-confirmed 2026-09-21): after the
family-balanced Stage-1 warm-up (arm ``W``, 80 epochs), the backbone is
refined for 20 anchored joint epochs in which the classification gradient
travels along the exact inference chain -- torch-mirrored inverse transform,
patient mean in raw space, dictionary-constrained linear head -- so the
trained path and the deployed path are the same computation:

    L = L_family + lambda_cls * BCE + lambda_a * L_anchor,  (0.5, 2) of record.

The head is a raw-space affine ``sigmoid((m - mu) / sigma @ w + b)`` with
``mu``/``sigma`` frozen from the train-only OOF standardizer of the frozen
Stage-2 endpoint and ``w``/``b`` initialized from its balanced logistic
(C=1, fit seed 42) and refined jointly. Every loss term is averaged within
each patient before the batch mean (whole-patient bags; singleton bags on
image-level cohorts degenerate to image-level exactly). Checkpoint selection
uses the Stage-1 regression monitor only (validation mean transformed R^2
over the 102 coordinates); no classification-based selection, no test
access. The frozen v1 endpoint pipeline is retained inside the run as a
diagnostic control, never as the record.

Anchoring: per-feature SmoothL1 against the frozen warm start, weighted
``(1 + rho_j) * mu_j`` with ``mu_j = 2`` on the dictionary 36 and ``mu_j =
1`` elsewhere, family-renormalized to mean one so the family-equal
aggregation structure of the Stage-1 loss is preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

from cbm.data.selection import select_train_validation
from cbm.data.ultrasound_augment import UltrasoundAugment
from cbm.eval.concept_groups import select_dictionary_members
from cbm.eval.metrics import extended_binary_metrics
from cbm.eval.panel import (
    frozen_panel_endpoint_pipeline,
    relationship_hypergraph,
    reliability_scores,
)
from cbm.eval.stage1 import (
    PREDICTION_PREFIX,
    stage1_quality,
    training_patient_sd,
)
from cbm.models.zoo import ThyroidXLConvNeXtRadiomicsRegressor, radiomics_feature_family
from cbm.train.stage1 import (
    build_warmup_cosine_scheduler,
    seed_everything,
    seeded_worker_init,
)
from cbm.utils import atomic_csv, atomic_json, canonical_config_hash, sha256_file

EXPECTED_FEATURES = 102
INVERSE_MIRROR_TOLERANCE = 1e-4


@dataclass(frozen=True)
class MemberCell:
    """One Stage-2 selection cell (cohort x seed) under the v1 member rule."""

    student: pd.DataFrame
    features: list[str]
    reliability: pd.Series
    dictionary_features: frozenset[str]
    members: list[str]


def load_protocol(config: dict, root: Path) -> dict[str, Any]:
    """The frozen Stage-1 final-fit data protocol (train/validation only)."""

    data = config["data"]
    config_root = root / data["root"]
    train, validation = select_train_validation(str(config["dataset"]), config_root, data)
    features = (
        pd.read_csv(config_root / data["inventory"]).pyradiomics_name.astype(str).tolist()
    )
    radiomics_settings = data["radiomics"]
    oracle = pd.read_csv(
        config_root / radiomics_settings["features"], dtype={"patient_id": str}
    )
    if oracle["sample_id"].duplicated().any():
        raise ValueError(f"{config['dataset']}: oracle radiomics has duplicate sample IDs")
    for column, expected in (radiomics_settings.get("required_columns") or {}).items():
        if column not in oracle or not oracle[column].eq(str(expected)).all():
            raise ValueError(f"{config['dataset']}: oracle rows outside {column}={expected!r}")
    selected = pd.concat([train, validation])
    oracle = oracle.set_index("sample_id")
    missing = sorted(set(selected["sample_id"]) - set(oracle.index))
    if missing:
        raise ValueError(f"{config['dataset']}: oracle misses {len(missing)} selected frames")
    oracle = oracle.loc[selected["sample_id"], ["patient_id", *features]].reset_index()
    families_per_group = {
        group: set(families) for group, families in config["model"]["groups"].items()
    }
    groups: dict[str, list[int]] = {group: [] for group in families_per_group}
    for index, name in enumerate(features):
        family = radiomics_feature_family(name)
        members = [g for g, fs in families_per_group.items() if family in fs]
        if len(members) != 1:
            raise ValueError(f"feature {name!r} matches groups {members}")
        groups[members[0]].append(index)
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "features": features,
        "groups": groups,
        "oracle": oracle.reset_index(drop=True),
    }


def load_member_cell(config: dict, root: Path, seed: int) -> MemberCell:
    """The unchanged v1 label-free member rule on the frozen OOF tables.

    Reliability is patient-level Spearman between the seed's train-only OOF
    Stage-1 predictions and the training-side oracle radiomics; members are
    the dictionary support filtered by ``min_reliability = 0.3`` and student
    redundancy ``0.9``. The student table carries the train labels; they are
    consumed only by the endpoint logistic fit, never by selection.
    """

    stage2 = config["stage2"]
    dataset = str(config["dataset"])
    student_path = root / str(stage2["student_table"]).format(seed=seed)
    student = pd.read_csv(student_path, dtype={"patient_id": str})
    predictions = [c for c in student.columns if c.startswith(PREDICTION_PREFIX)]
    student = student.rename(
        columns={c: c.removeprefix(PREDICTION_PREFIX) for c in predictions}
    ).sort_values("patient_id").reset_index(drop=True)
    if student["patient_id"].duplicated().any():
        raise ValueError(f"{student_path} is not patient-level")

    oracle = pd.read_csv(root / str(stage2["oracle_table"]), dtype={"patient_id": str})
    split_filter = stage2.get("oracle_split")
    if split_filter:
        oracle = oracle.loc[oracle[split_filter["column"]].eq(str(split_filter["value"]))].copy()
    features = sorted(
        set(c for c in student.columns if c.startswith("original_"))
        & set(c for c in oracle.columns if c.startswith("original_"))
    )
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"{dataset}: expected 102 common features, got {len(features)}")
    pooled = oracle.groupby("patient_id", sort=True)[features].mean().reset_index()
    pooled = pooled.set_index("patient_id").reindex(student["patient_id"]).reset_index()
    if pooled["patient_id"].tolist() != student["patient_id"].tolist():
        raise ValueError(f"{dataset}/seed{seed}: oracle does not cover the student patients")
    pooled["label"] = student["label"].to_numpy()
    if not np.isfinite(pooled[features].to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset}: oracle patient table has non-finite entries")

    audit = pd.read_csv(root / str(stage2["dictionary_audit"]))
    support, categories, _ = relationship_hypergraph(audit, features)
    reliability, _ = reliability_scores(student, pooled, features)
    correlation = student[features].corr(method="spearman").abs()
    members, _ = select_dictionary_members(
        support, categories, features, reliability, correlation
    )
    dictionary_features = {
        features[index]
        for index in np.flatnonzero(np.asarray(support).sum(axis=0) > 0)
    }
    return MemberCell(
        student=student,
        features=features,
        reliability=reliability,
        dictionary_features=frozenset(dictionary_features),
        members=list(members),
    )


# ------------------------------------------------------- batching primitives


class FrameDataset(torch.utils.data.Dataset):
    """Frame dataset with the frozen augmentation policy and frame labels."""

    def __init__(
        self,
        rows: pd.DataFrame,
        targets: np.ndarray,
        augment: UltrasoundAugment | None,
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
        if self.augment is not None:
            image = self.augment(image.clone())
        return {
            "sample_id": str(row.sample_id),
            "patient_id": str(row.patient_id),
            "image": image,
            "target": torch.from_numpy(self.targets[index]),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
        }


class PatientBagBatchSampler(torch.utils.data.Sampler):
    """Whole-patient bags packed greedily to a frame budget.

    Yields index lists in which each patient's frames form one contiguous
    block (patients in permuted order), so per-patient means are computable
    with deterministic cumsum ops in the train loop. Bags are reshuffled
    each epoch by the seeded generator; a patient's frames never split
    across batches.
    """

    def __init__(self, patient_ids, frame_budget: int, seed: int) -> None:
        frame_patients = pd.Series([str(value) for value in patient_ids])
        self._bags = [
            np.flatnonzero(frame_patients.eq(patient).to_numpy())
            for patient in frame_patients.unique()
        ]
        self._budget = int(frame_budget)
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        batch: list[int] = []
        count = 0
        for position in self._rng.permutation(len(self._bags)):
            bag = self._bags[position]
            if batch and count + len(bag) > self._budget:
                yield batch
                batch, count = [], 0
            batch.extend(int(index) for index in bag)
            count += len(bag)
        if batch:
            yield batch

    def __len__(self) -> int:
        return len(self._bags)


def patient_block_lengths(patient_ids: list[str]) -> np.ndarray:
    """Run lengths of contiguous patient blocks; validates one block each."""

    blocks: list[str] = []
    lengths: list[int] = []
    previous = None
    for value in patient_ids:
        if value != previous:
            blocks.append(value)
            lengths.append(1)
            previous = value
        else:
            lengths[-1] += 1
    if len(set(blocks)) != len(blocks):
        raise ValueError("patient frames split across blocks; bag sampler broken")
    return np.asarray(lengths, dtype=np.int64)


def block_means(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean along dim 0 within contiguous blocks (deterministic cumsum ops)."""

    squeeze = values.dim() == 1
    if squeeze:
        values = values.unsqueeze(1)
    cumsum = torch.cumsum(values, dim=0)
    ends = torch.cumsum(lengths, dim=0) - 1
    totals = cumsum[ends]
    starts = torch.cat([torch.zeros_like(totals[:1]), totals[:-1]], dim=0)
    means = (totals - starts) / lengths.unsqueeze(1).to(values.dtype)
    return means.squeeze(1) if squeeze else means


def patient_pos_weight(train_rows: pd.DataFrame) -> float:
    """Class-balance weight at the patient level."""

    labels = train_rows.groupby("patient_id")["label"].first().to_numpy(float)
    return float((len(labels) - labels.sum()) / labels.sum())


# -------------------------------------------------------- loss-side helpers


def anchor_weights(
    features: list[str],
    groups: dict[str, list[int]],
    reliability: pd.Series,
    dictionary_features: frozenset[str],
    *,
    dict_multiplier: float = 2.0,
    member_features: frozenset[str] | None = None,
    member_multiplier: float = 1.0,
) -> np.ndarray:
    """Per-feature anchor weights, family-renormalized to mean one.

    Weight = ``(1 + rho_j) * mu_j`` with ``mu_j = dict_multiplier`` on the
    dictionary coordinates and 1 elsewhere, times an optional extra
    ``member_multiplier`` on the selected member coordinates; within each
    family the weights are renormalized to mean 1 so the family-equal
    aggregation structure of the Stage-1 loss is preserved.
    """

    raw = np.array(
        [
            (1.0 + float(reliability[name]))
            * (float(dict_multiplier) if name in dictionary_features else 1.0)
            * (
                float(member_multiplier)
                if member_features and name in member_features
                else 1.0
            )
            for name in features
        ],
        dtype=np.float64,
    )
    weights = raw.copy()
    for indices in groups.values():
        block = raw[indices]
        weights[indices] = block / block.mean()
    return weights


def frozen_copy(model: torch.nn.Module) -> torch.nn.Module:
    """An eval-mode frozen deepcopy for the anchor targets."""

    frozen = copy.deepcopy(model)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    frozen.eval()
    return frozen


class DifferentiableInverse:
    """Torch mirror of the rank-gauss ``inverse_transform``.

    Piecewise-linear interpolation on the fitted per-feature knots with
    ``np.interp`` clamp semantics, float64. Gradients flow with the local
    knot-segment slope; outside the fitted range the clamp zeroes the slope,
    matching the numpy behaviour. Constant features (the epsilon knot
    signature) map to their constant with no gradient, matching the numpy
    ``constants_`` override.
    """

    def __init__(self, transform, feature_indices: list[int], device) -> None:
        if transform.method != "rank_gauss" or transform.knots_ is None:
            raise ValueError("only fitted rank_gauss transforms are supported")
        columns = [transform.knots_[index] for index in feature_indices]
        width = max(len(r) for r, _ in columns)
        gauss = np.full((len(columns), width), np.nan)
        raw = np.full((len(columns), width), np.nan)
        last_index = np.zeros(len(columns), dtype=np.int64)
        self.constants: dict[int, float] = {}
        for row, (r, g) in enumerate(columns):
            if r.size == 2 and float(g[0]) == -1e-6 and float(g[1]) == 1e-6:
                self.constants[row] = 0.5 * (float(r[0]) + float(r[1]))
                g = np.array([-1.0, 1.0])
                r = np.array([0.0, 0.0])
            gauss[row, : len(g)] = g
            gauss[row, len(g) :] = g[-1]
            raw[row, : len(r)] = r
            raw[row, len(r) :] = r[-1]
            last_index[row] = len(g) - 2
        self.device = device
        self.feature_indices = list(feature_indices)
        self.gauss = torch.tensor(gauss, dtype=torch.float64, device=device)
        self.raw = torch.tensor(raw, dtype=torch.float64, device=device)
        self.last_index = torch.tensor(last_index, dtype=torch.long, device=device)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Member-column tensor (n, k) float64 -> raw-space (n, k) float64."""

        z = values.to(dtype=torch.float64, device=self.device).t()  # (k, n)
        rows = torch.arange(self.gauss.shape[0], device=self.device)
        lower = self.gauss[:, :1]
        upper = self.gauss[rows, self.last_index + 1].unsqueeze(1)
        clamped = torch.clamp(z, lower, upper)
        position = torch.searchsorted(self.gauss.contiguous(), clamped.contiguous())
        index = torch.minimum(
            (position - 1).clamp(min=0), self.last_index.unsqueeze(1)
        )

        def gather(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            expanded = table.unsqueeze(1).expand(-1, idx.shape[1], -1)
            return torch.gather(expanded, 2, idx.unsqueeze(2)).squeeze(2)

        g_lo, g_hi = gather(self.gauss, index), gather(self.gauss, index + 1)
        r_lo, r_hi = gather(self.raw, index), gather(self.raw, index + 1)
        slope = (r_hi - r_lo) / (g_hi - g_lo)
        result = (r_lo + (clamped - g_lo) * slope).t()
        for row, constant in self.constants.items():
            result[:, row] = constant
        return result

    def verify(self, transform, values: np.ndarray) -> float:
        """Max scaled |torch - numpy| inverse difference over member columns."""

        values = np.asarray(values, dtype=np.float64)
        columns = self.feature_indices
        numpy_raw = transform.inverse_transform(values)[:, columns]
        torch_raw = self(
            torch.tensor(values[:, columns], device=self.device)
        ).cpu().numpy()
        scale = np.maximum(numpy_raw.std(axis=0), 1e-12)
        return float(np.max(np.abs(torch_raw - numpy_raw) / scale))


def pipeline_affine(pipeline) -> dict[str, Any]:
    """The fitted z-score + logistic endpoint as an explicit affine map."""

    scaler = pipeline.named_steps["standardscaler"]
    logistic = pipeline.named_steps["logisticregression"]
    return {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "weight": logistic.coef_[0].astype(np.float64),
        "bias": float(logistic.intercept_[0]),
    }


def unified_head_logits(values: np.ndarray, affine: dict[str, Any]) -> np.ndarray:
    """Exact float64 head logits: ledger-exact by construction."""

    values = np.asarray(values, dtype=np.float64)
    return (
        (values - affine["mean"]) / affine["scale"]
    ) @ affine["weight"] + affine["bias"]


def unified_head_probability(values: np.ndarray, affine: dict[str, Any]) -> np.ndarray:
    """The endpoint of record: sigmoid over the unified head logits."""

    logits = unified_head_logits(values, affine)
    return 1.0 / (1.0 + np.exp(-logits))


# ------------------------------------------------------------- run assembly


def load_warm_start(config: dict, root: Path, seed: int, *, device: str):
    """The frozen warm-up checkpoint, transform, and metadata.

    The method of record warm-up is family-balanced (arm ``W``); the
    conventional-arm ablation (C) opts in with ``warm_start.expected_arm``
    set to ``J`` and ``joint.stage1_loss: conventional``.
    """

    warm = config["warm_start"]
    expected_arm = str(warm.get("expected_arm", "W"))
    run_dir = root / str(warm["run_dir"]).format(seed=seed)
    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if str(checkpoint.get("arm")) != expected_arm:
        raise ValueError(
            f"warm start {run_dir} is not an arm-{expected_arm} run"
        )
    metadata = json.loads((run_dir / "metadata.json").read_text())
    required_epochs = int(warm["required_epochs"])
    if int(metadata.get("epochs", -1)) != required_epochs:
        raise ValueError(
            f"warm start trained {metadata.get('epochs')} epochs; "
            f"the method of record warm-up is {required_epochs}"
        )
    model = ThyroidXLConvNeXtRadiomicsRegressor(
        n_features=EXPECTED_FEATURES, dropout=0.1, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    transform = joblib.load(run_dir / "target_transform.joblib")
    return model, transform, checkpoint, run_dir


@torch.no_grad()
def predict_frames(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    *,
    amp: bool = True,
) -> dict[str, np.ndarray]:
    """Transformed-space frame predictions plus ids/labels/patients."""

    model.eval()
    ids, patients, labels, predictions = [], [], [], []
    for batch in loader:
        with torch.autocast("cuda", enabled=amp and str(device).startswith("cuda")):
            prediction = model(batch["image"].to(device))
        predictions.append(prediction.float().cpu().numpy())
        ids.extend(map(str, batch["sample_id"]))
        patients.extend(map(str, batch["patient_id"]))
        labels.extend(batch["label"].numpy().tolist())
    return {
        "sample_id": np.array(ids),
        "patient_id": np.array(patients),
        "label": np.array(labels, dtype=float),
        "prediction": np.concatenate(predictions),
    }


def evaluate_validation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    protocol: dict[str, Any],
    *,
    dataset: str,
    transform,
    member_indices: list[int],
    member_names: list[str],
    dict_features: frozenset[str],
    device: str,
    amp: bool,
    oracle: pd.DataFrame,
    training_sd: pd.Series,
    frozen_prediction: np.ndarray,
    frozen_pipeline,
    unified_head: dict[str, Any],
    unified_head_init: dict[str, Any],
    inverse: DifferentiableInverse,
) -> dict[str, Any]:
    """Endpoints, raw-space fidelity, drift, and ledger closure (validation only).

    The record endpoint is ``endpoint_unified``; ``endpoint_unified_init``
    (the unrefined OOF-init head) and ``endpoint_frozen_v1`` (the frozen v1
    pipeline) are diagnostics that isolate where the gain lives.
    """

    features: list[str] = protocol["features"]
    predicted = predict_frames(model, loader, device, amp=amp)
    frame = pd.DataFrame(
        {
            "unit_key": (
                predicted["patient_id"] if dataset == "thyroidxl" else predicted["sample_id"]
            ),
            "label": predicted["label"].astype(int),
        }
    )
    pooled = (
        frame.groupby("unit_key", sort=True)
        .agg(label=("label", "first"))
        .reset_index()
    )
    raw_frame = pd.DataFrame(
        transform.inverse_transform(predicted["prediction"]), columns=features
    )
    raw_frame.insert(0, "patient_id", predicted["patient_id"])
    raw_frame.insert(0, "sample_id", predicted["sample_id"])
    if dataset == "thyroidxl":
        pred_unit = raw_frame.groupby("patient_id", sort=True)[features].mean()
        truth_unit = oracle.groupby("patient_id", sort=True)[features].mean()
    else:
        pred_unit = raw_frame.set_index("sample_id")[features]
        truth_unit = oracle.set_index("sample_id")[features]
    truth_unit = truth_unit.reindex(pred_unit.index)
    if truth_unit.isna().any().any():
        raise ValueError("oracle unit truth missing rows for predicted units")
    if list(pooled["unit_key"]) != list(pred_unit.index):
        raise ValueError("unit ordering differs between head and raw pipelines")
    labels = pooled["label"].to_numpy(int)

    frozen_probability = frozen_pipeline.predict_proba(
        pred_unit.loc[:, member_names]
    )[:, 1]
    frozen_metrics = extended_binary_metrics(labels, frozen_probability)
    frozen_metrics["n_units"] = int(len(pooled))

    member_unit = pred_unit.loc[:, member_names].to_numpy(np.float64)
    logits64 = unified_head_logits(member_unit, unified_head)
    unified_probability = 1.0 / (1.0 + np.exp(-logits64))
    unified_metrics = extended_binary_metrics(labels, unified_probability)
    unified_metrics["n_units"] = int(len(pooled))
    logits32 = unified_head_logits(
        member_unit.astype(np.float32).astype(np.float64), unified_head
    )
    unified_closure = float(np.max(np.abs(logits64 - logits32)))
    logits_init = unified_head_logits(member_unit, unified_head_init)
    unified_init_probability = 1.0 / (1.0 + np.exp(-logits_init))
    unified_init_metrics = extended_binary_metrics(labels, unified_init_probability)
    unified_init_metrics["n_units"] = int(len(pooled))

    torch_raw = inverse(
        torch.tensor(predicted["prediction"][:, member_indices])
    ).cpu().numpy()
    torch_frame = pd.DataFrame(torch_raw, columns=member_names)
    if dataset == "thyroidxl":
        torch_unit = (
            torch_frame.assign(patient_id=predicted["patient_id"])
            .groupby("patient_id", sort=True)[member_names]
            .mean()
        )
    else:
        torch_unit = torch_frame.assign(patient_id=predicted["sample_id"]).set_index(
            "patient_id"
        )
    torch_unit = torch_unit.reindex(pred_unit.index)
    path_consistency = float(
        np.max(
            np.abs(
                unified_head_logits(torch_unit.to_numpy(np.float64), unified_head)
                - logits64
            )
        )
    )

    rhos, maes = [], []
    for name in features:
        truth = truth_unit[name].to_numpy()
        pred = pred_unit[name].to_numpy()
        rhos.append(float(spearmanr(truth, pred).statistic))
        maes.append(float(np.mean(np.abs(truth - pred))) / float(training_sd[name]))
    positions = {
        "members": [features.index(n) for n in member_names],
        "dict36": [features.index(n) for n in sorted(dict_features)],
    }
    fidelity = {
        "mean_rho_102": float(np.nanmean(rhos)),
        "mean_mae_over_sd_102": float(np.nanmean(maes)),
    }
    for label, idx in positions.items():
        fidelity[f"mean_rho_{label}"] = float(np.nanmean([rhos[i] for i in idx]))
        fidelity[f"mean_mae_over_sd_{label}"] = float(
            np.nanmean([maes[i] for i in idx])
        )

    drift_rhos, drift_abs = [], []
    for column in member_indices:
        drift_rhos.append(
            float(
                spearmanr(
                    frozen_prediction[:, column], predicted["prediction"][:, column]
                ).statistic
            )
        )
        drift_abs.append(
            float(
                np.mean(
                    np.abs(
                        frozen_prediction[:, column] - predicted["prediction"][:, column]
                    )
                )
            )
        )
    return {
        "endpoint_unified": unified_metrics,
        "endpoint_unified_init": unified_init_metrics,
        "endpoint_frozen_v1": frozen_metrics,
        "path_consistency": path_consistency,
        "fidelity": fidelity,
        "drift": {
            "member_mean_abs": float(np.mean(drift_abs)),
            "member_mean_spearman": float(np.nanmean(drift_rhos)),
        },
        "ledger_closure_error": unified_closure,
        "unified_probability": unified_probability,
        "unified_init_probability": unified_init_probability,
        "frozen_probability": frozen_probability,
        "unit_ids": pooled["unit_key"].to_numpy(),
        "unit_labels": labels,
        "frame_ids": predicted["sample_id"],
        "frame_prediction_transformed": predicted["prediction"],
        "frame_prediction_raw": raw_frame,
    }


def train_stage2_joint(
    config: dict,
    root: Path,
    output: Path,
    seed: int,
    *,
    device: str | None = None,
    warmstart_dir: Path | None = None,
    workers: int | None = None,
    epochs_override: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """One (cohort, seed) run of the method of record; returns the summary."""

    dataset = str(config["dataset"])
    settings = config["training"]
    joint = config["joint"]
    lambda_cls = float(joint["lambda_cls"])
    lambda_anchor = float(joint["lambda_anchor"])
    stage1_loss = str(joint.get("stage1_loss", "family_balanced"))
    if stage1_loss not in ("family_balanced", "conventional"):
        raise ValueError(f"unknown joint.stage1_loss {stage1_loss!r}")
    member_multiplier = float(joint["member_anchor_multiplier"])
    dict_multiplier = float(joint.get("dict_anchor_multiplier", 2.0))
    epochs = int(epochs_override or settings["epochs"])
    if smoke:
        epochs = 1
    device = str(device or settings["device"])
    workers = int(workers or settings["workers"])
    amp = bool(settings["amp"]) and device.startswith("cuda")
    output.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(config, root)
    features: list[str] = protocol["features"]
    cell = load_member_cell(config, root, seed)
    member_names = list(cell.members)
    member_indices = [features.index(name) for name in member_names]
    member_features = frozenset(member_names)
    group_weights = anchor_weights(
        features,
        protocol["groups"],
        cell.reliability,
        cell.dictionary_features,
        dict_multiplier=dict_multiplier,
        member_features=member_features,
        member_multiplier=member_multiplier,
    )

    oracle = protocol["oracle"]
    train_rows, validation = protocol["train"], protocol["validation"]
    oracle_indexed = oracle.set_index("sample_id")
    train_raw = oracle_indexed.loc[train_rows["sample_id"], features].to_numpy(
        np.float64
    )
    val_raw = oracle_indexed.loc[validation["sample_id"], features].to_numpy(np.float64)

    seed_everything(seed)
    model, transform, checkpoint, warm_dir = load_warm_start(
        config, root, seed, device=device
    )
    if warmstart_dir is not None:
        # Explicit-directory override for reproduction/debug; skips the
        # config's warm-up metadata checks but still checks the expected arm.
        model, transform, checkpoint, warm_dir = _load_warm_start_dir(
            warmstart_dir, device,
            expected_arm=str(config["warm_start"].get("expected_arm", "W")),
        )
    frozen = frozen_copy(model)
    train_y = transform.transform(train_raw).astype(np.float32)
    val_y = transform.transform(val_raw).astype(np.float32)

    augment = UltrasoundAugment(
        **{
            key: (tuple(value) if isinstance(value, list) else value)
            for key, value in config["augmentation"].items()
        }
    )
    train_loader = torch.utils.data.DataLoader(
        FrameDataset(train_rows, train_y, augment),
        batch_sampler=PatientBagBatchSampler(
            train_rows["patient_id"], int(settings["bag_frame_budget"]), seed
        ),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seeded_worker_init,
    )
    val_loader = torch.utils.data.DataLoader(
        FrameDataset(validation, val_y, None),
        shuffle=False,
        batch_size=int(settings["batch_size"]),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seeded_worker_init,
    )
    frozen_val_prediction = predict_frames(frozen, val_loader, device, amp=amp)[
        "prediction"
    ]

    frozen_pipeline = frozen_panel_endpoint_pipeline()
    frozen_pipeline.fit(cell.student[member_names], cell.student["label"].astype(int))
    init_affine = pipeline_affine(frozen_pipeline)

    inverse = DifferentiableInverse(transform, member_indices, device)
    inverse_check = inverse.verify(transform, frozen_val_prediction)
    if inverse_check >= INVERSE_MIRROR_TOLERANCE:
        raise SystemExit(f"torch inverse mirror deviates: {inverse_check}")

    head_weight = torch.nn.Parameter(
        torch.tensor(init_affine["weight"], dtype=torch.float32, device=device)
    )
    head_bias = torch.nn.Parameter(
        torch.tensor(float(init_affine["bias"]), dtype=torch.float32, device=device)
    )
    head_mean = torch.tensor(init_affine["mean"], dtype=torch.float32, device=device)
    head_scale = torch.tensor(init_affine["scale"], dtype=torch.float32, device=device)
    member_tensor = torch.tensor(member_indices, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(group_weights, dtype=torch.float32, device=device)
    group_tensors = {
        name: torch.tensor(indices, dtype=torch.long, device=device)
        for name, indices in protocol["groups"].items()
    }
    pos_weight_value = patient_pos_weight(train_rows)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    parameters = list(model.parameters()) + [head_weight, head_bias]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs=epochs, warmup_epochs=int(settings["warmup_epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history: list[dict[str, Any]] = []
    best = {"score": -float("inf"), "epoch": 0}
    started = time.time()
    for epoch in range(epochs):
        model.train()
        losses, cls_losses, anchor_losses = [], [], []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            lengths = torch.tensor(
                patient_block_lengths(list(batch["patient_id"])),
                dtype=torch.long,
                device=device,
            )
            ends = torch.cumsum(lengths, dim=0) - 1
            patient_label = batch["label"].to(device, non_blocking=True)[ends]
            with torch.autocast("cuda", enabled=amp):
                prediction = model(image)
                elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
                if stage1_loss == "conventional":
                    # C arm: one all-targets group, patient block means — the
                    # target-wise mean loss with the JR patient-level semantics.
                    per_frame_family = elementwise.mean(dim=1, keepdim=True)
                else:
                    per_frame_family = torch.stack(
                        [
                            elementwise[:, indices].mean(dim=1)
                            for indices in group_tensors.values()
                        ],
                        dim=1,
                    )
                loss = block_means(per_frame_family, lengths).mean()
                # The exact inference path: inverse transform -> patient
                # mean in raw space -> affine head, gradient flowing through.
                raw_members = inverse(prediction[:, member_tensor].double()).float()
                pooled_members = block_means(raw_members, lengths)
                patient_logit = (
                    (pooled_members - head_mean) / head_scale
                ) @ head_weight + head_bias
                cls_loss = bce(patient_logit, patient_label)
                loss = loss + lambda_cls * cls_loss
                cls_losses.append(float(cls_loss.detach()))
                if lambda_anchor > 0:
                    frozen_prediction = frozen(image)
                    diff = F.smooth_l1_loss(prediction, frozen_prediction, reduction="none")
                    per_frame_anchor = torch.stack(
                        [
                            (diff[:, indices] * weight_tensor[indices]).mean(dim=1)
                            for indices in group_tensors.values()
                        ],
                        dim=1,
                    )
                    anchor_loss = block_means(per_frame_anchor, lengths).mean()
                    loss = loss + lambda_anchor * anchor_loss
                    anchor_losses.append(float(anchor_loss.detach()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        predicted = predict_frames(model, val_loader, device, amp=amp)
        quality = stage1_quality(
            val_y,
            predicted["prediction"],
            reference_mean=np.zeros(val_y.shape[1], dtype=np.float64),
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_mean_r2": float(quality["mean_r2"]),
            "validation_mse": float(quality["mse"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "cls_loss": float(np.mean(cls_losses)),
            "anchor_loss": float(np.mean(anchor_losses)) if anchor_losses else None,
        }
        history.append(row)
        atomic_csv(output / "history.csv", pd.DataFrame(history))
        if row["validation_mean_r2"] > best["score"]:
            best = {"score": row["validation_mean_r2"], "epoch": epoch + 1}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "head_weight": head_weight.detach().cpu().numpy(),
                    "head_bias": float(head_bias.detach().item()),
                    "epoch": epoch + 1,
                    "validation_mean_r2": row["validation_mean_r2"],
                    "seed": seed,
                    "arm": "stage2_joint",
                },
                output / "best_checkpoint.pt",
            )
        print(
            f"stage2-joint {dataset} seed{seed} epoch={epoch + 1}/{epochs} "
            f"loss={row['train_loss']:.5f} val_meanR2={row['validation_mean_r2']:.4f} "
            f"cls={row['cls_loss']:.4f}"
            + (f" anc={row['anchor_loss']:.5f}" if row["anchor_loss"] is not None else ""),
            flush=True,
        )
        scheduler.step()

    best_checkpoint = torch.load(
        output / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    model.to(device)
    unified_head = {
        "mean": init_affine["mean"],
        "scale": init_affine["scale"],
        "weight": np.asarray(best_checkpoint["head_weight"], dtype=np.float64),
        "bias": float(best_checkpoint["head_bias"]),
    }

    oracle_val = oracle.loc[
        oracle["sample_id"].isin(set(validation["sample_id"])),
        ["sample_id", "patient_id", *features],
    ]
    training_sd = training_patient_sd(oracle, train_rows, features)
    result = evaluate_validation(
        model,
        val_loader,
        protocol,
        dataset=dataset,
        transform=transform,
        member_indices=member_indices,
        member_names=member_names,
        dict_features=cell.dictionary_features,
        device=device,
        amp=amp,
        oracle=oracle_val,
        training_sd=training_sd,
        frozen_prediction=frozen_val_prediction,
        frozen_pipeline=frozen_pipeline,
        unified_head=unified_head,
        unified_head_init=init_affine,
        inverse=inverse,
    )
    for endpoint, probability in (
        ("unified", result["unified_probability"]),
        ("unified_init", result["unified_init_probability"]),
        ("frozen_v1", result["frozen_probability"]),
    ):
        atomic_csv(
            output / f"validation_endpoint_{endpoint}_predictions.csv",
            pd.DataFrame(
                {
                    "unit_id": result["unit_ids"],
                    "label": result["unit_labels"],
                    "probability": probability,
                }
            ),
        )
    frame_predictions = pd.DataFrame(
        result["frame_prediction_transformed"], columns=features
    )
    frame_predictions.insert(0, "sample_id", result["frame_ids"])
    frame_predictions.to_csv(
        output / "validation_frame_predictions_transformed.csv", index=False
    )
    result["frame_prediction_raw"].to_csv(
        output / "validation_frame_predictions_raw.csv", index=False
    )

    resolved_config = json.loads(json.dumps(config))
    resolved_config["training"] = {**resolved_config["training"], "seed": seed, "device": device}
    summary = {
        "status": "complete",
        "method": "train_inference_consistent_stage2_joint",
        "cohort": dataset,
        "arm": "stage2_joint",
        "seed": seed,
        "smoke": bool(smoke),
        "epochs": epochs,
        "lambda_cls": lambda_cls,
        "lambda_anchor": lambda_anchor,
        "stage1_loss": stage1_loss,
        "member_anchor_multiplier": member_multiplier,
        "dict_anchor_multiplier": dict_multiplier,
        "bag_mode": True,
        "bag_frame_budget": int(settings["bag_frame_budget"]),
        "cls_pos_weight": float(pos_weight_value),
        "cls_pos_weight_level": "patient",
        "cls_path": "raw_space_patient_mean_dictionary_head",
        "best_epoch": best["epoch"],
        "best_validation_mean_r2": best["score"],
        "warm_start": {
            "run_dir": str(warm_dir),
            "checkpoint": str(warm_dir / "best_checkpoint.pt"),
            "checkpoint_sha256": sha256_file(warm_dir / "best_checkpoint.pt"),
            "warm_start_epoch": int(checkpoint["epoch"]) + 1,
            "warm_start_val_mean_r2": float(checkpoint["validation_mean_r2"]),
        },
        "head": {
            "n_members": len(member_names),
            "head_source": "unified_joint_finetuned",
            "unified_init": "v1_oof_pipeline_affine",
            "standardization": "frozen (mean/scale from the train-only OOF z-score fit)",
        },
        "torch_inverse_max_scaled_diff": inverse_check,
        "path_consistency": result["path_consistency"],
        "endpoint_unified": result["endpoint_unified"],
        "endpoint_unified_init": result["endpoint_unified_init"],
        "endpoint_frozen_v1": result["endpoint_frozen_v1"],
        "fidelity": result["fidelity"],
        "drift": result["drift"],
        "ledger_closure_error": result["ledger_closure_error"],
        "checkpoint_rule": {
            "metric": str(settings["checkpoint_metric"]),
            "rule": "best epoch by validation mean transformed R2 over all 102 targets",
        },
        "config_sha256": canonical_config_hash(resolved_config),
        "elapsed_seconds": time.time() - started,
        "official_test_used": False,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def _load_warm_start_dir(run_dir: Path, device: str, expected_arm: str = "W"):
    """Explicit-directory warm start (reproduction/debug override)."""

    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if str(checkpoint.get("arm")) != expected_arm:
        raise ValueError(f"warm start {run_dir} is not an arm-{expected_arm} run")
    model = ThyroidXLConvNeXtRadiomicsRegressor(
        n_features=EXPECTED_FEATURES, dropout=0.1, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    transform = joblib.load(run_dir / "target_transform.joblib")
    return model, transform, checkpoint, run_dir


def main(argv: list[str] | None = None, *, root: Path) -> None:
    """CLI entry; the wrapper script supplies the repository root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmstart-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="override (smoke/debug only)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="smoke override only")
    parser.add_argument("--smoke", action="store_true", help="1-epoch smoke only")
    args = parser.parse_args(argv)
    from setproctitle import setproctitle  # noqa: PLC0415

    config = yaml.safe_load(args.config.read_text())
    dataset = str(config["dataset"])
    setproctitle(f"cbm_stage2_joint_{dataset}_seed{args.seed}")
    summary = train_stage2_joint(
        config,
        root,
        args.output,
        args.seed,
        device=args.device,
        warmstart_dir=args.warmstart_dir,
        workers=args.workers,
        epochs_override=args.epochs,
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "cohort",
                    "seed",
                    "best_epoch",
                    "endpoint_unified",
                    "endpoint_unified_init",
                    "endpoint_frozen_v1",
                    "path_consistency",
                    "fidelity",
                    "drift",
                    "ledger_closure_error",
                )
            },
            indent=2,
        )
    )
