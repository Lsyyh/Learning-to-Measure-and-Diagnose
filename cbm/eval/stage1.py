"""Stage-1 evaluation stack: quality metrics, endpoint metrics, patient
primitives, leakage-safe OOF cross-fitting, and deterministic-replay
comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# ------------------------------------------------------- Stage-1 quality

#: Column prefix of Stage-1 radiomics prediction tables: one
#: ``prediction__<feature_name>`` column per inventory feature. The trainers
#: write it, the Stage-2 loaders strip it — this is its single definition.
PREDICTION_PREFIX = "prediction__"

#: A feature counts as "predictable" when its R^2 clears this threshold.
DEFAULT_R2_THRESHOLD = 0.1


def per_feature_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error per feature."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return np.mean((y_true - y_pred) ** 2, axis=0)


def per_feature_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Coefficient of determination per feature, against the evaluated split's own mean.

    Returns ``nan`` for features that are constant on this split (no variance
    to explain). Negative values mean the model is worse than that split's
    mean predictor. R^2 is invariant to affine rescaling of each feature,
    which is exactly why it is the selection metric for a high-dimensional
    radiomics vector whose features have very different variances.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    residual = np.sum((y_true - y_pred) ** 2, axis=0)
    total = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - residual / total
    return np.where(total < 1e-12, np.nan, r2)


def mean_predictor_mse(y_true: np.ndarray, reference_mean: np.ndarray | None = None) -> float:
    """MSE of the constant predictor, the baseline Stage-1 must beat.

    ``reference_mean`` defaults to the evaluated split's own mean. Pass the
    train mean (typically zeros in standardized space) to reproduce exactly
    what a model that only learned the training bias would score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    center = (
        y_true.mean(axis=0)
        if reference_mean is None
        else np.asarray(reference_mean, dtype=np.float64)
    )
    return float(np.mean((y_true - center) ** 2))


def stage1_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    reference_mean: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize Stage-1 quality with the mean predictor as the reference.

    Keys:
        ``mse``                  aggregated MSE over all features
        ``mean_predictor_mse``   the baseline the model must beat
        ``beats_mean_predictor`` bool, ``mse < mean_predictor_mse``
        ``mean_r2`` / ``median_r2``  aggregated per-feature R^2 (nan-safe)
        ``n_features`` / ``n_features_above_r2`` / ``fraction_above_r2``
        ``worst_features``       the five lowest-R^2 feature names (when given)
    """
    r2 = per_feature_r2(y_true, y_pred)
    mse = per_feature_mse(y_true, y_pred)
    finite = np.isfinite(r2)
    above = int(np.sum(r2[finite] > r2_threshold)) if finite.any() else 0
    summary: dict[str, object] = {
        "mse": float(np.mean(mse)),
        "mean_predictor_mse": mean_predictor_mse(y_true, reference_mean),
        "mean_r2": float(np.nanmean(r2)) if finite.any() else float("nan"),
        "median_r2": float(np.nanmedian(r2)) if finite.any() else float("nan"),
        "n_features": int(r2.size),
        "n_features_above_r2": above,
        "fraction_above_r2": float(above / r2.size) if r2.size else 0.0,
        "r2_threshold": float(r2_threshold),
    }
    summary["beats_mean_predictor"] = bool(
        summary["mse"] < summary["mean_predictor_mse"]  # type: ignore[operator]
    )
    if feature_names is not None:
        if len(feature_names) != r2.size:
            raise ValueError("feature_names length does not match the feature dimension")
        order = np.argsort(np.where(finite, r2, np.inf))
        summary["worst_features"] = [
            {"feature": feature_names[int(index)], "r2": float(r2[int(index)])}
            for index in order[:5]
        ]
    return summary


def stage1_quality_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
):
    """Per-feature quality table (feature, r2, mse, target_variance)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    if len(feature_names) != y_true.shape[1]:
        raise ValueError("feature_names length does not match the feature dimension")
    return pd.DataFrame(
        {
            "feature": feature_names,
            "r2": per_feature_r2(y_true, y_pred),
            "mse": per_feature_mse(y_true, y_pred),
            "target_variance": y_true.var(axis=0),
        }
    ).sort_values("r2", ascending=False, kind="stable", ignore_index=True)


# ------------------------------------------------------ endpoint metrics
# binary_metrics / extended_binary_metrics / expected_calibration_error moved
# to cbm.eval.metrics (single definition shared with the report builder).


# -------------------------------------------------- patient-level primitives


def validate_endpoint_arrays(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned, finite, binary endpoint arrays."""
    labels_value = np.asarray(labels, dtype=int)
    probabilities_value = np.asarray(probabilities, dtype=float)
    if labels_value.ndim != 1 or probabilities_value.ndim != 1:
        raise ValueError("Endpoint labels and probabilities must be one-dimensional")
    if len(labels_value) != len(probabilities_value) or not len(labels_value):
        raise ValueError("Endpoint labels and probabilities must be aligned and nonempty")
    if set(np.unique(labels_value)) != {0, 1}:
        raise ValueError("Endpoint evaluation requires both binary classes")
    if not np.isfinite(probabilities_value).all():
        raise ValueError("Endpoint probabilities must be finite")
    if ((probabilities_value < 0.0) | (probabilities_value > 1.0)).any():
        raise ValueError("Endpoint probabilities must be within [0, 1]")
    return labels_value, probabilities_value


def assert_single_label_per_patient(
    frame: pd.DataFrame,
    *,
    source: str,
    label_column: str = "label",
    patient_column: str = "patient_id",
) -> None:
    """Reject any patient carrying more than one endpoint label."""
    if frame.groupby(patient_column)[label_column].nunique().max() != 1:
        raise ValueError(f"{source} patient has conflicting endpoint labels")


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one class-stratified bootstrap replicate over aligned positions.

    Classes are resampled in ascending label order (0 then 1), each with
    replacement at its own size; the per-class draws are concatenated. This
    call order is behaviour-frozen and pinned by unit tests.
    """
    labels_value = np.asarray(labels, dtype=int)
    positions = {
        endpoint: np.flatnonzero(labels_value == endpoint) for endpoint in (0, 1)
    }
    if any(len(value) == 0 for value in positions.values()):
        raise ValueError("Stratified bootstrap requires both binary classes")
    return np.concatenate(
        [rng.choice(value, size=len(value), replace=True) for value in positions.values()]
    )


# --------------------------------------- ThyroidXL patient-level evaluation


def aggregate_stage1_by_patient(
    patient_ids: list[str] | np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    reference_mean: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mean-pool frame-level radiomics per patient and evaluate Stage-1 quality."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or len(patient_ids) != len(truth):
        raise ValueError("Patient Stage-1 arrays are not aligned")
    rows = []
    for patient_id in sorted(set(str(value) for value in patient_ids)):
        selected = np.asarray([str(value) == patient_id for value in patient_ids])
        rows.append(
            {
                "patient_id": patient_id,
                "frames": int(selected.sum()),
                "truth": truth[selected].mean(axis=0),
                "prediction": prediction[selected].mean(axis=0),
            }
        )
    patient_truth = np.stack([row.pop("truth") for row in rows])
    patient_prediction = np.stack([row.pop("prediction") for row in rows])
    return pd.DataFrame(rows), stage1_quality(
        patient_truth, patient_prediction, reference_mean=reference_mean
    )


def aggregate_endpoint_by_patient(
    patient_ids: list[str] | np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Mean-pool frame probabilities and report patient-level binary metrics."""
    source = pd.DataFrame(
        {
            "patient_id": [str(value) for value in patient_ids],
            "label": np.asarray(labels, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    assert_single_label_per_patient(source, source="ThyroidXL")
    aggregate = source.groupby("patient_id", sort=True).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
        frames=("probability", "size"),
    )
    aggregate = aggregate.reset_index()
    labels_value = aggregate["label"].to_numpy()
    probabilities_value = aggregate["probability"].to_numpy()
    metrics = {
        "roc_auc": float(roc_auc_score(labels_value, probabilities_value)),
        "average_precision": float(average_precision_score(labels_value, probabilities_value)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(labels_value, probabilities_value >= 0.5)
        ),
    }
    return aggregate, metrics


def training_patient_sd(
    oracle: pd.DataFrame, train: pd.DataFrame, features: list[str]
) -> pd.Series:
    """Per-feature SD of patient-pooled oracle values over the training patients.

    ``oracle`` is frame-level (one row per ``sample_id``; the same patients span
    multiple frames), so the slice must match ``patient_id`` columns on both
    sides — matching ``sample_id`` against patient ids silently empties the
    slice and NaNs the whole ``training_patient_sd`` column.
    """
    train_patients = set(train["patient_id"].astype(str))
    training = oracle.loc[oracle["patient_id"].astype(str).isin(train_patients)]
    if training.empty:
        raise ValueError(
            "training-patient oracle slice is empty — patient_id matching failed"
        )
    return (
        training.assign(patient_id=training["patient_id"].astype(str))
        .groupby("patient_id", sort=True)[features]
        .mean()
        .std(ddof=0)
    )


# -------------------------------------- leakage-safe OOF cross-fitting

#: Frozen provenance string stamped on every cross-fitted OOF prediction row.
OOF_PREDICTION_SOURCE = "stage1_patient_3fold_oof"


def build_stage1_oof_assignment(
    split_manifest: pd.DataFrame,
    *,
    n_splits: int = 3,
    seed: int = 42,
    split_column: str = "development_partition",
    split: str = "train",
    required_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assign every selected training patient to exactly one Stage-1 OOF fold.

    The default selector preserves the ThyroidXL development-train contract.
    Other locked protocols may select an explicitly named partition, for
    example ``split_column='final_partition', split='train'`` for TN3K R1b.
    ``required_columns`` applies additional exact-value guards before folds
    are assigned, so the selector cannot silently include another source.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    required = {
        "sample_id",
        "patient_id",
        "label",
        "status",
        "preprocessing_status",
        split_column,
    }
    if not required.issubset(split_manifest):
        raise ValueError("Stage-1 OOF split manifest schema is incomplete")
    selected = split_manifest.loc[
        split_manifest["status"].eq("included")
        & split_manifest["preprocessing_status"].eq("complete")
        & split_manifest[split_column].eq(split)
    ].copy()
    if selected.empty:
        raise ValueError("Stage-1 OOF cohort is empty")
    guards = required_columns or {"official_split": "train", "split": "development"}
    for column, value in guards.items():
        if column not in selected:
            raise ValueError(f"Stage-1 OOF guard column is missing: {column}")
        if not selected[column].eq(value).all():
            if column == "official_split" and value == "train":
                raise ValueError("Stage-1 OOF assignment selected locked official-test records")
            raise ValueError(
                f"Stage-1 OOF assignment selected rows outside {column}={value!r}"
            )
    if selected["sample_id"].duplicated().any():
        raise ValueError("ThyroidXL OOF sample IDs are not unique")
    patient_labels = selected[["patient_id", "label"]].drop_duplicates()
    if patient_labels.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("ThyroidXL OOF patients have conflicting labels")
    patient_labels = patient_labels.drop_duplicates("patient_id").sort_values("patient_id")
    patient_labels["oof_fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_ids = patient_labels["patient_id"].astype(str).to_numpy()
    labels = patient_labels["label"].to_numpy(int)
    for fold, (_, selection_positions) in enumerate(
        splitter.split(patient_ids, labels, groups=patient_ids)
    ):
        patient_labels.iloc[selection_positions, patient_labels.columns.get_loc("oof_fold")] = fold
    if (patient_labels["oof_fold"] < 0).any():
        raise ValueError("A Stage-1 training patient lacks an OOF fold")
    assignment = selected[["sample_id", "patient_id", "label"]].copy()
    fold_by_patient = patient_labels.set_index("patient_id")["oof_fold"]
    assignment["oof_fold"] = assignment["patient_id"].map(fold_by_patient).astype(int)
    assignment["prediction_source"] = OOF_PREDICTION_SOURCE
    assignment["crossfit_seed"] = int(seed)
    assignment["crossfit_folds"] = int(n_splits)
    return assignment.sort_values(["oof_fold", "patient_id", "sample_id"]).reset_index(drop=True)


def validate_oof_frame_predictions(
    assignment: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Require one finite held-out Stage-1 prediction for every train frame."""
    predicted_columns = [f"{PREDICTION_PREFIX}{name}" for name in feature_names]
    required = {
        "sample_id",
        "patient_id",
        "label",
        "oof_fold",
        "prediction_source",
        *predicted_columns,
    }
    if not required.issubset(predictions):
        raise ValueError("ThyroidXL Stage-1 OOF prediction schema is incomplete")
    if predictions["sample_id"].duplicated().any():
        raise ValueError("ThyroidXL Stage-1 OOF predictions contain duplicate samples")
    if set(predictions["sample_id"]) != set(assignment["sample_id"]):
        raise ValueError("ThyroidXL Stage-1 OOF predictions do not cover 100% of train frames")
    aligned = predictions.set_index("sample_id").loc[assignment["sample_id"]].reset_index()
    for column in ("patient_id", "label", "oof_fold", "prediction_source"):
        expected = assignment[column].astype(str).tolist()
        observed = aligned[column].astype(str).tolist()
        if observed != expected:
            raise ValueError(f"ThyroidXL Stage-1 OOF {column} disagrees with fold assignment")
    values = aligned[predicted_columns].to_numpy(np.float64)
    if values.shape != (len(assignment), len(feature_names)) or not np.isfinite(values).all():
        raise ValueError("ThyroidXL Stage-1 OOF features are incomplete or non-finite")
    return aligned


# -------------------------------------------- deterministic-replay compare


def _tensor_bytes(tensor) -> bytes:
    import torch

    tensor = tensor.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes()
    except (TypeError, RuntimeError):  # bfloat16 and other NumPy-opaque dtypes
        return tensor.view(torch.uint8).numpy().tobytes()


def checkpoint_digest(path: Path) -> dict[str, object]:
    """Digest the model_state tensors of a training checkpoint."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    digest = hashlib.sha256()
    tensors = 0
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
        tensors += 1
    return {
        "digest": digest.hexdigest(),
        "tensors": tensors,
        "epoch": checkpoint.get("epoch"),
        "seed": checkpoint.get("seed"),
    }


def _compare_columns(frame_a: pd.DataFrame, frame_b: pd.DataFrame, id_column: str) -> dict:
    if list(frame_a.columns) != list(frame_b.columns):
        return {"identical": False, "reason": "column names differ"}
    if len(frame_a) != len(frame_b):
        return {
            "identical": False,
            "reason": "row counts differ",
            "rows_a": len(frame_a),
            "rows_b": len(frame_b),
        }
    if not frame_a[id_column].astype(str).equals(frame_b[id_column].astype(str)):
        return {"identical": False, "reason": f"{id_column} order or values differ"}
    columns: dict[str, object] = {}
    identical = True
    worst = 0.0
    for column in frame_a.columns:
        series_a, series_b = frame_a[column], frame_b[column]
        if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b):
            difference = (series_a.astype(float) - series_b.astype(float)).abs()
            max_abs = float(difference.max()) if len(difference) else 0.0
            columns[column] = max_abs
            worst = max(worst, max_abs)
        else:
            equal = bool(series_a.astype(str).equals(series_b.astype(str)))
            columns[column] = "equal" if equal else "differ"
            identical = identical and equal
    return {"identical": identical and worst == 0.0, "max_abs_error": worst, "columns": columns}


def compare_csv(path_a: Path, path_b: Path, id_column: str) -> dict:
    if not path_a.exists() or not path_b.exists():
        return {
            "identical": False,
            "reason": "missing file",
            "present_a": path_a.exists(),
            "present_b": path_b.exists(),
        }
    return _compare_columns(pd.read_csv(path_a), pd.read_csv(path_b), id_column)


def compare_checkpoints(path_a: Path, path_b: Path) -> dict:
    if not path_a.exists() or not path_b.exists():
        return {
            "identical": False,
            "reason": "missing file",
            "present_a": path_a.exists(),
            "present_b": path_b.exists(),
        }
    digest_a, digest_b = checkpoint_digest(path_a), checkpoint_digest(path_b)
    return {
        "identical": digest_a["digest"] == digest_b["digest"],
        "digest_a": digest_a["digest"],
        "digest_b": digest_b["digest"],
        "tensors": digest_a["tensors"],
        "epoch_a": digest_a["epoch"],
        "epoch_b": digest_b["epoch"],
        "seed_a": digest_a["seed"],
        "seed_b": digest_b["seed"],
    }


def _merge(components: dict[str, object], kind: str, run_a: Path, run_b: Path) -> dict:
    identical = all(
        bool(component.get("identical")) for component in components.values()
    )
    numeric_errors = [
        float(component["max_abs_error"])
        for component in components.values()
        if "max_abs_error" in component
    ]
    return {
        "kind": kind,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "identical": identical,
        "max_abs_error": max(numeric_errors) if numeric_errors else None,
        "components": components,
    }


def compare_ours_oof_runs(
    run_a: Path, run_b: Path, snapshots: tuple[int, ...] = (80, 100)
) -> dict:
    """Compare two ``train_ours_oof.py`` fold directories snapshot by snapshot."""
    run_a, run_b = Path(run_a), Path(run_b)
    components: dict[str, object] = {}
    components["history"] = compare_csv(run_a / "history.csv", run_b / "history.csv", "epoch")
    for epoch in snapshots:
        components[f"oof{epoch}_predictions"] = compare_csv(
            run_a / f"oof{epoch}.csv", run_b / f"oof{epoch}.csv", "sample_id"
        )
        components[f"oof{epoch}_checkpoint"] = compare_checkpoints(
            run_a / f"checkpoint_oof{epoch}.pt", run_b / f"checkpoint_oof{epoch}.pt"
        )
    return _merge(components, "ours_oof", run_a, run_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[80, 100])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verdict = compare_ours_oof_runs(args.run_a, args.run_b, tuple(args.snapshots))
    payload = json.dumps(verdict, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
