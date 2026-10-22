"""Endpoint metric blocks shared by evaluators and the report builder.

``binary_metrics`` is the seven-key block every locked-test opening and the
final-result manifest report; ``extended_binary_metrics`` is the wider
two-stage evaluate schema (adds ECE). They live here so the exact metric
definitions cannot drift between one-shot evaluators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

#: The seven report-matrix metric keys, in canonical order.
METRIC_KEYS = (
    "auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "sensitivity_0.5",
    "specificity_0.5",
    "balanced_accuracy_0.5",
)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Seven-key endpoint block; threshold metrics are cut at 0.5."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    positive, negative = labels == 1, labels == 0
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "sensitivity_0.5": float((predicted[positive] == 1).mean()),
        "specificity_0.5": float((predicted[negative] == 0).mean()),
        "balanced_accuracy_0.5": float(
            0.5 * (predicted[positive] == 1).mean() + 0.5 * (predicted[negative] == 0).mean()
        ),
    }


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Equal-width binned ECE: sum of bin |accuracy − confidence| weighted by size."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(
        np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1
    )
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        member = indices == bin_index
        if not member.any():
            continue
        accuracy = float(labels[member].mean())
        confidence = float(probabilities[member].mean())
        error += (member.sum() / total) * abs(accuracy - confidence)
    return error


def extended_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """The frozen two-stage evaluate schema.

    Keys: ``auc``, ``balanced_accuracy`` (at 0.5), ``average_precision``,
    ``brier_score``, ``log_loss``, ``ece_10``. Kept distinct from
    :func:`binary_metrics` because downstream tables consume these key names.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, probabilities >= 0.5)
        ),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece_10": expected_calibration_error(labels, probabilities, bins=10),
    }


def pool_by_patient(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Mean-pool frame rows per patient; one label per patient (checked).

    Returns a table indexed by ``patient_id`` with a leading ``label`` column
    followed by the pooled features.
    """
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    grouped = frame.groupby("patient_id", sort=True)
    table = grouped[features].mean()
    labels = grouped["label"].first()
    if labels.isna().any() or frame.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("a patient carries conflicting or missing labels")
    table.insert(0, "label", labels.astype(int))
    return table
