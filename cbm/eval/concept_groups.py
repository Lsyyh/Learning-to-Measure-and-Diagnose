"""Reliability-constrained dictionary concept groups (TODO stage-1 candidate B1).

The TU1.0 dictionary fixes which named features may enter a concept group;
the Stage-1 out-of-fold prediction reliability decides which of them
actually do. The endpoint is one shared coefficient per named feature under
a reliability-weighted L1 penalty (less reliable members are penalized
harder), and every patient logit decomposes exactly into per-category
subtotals through equal-split membership allocation, reusing the audited
ledger of :mod:`cbm.eval.panel`.

Fixed minimal-version rules (recorded in the dev candidate note before any
run): non-finite or negative reliability never rewards a feature — such
features are excluded outright; members must reach point Spearman
reliability ``MIN_MEMBER_RELIABILITY``; within a category a candidate whose
student-prediction correlation with an already-kept member exceeds
``REDUNDANCY_THRESHOLD`` is dropped in favor of the more reliable one;
categories left without members are reported as a coverage shortfall and
are never backfilled from outside the dictionary. Selection and penalty
weights are label-free; only the logistic fit sees labels.

Endpoint estimator (dev revision v2, 2026-09-05): the concept endpoint
defaults to the SAME frozen estimator as the panel baseline of record
(``frozen_panel_endpoint_pipeline``), so the concept arm differs from the
panel arm only in its reliability-gated member set. The a priori
reliability-weighted L1 (``PENALTY_LAMBDA``) was evaluated on the incumbent
chain and hurt both discrimination and calibration without producing exact
zeros; it is retained as an explicitly requested diagnostic path
(``penalty_lambda=...``), not as the endpoint of record.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from cbm.eval.metrics import extended_binary_metrics
from cbm.eval.panel import (
    additive_dictionary_attributions,
    frozen_panel_endpoint_pipeline,
    reliability_scores,
)

# ------------------------------------------------- fixed minimal-version rules

MIN_MEMBER_RELIABILITY = 0.3
REDUNDANCY_THRESHOLD = 0.9
# A priori weighted-L1 strength; diagnostic-only after the dev finding that
# it cost ~0.02 OOF AUC and 2x ECE on the incumbent chain without sparsifying.
PENALTY_LAMBDA = 0.1
FIT_STEPS = 2000
FIT_LEARNING_RATE = 0.05
ENDPOINT_FOLDS = 5
ENDPOINT_SEED = 42


def provenance() -> dict[str, Any]:
    """The fixed operator constants of the concept-group endpoint."""
    return {
        "min_member_reliability": MIN_MEMBER_RELIABILITY,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "endpoint_model": (
            "shared frozen estimator: median-impute + train-z-score + "
            "balanced logistic C=1 max_iter=5000 seed 42, "
            "StratifiedKFold(5, shuffle, seed 42)"
        ),
        "weighted_l1_diagnostic_only": {
            "penalty_lambda": PENALTY_LAMBDA,
            "fit_steps": FIT_STEPS,
            "fit_learning_rate": FIT_LEARNING_RATE,
        },
        "selection_uses_endpoint_labels": False,
        "redundancy_computed_on": "stage1_oof_predictions",
    }


# ------------------------------------------------------ member selection


def reliability_gate(reliability: pd.Series, *, min_reliability: float = MIN_MEMBER_RELIABILITY) -> pd.Series:
    """Features whose reliability is finite and at or above the gate."""
    return reliability.notna() & (reliability >= float(min_reliability))


def select_dictionary_members(
    support: np.ndarray,
    categories: list[str],
    feature_names: list[str],
    reliability: pd.Series,
    student_correlation: pd.DataFrame,
    *,
    min_reliability: float = MIN_MEMBER_RELIABILITY,
    redundancy_threshold: float = REDUNDANCY_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    """Select reliable, mutually non-redundant dictionary members per category.

    Accepts no labels. Candidates are the category's dictionary support
    filtered by the reliability gate; within a category they are ordered by
    reliability (descending, stable) and a candidate is dropped when its
    absolute student-prediction Spearman correlation with an already-kept
    member exceeds ``redundancy_threshold``. Members of several categories
    are kept once and counted for each membership.
    """
    support = np.asarray(support)
    if support.shape != (len(categories), len(feature_names)):
        raise ValueError("Support shape does not match categories and feature names")
    if not student_correlation.index.equals(student_correlation.columns):
        raise ValueError("Student correlation must be a square feature matrix")
    passing = reliability_gate(reliability, min_reliability=min_reliability)
    members: list[str] = []
    rows: list[dict[str, Any]] = []
    for row, category in enumerate(categories):
        candidates = [
            feature_names[index] for index in np.flatnonzero(support[row])
        ]
        ordered = sorted(candidates, key=lambda name: (-float(reliability[name]), name))
        kept: list[str] = []
        for feature in ordered:
            if not bool(passing.loc[feature]):
                rows.append(
                    {
                        "category": category,
                        "feature": feature,
                        "reliability": float(reliability[feature]),
                        "kept": False,
                        "reason": "below_reliability_gate",
                    }
                )
                continue
            blocker = next(
                (
                    other
                    for other in kept
                    if abs(float(student_correlation.loc[feature, other]))
                    > redundancy_threshold
                ),
                None,
            )
            if blocker is not None:
                rows.append(
                    {
                        "category": category,
                        "feature": feature,
                        "reliability": float(reliability[feature]),
                        "kept": False,
                        "reason": f"redundant_with_{blocker}",
                    }
                )
                continue
            kept.append(feature)
            members.append(feature)
            rows.append(
                {
                    "category": category,
                    "feature": feature,
                    "reliability": float(reliability[feature]),
                    "kept": True,
                    "reason": "",
                }
            )
    selected = list(dict.fromkeys(members))
    if not selected:
        raise ValueError("No dictionary member passed the reliability gate")
    return selected, pd.DataFrame(rows)


def penalty_weights(reliability: pd.Series, members: list[str]) -> np.ndarray:
    """L1 penalty weights: 1 + (1 - rho), so less reliable members shrink harder.

    Reliability is clipped to [0, 1]; the rule never rewards negative or
    undefined values because such features are already excluded upstream.
    """
    rho = np.clip(
        reliability.loc[members].to_numpy(dtype=float), 0.0, 1.0
    )
    return 1.0 + (1.0 - rho)


def select_reliability_redundancy_features(
    feature_names: list[str],
    reliability: pd.Series,
    student_correlation: pd.DataFrame,
    *,
    width: int,
    redundancy_threshold: float = REDUNDANCY_THRESHOLD,
) -> list[str]:
    """Dictionary-free control selection: reliability order + redundancy pruning.

    Greedy flat selection over the full inventory (no dictionary supports, no
    category organization): candidates in descending point reliability, a
    candidate is skipped when its absolute student-prediction Spearman with an
    already-kept feature exceeds ``redundancy_threshold``, stopping at
    ``width``. Accepts no labels; only the candidate range and grouping differ
    from the concept-member selection.
    """
    if width < 1:
        raise ValueError("width must be positive")
    ordered = sorted(feature_names, key=lambda name: (-float(reliability[name]), name))
    kept: list[str] = []
    for feature in ordered:
        if len(kept) >= width:
            break
        if any(
            abs(float(student_correlation.loc[feature, other])) > redundancy_threshold
            for other in kept
        ):
            continue
        kept.append(feature)
    return kept


# ------------------------------------------------- weighted-L1 logistic fit


def fit_weighted_l1_logistic(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    penalty_lambda: float = PENALTY_LAMBDA,
    steps: int = FIT_STEPS,
    learning_rate: float = FIT_LEARNING_RATE,
) -> tuple[np.ndarray, float]:
    """Class-balanced logistic regression with a per-feature weighted L1.

    Full-batch Adam from a zero initialization on CPU (float64) — the
    objective is convex and the construction carries no randomness, so the
    fit is deterministic. Returns ``(beta, bias)``.
    """
    import torch

    values = torch.as_tensor(np.asarray(standardized, dtype=np.float64))
    target = torch.as_tensor(np.asarray(labels, dtype=np.float64))
    penalty = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    if values.ndim != 2 or target.ndim != 1 or len(values) != len(target):
        raise ValueError("Design matrix and labels must be aligned")
    if penalty.shape != (values.shape[1],):
        raise ValueError("Penalty weights must match the feature columns")

    positives = target.sum()
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Weighted logistic requires both classes")
    sample_weight = torch.where(
        target > 0.5, len(target) / (2.0 * positives), len(target) / (2.0 * negatives)
    )

    beta = torch.zeros(values.shape[1], dtype=torch.float64)
    beta.requires_grad_(True)
    bias = torch.zeros(1, dtype=torch.float64)
    bias.requires_grad_(True)
    optimizer = torch.optim.Adam([beta, bias], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        logits = values @ beta + bias
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        objective = (losses * sample_weight).mean() + float(penalty_lambda) * (
            penalty * beta.abs()
        ).sum()
        objective.backward()
        optimizer.step()
    return beta.detach().numpy().copy(), float(bias.detach().item())


def _prepare_standardized(
    fit_frame: pd.DataFrame, apply_frame: pd.DataFrame, members: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute and z-score ``apply_frame`` with statistics from ``fit_frame``."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit_values = imputer.fit_transform(fit_frame[members])
    apply_values = imputer.transform(apply_frame[members])
    return scaler.fit_transform(fit_values), scaler.transform(apply_values)


def crossfit_concept_endpoint(
    patients: pd.DataFrame,
    members: list[str],
    reliability: pd.Series,
    *,
    penalty_lambda: float | None = None,
    folds: int = ENDPOINT_FOLDS,
    seed: int = ENDPOINT_SEED,
) -> dict[str, Any]:
    """Out-of-fold concept-group endpoint probabilities over train patients.

    StratifiedKFold over patients (the same honest-OOF convention as the
    frozen panel endpoint); each fold refits imputation/scaling and the
    logistic on its training folds. The member set itself is fixed upstream
    by the label-free reliability rule. With ``penalty_lambda=None`` (the
    default) each fold uses the shared frozen estimator so the concept arm
    is directly comparable to the panel arm; a float selects the
    diagnostic-only reliability-weighted L1 path.
    """
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    labels = patients["label"].to_numpy(dtype=int)
    probabilities = np.full(len(patients), np.nan)
    fold_betas: list[np.ndarray] = []
    for fit_index, heldout_index in splitter.split(patients, labels):
        fit = patients.iloc[fit_index]
        heldout = patients.iloc[heldout_index]
        if penalty_lambda is None:
            pipeline = frozen_panel_endpoint_pipeline()
            pipeline.fit(fit[members], labels[fit_index])
            probabilities[heldout_index] = pipeline.predict_proba(heldout[members])[:, 1]
            logistic = pipeline.named_steps["logisticregression"]
            fold_betas.append(logistic.coef_[0].copy())
        else:
            weights = penalty_weights(reliability, members)
            fit_values, heldout_values = _prepare_standardized(fit, heldout, members)
            beta, bias = fit_weighted_l1_logistic(
                fit_values, labels[fit_index], weights, penalty_lambda=penalty_lambda
            )
            logits = heldout_values @ beta + bias
            probabilities[heldout_index] = 1.0 / (1.0 + np.exp(-logits))
            fold_betas.append(beta.copy())
    if np.isnan(probabilities).any():
        raise RuntimeError("concept endpoint left patients without a probability")
    return {
        "oof_probability": probabilities,
        "fold_coefficients": np.stack(fold_betas),
    }


def concept_group_ledger(
    patients: pd.DataFrame,
    members: list[str],
    reliability: pd.Series,
    support: np.ndarray,
    categories: list[str],
    feature_names: list[str],
    *,
    penalty_lambda: float | None = None,
) -> dict[str, pd.DataFrame | float | np.ndarray]:
    """Full-train concept fit with the exact category ledger and member vectors.

    Returns the standardized member values (``concept_members``), the feature
    contributions, the equal-split category subtotals, the patient logits,
    the fitted ``(beta, bias)``, and the coverage report. Category
    contributions are model risk allocations, not physician scores. The
    default estimator matches :func:`crossfit_concept_endpoint`.
    """
    positions = [feature_names.index(name) for name in members]
    support_members = np.asarray(support)[:, positions]
    # Label-free reliability record, reported whichever estimator is used.
    weights = penalty_weights(reliability, members)
    if penalty_lambda is None:
        pipeline = frozen_panel_endpoint_pipeline()
        pipeline.fit(patients[members], patients["label"].to_numpy(dtype=int))
        imputer, scaler, logistic = pipeline.named_steps.values()
        fit_values = scaler.transform(imputer.transform(patients[members]))
        beta = logistic.coef_[0]
        bias = float(logistic.intercept_[0])
    else:
        fit_values, _ = _prepare_standardized(patients, patients, members)
        beta, bias = fit_weighted_l1_logistic(
            fit_values, patients["label"].to_numpy(dtype=int), weights,
            penalty_lambda=penalty_lambda,
        )
    attributions = additive_dictionary_attributions(
        pd.DataFrame(fit_values, columns=members, index=patients["patient_id"].to_numpy()),
        beta,
        bias,
        support_members,
        categories,
        members,
    )
    kept_support = support_members.sum(axis=1)
    coverage = pd.DataFrame(
        {
            "category": categories,
            "dictionary_support": np.asarray(support).sum(axis=1).astype(int),
            "kept_members": kept_support.astype(int),
        }
    )
    return {
        "concept_members": pd.DataFrame(
            fit_values, columns=members, index=patients["patient_id"].to_numpy()
        ),
        "feature_contributions": attributions.feature_contributions,
        "category_subtotals": attributions.category_subtotals,
        "patient_logits": attributions.patient_logits,
        "beta": beta,
        "bias": bias,
        "penalty_weights": weights,
        "coverage": coverage,
    }


def run_concept_group_endpoint(
    train_student: pd.DataFrame,
    train_oracle: pd.DataFrame,
    support: np.ndarray,
    categories: list[str],
    feature_names: list[str],
    *,
    min_reliability: float = MIN_MEMBER_RELIABILITY,
    penalty_lambda: float | None = None,
) -> dict[str, Any]:
    """Full B1 pipeline on one Stage-1 source: selection, OOF endpoint, ledger.

    ``train_student``/``train_oracle`` are patient-pooled Stage-1 OOF
    predictions and offline oracle radiomics over the same training patients
    (aligned ``patient_id``); the oracle is used only to estimate prediction
    reliability, never as endpoint input. Returns the selected members, the
    honest-OOF endpoint metrics, and the exact concept tables.
    """
    if not train_student["patient_id"].equals(train_oracle["patient_id"]):
        raise ValueError("Student and oracle patients are misaligned")
    reliability, _ = reliability_scores(train_student, train_oracle, feature_names)
    student_correlation = (
        train_student[feature_names].corr(method="spearman").abs()
    )
    members, audit = select_dictionary_members(
        support,
        categories,
        feature_names,
        reliability,
        student_correlation,
        min_reliability=min_reliability,
    )
    crossfit = crossfit_concept_endpoint(
        train_student, members, reliability, penalty_lambda=penalty_lambda
    )
    metrics = extended_binary_metrics(
        train_student["label"].to_numpy(dtype=int), crossfit["oof_probability"]
    )
    ledger = concept_group_ledger(
        train_student,
        members,
        reliability,
        support,
        categories,
        feature_names,
        penalty_lambda=penalty_lambda,
    )
    shortfall = ledger["coverage"].loc[ledger["coverage"]["kept_members"].eq(0), "category"]
    return {
        "members": members,
        "selection_audit": audit,
        "reliability": reliability,
        "oof_probability": crossfit["oof_probability"],
        "fold_coefficients": crossfit["fold_coefficients"],
        "metrics": metrics,
        "ledger": ledger,
        "coverage_shortfall_categories": shortfall.tolist(),
        "provenance": provenance(),
    }
