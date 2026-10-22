"""The TU1.0 dictionary-panel Stage-2: reliability selection, the additive
endpoint with its exact per-feature ledger, and the frozen-operator panel
pipeline.

Three layers in one module: the label-blind hypergraph construction from the
audited TU1.0 relationships, the reliability-aware selection within frozen
dictionary supports (bootstrap Spearman lower bounds, redundancy-penalized
greedy picks), and the cohort pipeline that fits the additive logistic
endpoint and writes the exact feature/category ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cbm.eval.metrics import extended_binary_metrics
from cbm.eval.stage1 import PREDICTION_PREFIX
from cbm.utils import sha256_file

# ------------------------------------------------- dictionary hypergraph


def relationship_hypergraph(
    relationship_audit: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Convert audited TU relationship candidates into clinical hyperedges.

    Directions are deliberately ignored.  An ambiguous relationship contributes
    all author-audited candidate features to a set-valued concept support.  The
    support can then constrain coordinate selection without rotating named
    radiomics features into anonymous latent axes.
    """
    required = {
        "concept_category",
        "relationship_evidence_class",
        "candidate_inventory_features",
    }
    if required - set(relationship_audit):
        raise ValueError(
            f"Relationship audit lacks columns: {sorted(required - set(relationship_audit))}"
        )
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Radiomics feature inventory must be unique")
    positions = {name: index for index, name in enumerate(feature_names)}
    eligible = relationship_audit.loc[
        relationship_audit["relationship_evidence_class"].isin(
            ["author_explicit", "semantic_but_no_direction"]
        )
    ].copy()
    categories = sorted(eligible["concept_category"].dropna().astype(str).unique())
    support = np.zeros((len(categories), len(feature_names)), dtype=np.float32)
    evidence_rows: dict[str, int] = {}
    evidence_classes: dict[str, dict[str, int]] = {}
    for category_index, category in enumerate(categories):
        rows = eligible.loc[eligible["concept_category"].astype(str).eq(category)]
        candidates: set[str] = set()
        for value in rows["candidate_inventory_features"].fillna("").astype(str):
            candidates.update(name for name in value.split("|") if name)
        missing = sorted(candidates - set(positions))
        if missing:
            raise ValueError(f"Hyperedge {category!r} references unavailable features: {missing}")
        if not candidates:
            raise ValueError(f"Hyperedge {category!r} has no supported candidates")
        support[category_index, [positions[name] for name in sorted(candidates)]] = 1.0
        evidence_rows[category] = int(len(rows))
        evidence_classes[category] = {
            str(key): int(value)
            for key, value in rows["relationship_evidence_class"].value_counts().items()
        }
    if len(categories) < 2 or np.any(support.sum(axis=1) == 0):
        raise ValueError("Clinical hypergraph is degenerate")
    return support, categories, {
        "categories": categories,
        "support_sizes": {
            category: int(support[index].sum())
            for index, category in enumerate(categories)
        },
        "relationship_rows": evidence_rows,
        "evidence_classes": evidence_classes,
        "mapped_features": int(np.any(support > 0, axis=0).sum()),
        "feature_membership_histogram": {
            str(degree): int(np.sum(support.sum(axis=0) == degree))
            for degree in sorted(set(support.sum(axis=0).astype(int)))
        },
        "direction_used": False,
        "construction": "set_valued_author_audited_candidate_hypergraph",
    }


# ------------------------------------------- reliability-aware selection


@dataclass(frozen=True)
class AdditiveDictionaryAttributions:
    """Additive feature, category, and patient-logit tables for one linear model."""

    feature_contributions: pd.DataFrame
    category_subtotals: pd.DataFrame
    patient_logits: pd.DataFrame


def spearman_reliability_lower_bound(
    student: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    iterations: int,
    lower_quantile: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap paired-patient Spearman reliability lower bounds by feature.

    The two matrices must already be patient-aligned: their index and feature
    columns must match exactly. Each replicate resamples patient rows once and
    applies that same draw to student and oracle values, preserving pairings.
    The returned lower quantile is one-sided and entirely label-free.
    """
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be strictly between 0 and 0.5")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not student.index.equals(oracle.index):
        raise ValueError("Student and oracle patient indices must align exactly")
    if not student.columns.is_unique or not oracle.columns.is_unique:
        raise ValueError("Student and oracle feature columns must be unique")
    if not student.columns.equals(oracle.columns):
        raise ValueError("Student and oracle feature columns must align exactly")
    if len(student) < 2 or not len(student.columns):
        raise ValueError("Paired matrices require at least two patients and one feature")
    student_values = student.to_numpy(dtype=float)
    oracle_values = oracle.to_numpy(dtype=float)
    if not np.isfinite(student_values).all() or not np.isfinite(oracle_values).all():
        raise ValueError("Student and oracle feature matrices must be finite")

    def spearman_columns(left: np.ndarray, right: np.ndarray, axis: int) -> np.ndarray:
        left_rank = rankdata(left, axis=axis)
        right_rank = rankdata(right, axis=axis)
        left_centered = left_rank - left_rank.mean(axis=axis, keepdims=True)
        right_centered = right_rank - right_rank.mean(axis=axis, keepdims=True)
        numerator = (left_centered * right_centered).sum(axis=axis)
        denominator = np.sqrt((left_centered**2).sum(axis=axis) * (right_centered**2).sum(axis=axis))
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0.0,
        )

    raw = spearman_columns(student_values, oracle_values, axis=0)
    if not np.isfinite(raw).all():
        bad = student.columns[~np.isfinite(raw)].tolist()
        raise ValueError(f"Spearman reliability is undefined for constant feature(s): {bad}")
    draws = np.empty((iterations, student_values.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    batch_size = 128
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        indices = rng.integers(0, len(student_values), size=(stop - start, len(student_values)))
        draws[start:stop] = spearman_columns(student_values[indices], oracle_values[indices], axis=1)
    valid = np.isfinite(draws).sum(axis=0)
    if np.any(valid == 0):
        bad = student.columns[valid == 0].tolist()
        raise ValueError(f"No finite bootstrap Spearman values for feature(s): {bad}")
    return pd.DataFrame(
        {
            "raw_spearman": raw,
            "lower_bound": np.nanquantile(draws, lower_quantile, axis=0),
            "n_valid_bootstrap": valid,
        },
        index=pd.Index(student.columns, name="feature"),
    )


def additive_dictionary_attributions(
    feature_values: pd.DataFrame | np.ndarray,
    coefficients: pd.Series | np.ndarray,
    intercept: float,
    support: np.ndarray,
    categories: list[str],
    feature_names: list[str],
) -> AdditiveDictionaryAttributions:
    """Explain an additive logistic logit with equal allocation over memberships.

    ``feature_values`` must contain the values actually passed to the linear
    model (for example, train-standardized values). A feature contribution is
    ``value * coefficient``. If a feature belongs to *m* dictionary
    categories, precisely ``1/m`` of that contribution is assigned to each.
    Thus category subtotals plus the intercept reconstruct every patient logit
    without double counting. This function accepts no endpoint labels.
    """
    support_value = np.asarray(support)
    if support_value.shape != (len(categories), len(feature_names)):
        raise ValueError("Support shape does not match categories and feature names")
    if support_value.ndim != 2 or not np.isin(support_value, (0, 1)).all():
        raise ValueError("Support must be a two-dimensional binary membership matrix")
    if len(categories) != len(set(categories)) or len(feature_names) != len(set(feature_names)):
        raise ValueError("Categories and feature names must be unique")
    membership_count = support_value.sum(axis=0).astype(int)
    unmapped = [feature_names[index] for index in np.flatnonzero(membership_count == 0)]
    if unmapped:
        raise ValueError(f"Selected feature(s) are not mapped to a dictionary category: {unmapped}")

    if isinstance(feature_values, pd.DataFrame):
        missing = set(feature_names) - set(feature_values.columns)
        if missing:
            raise ValueError(f"Feature values lack columns: {sorted(missing)}")
        patient_ids = feature_values.index.to_numpy(copy=True)
        values = feature_values.loc[:, feature_names].to_numpy(dtype=float)
    else:
        values = np.asarray(feature_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise ValueError("Feature-value shape does not match feature names")
        patient_ids = np.arange(values.shape[0])
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Feature values must be a finite two-dimensional matrix")

    if isinstance(coefficients, pd.Series):
        if not coefficients.index.is_unique or set(coefficients.index) != set(feature_names):
            raise ValueError("Named coefficients must match feature names exactly")
        coefficient_values = coefficients.loc[feature_names].to_numpy(dtype=float)
    else:
        coefficient_values = np.asarray(coefficients, dtype=float)
        if coefficient_values.ndim != 1 or len(coefficient_values) != len(feature_names):
            raise ValueError("Coefficient shape does not match feature names")
    intercept_value = np.asarray(intercept, dtype=float)
    if intercept_value.ndim != 0:
        raise ValueError("Intercept must be a finite scalar")
    intercept_float = float(intercept_value)
    if not np.isfinite(coefficient_values).all() or not np.isfinite(intercept_float):
        raise ValueError("Coefficients and intercept must be finite")

    feature_contribution = values * coefficient_values
    logits = intercept_float + feature_contribution.sum(axis=1)
    feature_rows: list[dict[str, object]] = []
    category_totals = np.zeros((len(values), len(categories)), dtype=float)
    for feature_index, feature in enumerate(feature_names):
        memberships = np.flatnonzero(support_value[:, feature_index])
        allocation_fraction = 1.0 / len(memberships)
        for category_index in memberships:
            allocated = feature_contribution[:, feature_index] * allocation_fraction
            category_totals[:, category_index] += allocated
            for patient_index, patient_id in enumerate(patient_ids):
                feature_rows.append(
                    {
                        "patient_id": patient_id,
                        "feature": feature,
                        "feature_value": values[patient_index, feature_index],
                        "coefficient": coefficient_values[feature_index],
                        "membership_count": len(memberships),
                        "category": categories[category_index],
                        "allocation_fraction": allocation_fraction,
                        "feature_contribution": feature_contribution[patient_index, feature_index],
                        "allocated_contribution": allocated[patient_index],
                    }
                )
    reconstructed = intercept_float + category_totals.sum(axis=1)
    if not np.allclose(reconstructed, logits, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Allocated category subtotals do not reconstruct patient logits")
    category_rows = [
        {
            "patient_id": patient_ids[patient_index],
            "category": category,
            "category_subtotal": category_totals[patient_index, category_index],
        }
        for patient_index in range(len(values))
        for category_index, category in enumerate(categories)
    ]
    patient_logits = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "intercept": intercept_float,
            "logit": logits,
            "reconstructed_logit": reconstructed,
            "reconstruction_residual": logits - reconstructed,
        }
    )
    return AdditiveDictionaryAttributions(
        feature_contributions=pd.DataFrame(feature_rows),
        category_subtotals=pd.DataFrame(category_rows),
        patient_logits=patient_logits,
    )


def reliability_scores(
    student: pd.DataFrame,
    oracle: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Return patient-aligned student-oracle Spearman scores and oracle redundancy.

    The inputs must already be patient-pooled and aligned; no endpoint label is
    accepted or consumed.  Selection is complete-case by feature so reliability
    cannot be affected by outcome-aware imputation.
    """
    student_values = student[feature_names].replace([np.inf, -np.inf], np.nan)
    oracle_values = oracle[feature_names].replace([np.inf, -np.inf], np.nan)
    valid = [
        feature
        for feature in feature_names
        if student_values[feature].notna().all() and oracle_values[feature].notna().all()
    ]
    if len(valid) != len(feature_names):
        raise ValueError(f"Non-finite training feature(s): {sorted(set(feature_names) - set(valid))}")
    return (
        student_values.corrwith(oracle_values, method="spearman").astype(float),
        oracle_values.corr(method="spearman").abs().fillna(1.0),
    )


def select_reliability_panel(
    support: np.ndarray,
    categories: list[str],
    feature_names: list[str],
    reliability: pd.Series,
    oracle_abs_correlation: pd.DataFrame,
    *,
    k_per_category: int = 8,
    redundancy_lambda: float = 0.25,
) -> tuple[list[str], pd.DataFrame]:
    """Greedily select reliable, nonredundant features in each frozen support.

    Supports may overlap.  Category selections are made independently, then
    feature names are deduplicated in first-occurrence order for the final
    panel.  Neither labels nor outcome-model coefficients influence selection.
    """
    support = np.asarray(support)
    if support.shape != (len(categories), len(feature_names)):
        raise ValueError("Support shape does not match categories and feature names")
    if k_per_category < 1 or redundancy_lambda < 0:
        raise ValueError("k_per_category must be positive and redundancy_lambda nonnegative")
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for row, category in enumerate(categories):
        candidates = [feature_names[i] for i in np.flatnonzero(support[row])]
        chosen: list[str] = []
        while candidates and len(chosen) < k_per_category:
            def score(feature: str) -> tuple[float, str]:
                penalty = max(
                    (float(oracle_abs_correlation.loc[feature, other]) for other in chosen),
                    default=0.0,
                )
                return float(reliability[feature] - redundancy_lambda * penalty), feature

            winner = max(candidates, key=score)
            penalty = max(
                (float(oracle_abs_correlation.loc[winner, other]) for other in chosen), default=0.0
            )
            chosen.append(winner)
            candidates.remove(winner)
            rows.append(
                {
                    "category": category,
                    "selection_rank": len(chosen),
                    "feature": winner,
                    "selection_reliability": float(reliability[winner]),
                    "max_abs_oracle_corr_to_prior": penalty,
                    "objective": float(reliability[winner] - redundancy_lambda * penalty),
                }
            )
        selected.extend(chosen)
    return list(dict.fromkeys(selected)), pd.DataFrame(rows)


# --------------------------------------------------- panel cohort pipeline


@dataclass(frozen=True)
class PanelSpec:
    """Operator settings of the dictionary-panel selection.

    ``k_per_category`` caps picks per dictionary category,
    ``redundancy_lambda`` penalizes feature-correlated picks,
    ``reliability_bootstraps`` sets the bootstrap count of the Spearman
    lower bound, and ``endpoint_c``/``seed`` parametrize the additive
    logistic endpoint. Defaults are the frozen operator of record.
    """

    k_per_category: int = 8
    redundancy_lambda: float = 0.25
    reliability_bootstraps: int = 400
    reliability_lower_quantile: float = 0.025
    seed: int = 42
    endpoint_c: float = 0.1
    stability_folds: int = 5

    def provenance(self) -> dict[str, Any]:
        return {
            "k_per_category": self.k_per_category,
            "redundancy_lambda": self.redundancy_lambda,
            "reliability_bootstraps": self.reliability_bootstraps,
            "reliability_lower_quantile": self.reliability_lower_quantile,
            "seed": self.seed,
            "endpoint_model": (
                "SimpleImputer(median)+StandardScaler+LogisticRegression("
                f"C={self.endpoint_c},class_weight=balanced)"
            ),
            "stability_folds": self.stability_folds,
            "selection_uses_endpoint_labels": False,
        }


@dataclass
class Cohort:
    name: str
    train_student: pd.DataFrame
    train_oracle: pd.DataFrame
    validation_student: pd.DataFrame
    feature_names: list[str]
    support: np.ndarray
    categories: list[str]
    # Free-form provenance record of the cohort's input boundaries: path
    # strings for the hashed inputs plus small annotation values (e.g.
    # ``official_test_used``). Consumed verbatim by run_cohort's provenance
    # output; not a typed contract.
    boundary: dict[str, Any] = field(default_factory=dict)


def patient_mean(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Pool frame rows to patients as feature means, one label per patient."""
    required = {"sample_id", "patient_id", "label", *features}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("A patient has inconsistent labels")
    return (
        frame.groupby("patient_id", sort=True)
        .agg(
            label=("label", "first"),
            frames=("sample_id", "size"),
            **{feature: (feature, "mean") for feature in features},
        )
        .reset_index()
    )


def numeric_common(*frames: pd.DataFrame) -> list[str]:
    shared = set(frames[0].columns)
    for frame in frames[1:]:
        shared &= set(frame.columns)
    return sorted(
        column
        for column in shared
        if column.startswith("original_")
        and all(pd.api.types.is_numeric_dtype(frame[column]) for frame in frames)
    )


def _strip_prediction_prefix(frame: pd.DataFrame, expected: int) -> pd.DataFrame:
    prediction_columns = [
        column for column in frame.columns if column.startswith(PREDICTION_PREFIX)
    ]
    if len(prediction_columns) != expected:
        raise ValueError(f"Expected exactly {expected} '{PREDICTION_PREFIX}*' columns")
    return frame.rename(
        columns={column: column.removeprefix(PREDICTION_PREFIX) for column in prediction_columns}
    )


def _align_pair(student: pd.DataFrame, oracle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    student = student.merge(oracle[["patient_id"]], on="patient_id", how="inner", validate="one_to_one")
    oracle = oracle.set_index("patient_id").loc[student["patient_id"]].reset_index()
    return student.reset_index(drop=True), oracle


def _restricted_support(
    builder, mapping: pd.DataFrame, dictionary_names: list[str], features: list[str]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build the hypergraph at full width, then restrict columns to ``features``.

    Panel-width student tables cannot host the full dictionary (categories
    reference features outside any panel), so the support is built over the
    full inventory and subset; categories empty after subsetting are dropped.
    """
    support_full, categories_full, audit = builder(mapping, dictionary_names)
    positions = [dictionary_names.index(name) for name in features]
    support = support_full[:, positions]
    keep = [index for index in range(len(categories_full)) if support[index].sum() > 0]
    categories = [categories_full[index] for index in keep]
    return support[keep, :], categories, audit


def load_thyroidxl_cohort(
    *,
    oof_predictions: Path,
    validation_predictions: Path,
    oracle_radiomics: Path,
    dictionary_audit: Path,
    expected_features: int = 102,
) -> Cohort:
    """Load the OOF Stage-1 predictions ("student") against the true
    development radiomics ("oracle"), strict sample join.

    In this module "student" always means Stage-1 predicted radiomics and
    "oracle" always means the offline PyRadiomics ground truth.
    """
    student_train = _strip_prediction_prefix(
        pd.read_csv(oof_predictions, dtype={"patient_id": str}), expected_features
    )
    student_validation = _strip_prediction_prefix(
        pd.read_csv(validation_predictions, dtype={"patient_id": str}), expected_features
    )
    oracle_all = pd.read_csv(oracle_radiomics, dtype={"patient_id": str})
    oracle_all = oracle_all.loc[oracle_all["split"].eq("development")].copy()
    features = numeric_common(student_train, student_validation, oracle_all)
    oracle_columns = ["sample_id", "patient_id", "label", *features]
    oracle_train = student_train[["sample_id", "patient_id", "label"]].merge(
        oracle_all[oracle_columns],
        on="sample_id",
        how="left",
        suffixes=("_student", "_oracle"),
        validate="one_to_one",
        indicator=True,
    )
    if not oracle_train["_merge"].eq("both").all():
        raise ValueError("A Stage-1 OOF sample has no development oracle radiomics")
    if not oracle_train["patient_id_student"].equals(oracle_train["patient_id_oracle"]):
        raise ValueError("Student and oracle patient IDs differ after sample_id join")
    if not oracle_train["label_student"].equals(oracle_train["label_oracle"]):
        raise ValueError("Student and oracle labels differ after sample_id join")
    oracle_train = oracle_train.rename(
        columns={"patient_id_student": "patient_id", "label_student": "label"}
    ).drop(columns=["patient_id_oracle", "label_oracle", "_merge"])
    train_student = patient_mean(student_train, features)
    validation_student = patient_mean(student_validation, features)
    train_oracle = patient_mean(oracle_train, features)
    train_student, train_oracle = _align_pair(train_student, train_oracle)
    dictionary_names = [column for column in oracle_all if column.startswith("original_")]
    support, categories, support_audit = _restricted_support(
        relationship_hypergraph, pd.read_csv(dictionary_audit), dictionary_names, features
    )
    return Cohort(
        "thyroidxl",
        train_student,
        train_oracle,
        validation_student,
        features,
        support,
        categories,
        {
            "student_train": str(oof_predictions),
            "student_validation": str(validation_predictions),
            "oracle_train_only": str(oracle_radiomics),
            "dictionary": str(dictionary_audit),
            "official_test_used": False,
            "support_audit": support_audit,
        },
    )


def reliability_frame(
    student: pd.DataFrame,
    oracle: pd.DataFrame,
    features: list[str],
    spec: PanelSpec,
) -> pd.DataFrame:
    """Point Spearman, bootstrap lower bound, and raw-unit MAE per feature."""
    if not student["patient_id"].equals(oracle["patient_id"]):
        raise ValueError("Student and oracle patients are misaligned")
    point, redundancy = reliability_scores(student, oracle, features)
    lower = spearman_reliability_lower_bound(
        student.set_index("patient_id")[features],
        oracle.set_index("patient_id")[features],
        iterations=spec.reliability_bootstraps,
        lower_quantile=spec.reliability_lower_quantile,
        seed=spec.seed,
    )
    difference = (
        student.set_index("patient_id")[features] - oracle.set_index("patient_id")[features]
    )
    return pd.DataFrame(
        {
            "raw_spearman": point,
            "lower_bound": lower["lower_bound"],
            "oracle_mae": difference.abs().mean(),
        }
    ).sort_index()


def frozen_panel_endpoint_pipeline():
    """The frozen Stage-2 endpoint estimator of record (single source).

    Median-impute -> z-score -> balanced logistic (C=1, max_iter=5000,
    seed 42). Shared by the frozen panel endpoint below and by the
    concept-group endpoint in :mod:`cbm.eval.concept_groups` so both arms
    differ only in their feature sets. Changing any constant here requires
    a new preregistration.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
    )


def fit_frozen_panel_endpoint(
    patients: pd.DataFrame,
    panel: list[str],
) -> tuple[np.ndarray, float]:
    """Cross-fit the frozen patient-level panel endpoint over OOF folds.

    StratifiedKFold(5, shuffle, seed 42) over patients; each fold fits
    median-impute -> z-score -> balanced logistic (C=1, max_iter=5000,
    seed 42) on the training folds and predicts the held-out patients.
    Returns ``(oof_probabilities, max_closure_error)`` where the closure
    error is the max absolute gap between ``transform @ coef + intercept``
    and the pipeline's decision function over held-out patients — the
    exactness check of the additive ledger. Every constant here is frozen
    protocol, shared by the per-cell five-gate evaluator and the seed
    replication audit; changing any of it requires a new preregistration.
    """
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = patients["label"].to_numpy(dtype=int)
    probabilities = np.full(len(patients), np.nan)
    max_closure = 0.0
    for fit_index, heldout_index in splitter.split(patients, labels):
        fit = patients.iloc[fit_index]
        heldout = patients.iloc[heldout_index]
        pipeline = frozen_panel_endpoint_pipeline()
        pipeline.fit(fit[panel], fit["label"].astype(int))
        probabilities[heldout_index] = pipeline.predict_proba(heldout[panel])[:, 1]
        logistic = pipeline.named_steps["logisticregression"]
        transformed = pipeline[:-1].transform(heldout[panel])
        reconstructed = transformed @ logistic.coef_[0] + logistic.intercept_[0]
        max_closure = max(
            max_closure,
            float(np.abs(reconstructed - pipeline.decision_function(heldout[panel])).max()),
        )
    if np.isnan(probabilities).any():
        raise RuntimeError("endpoint left patients without an OOF probability")
    return probabilities, max_closure


def fit_additive_endpoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    *,
    spec: PanelSpec,
) -> dict[str, Any]:
    """Median-imputed, train-standardized, balanced additive logistic model."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=spec.endpoint_c, class_weight="balanced", max_iter=5000, random_state=spec.seed
        ),
    )
    model.fit(train[features], train["label"].astype(int))
    probability = model.predict_proba(validation[features])[:, 1]
    metrics = extended_binary_metrics(validation["label"].astype(int).to_numpy(), probability)
    metrics.update(
        n_features=len(features),
        n_train_patients=len(train),
        n_validation_patients=len(validation),
    )
    return {"model": model, "probability": probability, "metrics": metrics}


def exact_ledger(
    model: Any,
    validation: pd.DataFrame,
    panel: list[str],
    cohort: Cohort,
) -> dict[str, pd.DataFrame]:
    """Exact feature/category ledger for the panel arm on validation patients."""
    imputer, scaler, logistic = model.named_steps.values()
    values = imputer.transform(validation[panel])
    standardized = scaler.transform(values)
    support_columns = cohort.support[:, [cohort.feature_names.index(f) for f in panel]]
    memberships = {
        feature: {
            cohort.categories[g]
            for g in np.flatnonzero(support_columns[:, column])
        }
        for column, feature in enumerate(panel)
    }
    categories = sorted({category for members in memberships.values() for category in members})
    remapped = np.array(
        [
            [1 if category in members else 0 for members in memberships.values()]
            for category in categories
        ],
        dtype=float,
    )
    attributions = additive_dictionary_attributions(
        pd.DataFrame(standardized, columns=panel, index=validation["patient_id"].to_numpy()),
        logistic.coef_[0],
        float(logistic.intercept_[0]),
        remapped,
        categories,
        panel,
    )
    return {
        "feature_ledger": attributions.feature_contributions,
        "category_ledger": attributions.category_subtotals,
        "logit_audit": attributions.patient_logits,
    }


def panel_stability(cohort: Cohort, spec: PanelSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit reliability and selection on training-only folds (no labels used)."""
    splitter = StratifiedKFold(
        n_splits=spec.stability_folds, shuffle=True, random_state=spec.seed
    )
    selected_by_fold: list[set[str]] = []
    category_sets: dict[str, list[set[str]]] = {category: [] for category in cohort.categories}
    rows: list[dict[str, Any]] = []
    labels = cohort.train_student["label"].astype(int).to_numpy()
    for fold, (indices, _) in enumerate(
        splitter.split(cohort.train_student, labels), start=1
    ):
        student = cohort.train_student.iloc[indices].reset_index(drop=True)
        oracle = cohort.train_oracle.iloc[indices].reset_index(drop=True)
        reliability = reliability_frame(student, oracle, cohort.feature_names, spec)
        _, redundancy = reliability_scores(student, oracle, cohort.feature_names)
        selected, details = select_reliability_panel(
            cohort.support,
            cohort.categories,
            cohort.feature_names,
            reliability["lower_bound"],
            redundancy,
            k_per_category=spec.k_per_category,
            redundancy_lambda=spec.redundancy_lambda,
        )
        selected_by_fold.append(set(selected))
        for category in cohort.categories:
            feature_set = set(details.loc[details["category"].eq(category), "feature"])
            category_sets[category].append(feature_set)
        for detail in details.to_dict(orient="records"):
            rows.append(
                {
                    "fold": fold,
                    **detail,
                    "reliability_sign": int(detail["selection_reliability"] >= 0),
                }
            )
    selected_frame = pd.DataFrame(rows)
    frequency = (
        selected_frame.groupby(["category", "feature"], sort=True)
        .agg(
            selected_folds=("fold", "nunique"),
            selection_frequency=("fold", lambda x: x.nunique() / float(spec.stability_folds)),
            positive_reliability_folds=("reliability_sign", "sum"),
        )
        .reset_index()
    )
    frequency["direction_consistency"] = frequency.apply(
        lambda row: max(
            row["positive_reliability_folds"],
            row["selected_folds"] - row["positive_reliability_folds"],
        )
        / row["selected_folds"],
        axis=1,
    )
    jaccard_rows = []
    for category, sets in {"all_deduplicated": selected_by_fold, **category_sets}.items():
        pairs = [
            len(a & b) / len(a | b)
            for i, a in enumerate(sets)
            for b in sets[i + 1 :]
            if a | b
        ]
        jaccard_rows.append(
            {
                "category": category,
                "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else float("nan"),
                "n_pairs": len(pairs),
            }
        )
    return selected_frame, frequency, pd.DataFrame(jaccard_rows)


# Frozen panel operator of record (TODO §4 baseline): point reliability,
# independent per-category greedy selection, k=4, redundancy penalty 0.5.
PANEL_K_PER_CATEGORY = 4
PANEL_REDUNDANCY_LAMBDA = 0.5


def panel_arm(
    student: pd.DataFrame,
    oracle: pd.DataFrame,
    support: np.ndarray,
    categories: list[str],
    feature_names: list[str],
) -> dict[str, Any]:
    """Fit the reliability-selected panel endpoint on train-patient tables.

    ``student`` and ``oracle`` must already be patient-aligned. Returns the
    selected members, the greedy selection audit, the point reliability
    scores, the cross-fit endpoint metrics, and the additive-ledger closure
    error.
    """
    reliability, redundancy = reliability_scores(student, oracle, feature_names)
    panel, selection = select_reliability_panel(
        support,
        categories,
        feature_names,
        reliability,
        redundancy,
        k_per_category=PANEL_K_PER_CATEGORY,
        redundancy_lambda=PANEL_REDUNDANCY_LAMBDA,
    )
    probabilities, closure = fit_frozen_panel_endpoint(student, panel)
    metrics = extended_binary_metrics(student["label"].to_numpy(dtype=int), probabilities)
    return {
        "members": panel,
        "selection": selection,
        "reliability": reliability,
        "metrics": metrics,
        "max_closure_error": closure,
    }


def run_cohort(
    cohort: Cohort,
    output_dir: Path,
    spec: PanelSpec,
    *,
    panel: list[str] | None = None,
) -> dict[str, Any]:
    """Generate the selected panel, the full-inventory endpoint reference,
    and the exact attribution tables (feature/category "ledgers").

    With ``panel`` given (frozen-panel mode), the panel is frozen upstream:
    no re-selection or stability refit happens, and the endpoint model is
    fitted directly on the fixed coordinates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability = reliability_frame(
        cohort.train_student, cohort.train_oracle, cohort.feature_names, spec
    )
    _, redundancy = reliability_scores(
        cohort.train_student, cohort.train_oracle, cohort.feature_names
    )
    if panel is None:
        selected, details = select_reliability_panel(
            cohort.support,
            cohort.categories,
            cohort.feature_names,
            reliability["lower_bound"],
            redundancy,
            k_per_category=spec.k_per_category,
            redundancy_lambda=spec.redundancy_lambda,
        )
        details = details.join(reliability, on="feature")
        details["selected_deduplicated_panel"] = details["feature"].isin(selected)
    else:
        missing = sorted(set(panel) - set(cohort.feature_names))
        if missing:
            raise ValueError(f"Fixed panel features absent from the cohort: {missing}")
        selected = list(panel)
        details = reliability.loc[selected].reset_index(names="feature")
        details.insert(0, "category", "frozen_upstream_panel")
        details["selected_deduplicated_panel"] = True
        details["selection_rank"] = range(len(selected))
    generic = (
        reliability["lower_bound"]
        .sort_values(ascending=False, kind="stable")
        .index[: len(selected)]
        .tolist()
    )
    arms = {
        "true_dictionary": selected,
        "generic_top_reliability": generic,
        "raw_radiomics": cohort.feature_names,
    }
    if panel is not None:
        arms = {
            "s2_panel_frozen": selected,
            "generic_top_reliability": generic,
        }
    metric_rows, prediction_rows = [], []
    ledger = None
    for arm, features in arms.items():
        fitted = fit_additive_endpoint(
            cohort.train_student, cohort.validation_student, features, spec=spec
        )
        metric_rows.append({"arm": arm, **fitted["metrics"]})
        prediction_rows.append(
            pd.DataFrame(
                {
                    "arm": arm,
                    "patient_id": cohort.validation_student["patient_id"],
                    "label": cohort.validation_student["label"].astype(int),
                    "probability": fitted["probability"],
                }
            )
        )
        if arm in ("true_dictionary", "s2_panel_frozen"):
            ledger = exact_ledger(fitted["model"], cohort.validation_student, selected, cohort)
    if panel is None:
        fold_selections, stability, jaccard = panel_stability(cohort, spec)
        fold_selections.to_csv(output_dir / "stability_fold_selections.csv", index=False)
        stability.to_csv(output_dir / "stability_summary.csv", index=False)
        jaccard.to_csv(output_dir / "stability_jaccard.csv", index=False)

    details.to_csv(output_dir / "preliminary_panel.csv", index=False)
    pd.DataFrame({"pyradiomics_name": selected}).to_csv(
        output_dir / "panel_inventory.csv", index=False
    )
    reliability.to_csv(output_dir / "reliability_detail.csv")
    pd.DataFrame(metric_rows).to_csv(output_dir / "endpoint_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    if ledger is not None:
        ledger["feature_ledger"].to_csv(output_dir / "feature_ledger.csv", index=False)
        ledger["category_ledger"].to_csv(output_dir / "category_ledger.csv", index=False)
        ledger["logit_audit"].to_csv(output_dir / "logit_audit.csv", index=False)
    provenance = {
        "cohort": cohort.name,
        "spec": spec.provenance(),
        "panel": selected,
        "panel_mode": "frozen_upstream" if panel is not None else "selected_here",
        "n_features_student_oracle_intersection": len(cohort.feature_names),
        "n_train_patients": len(cohort.train_student),
        "n_validation_patients": len(cohort.validation_student),
        "n_train_positive": int(cohort.train_student["label"].sum()),
        "n_validation_positive": int(cohort.validation_student["label"].sum()),
        "boundary": cohort.boundary,
        "input_sha256": {
            key: sha256_file(Path(value))
            for key, value in cohort.boundary.items()
            if key in {"student_train", "student_validation", "dictionary", "stage1_metadata"}
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return provenance
