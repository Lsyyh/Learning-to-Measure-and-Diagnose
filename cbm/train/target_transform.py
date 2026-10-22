"""Train-only target transforms for Stage-1 radiomics regression.

Radiomics targets are heavy tailed (statistics below are from the motivating
prior 3D cohort, kept as design provenance): on that cohort 26 % of its 107
features have |skew| > 3, and ``glszm_LargeAreaLowGrayLevelEmphasis`` carries
nine times more variance on validation than on train. Under a plain z-score
those few features dominate the SmoothL1 gradient and the aggregated
validation MSE, which is how Stage-1 ended up worse than the mean predictor
while the training loss kept falling.

``rank_gauss`` maps each feature to a normal distribution through its train
empirical CDF, so every feature contributes comparable gradient magnitude and
outliers are bounded. ``zscore`` reproduces the previous behaviour exactly and
stays available for ablation.

Both directions are fit on train only and persisted next to the checkpoint, so
predictions can be inverted back to raw radiomics units for Stage-2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("zscore", "rank_gauss")


class TargetTransform:
    """Invertible, train-only per-feature target transform."""

    def __init__(self, method: str = "rank_gauss") -> None:
        if method not in METHODS:
            raise ValueError(f"unknown target transform {method!r}; options are {list(METHODS)}")
        self.method = method
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        # Per-feature interpolation knots (raw value -> normal score).
        self.knots_: list[tuple[np.ndarray, np.ndarray]] | None = None
        # Per-feature training constants (NaN where the feature varies). A
        # constant feature must inverse-transform to the constant itself no
        # matter what the network outputs; without this record the constant's
        # epsilon knots clamp out-of-range predictions to constant +- 0.5.
        self.constants_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, values: np.ndarray) -> TargetTransform:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("target values must be 2-D (n_samples, n_features)")
        self.constants_ = np.where(np.ptp(values, axis=0) == 0, values[0], np.nan)
        if self.method == "zscore":
            self.mean_ = values.mean(axis=0).astype(np.float32)
            std = values.std(axis=0)
            self.std_ = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            return self
        self.knots_ = [_rank_gauss_knots(values[:, index]) for index in range(values.shape[1])]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.method == "zscore":
            self._require_fitted(self.mean_)
            return ((values - self.mean_) / self.std_).astype(np.float32)
        self._require_fitted(self.knots_)
        columns = [
            np.interp(values[:, index], raw, gauss)
            for index, (raw, gauss) in enumerate(self.knots_)  # type: ignore[arg-type]
        ]
        return np.column_stack(columns).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map predictions back to raw radiomics units for Stage-2."""
        values = np.asarray(values, dtype=np.float64)
        if self.method == "zscore":
            self._require_fitted(self.mean_)
            result = (values * self.std_ + self.mean_).astype(np.float32)
        else:
            self._require_fitted(self.knots_)
            columns = [
                np.interp(values[:, index], gauss, raw)
                for index, (raw, gauss) in enumerate(self.knots_)  # type: ignore[arg-type]
            ]
            result = np.column_stack(columns).astype(np.float32)
        constants = self._resolved_constants()
        if constants is not None:
            constant_columns = ~np.isnan(constants)
            if constant_columns.any():
                result[:, constant_columns] = np.float32(constants[constant_columns])
        return result

    def _resolved_constants(self) -> np.ndarray | None:
        """Fitted constants, deriving them for legacy pickled instances.

        Instances fitted before ``constants_`` existed (every frozen run's
        ``target_transform.joblib``) lack the attribute -- unpickling skips
        ``__init__``, so the attribute reads as missing, not ``None``; their
        rank-gauss constants are recoverable from the constant-column
        epsilon-knot signature. Legacy ``zscore`` instances cannot be
        recovered this way and keep their original behaviour (no frozen run
        uses zscore).
        """
        if getattr(self, "constants_", None) is not None:
            return self.constants_
        if self.method == "rank_gauss" and self.knots_ is not None:
            constants = np.full(len(self.knots_), np.nan)
            for index, (raw, gauss) in enumerate(self.knots_):
                if (
                    raw.size == 2
                    and float(gauss[0]) == -1e-6
                    and float(gauss[1]) == 1e-6
                ):
                    constants[index] = 0.5 * (float(raw[0]) + float(raw[1]))
            self.constants_ = constants
        return self.constants_

    # ---------------------------------------------------------- persistence
    def save(self, path: Path | str) -> Path:
        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @staticmethod
    def load(path: Path | str) -> TargetTransform:
        import joblib

        return joblib.load(str(path))

    def to_metadata(self) -> dict[str, Any]:
        """JSON-serializable description recorded in ``stage1_metadata.json``."""
        metadata: dict[str, Any] = {"method": self.method}
        if self.method == "zscore" and self.mean_ is not None:
            metadata["target_mean"] = self.mean_.tolist()
            metadata["target_std"] = self.std_.tolist()  # type: ignore[union-attr]
        return metadata

    @staticmethod
    def _require_fitted(attribute: Any) -> None:
        if attribute is None:
            raise RuntimeError("TargetTransform has not been fitted")


def _rank_gauss_knots(column: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone knots mapping one feature's train values to normal scores.

    Ranks use the mid-point plotting position ``(rank - 0.5) / n``, so the
    extremes land near +-Phi^-1(1 - 0.5/n) (about +-2.9 for n = 600) instead of
    the +-5.2 saturation that ``QuantileTransformer`` produces at the training
    minimum and maximum. Ties share one knot with their averaged score, which
    keeps the mapping strictly increasing and therefore invertible by
    interpolation.
    """
    from scipy.special import ndtri

    column = np.asarray(column, dtype=np.float64)
    n = column.size
    order = np.argsort(np.argsort(column, kind="stable"), kind="stable")
    scores = ndtri((order + 0.5) / n)
    raw, inverse = np.unique(column, return_inverse=True)
    if len(raw) == 1:
        # Constant feature: map everything to 0 and invert back to the constant.
        # The knots must stay strictly increasing for np.interp in both
        # directions, hence the symmetric epsilon around the single value.
        return (
            np.array([raw[0] - 0.5, raw[0] + 0.5]),
            np.array([-1e-6, 1e-6]),
        )
    summed = np.bincount(inverse, weights=scores, minlength=len(raw))
    counts = np.bincount(inverse, minlength=len(raw))
    return raw, summed / counts


def fit_target_transform(values: np.ndarray, method: str = "rank_gauss") -> TargetTransform:
    """Fit a train-only target transform."""
    return TargetTransform(method).fit(values)
