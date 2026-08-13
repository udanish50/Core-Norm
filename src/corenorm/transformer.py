"""Core-Norm: bounded, asymmetric, invertible numeric feature preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class CoreNorm(BaseEstimator, TransformerMixin):
    """Core-Norm transformer.

    Core-Norm maps every input feature to two bounded coordinates: a central
    coordinate in [-1, 1] and an invertible residual coordinate in (-1, 1).
    The fitted representation uses a median, separate lower/upper quartile
    scales, and a capped empirical transition threshold.

    Parameters
    ----------
    q : float, default=0.95
        Quantile of ``abs(u)`` used to estimate the candidate transition
        threshold before clipping.
    tau_min : float, default=1.5
        Minimum standardized transition threshold.
    tau_max : float, default=3.0
        Maximum standardized transition threshold.
    eps : float, default=1e-8
        Positive numerical stabilizer added to fitted side scales.

    Notes
    -----
    * NaN values are preserved. Core-Norm does not perform imputation.
    * Infinite values are rejected.
    * A p-feature matrix is transformed to 2p features.
    """

    def __init__(
        self, q: float = 0.95, tau_min: float = 1.5, tau_max: float = 3.0, eps: float = 1e-8
    ):
        self.q = q
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.eps = eps

    def _validate_params(self) -> None:
        if not (0.5 < float(self.q) < 1.0):
            raise ValueError("q must be strictly between 0.5 and 1.0.")
        if float(self.tau_min) <= 0:
            raise ValueError("tau_min must be positive.")
        if float(self.tau_max) < float(self.tau_min):
            raise ValueError("tau_max must be greater than or equal to tau_min.")
        if float(self.eps) <= 0:
            raise ValueError("eps must be positive.")

    @staticmethod
    def _as_2d_float(X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be a 2-D array; received shape {arr.shape!r}.")
        if arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if np.isinf(arr).any():
            raise ValueError("Core-Norm does not accept +inf or -inf values.")
        return arr

    def _require_fitted(self) -> None:
        if not hasattr(self, "median_"):
            raise NotFittedError("This CoreNorm instance is not fitted yet.")

    def fit(self, X: Any, y: Any = None) -> CoreNorm:
        """Fit Core-Norm statistics from training features only."""
        self._validate_params()
        arr = self._as_2d_float(X)
        if np.any(np.all(np.isnan(arr), axis=0)):
            bad = np.flatnonzero(np.all(np.isnan(arr), axis=0)).tolist()
            raise ValueError(f"All-NaN feature columns cannot be fitted: {bad}.")

        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)

        self.median_ = np.nanmedian(arr, axis=0)
        q25 = np.nanquantile(arr, 0.25, axis=0)
        q75 = np.nanquantile(arr, 0.75, axis=0)

        mad = np.nanmedian(np.abs(arr - self.median_), axis=0)
        fallback = np.where(mad > self.eps, mad, 1.0)
        left = self.median_ - q25
        right = q75 - self.median_
        self.scale_lower_ = np.where(left > self.eps, left, fallback)
        self.scale_upper_ = np.where(right > self.eps, right, fallback)

        u = self._standardize(arr)
        tau = np.nanquantile(np.abs(u), self.q, axis=0)
        tau = np.nan_to_num(
            tau,
            nan=float(self.tau_min),
            posinf=float(self.tau_max),
            neginf=float(self.tau_min),
        )
        self.tau_ = np.clip(tau, self.tau_min, self.tau_max)
        return self

    def _check_feature_count(self, arr: np.ndarray) -> None:
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {arr.shape[1]}.")

    def _standardize(self, arr: np.ndarray) -> np.ndarray:
        return np.where(
            arr < self.median_,
            (arr - self.median_) / (self.scale_lower_ + self.eps),
            (arr - self.median_) / (self.scale_upper_ + self.eps),
        )

    def transform(self, X: Any) -> np.ndarray:
        """Transform X to the 2p-dimensional Core-Norm representation."""
        self._require_fitted()
        arr = self._as_2d_float(X)
        self._check_feature_count(arr)
        u = self._standardize(arr)

        central = np.clip(u / self.tau_, -1.0, 1.0)
        excess = np.maximum(np.abs(u) - self.tau_, 0.0)
        log_excess = np.log1p(excess)
        residual = np.sign(u) * log_excess / (1.0 + log_excess)
        return np.concatenate((central, residual), axis=1)

    def inverse_transform(self, Z: Any) -> np.ndarray:
        """Reconstruct original-scale features from a valid Core-Norm encoding."""
        self._require_fitted()
        z = self._as_2d_float(Z)
        expected = 2 * self.n_features_in_
        if z.shape[1] != expected:
            raise ValueError(f"Expected {expected} encoded columns, received {z.shape[1]}.")

        p = self.n_features_in_
        central = z[:, :p]
        residual = z[:, p:]

        finite_c = central[np.isfinite(central)]
        finite_r = residual[np.isfinite(residual)]
        tol = 1e-12
        if finite_c.size and np.max(np.abs(finite_c)) > 1.0 + tol:
            raise ValueError("Central coordinates must lie in [-1, 1].")
        if finite_r.size and np.max(np.abs(finite_r)) >= 1.0:
            raise ValueError("Residual coordinates must lie strictly inside (-1, 1).")

        u = np.full_like(central, np.nan, dtype=np.float64)
        central_mask = (residual == 0.0) & np.isfinite(central)
        u[central_mask] = (self.tau_ * central)[central_mask]

        tail_mask = (residual != 0.0) & np.isfinite(residual)
        if np.any(tail_mask):
            e = np.abs(residual[tail_mask])
            a = e / (1.0 - e)
            distance = np.expm1(a)
            tau_grid = np.broadcast_to(self.tau_, residual.shape)
            u[tail_mask] = np.sign(residual[tail_mask]) * (tau_grid[tail_mask] + distance)

        out = np.where(
            u < 0.0,
            self.median_ + u * (self.scale_lower_ + self.eps),
            self.median_ + u * (self.scale_upper_ + self.eps),
        )
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return names for the central and residual coordinates."""
        self._require_fitted()
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                names = [str(v) for v in self.feature_names_in_]
            else:
                names = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            names = [str(v) for v in input_features]
            if len(names) != self.n_features_in_:
                raise ValueError("input_features has the wrong length.")
        return np.asarray(
            [f"{n}__central" for n in names] + [f"{n}__residual" for n in names],
            dtype=object,
        )

    def to_state(self) -> dict[str, Any]:
        """Return a JSON-serializable fitted-state dictionary."""
        self._require_fitted()
        return {
            "schema_version": 1,
            "method": "Core-Norm",
            "params": {
                "q": float(self.q),
                "tau_min": float(self.tau_min),
                "tau_max": float(self.tau_max),
                "eps": float(self.eps),
            },
            "n_features_in": int(self.n_features_in_),
            "feature_names_in": (
                [str(v) for v in self.feature_names_in_]
                if hasattr(self, "feature_names_in_")
                else None
            ),
            "median": self.median_.tolist(),
            "scale_lower": self.scale_lower_.tolist(),
            "scale_upper": self.scale_upper_.tolist(),
            "tau": self.tau_.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CoreNorm:
        """Restore a fitted CoreNorm object from ``to_state`` output."""
        if state.get("method") != "Core-Norm" or state.get("schema_version") != 1:
            raise ValueError("Unsupported or invalid Core-Norm state.")
        obj = cls(**state["params"])
        obj.n_features_in_ = int(state["n_features_in"])
        obj.median_ = np.asarray(state["median"], dtype=np.float64)
        obj.scale_lower_ = np.asarray(state["scale_lower"], dtype=np.float64)
        obj.scale_upper_ = np.asarray(state["scale_upper"], dtype=np.float64)
        obj.tau_ = np.asarray(state["tau"], dtype=np.float64)
        names = state.get("feature_names_in")
        if names is not None:
            obj.feature_names_in_ = np.asarray(names, dtype=object)
        for a in (obj.median_, obj.scale_lower_, obj.scale_upper_, obj.tau_):
            if a.shape != (obj.n_features_in_,):
                raise ValueError("Invalid Core-Norm state dimensions.")
        obj._validate_params()
        return obj

    def save(self, path: str | Path) -> Path:
        """Persist fitted state as human-readable JSON."""
        target = Path(path)
        target.write_text(json.dumps(self.to_state(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> CoreNorm:
        """Load fitted state created by ``save``."""
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))
