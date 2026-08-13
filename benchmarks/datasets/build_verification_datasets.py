"""Rebuild the public Core-Norm verification datasets deterministically.

The canonical scikit-learn datasets are the same source datasets named in the
archived Phase-1 result tables. Synthetic files are public replay fixtures for
the same distribution families; they are not claimed to be byte-identical to
the exploratory matrices, which were not archived during the first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)

SEED = 20260813
N = 800
P = 10
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "datasets" / "files"
OUT.mkdir(parents=True, exist_ok=True)


def _robust_latent(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    q25 = np.nanquantile(X, 0.25, axis=0)
    q75 = np.nanquantile(X, 0.75, axis=0)
    scale = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
    return np.clip((X - med) / scale, -4, 4)


def _target(
    X: np.ndarray,
    rng: np.random.Generator,
    task: str,
) -> np.ndarray:
    Z = _robust_latent(X)
    if task == "classification":
        weights = np.array([1.15, -0.9, 0.65, 0.45, -0.35, 0.3, -0.25, 0.2, 0.15, -0.1])
        weights = weights[: Z.shape[1]]
        score = (
            Z[:, : len(weights)] @ weights + 0.45 * np.sin(Z[:, 0]) + rng.normal(0, 0.55, len(Z))
        )
        return (score > np.median(score)).astype(int)

    weights = np.array([2.0, -1.4, 1.1, 0.8, -0.65, 0.55, -0.45, 0.35, 0.25, -0.15])
    weights = weights[: Z.shape[1]]
    return (
        Z[:, : len(weights)] @ weights
        + 0.8 * np.sin(Z[:, 0])
        + 0.35 * Z[:, 1] ** 2
        + rng.normal(0, 0.45, len(Z))
    )


def _synthetic(
    family: str,
    seed: int,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if family == "gaussian":
        X = rng.normal(size=(N, P))
    elif family == "correlated":
        covariance = np.fromfunction(
            lambda i, j: 0.72 ** np.abs(i - j),
            (P, P),
        )
        X = rng.multivariate_normal(np.zeros(P), covariance, size=N)
    elif family == "lognormal":
        X = rng.lognormal(0, 1.05, size=(N, P))
    elif family == "student_t":
        X = rng.standard_t(2.5, size=(N, P))
    elif family == "laplace":
        X = rng.laplace(size=(N, P))
    elif family == "bimodal":
        centers = rng.choice([-2.2, 2.2], size=(N, P))
        X = rng.normal(centers, 0.75)
    elif family == "zero_inflated":
        X = rng.normal(size=(N, P))
        X[rng.random((N, P)) < 0.68] = 0
    elif family == "bounded_beta":
        X = rng.beta(2, 5, size=(N, P)) * 8 - 2
    elif family == "mixed":
        generators = [
            lambda: rng.normal(size=N),
            lambda: rng.lognormal(0, 0.9, size=N),
            lambda: rng.standard_t(3, size=N),
            lambda: rng.laplace(size=N),
            lambda: rng.beta(2, 5, size=N) * 5,
        ]
        X = np.column_stack([generators[j % len(generators)]() for j in range(P)])
    elif family == "heteroscedastic":
        base = rng.normal(size=(N, P))
        row_scale = (np.abs(base[:, 0]) + 0.35)[:, None]
        column_scale = np.linspace(0.5, 2.0, P)[None, :]
        X = base * row_scale * column_scale
    else:
        raise ValueError(f"Unknown family: {family}")

    return X, _target(X, rng, task)


def _write(
    task: str,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
) -> Path:
    frame = pd.DataFrame(X, columns=columns)
    frame["target"] = y
    path = OUT / f"{task}_{name}.csv"
    frame.to_csv(path, index=False, float_format="%.12g")
    return path


def _clean_feature_names(raw_names: object, n_features: int) -> list[str]:
    names = raw_names if raw_names is not None else [f"x{i + 1}" for i in range(n_features)]
    cleaned = []
    for value in names:
        name = str(value).strip().lower()
        for old, new in ((" ", "_"), ("(", ""), (")", ""), ("/", "_"), ("-", "_")):
            name = name.replace(old, new)
        cleaned.append(name)
    return cleaned


def main() -> None:
    families = [
        "gaussian",
        "correlated",
        "lognormal",
        "student_t",
        "laplace",
        "bimodal",
        "zero_inflated",
        "bounded_beta",
        "mixed",
        "heteroscedastic",
    ]

    for task in ("classification", "regression"):
        seed_offset = 0 if task == "classification" else 1000
        for index, family in enumerate(families):
            X, y = _synthetic(family, SEED + index + seed_offset, task)
            _write(task, family, X, y, [f"x{j + 1}" for j in range(P)])

    real_datasets = [
        ("classification", "iris", load_iris),
        ("classification", "wine", load_wine),
        ("classification", "breast", load_breast_cancer),
        ("classification", "digits", load_digits),
        ("regression", "diabetes", load_diabetes),
    ]
    for task, name, loader in real_datasets:
        bunch = loader()
        data = np.asarray(bunch.data, dtype=float)
        target = np.asarray(bunch.target)
        feature_names = _clean_feature_names(
            getattr(bunch, "feature_names", None),
            data.shape[1],
        )
        _write(task, name, data, target, feature_names)


if __name__ == "__main__":
    main()
