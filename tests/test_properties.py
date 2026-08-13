import numpy as np
import pytest

from corenorm import CoreNorm


def test_bounded_encoding():
    rng = np.random.default_rng(7)
    X = rng.standard_t(1.5, size=(3000, 4)) * 100
    Z = CoreNorm().fit(X).transform(X)
    p = X.shape[1]
    assert np.nanmax(np.abs(Z[:, :p])) <= 1.0
    assert np.nanmax(np.abs(Z[:, p:])) < 1.0


def test_positive_affine_invariance_up_to_eps():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(2000, 3))
    a, b = 17.0, -300.0
    z1 = CoreNorm(eps=1e-12).fit_transform(X)
    z2 = CoreNorm(eps=1e-12).fit_transform(a * X + b)
    assert np.allclose(z1, z2, atol=1e-10, rtol=1e-10)


def test_constant_feature_and_nan_preservation():
    X = np.array([[5.0, 1.0], [5.0, np.nan], [5.0, 4.0], [5.0, 7.0]])
    scaler = CoreNorm().fit(X)
    Z = scaler.transform(X)
    X_hat = scaler.inverse_transform(Z)
    assert np.allclose(X, X_hat, equal_nan=True)
    assert np.all(Z[:, 0] == 0.0)


def test_reject_infinity():
    with pytest.raises(ValueError):
        CoreNorm().fit(np.array([[1.0], [np.inf]]))


def test_reject_all_nan_feature():
    with pytest.raises(ValueError):
        CoreNorm().fit(np.array([[1.0, np.nan], [2.0, np.nan]]))
