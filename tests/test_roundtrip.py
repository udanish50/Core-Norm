import numpy as np

from corenorm import CoreNorm


def test_roundtrip_diverse_distributions():
    rng = np.random.default_rng(20260813)
    n = 5000
    datasets = [
        rng.normal(10, 3, (n, 5)),
        rng.lognormal(2, 1.2, (n, 5)),
        rng.standard_t(2, (n, 5)) * 5 + 2,
        rng.exponential(5, (n, 5)) - 2,
        np.concatenate([rng.normal(-3, 1, (n // 2, 5)), rng.normal(4, 1.5, (n // 2, 5))]),
    ]
    for X in datasets:
        scaler = CoreNorm().fit(X[:3000])
        Z = scaler.transform(X)
        X_hat = scaler.inverse_transform(Z)
        assert np.allclose(X, X_hat, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_extreme_values_roundtrip():
    train = np.arange(100.0).reshape(-1, 1)
    values = np.array([[-1e100], [-1e12], [-1000.0], [0.0], [50.0], [1000.0], [1e12], [1e100]])
    scaler = CoreNorm().fit(train)
    recovered = scaler.inverse_transform(scaler.transform(values))
    assert np.allclose(values, recovered, rtol=5e-12, atol=1e-10)
