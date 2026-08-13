# Core-Norm

**Bounded, asymmetric, invertible preprocessing for numeric machine-learning features.**

[![CI](https://github.com/udanish50/Core-Norm/actions/workflows/ci.yml/badge.svg)](https://github.com/udanish50/Core-Norm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Research software](https://img.shields.io/badge/status-research%20software-5f3bb5.svg)](docs/limitations.md)

Core-Norm is an experimental feature-preprocessing method designed around a specific tension in conventional scaling: suppressing the numerical leverage of extreme observations often discards or distorts information about *how extreme* those observations were. Core-Norm separates those two jobs into two bounded coordinates per numeric feature while retaining an exact inverse for valid encodings.

> **Research status.** Core-Norm is research software. The included benchmark is preliminary and is not a claim of universal superiority or peer-reviewed state of the art. See [Limitations](docs/limitations.md).

## Why Core-Norm?

For each fitted numeric feature, Core-Norm uses a median, separate lower/upper quartile scales, and a capped empirical transition threshold. It then produces:

- a **central coordinate** in `[-1, 1]`, which protects the main representation from arbitrarily large raw values; and
- a **residual coordinate** in `(-1, 1)`, which encodes excess magnitude beyond the transition boundary using a monotone, invertible compression.

The complete transform is therefore bounded but does not rely on destructive tail clipping.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## 30-second example

```python
import numpy as np
from corenorm import CoreNorm

X_train = np.array([
    [18.0, 42.0, 110.0],
    [19.0, 45.0, 180.0],
    [20.0, 47.0, 260.0],
    [21.0, 50.0, 420.0],
    [37.0, 85.0, 1180.0],  # extreme but potentially meaningful
])

scaler = CoreNorm().fit(X_train)
Z = scaler.transform(X_train)
X_reconstructed = scaler.inverse_transform(Z)

print(Z.shape)  # (5, 6): two bounded coordinates per original feature
print(np.max(np.abs(X_train - X_reconstructed)))
```

### scikit-learn pipeline

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from corenorm import CoreNorm

model = make_pipeline(CoreNorm(), Ridge())
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

Core-Norm should be **fit on training data only**, exactly like any other fitted preprocessing transformation.

## Method

Let `m` be the feature median and define separate side scales

\[
s^- = m-Q_{0.25}, \qquad s^+=Q_{0.75}-m.
\]

The asymmetric standardized coordinate is

\[
u = \begin{cases}
(x-m)/(s^-+\varepsilon), & x<m,\\
(x-m)/(s^++\varepsilon), & x\ge m.
\end{cases}
\]

The transition threshold is

\[
\tau=\operatorname{clip}\left(Q_q(|u|),\tau_{min},\tau_{max}\right).
\]

Core-Norm returns

\[
C=\operatorname{clip}(u/\tau,-1,1)
\]

and, with `d=max(|u|-tau,0)`,

\[
R=\operatorname{sign}(u)\frac{\log(1+d)}{1+\log(1+d)}.
\]

The output is `[C, R]`.

See [Method and derivation](docs/method.md) for the inverse and mathematical properties.

## Preliminary benchmark

The repository preserves the Phase-1 benchmark generated during method development. Across the saved contamination curve, Core-Norm retained substantially more performance than conventional fitted scalers as synthetic feature corruption increased.

| Task | Clean | 20% cell corruption | Robustness AUC |
|---|---:|---:|---:|
| Classification · Core-Norm | 0.838 | 0.710 | **0.762** |
| Classification · Quantile-Normal | 0.835 | 0.677 | 0.738 |
| Classification · Z-score | 0.838 | 0.476 | 0.558 |
| Regression · Core-Norm | 0.595 | 0.299 | **0.417** |
| Regression · Quantile-Normal | 0.583 | 0.132 | 0.310 |
| Regression · Z-score | 0.578 | -0.096 | 0.065 |

These are **preliminary internal benchmark results**, not a universal performance guarantee. The complete canonicalized evaluation tables are under [`benchmarks/results/`](benchmarks/results/), including clean data, missingness, drift, row-wise corruption, cell-wise corruption, several synthetic distribution families, and real scikit-learn datasets.

## Verified round-trip

`tests/test_roundtrip.py` verifies

```text
X -> Core-Norm -> inverse_transform -> X
```

across ordinary, skewed, heavy-tailed, multimodal, zero-inflated, constant, missing-value, and extreme-value cases. The development stress test is archived in [`benchmarks/results/inverse_roundtrip.csv`](benchmarks/results/inverse_roundtrip.csv).

## Repository map

```text
src/corenorm/              implementation
examples/                  minimal and applied examples
tests/                     property, inverse, edge-case and sklearn tests
benchmarks/results/        canonicalized Phase-1 evaluation outputs
benchmarks/                result summarization and reproducibility notes
docs/                      method, API, benchmark interpretation, limitations
.github/workflows/         continuous integration
```

## Important limitations

Core-Norm doubles the numeric feature dimension, does not impute missing data, does not by itself solve train/test distribution shift, and is not guaranteed to improve every learner or every distribution. Distance-based models can be sensitive to the added coordinate geometry. Quartile-based scale estimation also has finite contamination robustness. Read the full [limitations](docs/limitations.md) before using Core-Norm in a study.

## Citation

If you use this research software before an archival paper is available, cite the software metadata in [`CITATION.cff`](CITATION.cff). Update the citation to the peer-reviewed paper once one exists.

## License

MIT. See [LICENSE](LICENSE).
