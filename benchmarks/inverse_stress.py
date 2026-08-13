"""Generate the archived Core-Norm inverse round-trip stress table."""

import csv
from pathlib import Path

import numpy as np

from corenorm import CoreNorm

rng = np.random.default_rng(20260813)
n = 20_000
p = 5
sets = {
    "gaussian": rng.normal(size=(n, p)),
    "lognormal": rng.lognormal(0, 1.2, size=(n, p)),
    "student_t": rng.standard_t(2.2, size=(n, p)),
    "exponential": rng.exponential(size=(n, p)),
    "bimodal": rng.normal(rng.choice([-3.0, 3.0], size=(n, p)), 0.8),
    "zero_inflated": rng.normal(size=(n, p)),
    "contaminated_5pct": rng.normal(size=(n, p)),
}
sets["zero_inflated"][rng.random((n, p)) < 0.8] = 0
mask = rng.random((n, p)) < 0.05
contamination = rng.choice([-1.0, 1.0], size=mask.sum()) * rng.lognormal(
    6,
    1,
    size=mask.sum(),
)
sets["contaminated_5pct"][mask] += contamination

rows = []
for name, X in sets.items():
    scaler = CoreNorm().fit(X)
    reconstructed = scaler.inverse_transform(scaler.transform(X))
    errors = np.abs(X - reconstructed)
    rows.append(
        [
            name,
            int(X.size),
            float(np.max(errors)),
            float(np.mean(errors)),
        ]
    )

out = Path(__file__).resolve().parent / "results" / "inverse_roundtrip.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "distribution",
            "scalar_values",
            "max_abs_error",
            "mean_abs_error",
        ]
    )
    writer.writerows(rows)

print(out)
print("scalar values:", sum(row[1] for row in rows))
