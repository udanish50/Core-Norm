"""Generate the archived Core-Norm inverse round-trip stress table."""
from pathlib import Path
import csv
import numpy as np
from corenorm import CoreNorm

rng = np.random.default_rng(20260813)
n, p = 20000, 5
sets = {
    "gaussian": rng.normal(10, 3, (n, p)),
    "lognormal": rng.lognormal(2, 1.2, (n, p)),
    "student_t": rng.standard_t(2, (n, p)) * 5 + 2,
    "exponential": rng.exponential(5, (n, p)) - 2,
    "bimodal": np.concatenate([rng.normal(-3, 1, (n//2, p)), rng.normal(4, 1.5, (n//2, p))]),
    "zero_inflated": np.where(rng.random((n,p)) < 0.8, 0.0, rng.lognormal(1, 1, (n,p))),
    "contaminated_5pct": rng.normal(0, 1, (n, p)),
}
mask = rng.random((n,p)) < 0.05
sets["contaminated_5pct"][mask] += rng.choice([-1.0,1.0], size=mask.sum()) * rng.lognormal(6,1,size=mask.sum())

rows=[]
for name, X in sets.items():
    scaler=CoreNorm().fit(X[:12000])
    Z=scaler.transform(X)
    Xhat=scaler.inverse_transform(Z)
    err=np.abs(Xhat-X)
    rows.append((name, X.size, float(np.nanmax(err)), float(np.nanmean(err))))

out=Path(__file__).resolve().parent/'results'/'inverse_roundtrip.csv'
with out.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['distribution','scalar_values','max_abs_error','mean_abs_error']); w.writerows(rows)
print(out)
print('scalar values:', sum(r[1] for r in rows))
print('max error:', max(r[2] for r in rows))
