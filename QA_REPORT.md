# Core-Norm QA report

Release preparation date: 2026-08-13

## Implementation

- `pytest`: **12 passed**.
- Python compilation: passed for `src/`, `tests/`, and `benchmarks/`.
- Core-Norm remains the sole proposed method name throughout the public documentation.
- Existing scikit-learn compatibility, serialization, feature naming, boundedness, affine-invariance, NaN/constant handling, and forward/inverse tests remain included.

## Public evidence layer

- Archived Phase-1 classification rows: **6,804**.
- Archived Phase-1 regression rows: **5,346**.
- Total archived evaluation rows: **12,150**.
- Public verification datasets: **25**.
- Live Python verification: **25/25 datasets passed** boundedness and inverse checks.
- Worst relative inverse error across the 25 public verification datasets in this run: **2.79e-15**.
- Every public dataset CSV is SHA-256 pinned in `benchmarks/datasets/manifest.json`.
- The five canonical real datasets correspond to the scikit-learn datasets named in the archived benchmark.
- Synthetic files are explicitly documented as deterministic replay fixtures for the same distribution families; they are not represented as byte-identical exploratory matrices.

## CI

- Main CI now verifies the public evidence manifest and reruns Core-Norm dataset property checks.
- A dedicated `Public Evidence` GitHub Actions workflow is included and uploads the generated evidence ledger as an artifact.
