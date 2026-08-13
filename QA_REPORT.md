# Core-Norm QA report

Validation date: 2026-08-13

- Python unit/property suite: **10 passed**.
- Forward/inverse stress generation: **700,000 scalar values** across Gaussian, lognormal, Student-t, exponential, bimodal, zero-inflated and 5% contaminated distributions.
- Worst archived absolute round-trip error in that stress run: **2.4374458007514477e-10**.
- Python source/example/benchmark syntax: `compileall` passed after final fixes.
- Naming audit: no public `CORE-full`, `CORE-body`, `CORENormV2`, or `CORE-Norm v2` labels remain.
- Editable-install smoke test passed with the local environment using `--no-build-isolation`; normal CI installs from the declared `pyproject.toml`.
- scikit-learn Pipeline smoke test is part of the unit suite.
- NaN preservation, constant columns, invalid infinity, state serialization, boundedness, and positive-affine invariance are covered by tests.

The local sandbox did not contain Ruff and has no package-index network access, so Ruff itself was not executed locally. The GitHub Actions CI workflow runs Ruff on every push/pull request.
