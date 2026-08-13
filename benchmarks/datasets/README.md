# Public verification datasets

These CSV files make the Core-Norm website evidence inspectable in a browser. Visitors can fetch a dataset directly from this public repository, verify its SHA-256 digest, fit Core-Norm, transform it, invert the representation, and check the numerical bounds and reconstruction error locally.

- The **Iris, Wine, Breast Cancer Wisconsin, Digits, and Diabetes** files are canonical scikit-learn bundled datasets corresponding to the real dataset names in the archived Phase-1 result tables.
- The synthetic CSV files are **deterministic public replay fixtures for the same distribution families** (Gaussian, correlated, lognormal, Student-t, Laplace, bimodal, zero-inflated, bounded beta, mixed, and heteroscedastic). The first exploratory Phase-1 run preserved the result tables but did not preserve byte-identical generated matrices; these fixtures therefore support transparent live property verification without pretending to be the original unarchived random matrices.

`manifest.json` records source, task, row/feature counts, repository URLs, and SHA-256 hashes. `build_verification_datasets.py` regenerates the files deterministically.
