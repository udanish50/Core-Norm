# Benchmark interpretation

The archived Phase-1 benchmark is a development benchmark, not a peer-reviewed claim of state-of-the-art performance.

It includes several synthetic distribution families and real scikit-learn datasets, multiple model families, and conditions such as clean data, cell-wise extreme corruption, row-wise corruption, missingness, and distribution drift.

The strongest repeated pattern in the saved results is robustness under feature corruption. Core-Norm remained competitive on clean data and degraded more gradually than Z-score, Min-Max, and RobustScaler as extreme corruption increased. Quantile-Normal remained a strong comparator and can be preferable for some distributions.

The benchmark also surfaced limitations: compound distribution shift can erase the advantage of static preprocessing, and distance-based learners can be sensitive to the increased feature dimension.

See `results/README.md` for provenance and canonicalization rules.

## Public website evidence

The website does not embed hand-copied benchmark numbers as its primary evidence source. It can retrieve the full `classification_results.csv`, `regression_results.csv`, dataset manifest, and live verification ledger directly from the public GitHub repository. Visitors can inspect all saved run rows and can independently execute Core-Norm forward/inverse verification in their browser on repository-hosted datasets.

The synthetic public dataset files are deterministic replay fixtures for the same named distribution families; the first exploratory matrices were not archived. This distinction is surfaced in the manifest and website UI.
