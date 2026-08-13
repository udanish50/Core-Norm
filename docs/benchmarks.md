# Benchmark interpretation

The archived Phase-1 benchmark is a development benchmark, not a peer-reviewed claim of state-of-the-art performance.

It includes several synthetic distribution families and real scikit-learn datasets, multiple model families, and conditions such as clean data, cell-wise extreme corruption, row-wise corruption, missingness, and distribution drift.

The strongest repeated pattern in the saved results is robustness under feature corruption. Core-Norm remained competitive on clean data and degraded more gradually than Z-score, Min-Max, and RobustScaler as extreme corruption increased. Quantile-Normal remained a strong comparator and can be preferable for some distributions.

The benchmark also surfaced limitations: compound distribution shift can erase the advantage of static preprocessing, and distance-based learners can be sensitive to the increased feature dimension.

See `results/README.md` for provenance and canonicalization rules.
