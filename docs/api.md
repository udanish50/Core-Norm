# API

## `CoreNorm(q=0.95, tau_min=1.5, tau_max=3.0, eps=1e-8)`

`fit(X)` learns feature-wise robust statistics from a two-dimensional numeric matrix.

`transform(X)` returns a NumPy array with `2 * n_features` columns. The first block contains central coordinates; the second block contains residual coordinates.

`inverse_transform(Z)` reconstructs original-scale values from a valid complete Core-Norm encoding.

`get_feature_names_out()` returns names ending in `__central` and `__residual`.

`save(path)` writes fitted state as JSON. `CoreNorm.load(path)` restores it.

### Missing values

NaN values are preserved. Apply an imputer before or after Core-Norm according to the design of the predictive pipeline; do not fit preprocessing statistics on test data.

### Infinities

Positive and negative infinity are rejected because finite quantiles and a finite inverse are required.
