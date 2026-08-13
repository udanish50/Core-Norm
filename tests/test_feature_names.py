import numpy as np

from corenorm import CoreNorm


def test_feature_names_without_pandas():
    scaler = CoreNorm().fit(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
    assert scaler.get_feature_names_out().tolist() == [
        "x0__central",
        "x1__central",
        "x0__residual",
        "x1__residual",
    ]
