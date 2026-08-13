from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from corenorm import CoreNorm


def test_sklearn_pipeline_compatibility():
    X, y = load_diabetes(return_X_y=True)
    model = make_pipeline(CoreNorm(), Ridge())
    model.fit(X[:350], y[:350])
    pred = model.predict(X[350:])
    assert pred.shape == (X.shape[0] - 350,)
