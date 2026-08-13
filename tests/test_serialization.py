import numpy as np
from corenorm import CoreNorm


def test_json_state_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 4))
    original = CoreNorm().fit(X)
    path = original.save(tmp_path / "state.json")
    restored = CoreNorm.load(path)
    assert np.allclose(original.transform(X), restored.transform(X))
    assert np.allclose(X, restored.inverse_transform(restored.transform(X)))
