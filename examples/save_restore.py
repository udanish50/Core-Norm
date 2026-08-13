import numpy as np

from corenorm import CoreNorm

X = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 100.0]])
CoreNorm().fit(X).save("core_norm_state.json")
restored = CoreNorm.load("core_norm_state.json")
print(restored.transform(X))
