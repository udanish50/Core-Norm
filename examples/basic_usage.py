import numpy as np
from corenorm import CoreNorm

X = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0], [40.0, 60.0]])
scaler = CoreNorm().fit(X)
Z = scaler.transform(X)
X_hat = scaler.inverse_transform(Z)

print("encoded shape:", Z.shape)
print("max reconstruction error:", np.max(np.abs(X - X_hat)))
