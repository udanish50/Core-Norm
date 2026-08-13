"""Illustrative weather/energy feature preprocessing with Core-Norm."""

import numpy as np

from corenorm import CoreNorm

# temperature C, relative humidity %, GHI W/m^2, power kW
X_train = np.array(
    [
        [18.2, 52.0, 120.0, 1.8],
        [19.7, 48.0, 280.0, 2.2],
        [22.1, 45.0, 610.0, 3.9],
        [24.6, 42.0, 870.0, 5.4],
        [26.8, 40.0, 1115.0, 6.8],
    ]
)

scaler = CoreNorm().fit(X_train)
X_encoded = scaler.transform(X_train)
print(X_encoded)
print("round-trip:")
print(scaler.inverse_transform(X_encoded))
