from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from corenorm import CoreNorm

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = make_pipeline(CoreNorm(), Ridge(alpha=1.0))
model.fit(X_train, y_train)
print("R^2:", model.score(X_test, y_test))
