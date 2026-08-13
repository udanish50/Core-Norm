from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent / "results"

for filename, metric in [
    ("classification_results.csv", "bal_acc"),
    ("regression_results.csv", "r2"),
]:
    df = pd.read_csv(ROOT / filename)
    summary = (
        df.groupby(["condition", "method"], dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(["condition", metric], ascending=[True, False])
    )
    print("\n" + filename)
    print(summary.to_string(index=False))
