from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "results"


def main() -> None:
    """Summarize the archived classification and regression benchmark files."""
    classification = pd.read_csv(ROOT / "classification_results.csv")
    regression = pd.read_csv(ROOT / "regression_results.csv")
    print("classification rows:", len(classification))
    print("regression rows:", len(regression))
    print("total rows:", len(classification) + len(regression))


if __name__ == "__main__":
    main()
