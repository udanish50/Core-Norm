"""Re-run Core-Norm's public numerical-property checks on every published dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from corenorm import CoreNorm  # noqa: E402


def load_matrix(path: Path, target: str) -> np.ndarray:
    """Load numeric predictors from a public verification CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [name for name in reader.fieldnames or [] if name != target]
        rows = [[float(row[name]) for name in names] for row in reader]
    return np.asarray(rows, dtype=np.float64)


def rel_error(x: np.ndarray, xhat: np.ndarray) -> float:
    """Return maximum element-wise relative reconstruction error."""
    denominator = np.maximum(np.abs(x), 1.0)
    return float(np.nanmax(np.abs(x - xhat) / denominator))


def run() -> dict[str, object]:
    """Verify checksums, boundedness and inverse reconstruction on every dataset."""
    manifest_path = ROOT / "benchmarks" / "datasets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []

    for item in manifest["datasets"]:
        path = ROOT / item["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")

        X = load_matrix(path, item["target"])
        n_fit = max(2, int(np.ceil(0.70 * len(X))))
        scaler = CoreNorm().fit(X[:n_fit])
        transformed = scaler.transform(X)
        reconstructed = scaler.inverse_transform(transformed)

        n_features = X.shape[1]
        central = transformed[:, :n_features]
        residual = transformed[:, n_features:]
        max_abs = float(np.nanmax(np.abs(X - reconstructed)))
        max_rel = rel_error(X, reconstructed)
        central_max = float(np.nanmax(np.abs(central)))
        residual_max = float(np.nanmax(np.abs(residual)))
        tail_fraction = float(np.mean(np.abs(residual) > 0))
        passed = bool(central_max <= 1.0 + 1e-12 and residual_max < 1.0 and max_rel < 1e-8)

        records.append(
            {
                "id": item["id"],
                "name": item["name"],
                "task": item["task"],
                "rows": int(X.shape[0]),
                "features": int(X.shape[1]),
                "fit_rows": n_fit,
                "sha256": digest,
                "central_max_abs": central_max,
                "residual_max_abs": residual_max,
                "max_abs_inverse_error": max_abs,
                "max_relative_inverse_error": max_rel,
                "tail_cell_fraction": tail_fraction,
                "passed": passed,
            }
        )

    return {
        "schema_version": 1,
        "method": "Core-Norm",
        "generated_on": "2026-08-13",
        "fit_fraction": 0.70,
        "dataset_count": len(records),
        "all_passed": all(bool(record["passed"]) for record in records),
        "datasets": records,
    }


def main() -> None:
    """Write the public verification ledger and fail if any check fails."""
    result = run()
    output = ROOT / "benchmarks" / "results" / "live_dataset_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"verified {result['dataset_count']} datasets; all_passed={result['all_passed']}")
    datasets = result["datasets"]
    worst = max(
        float(record["max_relative_inverse_error"])
        for record in datasets  # type: ignore[union-attr]
    )
    print("worst relative inverse error", worst)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
