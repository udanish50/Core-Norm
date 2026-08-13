"""Validate the public Core-Norm evidence manifest and archived result files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def row_count(path: Path) -> int:
    """Count data rows in a CSV, excluding its header."""
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> None:
    """Verify dataset hashes and exact archived benchmark row counts."""
    dataset_manifest_path = ROOT / "benchmarks" / "datasets" / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))

    for dataset in dataset_manifest["datasets"]:
        path = ROOT / dataset["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != dataset["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")

    expected = {
        "classification_results.csv": 6804,
        "regression_results.csv": 5346,
    }
    files = []
    for name, expected_rows in expected.items():
        path = ROOT / "benchmarks" / "results" / name
        actual_rows = row_count(path)
        if actual_rows != expected_rows:
            raise RuntimeError(f"Unexpected row count for {name}: {actual_rows} != {expected_rows}")
        files.append(
            {
                "name": name,
                "rows": actual_rows,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_url": (
                    "https://raw.githubusercontent.com/udanish50/Core-Norm/main/"
                    f"benchmarks/results/{name}"
                ),
                "github_url": (
                    f"https://github.com/udanish50/Core-Norm/blob/main/benchmarks/results/{name}"
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "method": "Core-Norm",
        "total_saved_evaluations": sum(expected.values()),
        "files": files,
        "note": (
            "Archived Phase-1 result rows. The website filters and displays "
            "these files directly from the public GitHub repository."
        ),
    }
    output = ROOT / "benchmarks" / "results" / "public_evidence_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS public evidence:",
        manifest["total_saved_evaluations"],
        "saved evaluations and",
        len(dataset_manifest["datasets"]),
        "public datasets",
    )


if __name__ == "__main__":
    main()
