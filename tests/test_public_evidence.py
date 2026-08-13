from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_dataset_manifest_has_25_verified_files() -> None:
    manifest_path = ROOT / "benchmarks" / "datasets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["datasets"]) == 25

    for dataset in manifest["datasets"]:
        path = ROOT / dataset["file"]
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == dataset["sha256"]
