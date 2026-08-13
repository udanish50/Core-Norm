# Contributing

Please open an issue before changing the mathematical definition of Core-Norm. Implementation fixes, tests, documentation, benchmark extensions, and reproducibility improvements are welcome.

Development setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests examples
pytest
```

Any change to the forward transformation must include a corresponding inverse test and a clear mathematical rationale.
