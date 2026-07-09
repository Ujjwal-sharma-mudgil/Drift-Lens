name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install package + dev dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v --cov=driftlens --cov-report=term-missing

      - name: Smoke-test the CLI
        run: |
          python examples/generate_sample_data.py
          driftlens compare examples/sample_baseline.csv examples/sample_current.csv --output /tmp/report.html --fail-on-drift || true
