# 🔍 DriftLens

**Catch data drift before your model does.**

DriftLens is a lightweight Python library and CLI for detecting statistical
drift between two tabular datasets (e.g. training data vs. production
traffic, last month vs. this month) and flagging anomalous rows within a
single dataset — with zero heavyweight dependencies and a self-contained
HTML report you can open in any browser, no internet connection required.

```bash
pip install -e .
python examples/generate_sample_data.py
driftlens compare examples/sample_baseline.csv examples/sample_current.csv -o report.html
```

That last command produces a report that looks like this:

- **Overall verdict** (ok / warning / drift) at a glance
- **Per-column PSI overview chart**, sorted, color-coded
- **Per-column distribution overlays** (baseline vs. current), automatically
  histograms for numeric columns or bar charts for categorical ones
- **New-category detection** — did production traffic start sending values
  your training data never saw?

## Why this exists

Silent data drift is one of the most common ways ML systems degrade in
production — a model trained on one distribution slowly starts seeing a
different one, and nobody notices until accuracy has already dropped. Most
drift-detection tooling is either buried inside a large MLOps platform or
requires a paid SaaS. DriftLens is the ~15-minute version: a single `pip
install`, three statistical tests that are actually well-established
(PSI, KS test, chi-square, Jensen-Shannon divergence), and a report you can
attach to a Slack message or a CI failure.

## Features

- 📊 **Distribution drift detection** for both numeric and categorical
  columns (PSI + KS test + chi-square test + Jensen-Shannon divergence)
- 🚨 **Anomaly detection** within a single dataset (z-score, modified
  z-score / MAD, IQR / Tukey fences)
- 🖥️ **CLI** for quick checks and CI pipelines (`--fail-on-drift` exits
  non-zero when drift is detected)
- 📄 **Self-contained HTML reports** — charts are embedded as base64 PNGs,
  so the report is a single file with no external dependencies
- 🧩 **Small, typed, dependency-light codebase** — just numpy, pandas,
  scipy, matplotlib, and click
- ✅ **25 unit tests**, stdlib `unittest` (also pytest-compatible)

## Quick example

```python
import pandas as pd
from driftlens.core import compare_datasets

baseline = pd.read_csv("training_data.csv")
current = pd.read_csv("production_data.csv")

report = compare_datasets(baseline, current)
print(report.summary())
# DriftLens report: DRIFT
#   baseline rows: 2000, current rows: 2000
#   columns compared: 5
#   drifted: 3  warning: 0
# !! signup_age               [numeric    ] PSI=0.9203  JS=0.1653  KS test p=3.7e-148
# !! plan_type                [categorical] PSI=0.6715  JS=0.0531  Chi-square test p=6.2e-48
#    region                   [categorical] PSI=0.0001  JS=0.0000  Chi-square test p=0.96
```

See [`docs/USAGE.md`](docs/USAGE.md) for the full CLI/library reference and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the statistics behind it.

## Installation

```bash
git clone https://github.com/Ujjwal-sharma-mudgil/driftlens.git
cd driftlens
pip install -e ".[dev]"   # [dev] pulls in pytest for running tests
```

## Running the tests

```bash
pytest tests/ -v
# or, dependency-free:
python -m unittest discover -s tests -v
```

## Project layout
