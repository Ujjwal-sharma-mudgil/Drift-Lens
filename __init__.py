# Usage

## Installation

```bash
git clone https://github.com/yourname/driftlens.git
cd driftlens
pip install -e .
```

## As a library

```python
import pandas as pd
from driftlens.core import compare_datasets
from driftlens.report import save_html_report

baseline = pd.read_csv("baseline.csv")
current = pd.read_csv("current.csv")

report = compare_datasets(baseline, current)

print(report.summary())
print("Drifted columns:", report.drifted_columns)

save_html_report(report, baseline, current, "report.html")
```

### Anomaly detection

```python
from driftlens.anomaly import detect_dataframe

df = pd.read_csv("current.csv")
results = detect_dataframe(df, method="modified_zscore")

for column, result in results.items():
    print(column, result.n_anomalies, "anomalies")
```

## As a CLI

```bash
# Generate a human-readable summary + HTML report
driftlens compare baseline.csv current.csv --output report.html

# Only compare specific columns, with custom thresholds
driftlens compare baseline.csv current.csv \
  --columns "age,plan_type" \
  --psi-warning 0.15 \
  --psi-drift 0.3

# Also emit machine-readable JSON (for dashboards / alerting)
driftlens compare baseline.csv current.csv --json-output report.json

# Use in CI: exit non-zero if drift is detected
driftlens compare baseline.csv current.csv --fail-on-drift

# Find anomalous rows in a single file
driftlens anomalies data.csv --method modified_zscore --column session_count
```

## Try it immediately with the bundled demo data

```bash
python examples/generate_sample_data.py
driftlens compare examples/sample_baseline.csv examples/sample_current.csv -o demo_report.html
open demo_report.html   # or just double-click it
```

This generates a baseline/current pair with a few intentionally injected
drift patterns (an aging user base, a new pricing tier, heavier-tailed
session counts) alongside two stable "control" columns, so you can see
what both true positives and true negatives look like.

## Configuration reference

| Option              | Default | Meaning                                          |
|---------------------|---------|---------------------------------------------------|
| `psi_warning_threshold` | 0.1  | PSI above this → `warning`                        |
| `psi_drift_threshold`   | 0.25 | PSI above this → `drift`                          |
| `alpha`                 | 0.05 | p-value below this (KS / chi-square) → `drift`   |

See `docs/METHODOLOGY.md` for the statistical reasoning behind these
defaults.
