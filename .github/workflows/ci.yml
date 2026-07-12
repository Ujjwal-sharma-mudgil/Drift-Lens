"""
driftlens.cli
=============

Command-line interface:

    driftlens compare baseline.csv current.csv --output report.html
    driftlens anomalies data.csv --column price --method modified_zscore
"""

from __future__ import annotations

import json
import sys

import click
import pandas as pd

from anomaly import detect_dataframe
from core import compare_datasets
from report import save_html_report

__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__, prog_name="driftlens")
def main():
    """DriftLens: statistical drift and anomaly detection for tabular data."""


@main.command()
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("current_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Write an HTML report to this path.")
@click.option("--json-output", default=None, help="Write a machine-readable JSON report to this path.")
@click.option("--columns", default=None, help="Comma-separated list of columns to compare (default: all shared).")
@click.option("--psi-warning", default=0.1, show_default=True, help="PSI threshold for WARNING status.")
@click.option("--psi-drift", default=0.25, show_default=True, help="PSI threshold for DRIFT status.")
@click.option("--alpha", default=0.05, show_default=True, help="Significance level for KS / chi-square tests.")
@click.option("--fail-on-drift", is_flag=True, help="Exit with a non-zero status code if drift is detected (useful in CI).")
def compare(baseline_path, current_path, output, json_output, columns, psi_warning, psi_drift, alpha, fail_on_drift):
    """Compare BASELINE_PATH and CURRENT_PATH (CSV files) for distribution drift."""
    baseline = pd.read_csv(baseline_path)
    current = pd.read_csv(current_path)
    col_list = [c.strip() for c in columns.split(",")] if columns else None

    report = compare_datasets(
        baseline,
        current,
        columns=col_list,
        psi_warning_threshold=psi_warning,
        psi_drift_threshold=psi_drift,
        alpha=alpha,
    )

    click.echo(report.summary())

    if output:
        save_html_report(report, baseline, current, output)
        click.echo(f"\nHTML report written to {output}")

    if json_output:
        with open(json_output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        click.echo(f"JSON report written to {json_output}")

    if fail_on_drift and report.drifted_columns:
        sys.exit(1)


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--column", "-c", multiple=True, help="Column(s) to check. Default: all numeric columns.")
@click.option(
    "--method",
    type=click.Choice(["zscore", "modified_zscore", "iqr"]),
    default="modified_zscore",
    show_default=True,
)
@click.option("--threshold", default=None, type=float, help="Override the method's default threshold.")
@click.option("--top", default=10, show_default=True, help="Show the top N most anomalous rows per column.")
def anomalies(data_path, column, method, threshold, top):
    """Flag anomalous rows in DATA_PATH (a CSV file)."""
    df = pd.read_csv(data_path)
    cols = list(column) or None
    results = detect_dataframe(df, columns=cols, method=method, threshold=threshold)

    for col, result in results.items():
        click.echo(f"\n{col}: {result.n_anomalies} anomalies ({result.anomaly_rate:.1%} of rows), method={method}")
        if result.n_anomalies:
            ranked = result.scores.sort_values(ascending=False).head(top)
            for idx, score in ranked.items():
                if result.is_anomaly[idx]:
                    raw_value = df.loc[idx, col]
                    value = raw_value.item() if hasattr(raw_value, "item") else raw_value
                    click.echo(f"    row {idx}: value={value!r}  score={score:.3f}")


if __name__ == "__main__":
    main()
