"""
driftlens.report
================
Self-contained HTML report generation for a DriftReport.

No external dependencies at render time: charts are rendered with
matplotlib and embedded directly into the HTML as base64-encoded PNGs, so
the resulting file can be opened in any browser with no internet
connection and no other files alongside it.
"""
from __future__ import annotations

import base64
import io
from html import escape

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import pandas as pd

from core import ColumnType, DriftReport, DriftStatus


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _psi_overview_chart(report: DriftReport) -> str:
    results = sorted(report.results, key=lambda r: r.psi, reverse=True)
    names = [r.column for r in results]
    psis = [r.psi for r in results]
    colors = [
        "#d64545" if r.status == DriftStatus.DRIFT
        else "#e0a92e" if r.status == DriftStatus.WARNING
        else "#3f9142"
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(names))))
    ax.barh(names, psis, color=colors)
    ax.axvline(report.psi_warning_threshold, color="#e0a92e", linestyle="--", linewidth=1)
    ax.axvline(report.psi_drift_threshold, color="#d64545", linestyle="--", linewidth=1)
    ax.set_xlabel("PSI")
    ax.set_title("Population Stability Index by column")
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _numeric_distribution_chart(column: str, baseline: pd.Series, current: pd.Series) -> str:
    b = pd.to_numeric(baseline, errors="coerce").dropna()
    c = pd.to_numeric(current, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = 30
    ax.hist(b, bins=bins, alpha=0.5, label="baseline", density=True, color="#4472c4")
    ax.hist(c, bins=bins, alpha=0.5, label="current", density=True, color="#d64545")
    ax.set_title(column)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _categorical_distribution_chart(column: str, baseline: pd.Series, current: pd.Series) -> str:
    b = baseline.dropna().astype(str)
    c = current.dropna().astype(str)
    categories = sorted(set(b.unique()) | set(c.unique()), key=str)[:12]
    b_counts = b.value_counts(normalize=True)
    c_counts = c.value_counts(normalize=True)

    x = range(len(categories))
    width = 0.4
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([i - width / 2 for i in x], [b_counts.get(cat, 0) for cat in categories], width=width, label="baseline", color="#4472c4")
    ax.bar([i + width / 2 for i in x], [c_counts.get(cat, 0) for cat in categories], width=width, label="current", color="#d64545")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
    ax.set_title(column)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


_STATUS_COLORS = {
    DriftStatus.OK: "#3f9142",
    DriftStatus.WARNING: "#e0a92e",
    DriftStatus.DRIFT: "#d64545",
}


def _render_column_section(report: DriftReport, baseline: pd.DataFrame, current: pd.DataFrame) -> str:
    parts = []
    for r in report.results:
        color = _STATUS_COLORS[r.status]
        if r.column_type == ColumnType.NUMERIC:
            chart_b64 = _numeric_distribution_chart(r.column, baseline[r.column], current[r.column])
        else:
            chart_b64 = _categorical_distribution_chart(r.column, baseline[r.column], current[r.column])

        p_value_html = f"<div>p-value: {r.p_value:.4g}</div>" if r.p_value is not None else ""
        detail_html = f'<div class="detail">{escape(r.detail)}</div>' if r.detail else ""

        parts.append(f"""
        <div class="column-card" style="border-left: 6px solid {color}">
            <h3>{escape(r.column)} <span class="badge" style="background:{color}">{r.status.value.upper()}</span></h3>
            <div class="stats">
                <div>Type: {r.column_type.value}</div>
                <div>PSI: {r.psi:.4f}</div>
                <div>JS divergence: {r.js_divergence:.4f}</div>
                <div>Test: {escape(r.test_name)}</div>
                {p_value_html}
            </div>
            {detail_html}
            <img src="data:image/png;base64,{chart_b64}" alt="{escape(r.column)} distribution" />
        </div>
        """)
    return "\n".join(parts)


def render_html_report(report: DriftReport, baseline: pd.DataFrame, current: pd.DataFrame) -> str:
    """Render a DriftReport (plus the source dataframes, for charting) to a self-contained HTML string."""
    overview_b64 = _psi_overview_chart(report)
    overall_color = _STATUS_COLORS[report.overall_status]
    column_sections = _render_column_section(report, baseline, current)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DriftLens Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem; background: #f7f7f9; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .overall-badge {{ display: inline-block; padding: 0.35rem 0.9rem; border-radius: 6px; color: white; font-weight: 600; background: {overall_color}; }}
  .summary-stats {{ margin: 1rem 0 2rem; color: #555; }}
  .overview-chart {{ text-align: center; margin-bottom: 2rem; }}
  .overview-chart img {{ max-width: 100%; }}
  .column-card {{ background: white; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .column-card h3 {{ margin-top: 0; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px; color: white; font-size: 0.75rem; vertical-align: middle; }}
  .stats {{ display: flex; gap: 1.5rem; flex-wrap: wrap; color: #555; font-size: 0.9rem; margin-bottom: 0.5rem; }}
  .detail {{ color: #a05a00; font-size: 0.85rem; margin-bottom: 0.5rem; }}
  .column-card img {{ max-width: 100%; }}
</style>
</head>
<body>
  <h1>DriftLens Report</h1>
  <span class="overall-badge">{report.overall_status.value.upper()}</span>
  <div class="summary-stats">
    baseline rows: {report.n_baseline} &nbsp;|&nbsp;
    current rows: {report.n_current} &nbsp;|&nbsp;
    columns compared: {len(report.results)} &nbsp;|&nbsp;
    drifted: {len(report.drifted_columns)} &nbsp;|&nbsp;
    warning: {len(report.warning_columns)}
  </div>

  <div class="overview-chart">
    <img src="data:image/png;base64,{overview_b64}" alt="PSI overview chart" />
  </div>

  {column_sections}
</body>
</html>"""


def save_html_report(report: DriftReport, baseline: pd.DataFrame, current: pd.DataFrame, path: str) -> None:
    """Render and write a DriftReport to an HTML file at `path`."""
    html = render_html_report(report, baseline, current)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
