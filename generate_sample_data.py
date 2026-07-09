"""
driftlens.report
=================

Renders a self-contained HTML report (charts embedded as base64 PNGs, no
external JS/CSS dependencies, no internet required to view it) from a
DriftReport produced by driftlens.core.compare_datasets.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .core import ColumnType, DriftReport, DriftStatus

_STATUS_COLORS = {
    DriftStatus.OK: "#2e7d32",
    DriftStatus.WARNING: "#f9a825",
    DriftStatus.DRIFT: "#c62828",
}

_STATUS_BG = {
    DriftStatus.OK: "#e8f5e9",
    DriftStatus.WARNING: "#fff8e1",
    DriftStatus.DRIFT: "#ffebee",
}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _numeric_chart(name: str, baseline: pd.Series, current: pd.Series) -> str:
    b = pd.to_numeric(baseline, errors="coerce").dropna()
    c = pd.to_numeric(current, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = 25
    if len(b):
        ax.hist(b, bins=bins, alpha=0.55, label="baseline", color="#5c6bc0", density=True)
    if len(c):
        ax.hist(c, bins=bins, alpha=0.55, label="current", color="#ef6c00", density=True)
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _categorical_chart(name: str, baseline: pd.Series, current: pd.Series, top_n: int = 8) -> str:
    b = baseline.dropna().astype(str)
    c = current.dropna().astype(str)
    top_categories = b.value_counts().head(top_n).index.tolist()
    extra = [cat for cat in c.value_counts().index if cat not in top_categories]
    categories = top_categories + extra[: max(0, top_n - len(top_categories))]

    b_pct = [b.value_counts(normalize=True).get(cat, 0) for cat in categories]
    c_pct = [c.value_counts(normalize=True).get(cat, 0) for cat in categories]

    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    width = 0.38
    ax.bar([i - width / 2 for i in x], b_pct, width=width, label="baseline", color="#5c6bc0")
    ax.bar([i + width / 2 for i in x], c_pct, width=width, label="current", color="#ef6c00")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=40, ha="right", fontsize=7)
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _psi_overview_chart(report: DriftReport) -> str:
    columns = [r.column for r in report.results]
    psis = [r.psi for r in report.results]
    colors = [_STATUS_COLORS[r.status] for r in report.results]

    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.4 * len(columns))))
    ax.barh(columns, psis, color=colors)
    ax.axvline(report.psi_warning_threshold, color="#f9a825", linestyle="--", linewidth=1, label="warning")
    ax.axvline(report.psi_drift_threshold, color="#c62828", linestyle="--", linewidth=1, label="drift")
    ax.set_xlabel("PSI")
    ax.set_title("Population Stability Index by column", fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_base64(fig)


def render_html(
    report: DriftReport,
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    title: str = "DriftLens Report",
) -> str:
    """Render a DriftReport as a self-contained HTML document (string)."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overview_img = _psi_overview_chart(report)

    rows_html = []
    for r in report.results:
        if r.column_type == ColumnType.NUMERIC:
            chart_b64 = _numeric_chart(r.column, baseline[r.column], current[r.column])
        else:
            chart_b64 = _categorical_chart(r.column, baseline[r.column], current[r.column])

        p_value_str = f"{r.p_value:.4g}" if r.p_value is not None else "n/a"
        stat_str = f"{r.test_statistic:.4f}" if r.test_statistic is not None else "n/a"
        bg = _STATUS_BG[r.status]
        color = _STATUS_COLORS[r.status]

        rows_html.append(f"""
        <div class="card" style="background:{bg};">
          <div class="card-header">
            <span class="status-dot" style="background:{color};"></span>
            <h3>{r.column}</h3>
            <span class="badge">{r.column_type.value}</span>
            <span class="badge status-badge" style="color:{color}; border-color:{color};">
              {r.status.value.upper()}
            </span>
          </div>
          <div class="card-body">
            <img src="data:image/png;base64,{chart_b64}" alt="{r.column} distribution" />
            <table class="stats-table">
              <tr><th></th><th>value</th></tr>
              <tr><td>PSI</td><td>{r.psi:.4f}</td></tr>
              <tr><td>JS divergence</td><td>{r.js_divergence:.4f}</td></tr>
              <tr><td>{r.test_name}</td><td>stat={stat_str}, p={p_value_str}</td></tr>
              {f'<tr><td colspan="2">{r.detail}</td></tr>' if r.detail else ""}
            </table>
          </div>
        </div>
        """)

    status_color = _STATUS_COLORS[report.overall_status]
    status_bg = _STATUS_BG[report.overall_status]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    background: #fafafa; color: #212121; margin: 0; padding: 0 0 3rem 0;
  }}
  header {{
    background: {status_bg}; border-bottom: 4px solid {status_color};
    padding: 2rem 3rem; margin-bottom: 2rem;
  }}
  header h1 {{ margin: 0 0 0.25rem 0; font-size: 1.6rem; }}
  header .status {{ font-weight: 700; color: {status_color}; font-size: 1.1rem; }}
  header .meta {{ color: #616161; font-size: 0.9rem; margin-top: 0.5rem; }}
  .container {{ padding: 0 3rem; }}
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem; margin-bottom: 2rem;
  }}
  .summary-box {{
    background: white; border-radius: 10px; padding: 1rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .summary-box .num {{ font-size: 1.6rem; font-weight: 700; }}
  .summary-box .label {{ color: #757575; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  .overview-chart {{ background: white; border-radius: 10px; padding: 1rem; margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
  .overview-chart img {{ max-width: 100%; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1.25rem; }}
  .card {{ border-radius: 12px; padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card-header {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }}
  .card-header h3 {{ margin: 0; font-size: 1rem; flex-grow: 1; }}
  .status-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .badge {{ font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 999px; background: white;
    border: 1px solid #bdbdbd; color: #616161; }}
  .status-badge {{ font-weight: 700; }}
  .card img {{ width: 100%; border-radius: 6px; background: white; }}
  .stats-table {{ width: 100%; font-size: 0.82rem; margin-top: 0.5rem; border-collapse: collapse; }}
  .stats-table td, .stats-table th {{ padding: 0.25rem 0.4rem; text-align: left; }}
  .stats-table tr:nth-child(even) {{ background: rgba(0,0,0,0.03); }}
  footer {{ text-align: center; color: #9e9e9e; font-size: 0.8rem; margin-top: 3rem; }}
</style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="status">Overall status: {report.overall_status.value.upper()}</div>
    <div class="meta">Generated {generated_at} &middot; baseline rows: {report.n_baseline} &middot; current rows: {report.n_current}</div>
  </header>
  <div class="container">
    <div class="summary-grid">
      <div class="summary-box"><div class="num">{len(report.results)}</div><div class="label">columns</div></div>
      <div class="summary-box"><div class="num" style="color:{_STATUS_COLORS[DriftStatus.DRIFT]}">{len(report.drifted_columns)}</div><div class="label">drifted</div></div>
      <div class="summary-box"><div class="num" style="color:{_STATUS_COLORS[DriftStatus.WARNING]}">{len(report.warning_columns)}</div><div class="label">warning</div></div>
      <div class="summary-box"><div class="num" style="color:{_STATUS_COLORS[DriftStatus.OK]}">{len(report.results) - len(report.drifted_columns) - len(report.warning_columns)}</div><div class="label">stable</div></div>
    </div>
    <div class="overview-chart">
      <img src="data:image/png;base64,{overview_img}" alt="PSI overview" />
    </div>
    <div class="grid">
      {"".join(rows_html)}
    </div>
  </div>
  <footer>Generated by driftlens &middot; github.com/yourname/driftlens</footer>
</body>
</html>
"""
    return html


def save_html_report(
    report: DriftReport,
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str,
    title: str = "DriftLens Report",
) -> str:
    html = render_html(report, baseline, current, title=title)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
