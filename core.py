"""
driftlens.core
==============
Statistical data-drift detection for tabular data.

The central idea: you have a *baseline* dataset (e.g. your training data,
or last week's production traffic) and a *current* dataset (e.g. today's
production traffic). For each column, driftlens tells you whether the
distribution has shifted meaningfully, using well-established statistical
tests rather than vibes.

Numeric columns   -> Kolmogorov-Smirnov test + Population Stability Index (PSI)
Categorical cols  -> Chi-square test of independence + PSI on category shares
All columns       -> Jensen-Shannon divergence (bounded 0..1, easy to compare
                     across features regardless of type)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


class DriftStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    DRIFT = "drift"


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass
class ColumnDriftResult:
    """Drift-test results for a single column."""
    column: str
    column_type: ColumnType
    status: DriftStatus
    psi: float
    js_divergence: float
    test_name: str
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    baseline_summary: dict = field(default_factory=dict)
    current_summary: dict = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["column_type"] = self.column_type.value
        d["status"] = self.status.value
        return d


@dataclass
class DriftReport:
    """Aggregate drift report across every compared column."""
    results: list[ColumnDriftResult]
    n_baseline: int
    n_current: int
    psi_warning_threshold: float
    psi_drift_threshold: float
    alpha: float

    @property
    def drifted_columns(self) -> list[str]:
        return [r.column for r in self.results if r.status == DriftStatus.DRIFT]

    @property
    def warning_columns(self) -> list[str]:
        return [r.column for r in self.results if r.status == DriftStatus.WARNING]

    @property
    def overall_status(self) -> DriftStatus:
        if self.drifted_columns:
            return DriftStatus.DRIFT
        if self.warning_columns:
            return DriftStatus.WARNING
        return DriftStatus.OK

    def summary(self) -> str:
        lines = [
            f"DriftLens report: {self.overall_status.value.upper()}",
            f"  baseline rows: {self.n_baseline}, current rows: {self.n_current}",
            f"  columns compared: {len(self.results)}",
            f"  drifted: {len(self.drifted_columns)}  warning: {len(self.warning_columns)}",
        ]
        for r in self.results:
            marker = {"ok": "  ", "warning": " !", "drift": "!!"}[r.status.value]
            lines.append(
                f"{marker} {r.column:<24} [{r.column_type.value:<11}] "
                f"PSI={r.psi:.4f}  JS={r.js_divergence:.4f}  {r.test_name}"
                + (f"  p={r.p_value:.4g}" if r.p_value is not None else "")
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "n_baseline": self.n_baseline,
            "n_current": self.n_current,
            "psi_warning_threshold": self.psi_warning_threshold,
            "psi_drift_threshold": self.psi_drift_threshold,
            "alpha": self.alpha,
            "results": [r.to_dict() for r in self.results],
        }


def _psi_numeric(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for a numeric column, via shared quantile bins."""
    baseline = baseline.dropna()
    current = current.dropna()
    if baseline.empty or current.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 3:
        # Not enough distinct values to bin meaningfully.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(baseline, bins=edges)
    c_counts, _ = np.histogram(current, bins=edges)
    b_pct = np.where(b_counts == 0, 1e-4, b_counts / b_counts.sum())
    c_pct = np.where(c_counts == 0, 1e-4, c_counts / c_counts.sum())
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def _psi_categorical(baseline: pd.Series, current: pd.Series) -> float:
    baseline = baseline.dropna()
    current = current.dropna()
    categories = set(baseline.unique()) | set(current.unique())
    if not categories:
        return 0.0
    b_counts = baseline.value_counts()
    c_counts = current.value_counts()
    psi = 0.0
    for cat in categories:
        b_pct = max(b_counts.get(cat, 0) / max(len(baseline), 1), 1e-4)
        c_pct = max(c_counts.get(cat, 0) / max(len(current), 1), 1e-4)
        psi += (c_pct - b_pct) * np.log(c_pct / b_pct)
    return float(psi)


def _js_divergence_numeric(baseline: pd.Series, current: pd.Series, bins: int = 20) -> float:
    baseline = baseline.dropna()
    current = current.dropna()
    if baseline.empty or current.empty:
        return 0.0
    lo = min(baseline.min(), current.min())
    hi = max(baseline.max(), current.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(baseline, bins=edges, density=False)
    q, _ = np.histogram(current, bins=edges, density=False)
    return _js_from_counts(p, q)


def _js_divergence_categorical(baseline: pd.Series, current: pd.Series) -> float:
    baseline = baseline.dropna()
    current = current.dropna()
    categories = sorted(set(baseline.unique()) | set(current.unique()), key=str)
    if not categories:
        return 0.0
    b_counts = baseline.value_counts()
    c_counts = current.value_counts()
    p = np.array([b_counts.get(c, 0) for c in categories], dtype=float)
    q = np.array([c_counts.get(c, 0) for c in categories], dtype=float)
    return _js_from_counts(p, q)


def _js_from_counts(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = p_counts.astype(float)
    q = q_counts.astype(float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.clip(js, 0.0, 1.0))


def _classify(psi: float, p_value: Optional[float], psi_warn: float, psi_drift: float, alpha: float) -> DriftStatus:
    if psi >= psi_drift or (p_value is not None and p_value < alpha):
        return DriftStatus.DRIFT
    if psi >= psi_warn:
        return DriftStatus.WARNING
    return DriftStatus.OK


def _infer_column_type(series: pd.Series) -> ColumnType:
    # Booleans are technically numeric dtype in pandas/numpy, but treating
    # them as categorical gives far more sensible drift semantics (category
    # share shift) instead of quantile-based binning on {0, 1}.
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.CATEGORICAL
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC
    return ColumnType.CATEGORICAL


def compare_column(
    name: str,
    baseline: pd.Series,
    current: pd.Series,
    psi_warning_threshold: float = 0.1,
    psi_drift_threshold: float = 0.25,
    alpha: float = 0.05,
) -> ColumnDriftResult:
    """Run the appropriate drift tests for a single column."""
    col_type = _infer_column_type(baseline)

    if col_type == ColumnType.NUMERIC:
        b = pd.to_numeric(baseline, errors="coerce").dropna()
        c = pd.to_numeric(current, errors="coerce").dropna()
        psi = _psi_numeric(b, c)
        js = _js_divergence_numeric(b, c)
        if len(b) >= 2 and len(c) >= 2:
            ks_stat, p_value = stats.ks_2samp(b, c)
            test_name = "KS test"
        else:
            ks_stat, p_value = None, None
            test_name = "KS test (skipped: insufficient data)"
        status = _classify(psi, p_value, psi_warning_threshold, psi_drift_threshold, alpha)
        baseline_summary = {
            "mean": float(b.mean()) if len(b) else None,
            "std": float(b.std()) if len(b) else None,
            "min": float(b.min()) if len(b) else None,
            "max": float(b.max()) if len(b) else None,
        }
        current_summary = {
            "mean": float(c.mean()) if len(c) else None,
            "std": float(c.std()) if len(c) else None,
            "min": float(c.min()) if len(c) else None,
            "max": float(c.max()) if len(c) else None,
        }
        detail = ""
        return ColumnDriftResult(
            column=name,
            column_type=col_type,
            status=status,
            psi=psi,
            js_divergence=js,
            test_name=test_name,
            test_statistic=float(ks_stat) if ks_stat is not None else None,
            p_value=float(p_value) if p_value is not None else None,
            baseline_summary=baseline_summary,
            current_summary=current_summary,
            detail=detail,
        )

    # Categorical path
    b = baseline.dropna().astype(str)
    c = current.dropna().astype(str)
    psi = _psi_categorical(b, c)
    js = _js_divergence_categorical(b, c)
    categories = sorted(set(b.unique()) | set(c.unique()))
    b_counts = b.value_counts()
    c_counts = c.value_counts()

    if len(categories) >= 2 and len(b) > 0 and len(c) > 0:
        observed = np.array(
            [[b_counts.get(cat, 0), c_counts.get(cat, 0)] for cat in categories]
        )
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(observed)
            test_name = "Chi-square test"
        except ValueError:
            chi2, p_value = None, None
            test_name = "Chi-square test (skipped: degenerate table)"
    else:
        chi2, p_value = None, None
        test_name = "Chi-square test (skipped: insufficient categories)"

    status = _classify(psi, p_value, psi_warning_threshold, psi_drift_threshold, alpha)
    baseline_summary = {"top_categories": b_counts.head(5).to_dict(), "n_unique": int(b.nunique())}
    current_summary = {"top_categories": c_counts.head(5).to_dict(), "n_unique": int(c.nunique())}
    new_categories = set(c.unique()) - set(b.unique())
    detail = f"new categories in current: {sorted(new_categories)}" if new_categories else ""

    return ColumnDriftResult(
        column=name,
        column_type=col_type,
        status=status,
        psi=psi,
        js_divergence=js,
        test_name=test_name,
        test_statistic=float(chi2) if chi2 is not None else None,
        p_value=float(p_value) if p_value is not None else None,
        baseline_summary=baseline_summary,
        current_summary=current_summary,
        detail=detail,
    )


def compare_datasets(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    columns: Optional[list[str]] = None,
    psi_warning_threshold: float = 0.1,
    psi_drift_threshold: float = 0.25,
    alpha: float = 0.05,
) -> DriftReport:
    """Compare two dataframes column-by-column and return a DriftReport.

    Parameters
    ----------
    baseline, current:
        The two datasets to compare. Only columns present in *both* are
        compared, unless `columns` is given explicitly.
    columns:
        Restrict comparison to this list of column names.
    psi_warning_threshold, psi_drift_threshold:
        PSI cutoffs. Common industry defaults: <0.1 no shift, 0.1-0.25
        moderate shift (warning), >0.25 significant shift (drift).
    alpha:
        Significance level for the KS / chi-square tests.
    """
    shared = columns or [c for c in baseline.columns if c in current.columns]
    if not shared:
        raise ValueError("No shared columns between baseline and current datasets.")

    results = [
        compare_column(
            col,
            baseline[col],
            current[col],
            psi_warning_threshold=psi_warning_threshold,
            psi_drift_threshold=psi_drift_threshold,
            alpha=alpha,
        )
        for col in shared
    ]

    return DriftReport(
        results=results,
        n_baseline=len(baseline),
        n_current=len(current),
        psi_warning_threshold=psi_warning_threshold,
        psi_drift_threshold=psi_drift_threshold,
        alpha=alpha,
    )
