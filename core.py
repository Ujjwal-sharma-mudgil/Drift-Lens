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
    if p.sum()
