"""
driftlens.anomaly
==================

Lightweight, dependency-free (beyond numpy/pandas) anomaly detection for
tabular data. Three interchangeable scoring methods are provided:

- z-score:        classic mean/std based scoring. Fast, but sensitive to
                   outliers skewing the mean/std themselves.
- modified z-score: median/MAD based scoring. Much more robust to the
                   outliers you're trying to detect.
- iqr:             Tukey's fences (Q1 - k*IQR, Q3 + k*IQR).

All three return a per-row boolean mask plus a continuous anomaly score,
so you can either threshold quickly or rank rows by "how anomalous".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AnomalyResult:
    column: str
    method: str
    threshold: float
    scores: pd.Series
    is_anomaly: pd.Series

    @property
    def n_anomalies(self) -> int:
        return int(self.is_anomaly.sum())

    @property
    def anomaly_rate(self) -> float:
        return float(self.is_anomaly.mean()) if len(self.is_anomaly) else 0.0


def zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> AnomalyResult:
    values = pd.to_numeric(series, errors="coerce")
    mean, std = values.mean(), values.std(ddof=0)
    if std == 0 or np.isnan(std):
        scores = pd.Series(np.zeros(len(values)), index=values.index)
    else:
        scores = ((values - mean) / std).abs()
    return AnomalyResult(
        column=series.name or "value",
        method="zscore",
        threshold=threshold,
        scores=scores,
        is_anomaly=scores > threshold,
    )


def modified_zscore_anomalies(series: pd.Series, threshold: float = 3.5) -> AnomalyResult:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0 or np.isnan(mad):
        scores = pd.Series(np.zeros(len(values)), index=values.index)
    else:
        scores = (0.6745 * (values - median) / mad).abs()
    return AnomalyResult(
        column=series.name or "value",
        method="modified_zscore",
        threshold=threshold,
        scores=scores,
        is_anomaly=scores > threshold,
    )


def iqr_anomalies(series: pd.Series, k: float = 1.5) -> AnomalyResult:
    values = pd.to_numeric(series, errors="coerce")
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    if iqr == 0:
        scores = pd.Series(np.zeros(len(values)), index=values.index)
        is_anomaly = pd.Series(np.zeros(len(values), dtype=bool), index=values.index)
    else:
        # Distance outside the fence, normalized by IQR (0 = inside the fence).
        below = (lower - values).clip(lower=0) / iqr
        above = (values - upper).clip(lower=0) / iqr
        scores = below + above
        is_anomaly = scores > 0
    return AnomalyResult(
        column=series.name or "value",
        method="iqr",
        threshold=k,
        scores=scores,
        is_anomaly=is_anomaly,
    )


_METHODS = {
    "zscore": zscore_anomalies,
    "modified_zscore": modified_zscore_anomalies,
    "iqr": iqr_anomalies,
}


def detect(
    series: pd.Series,
    method: str = "modified_zscore",
    threshold: Optional[float] = None,
) -> AnomalyResult:
    """Detect anomalies in a single numeric column.

    Parameters
    ----------
    method: one of "zscore", "modified_zscore", "iqr".
    threshold: overrides the method's default threshold/k value.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_METHODS)}.")
    fn = _METHODS[method]
    kwargs = {}
    if threshold is not None:
        param_name = "k" if method == "iqr" else "threshold"
        kwargs[param_name] = threshold
    return fn(series, **kwargs)


def detect_dataframe(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "modified_zscore",
    threshold: Optional[float] = None,
) -> dict[str, AnomalyResult]:
    """Run anomaly detection across every numeric column in a dataframe."""
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return {col: detect(df[col], method=method, threshold=threshold) for col in cols}
