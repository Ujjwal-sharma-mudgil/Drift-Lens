# Methodology

DriftLens compares a **baseline** dataset against a **current** dataset,
column by column, and classifies each column as `ok`, `warning`, or `drift`.

## Numeric columns

1. **Population Stability Index (PSI)** — bins the baseline into deciles
   (by quantile), then measures how much the current data's distribution
   across those same bins has shifted, weighted logarithmically. This is
   the standard metric used in credit-risk and MLOps monitoring.

   | PSI range   | Interpretation          |
   |-------------|--------------------------|
   | < 0.1       | No significant shift     |
   | 0.1 – 0.25  | Moderate shift (warning) |
   | > 0.25      | Significant shift (drift)|

2. **Kolmogorov-Smirnov (KS) test** — a nonparametric test of whether two
   samples come from the same distribution. Used as a secondary, p-value
   based signal: if `p < alpha` (default 0.05), that's independent evidence
   of drift even if PSI is borderline.

3. **Jensen-Shannon divergence** — a symmetric, bounded (0 to 1) measure of
   distributional difference. Useful for ranking "how different" columns
   are on a common scale, independent of type.

## Categorical columns

1. **PSI on category shares** — same idea as numeric PSI, but bins are the
   categories themselves instead of quantiles.
2. **Chi-square test of independence** — tests whether the category
   frequency distribution differs significantly between baseline and
   current.
3. **New category detection** — if `current` contains categories absent
   from `baseline` (e.g. a new signup plan, a new error code), this is
   surfaced explicitly in the result's `detail` field regardless of
   whether it crosses the PSI/chi-square thresholds.

## Column status classification

A column is marked `drift` if **either**:
- its PSI exceeds `psi_drift_threshold` (default 0.25), **or**
- its statistical test's p-value is below `alpha` (default 0.05)

It's marked `warning` if its PSI is between `psi_warning_threshold` (default
0.1) and `psi_drift_threshold`, and otherwise `ok`.

Both thresholds are configurable — see `docs/USAGE.md`.

## Anomaly detection

`driftlens.anomaly` operates on a *single* dataset (not a comparison) and
flags individual anomalous rows within it:

- **z-score**: `|x - mean| / std`. Fast, intuitive, but the mean/std used
  to compute the score are themselves distorted by the outliers you're
  trying to catch.
- **modified z-score**: `0.6745 * |x - median| / MAD`. Uses the median and
  median absolute deviation instead, so a handful of extreme values don't
  drag the reference statistics along with them. This is the recommended
  default.
- **IQR (Tukey's fences)**: flags anything outside
  `[Q1 - k*IQR, Q3 + k*IQR]`, the same rule used to draw box-plot whiskers.

All three return both a boolean flag and a continuous score, so you can
either threshold immediately or rank rows by severity.
