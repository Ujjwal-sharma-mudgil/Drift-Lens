"""Tests for driftlens.anomaly"""
import unittest

import numpy as np
import pandas as pd

from anomaly import detect, detect_dataframe, zscore_anomalies, modified_zscore_anomalies, iqr_anomalies


def _with_outliers(n=100, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=50, scale=5, size=n)
    values[0] = 500  # obvious outlier
    values[1] = -400  # obvious outlier
    return pd.Series(values, name="value")


class TestZscoreMethods(unittest.TestCase):
    def test_zscore_flags_extreme_values(self):
        series = _with_outliers(seed=1)
        result = zscore_anomalies(series)
        self.assertTrue(result.is_anomaly.iloc[0])
        self.assertTrue(result.is_anomaly.iloc[1])
        self.assertGreaterEqual(result.n_anomalies, 2)

    def test_modified_zscore_flags_extreme_values(self):
        series = _with_outliers(seed=2)
        result = modified_zscore_anomalies(series)
        self.assertTrue(result.is_anomaly.iloc[0])
        self.assertTrue(result.is_anomaly.iloc[1])

    def test_iqr_flags_extreme_values(self):
        series = _with_outliers(seed=3)
        result = iqr_anomalies(series)
        self.assertTrue(result.is_anomaly.iloc[0])
        self.assertTrue(result.is_anomaly.iloc[1])

    def test_constant_series_has_no_anomalies(self):
        series = pd.Series([5.0] * 50)
        for fn in (zscore_anomalies, modified_zscore_anomalies, iqr_anomalies):
            result = fn(series)
            self.assertEqual(result.n_anomalies, 0)

    def test_anomaly_rate_is_between_zero_and_one(self):
        series = _with_outliers(n=200, seed=4)
        result = modified_zscore_anomalies(series)
        self.assertGreaterEqual(result.anomaly_rate, 0.0)
        self.assertLessEqual(result.anomaly_rate, 1.0)


class TestDetectDispatch(unittest.TestCase):
    def test_detect_dispatches_to_correct_method(self):
        series = _with_outliers(seed=5)
        result = detect(series, method="iqr")
        self.assertEqual(result.method, "iqr")

    def test_detect_raises_on_unknown_method(self):
        series = _with_outliers(seed=6)
        with self.assertRaises(ValueError):
            detect(series, method="not_a_real_method")

    def test_detect_respects_custom_threshold(self):
        series = _with_outliers(n=200, seed=7)
        loose = detect(series, method="modified_zscore", threshold=100.0)
        strict = detect(series, method="modified_zscore", threshold=0.1)
        self.assertLessEqual(loose.n_anomalies, strict.n_anomalies)


class TestDetectDataframe(unittest.TestCase):
    def test_runs_across_all_numeric_columns(self):
        df = pd.DataFrame({
            "a": _with_outliers(n=100, seed=8),
            "b": _with_outliers(n=100, seed=9),
            "label": ["x"] * 100,
        })
        results = detect_dataframe(df)
        self.assertIn("a", results)
        self.assertIn("b", results)
        self.assertNotIn("label", results)

    def test_restricts_to_given_columns(self):
        df = pd.DataFrame({
            "a": _with_outliers(n=100, seed=10),
            "b": _with_outliers(n=100, seed=11),
        })
        results = detect_dataframe(df, columns=["a"])
        self.assertEqual(list(results.keys()), ["a"])


if __name__ == "__main__":
    unittest.main()
