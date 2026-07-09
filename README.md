import unittest

import numpy as np
import pandas as pd

from driftlens.anomaly import (
    detect,
    detect_dataframe,
    iqr_anomalies,
    modified_zscore_anomalies,
    zscore_anomalies,
)

RNG = np.random.default_rng(1)


class TestZScore(unittest.TestCase):
    def test_flags_obvious_outlier(self):
        values = pd.Series(list(RNG.normal(0, 1, 500)) + [50.0])
        result = zscore_anomalies(values, threshold=3.0)
        self.assertTrue(result.is_anomaly.iloc[-1])
        self.assertEqual(result.n_anomalies, 1)

    def test_constant_series_has_no_anomalies(self):
        values = pd.Series([5.0] * 100)
        result = zscore_anomalies(values)
        self.assertEqual(result.n_anomalies, 0)


class TestModifiedZScore(unittest.TestCase):
    def test_flags_obvious_outlier_robustly(self):
        # A couple of extra big outliers shouldn't blind the robust method
        # the way they would blind a plain z-score.
        values = pd.Series(list(RNG.normal(0, 1, 500)) + [50.0, 60.0, 70.0])
        result = modified_zscore_anomalies(values, threshold=3.5)
        self.assertGreaterEqual(result.n_anomalies, 3)

    def test_constant_series_has_no_anomalies(self):
        values = pd.Series([5.0] * 100)
        result = modified_zscore_anomalies(values)
        self.assertEqual(result.n_anomalies, 0)


class TestIQR(unittest.TestCase):
    def test_flags_values_outside_fences(self):
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = iqr_anomalies(values, k=1.5)
        self.assertTrue(result.is_anomaly.iloc[-1])

    def test_score_zero_inside_fence(self):
        values = pd.Series([1, 2, 3, 4, 5])
        result = iqr_anomalies(values, k=1.5)
        self.assertEqual(result.n_anomalies, 0)


class TestDispatch(unittest.TestCase):
    def test_detect_dispatches_to_correct_method(self):
        values = pd.Series(RNG.normal(0, 1, 200))
        for method in ("zscore", "modified_zscore", "iqr"):
            result = detect(values, method=method)
            self.assertEqual(result.method, method)

    def test_detect_raises_on_unknown_method(self):
        values = pd.Series([1, 2, 3])
        with self.assertRaises(ValueError):
            detect(values, method="not_a_real_method")

    def test_detect_dataframe_covers_numeric_columns_only(self):
        df = pd.DataFrame(
            {
                "num": RNG.normal(0, 1, 100),
                "cat": RNG.choice(["a", "b"], size=100),
            }
        )
        results = detect_dataframe(df)
        self.assertIn("num", results)
        self.assertNotIn("cat", results)

    def test_threshold_override(self):
        values = pd.Series(list(RNG.normal(0, 1, 300)) + [4.0])
        loose = detect(values, method="zscore", threshold=10.0)
        tight = detect(values, method="zscore", threshold=1.0)
        self.assertLessEqual(loose.n_anomalies, tight.n_anomalies)


if __name__ == "__main__":
    unittest.main()
