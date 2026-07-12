"""Tests for driftlens.core"""
import unittest

import numpy as np
import pandas as pd

from core import (
    ColumnType,
    DriftStatus,
    compare_column,
    compare_datasets,
)


def _seeded(n, loc=0.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(loc=loc, scale=scale, size=n))


class TestCompareColumnNumeric(unittest.TestCase):
    def test_identical_distributions_are_ok(self):
        baseline = _seeded(500, seed=1)
        current = _seeded(500, seed=1)
        result = compare_column("value", baseline, current)
        self.assertEqual(result.column_type, ColumnType.NUMERIC)
        self.assertEqual(result.status, DriftStatus.OK)
        self.assertLess(result.psi, 0.1)

    def test_shifted_distribution_flags_drift(self):
        baseline = _seeded(500, loc=0.0, seed=2)
        current = _seeded(500, loc=5.0, seed=3)
        result = compare_column("value", baseline, current)
        self.assertEqual(result.status, DriftStatus.DRIFT)
        self.assertGreater(result.psi, 0.25)
        self.assertLess(result.p_value, 0.05)

    def test_moderate_shift_can_warn(self):
        baseline = _seeded(1000, loc=0.0, scale=1.0, seed=4)
        current = _seeded(1000, loc=0.5, scale=1.0, seed=5)
        result = compare_column("value", baseline, current)
        self.assertIn(result.status, (DriftStatus.WARNING, DriftStatus.DRIFT))

    def test_handles_empty_series(self):
        baseline = pd.Series([], dtype=float)
        current = pd.Series([], dtype=float)
        result = compare_column("value", baseline, current)
        self.assertEqual(result.psi, 0.0)


class TestCompareColumnCategorical(unittest.TestCase):
    def test_identical_categories_are_ok(self):
        baseline = pd.Series(["a", "b", "a", "c"] * 50)
        current = pd.Series(["a", "b", "a", "c"] * 50)
        result = compare_column("cat", baseline, current)
        self.assertEqual(result.column_type, ColumnType.CATEGORICAL)
        self.assertEqual(result.status, DriftStatus.OK)

    def test_new_category_is_flagged_in_detail(self):
        baseline = pd.Series(["a", "b"] * 50)
        current = pd.Series(["a", "b", "z"] * 50)
        result = compare_column("cat", baseline, current)
        self.assertIn("z", result.detail)

    def test_shifted_category_shares_flag_drift(self):
        baseline = pd.Series(["a"] * 90 + ["b"] * 10)
        current = pd.Series(["a"] * 10 + ["b"] * 90)
        result = compare_column("cat", baseline, current)
        self.assertEqual(result.status, DriftStatus.DRIFT)

    def test_boolean_column_treated_as_categorical(self):
        baseline = pd.Series([True, False] * 50)
        current = pd.Series([True, False] * 50)
        result = compare_column("flag", baseline, current)
        self.assertEqual(result.column_type, ColumnType.CATEGORICAL)


class TestCompareDatasets(unittest.TestCase):
    def setUp(self):
        self.baseline = pd.DataFrame({
            "age": _seeded(300, loc=30, seed=10),
            "plan": pd.Series(["basic", "pro"] * 150),
        })
        self.current_same = pd.DataFrame({
            "age": _seeded(300, loc=30, seed=10),
            "plan": pd.Series(["basic", "pro"] * 150),
        })
        self.current_drifted = pd.DataFrame({
            "age": _seeded(300, loc=60, seed=11),
            "plan": pd.Series(["enterprise", "pro"] * 150),
        })

    def test_no_drift_when_datasets_match(self):
        report = compare_datasets(self.baseline, self.current_same)
        self.assertEqual(report.overall_status, DriftStatus.OK)
        self.assertEqual(len(report.drifted_columns), 0)

    def test_detects_drift_across_columns(self):
        report = compare_datasets(self.baseline, self.current_drifted)
        self.assertEqual(report.overall_status, DriftStatus.DRIFT)
        self.assertIn("age", report.drifted_columns)

    def test_restricts_to_given_columns(self):
        report = compare_datasets(self.baseline, self.current_drifted, columns=["plan"])
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].column, "plan")

    def test_raises_on_no_shared_columns(self):
        other = pd.DataFrame({"unrelated": [1, 2, 3]})
        with self.assertRaises(ValueError):
            compare_datasets(self.baseline, other)

    def test_report_to_dict_roundtrip(self):
        report = compare_datasets(self.baseline, self.current_drifted)
        d = report.to_dict()
        self.assertIn("overall_status", d)
        self.assertEqual(len(d["results"]), len(report.results))

    def test_summary_is_nonempty_string(self):
        report = compare_datasets(self.baseline, self.current_drifted)
        summary = report.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("DriftLens report", summary)


if __name__ == "__main__":
    unittest.main()
