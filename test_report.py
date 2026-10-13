"""Unit and synthetic end-to-end tests for the report pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from aggregation_methods import elo, improvability
from make_report import run_report
from make_synthetic_results import generate
from report_data import load_report_data
from report_plots import runtime_step_curves
from report_tables import make_all_tables


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="hpo-report-tests-")
        cls.root = Path(cls.temp.name)
        cls.results = cls.root / "results"
        generate(cls.results, iterations=270, hyperband_iterations=11)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_loader_shapes_and_missing_smac(self):
        data = load_report_data(self.results, ["gpboost", "xgboost", "catboost"])
        self.assertEqual(len(data["tasks"]), 3)
        self.assertEqual(data["iterations"], 270)
        for library in data["libraries"]:
            self.assertEqual(len(data["seeds"][library]), 2)
            self.assertEqual(data["scores"][library]["R2"]["gp_bo"].shape, (4, 270, 3, 2))
            self.assertEqual(data["runtime"][library]["random_search"].shape, (4, 270, 3, 2))
            self.assertTrue(np.isnan(data["runtime"][library]["Default"]).all())
        for library in ("xgboost", "catboost"):
            self.assertTrue(np.isnan(data["scores"][library]["R2"]["SMAC"]).all())
        self.assertTrue(np.isfinite(data["scores"]["gpboost"]["R2"]["SMAC"]).any())

    def test_runtime_step_function_is_hand_calculable(self):
        grid = np.array([0.25, 0.5, 0.75, 1.0])
        curve = runtime_step_curves([1, 2], [0.2, 0.6], 2, grid, selection="oracle")
        np.testing.assert_allclose(curve, [np.nan, 0.2, 0.2, 0.6], equal_nan=True)

    def test_val_and_oracle_runtime_curves_differ(self):
        grid = np.array([1 / 3, 2 / 3, 1.0])
        val_curve = runtime_step_curves(
            [1, 2, 3], [0.9, 0.4, 0.5], 3, grid, selection="val"
        )
        oracle_curve = runtime_step_curves(
            [1, 2, 3], [0.9, 0.4, 0.9], 3, grid, selection="oracle"
        )
        np.testing.assert_allclose(val_curve, [0.9, 0.4, 0.5])
        np.testing.assert_allclose(oracle_curve, [0.9, 0.9, 0.9])
        self.assertFalse(np.array_equal(val_curve, oracle_curve))

    def test_budgeted_aggregations_select_requested_trials(self):
        result = {
            "Default": np.array([[10.0], [10.0], [10.0]]),
            "rmse": True,
            "logloss": False,
            "classification": False,
            "normalized": False,
            "random_search": np.array([[[[8.0]], [[6.0]], [[4.0]]]]),
        }
        values = improvability(result, budgets=[1, 3])
        np.testing.assert_allclose(values["random_search"]["num_leaves"], [0.0, 0.0])
        np.testing.assert_allclose(values["Default"]["num_leaves"], [0.2, 0.6])
        ratings = elo(result, bootstrap_rounds=5, budgets=[1, 3])
        self.assertEqual(len(ratings["random_search"]["num_leaves"]), 2)
        np.testing.assert_allclose(ratings["Default"]["num_leaves"], [1000.0, 1000.0])
        with self.assertRaises(ValueError):
            improvability(result, budgets=[4])

    def test_end_to_end_files_and_reproducible_tables(self):
        first = self.root / "report_one"
        second = self.root / "report_two"
        libraries = ["gpboost", "xgboost", "catboost"]
        budgets = [10, 30, 90, 270]
        report_one = run_report(self.results, libraries, first, budgets=budgets)
        make_all_tables(report_one, second, budgets)
        self.assertEqual(len(report_one["tasks"]), 3)
        self.assertEqual(report_one["iterations"], 270)
        self.assertTrue((first / "README.txt").exists())
        for relative in (
            "plots/overall_gpboost_regression_max_depth_R2.png",
            "plots/runtime/accuracy_gpboost_max_depth_val.png",
            "plots/library_regression_max_depth_R2_random_search.png",
            "tables/overall_metrics.csv",
            "tables/overall_metrics.tex",
            "tables/per_dataset_end_budget.csv",
            "tables/library_comparison.csv",
            "tables/runtime_summary.csv",
        ):
            self.assertTrue((first / relative).exists(), relative)
        for table in ("overall_metrics.csv", "per_dataset_end_budget.csv",
                      "library_comparison.csv", "runtime_summary.csv"):
            left = (first / "tables" / table).read_bytes()
            right = (second / "tables" / table).read_bytes()
            self.assertEqual(left, right, table)
        for relative in (
            "plots/overall_gpboost_regression_max_depth_R2.png",
            "plots/runtime/accuracy_gpboost_max_depth_val.png",
            "plots/library_regression_max_depth_R2_random_search.png",
        ):
            image = mpimg.imread(first / relative)
            self.assertGreater(image.shape[0], 200)
            self.assertGreater(image.shape[1], 200)
            self.assertGreater(float(np.std(image)), 0.01, relative)


if __name__ == "__main__":
    unittest.main()
