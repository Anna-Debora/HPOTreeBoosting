"""Focused checks for SMAC and Hyperband runtime accounting."""

import time
import unittest

import numpy as np
import pandas as pd

from SMAC_method import _align_config_runtimes
from methods import ParameterOptimization


class RuntimeAccountingTests(unittest.TestCase):
    def test_smac_repeated_configurations_align_to_result_rows(self):
        first = {"lambda_l1": 2.0, "max_depth": 4}
        second = {"lambda_l1": 3.0, "max_depth": 5}
        runtimes = {
            (("lambda_l1", 2.0), ("max_depth", 4)): [0.1, 0.2],
            (("lambda_l1", 3.0), ("max_depth", 5)): [0.3],
        }

        aligned = _align_config_runtimes([first, second], runtimes)

        self.assertEqual(len(aligned), 2)
        np.testing.assert_allclose(aligned, [0.3, 0.3])
        self.assertFalse(np.isnan(aligned).any())
        result = pd.DataFrame({"val_score": [1.0, 2.0], "runtime": aligned})
        self.assertEqual(result.columns.get_loc("runtime"), result.columns.get_loc("val_score") + 1)

    def test_hyperband_runtime_sum_covers_all_validation_training_calls(self):
        X = pd.DataFrame({"feature_a": np.arange(320), "feature_b": np.arange(320) % 7})
        y = pd.Series(np.arange(320, dtype=float) / 10)
        optimizer = ParameterOptimization(
            X=X,
            y=y,
            categorical_indicator=np.array([False, False]),
            suite_id=335,
            seed=11,
            try_max_depth=True,
        )
        train_indices, test_indices = next(iter(optimizer.splits))
        measured_training_time = 0.0

        def fake_train(X_train, y_train, X_val, y_val, params, num_boost_round=1000):
            nonlocal measured_training_time
            started = time.perf_counter()
            time.sleep(0.001)
            measured_training_time += time.perf_counter() - started
            return float(params["learning_rate"]), 1

        def fake_test_scores(**kwargs):
            frame = kwargs["df"].copy()
            frame["test_score"] = 0.5
            frame["test_log_loss"] = np.nan
            frame["test_f1_score"] = np.nan
            frame["test_rmse"] = 1.0
            frame["current_best_test_score"] = 0.5
            frame["current_best_test_log_loss"] = np.nan
            frame["current_best_test_f1_score"] = np.nan
            frame["current_best_test_rmse"] = 1.0
            return frame

        optimizer._train_model_for_validation = fake_train
        optimizer._compute_test_scores = fake_test_scores
        result = optimizer.hyperband_method(
            X_train_full=optimizer.X.iloc[train_indices],
            y_train_full=optimizer.y.iloc[train_indices],
            X_test=optimizer.X.iloc[test_indices],
            y_test=optimizer.y.iloc[test_indices],
            R=16,
            factor=2,
        )

        self.assertIn("runtime", result.columns)
        self.assertEqual(result.columns.get_loc("runtime"), result.columns.get_loc("val_score") + 1)
        self.assertFalse(result["runtime"].isna().any())
        self.assertTrue((result["runtime"] > 0).all())
        self.assertGreater(measured_training_time, 0)
        self.assertAlmostEqual(result["runtime"].sum(), measured_training_time, delta=0.03)


if __name__ == "__main__":
    unittest.main()
