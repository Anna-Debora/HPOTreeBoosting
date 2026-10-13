"""Unit and synthetic end-to-end tests for the report pipeline."""

from __future__ import annotations

import argparse
import os
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
from make_euler_jobs import generate_jobs
import methods
import optuna
import SMAC_method
import run_default
import run_experiment_H_SMAC
import run_experiments
import utils_plot_maker as upm


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

    def test_mini_full_loader_matches_create_scores_dict(self):
        root = Path("/tmp/mini_full")
        suite_id, task_id, seed, library, trial_count = 335, 361288, 27225, "gpboost", 10
        main_path = root / "Results" / f"seed_{seed}" / library / f"{suite_id}_{task_id}.csv"
        self._create_mini_full(root, suite_id, task_id, seed, library, trial_count)

        original_folds = upm.FOLDS
        original_create_scores_dict = upm.create_scores_dict

        def one_fold_create_scores_dict(*args, **kwargs):
            saved_folds = upm.FOLDS
            saved_num_folds = upm.NUM_FOLDS
            upm.FOLDS = [0]
            upm.NUM_FOLDS = 1
            try:
                return original_create_scores_dict(*args, **kwargs)
            finally:
                upm.FOLDS = saved_folds
                upm.NUM_FOLDS = saved_num_folds

        upm.create_scores_dict = one_fold_create_scores_dict
        try:
            data = load_report_data(root, [library])
        finally:
            upm.FOLDS = original_folds
            upm.create_scores_dict = original_create_scores_dict
        self.assertEqual([(task["suite_id"], task["task_id"]) for task in data["tasks"]],
                         [(suite_id, task_id)])
        self.assertEqual(data["seeds"][library], [seed])
        self.assertEqual(data["iterations"], trial_count)
        self.assertFalse(any("Fehlt:" in warning for warning in data["warnings"]),
                         data["warnings"])
        self.assertTrue(np.isfinite(data["scores"][library]["R2"]["Default"]).any())

        reader_result = data["create_scores_dict"][library][(False, "score")][0]
        rmse_result = data["create_scores_dict"][library][(False, "rmse")][0]
        metric_map = {"R2": reader_result, "RMSE": rmse_result}
        for method in ("grid_search", "random_search", "tpe", "gp_bo", "SMAC"):
            strategy_count = 2 if method == "grid_search" else 3 if method == "SMAC" else 4
            for strategy_index in range(strategy_count):
                direct = data["scores"][library]["R2"][method][strategy_index, :, 0, 0]
                expected = metric_map["R2"][method][strategy_index, :, 0, 0]
                self.assertEqual(len(expected), trial_count)
                np.testing.assert_allclose(direct, expected, equal_nan=True)
                direct_rmse = data["scores"][library]["RMSE"][method][strategy_index, :, 0, 0]
                expected_rmse = metric_map["RMSE"][method][strategy_index, :, 0, 0]
                np.testing.assert_allclose(direct_rmse, expected_rmse, equal_nan=True)
                for budget_index in (0, trial_count - 1):
                    self.assertEqual(direct[budget_index], expected[budget_index])
        direct_default = data["scores"][library]["R2"]["Default"][:, 0, 0]
        np.testing.assert_allclose(
            direct_default, reader_result["Default"][:, 0], equal_nan=True
        )

        self.assertNotIn("runtime", reader_result)
        main_frame = __import__("pandas").read_csv(main_path)
        for method in ("grid_search", "random_search", "tpe", "gp_bo"):
            for strategy_index, (try_leaves, joint, num_iter) in enumerate((
                (True, False, False), (False, False, False),
                (False, True, False), (False, False, True),
            )):
                rows = main_frame.loc[
                    main_frame["method"].eq(method)
                    & main_frame["try_num_leaves"].eq(try_leaves)
                    & main_frame["joint_tuning_depth_leaves"].eq(joint)
                    & main_frame["try_num_iter"].eq(num_iter)
                ].sort_values("iter")
                if rows.empty:
                    continue
                expected_runtime = np.cumsum(rows["runtime"].to_numpy(dtype=float))
                actual_runtime = data["runtime"][library][method][strategy_index, :, 0, 0]
                np.testing.assert_allclose(
                    actual_runtime, expected_runtime, rtol=1e-10, atol=1e-10,
                    err_msg=f"{method}/{strategy_index}: actual={actual_runtime[:3]}, "
                            f"expected={expected_runtime[:3]}",
                )

        runtime_sources = {
            "hyperband": root / "Results_Hyperband_2007" / f"seed_{seed}" / library
            / f"{suite_id}_{task_id}.csv",
            "SMAC": root / "Results_SMAC" / f"seed_{seed}" / f"{suite_id}_{task_id}.csv",
        }
        for method, path in runtime_sources.items():
            frame = __import__("pandas").read_csv(path)
            flags_by_strategy = (
                (True, False, False), (False, False, False), (False, True, False),
            )
            for strategy_index, (try_leaves, joint, num_iter) in enumerate(flags_by_strategy):
                rows = frame.loc[
                    frame["method"].eq(method)
                    & frame["try_num_leaves"].eq(try_leaves)
                    & frame["joint_tuning_depth_leaves"].eq(joint)
                    & frame["try_num_iter"].eq(num_iter)
                ].sort_values("iter")
                self.assertFalse(rows.empty, f"{method}/{strategy_index} CSV block missing")
                expected_runtime = np.cumsum(rows["runtime"].to_numpy(dtype=float))
                actual_runtime = data["runtime"][library][method][strategy_index, :, 0, 0]
                if method == "hyperband":
                    positions = np.rint(np.linspace(0, trial_count - 1, len(rows))).astype(int)
                    actual_runtime = actual_runtime[positions]
                np.testing.assert_allclose(
                    actual_runtime, expected_runtime, rtol=1e-10, atol=1e-10,
                    err_msg=f"{method}/{strategy_index} cumulative runtimes",
                )

        default_frame = __import__("pandas").read_csv(
            root / "Results_Default_Run" / library / f"{suite_id}_{task_id}.csv"
        )
        hyperband_path = (root / "Results_Hyperband_2007" / f"seed_{seed}" / library
                          / f"{suite_id}_{task_id}.csv")
        smac_path = root / "Results_SMAC" / f"seed_{seed}" / f"{suite_id}_{task_id}.csv"
        self.assertIn("current_best_test_score", default_frame.columns)
        self.assertTrue(hyperband_path.exists())
        self.assertTrue(smac_path.exists())
        self.assertTrue(np.isfinite(data["scores"][library]["R2"]["hyperband"]).any())
        self.assertTrue(np.isfinite(data["scores"][library]["R2"]["SMAC"]).any())
        upm.create_scores_dict = one_fold_create_scores_dict
        try:
            report_data = run_report(root, [library], root / "report", budgets=[1, trial_count])
        finally:
            upm.FOLDS = original_folds
            upm.create_scores_dict = original_create_scores_dict
        self.assertFalse(any("Fehlt:" in warning for warning in report_data["warnings"]),
                 report_data["warnings"])
        overall = __import__("pandas").read_csv(root / "report" / "tables" / "overall_metrics.csv")
        elo_values = overall.loc[overall["domain"] == "regression", "elo"].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(elo_values).any(), overall.to_string())

    def test_euler_job_generator_only_writes_sbatch_text(self):
        task_file = self.root / "euler_tasks.txt"
        task_file.write_text("335 361288\n334 361110\n")
        output = self.root / "euler_jobs"
        script, manifest, command, job_count = generate_jobs(
            task_file, seeds=[27225, 34326], libraries=["gpboost", "xgboost"],
            results="Results", out=output,
        )
        text = script.read_text()
        self.assertEqual(job_count, 8)
        self.assertEqual(len(manifest.read_text().splitlines()), 8)
        self.assertIn("#SBATCH --partition=normal.4h", text)
        self.assertIn("#SBATCH --time=04:00:00", text)
        self.assertIn("stack/2024-06 gcc/12.2.0 python/3.9.18", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("run_experiments.py", text)
        self.assertIn("partial results are not saved and are lost", text)
        self.assertTrue(command.startswith("sbatch "))

    @staticmethod
    def _create_mini_full(root, suite_id, task_id, seed, library, trial_count):
        project_root = Path(__file__).resolve().parent
        root.mkdir(parents=True, exist_ok=True)
        old_cwd = os.getcwd()
        original_grid = methods.ParameterOptimization.grid_search_method
        original_optimize = optuna.study.Study.optimize
        original_gp_minimize = methods.gp_minimize
        original_hyperband_method = methods.ParameterOptimization.hyperband_method
        original_split = methods.ParameterOptimization._split_data
        original_smac_split = SMAC_method.ParameterOptimizationSMAC._split_data
        original_scenario = SMAC_method.Scenario

        def small_grid(self, *args, num_try_random=None, **kwargs):
            return original_grid(
                self, *args, num_try_random=trial_count if num_try_random in (None, 270)
                else min(num_try_random, trial_count), **kwargs
            )

        def small_optimize(self, func, *args, **kwargs):
            kwargs["n_trials"] = min(kwargs.get("n_trials", trial_count), trial_count)
            return original_optimize(self, func, *args, **kwargs)

        def small_gp_minimize(func, dimensions, *args, **kwargs):
            kwargs["n_calls"] = trial_count
            return original_gp_minimize(func, dimensions, *args, **kwargs)

        def small_hyperband_method(self, *args, **kwargs):
            kwargs["R"] = 9
            return original_hyperband_method(self, *args, **kwargs)

        methods.ParameterOptimization.grid_search_method = small_grid
        optuna.study.Study.optimize = small_optimize
        methods.gp_minimize = small_gp_minimize
        methods.ParameterOptimization.hyperband_method = small_hyperband_method
        methods.ParameterOptimization._split_data = lambda self: list(original_split(self))[:1]
        SMAC_method.ParameterOptimizationSMAC._split_data = (
            lambda self: list(original_smac_split(self))[:1]
        )
        SMAC_method.Scenario = lambda *args, **kwargs: original_scenario(
            *args, **{**kwargs, "n_trials": trial_count}
        )
        try:
            os.chdir(root)
            (root / "suite_335_tasks.txt").write_text(f"{task_id}\n{task_id}\n")
            data_link = root / "data"
            if not data_link.exists():
                data_link.symlink_to(project_root / "data", target_is_directory=True)

            run_experiments.main(argparse.Namespace(
                suite_id=str(suite_id), task_id=str(task_id), seed=str(seed),
                result_folder=str(root / "Results"), library=library,
            ))
            run_default.main(argparse.Namespace(
                task_id="7", result_folder=str(root / "Results_Default_Run"), library=library,
            ))
            default_path = root / "Results_Default_Run" / library / f"{suite_id}_{task_id}.csv"
            default_frame = __import__("pandas").read_csv(default_path)
            default_frame.groupby("fold", sort=False).head(trial_count).to_csv(default_path, index=False)

            (root / "suite_335_tasks.txt").write_text(f"{task_id}\n")
            run_experiment_H_SMAC.main(argparse.Namespace(
                suite_id=str(suite_id), seed=str(seed),
                result_folder_SMAC=str(root / "Results_SMAC"),
                result_folder_H=str(root / "Results_Hyperband_2007"), library=library,
            ))
        finally:
            os.chdir(old_cwd)
            methods.ParameterOptimization.grid_search_method = original_grid
            optuna.study.Study.optimize = original_optimize
            methods.gp_minimize = original_gp_minimize
            methods.ParameterOptimization.hyperband_method = original_hyperband_method
            methods.ParameterOptimization._split_data = original_split
            SMAC_method.ParameterOptimizationSMAC._split_data = original_smac_split
            SMAC_method.Scenario = original_scenario

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
