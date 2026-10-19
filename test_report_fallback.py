"""Report without Default/Hyperband/SMAC files must still give raw (non-normalized) plots."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from aggregation_methods import elo
from make_euler_jobs import generate_jobs
from make_report import run_report
from make_synthetic_results import generate
from report_plots import COLORS


def _color_pixels(png, hex_color):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    image = mpimg.imread(png)[..., :3]
    return int((np.abs(image - rgb).max(axis=2) < 0.04).sum())


class FallbackReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="hpo-fallback-", dir="/tmp")
        cls.root = Path(cls.temp.name)
        cls.results = cls.root / "results"
        generate(cls.results, iterations=30, hyperband_iterations=5)
        for folder in ("Results_Default_Run", "Results_Hyperband_2007", "Results_SMAC"):
            shutil.rmtree(cls.results / folder, ignore_errors=True)
        cls.out = cls.root / "report"
        cls.data = run_report(cls.results, ["gpboost", "xgboost"], cls.out, budgets=[10, 30])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_overall_plots_have_curve_pixels(self):
        for stem, method in (("overall_gpboost_regression_max_depth_R2", "random_search"),
                             ("overall_gpboost_classification_joint_Accuracy", "tpe"),
                             ("overall_xgboost_regression_num_leaves_R2", "gp_bo")):
            png = self.out / "plots" / f"{stem}.png"
            self.assertTrue(png.exists(), png)
            self.assertGreater(_color_pixels(png, COLORS[method]), 200, stem)

    def test_runtime_plot_has_curve_pixels(self):
        png = self.out / "plots" / "runtime" / "accuracy_gpboost_max_depth_val.png"
        self.assertGreater(_color_pixels(png, COLORS["random_search"]), 100)

    def test_readme_warns(self):
        text = (self.out / "README.txt").read_text()
        self.assertIn("NICHT normalisiert", text)
        self.assertIn("Elo", text)
        self.assertIn("<5", text)

    def test_default_elo_and_normalized_are_nan(self):
        table = pd.read_csv(self.out / "tables" / "overall_metrics.csv")
        default = table[table["method"] == "Default"]
        self.assertTrue(default["elo"].isna().all())
        self.assertTrue(table["normalized_score"].isna().all())
        self.assertTrue(table[table["method"] == "random_search"]["elo"].notna().any())

    def test_elo_nan_without_data(self):
        result = {
            "Default": np.full((3, 2), np.nan), "rmse": True, "normalized": False,
            "random_search": np.array([[1.0, 2.0]] * 3)[None, :, :, None],
            "tpe": np.array([[2.0, 1.0]] * 3)[None, :, :, None],
            "gp_bo": np.full((1, 3, 2, 1), np.nan),
        }
        values = elo(result, bootstrap_rounds=5)
        self.assertTrue(np.isnan(values["Default"]["num_leaves"]).all())
        self.assertTrue(np.isnan(values["gp_bo"]["num_leaves"]).all())
        self.assertTrue(np.isfinite(values["random_search"]["num_leaves"]).all())


class EulerKindTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hpo-euler-", dir="/tmp")
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks.txt"
        self.tasks.write_text("335 200\n")

    def tearDown(self):
        self.temp.cleanup()

    def _jobs(self, kind, **kwargs):
        return generate_jobs(self.tasks, kwargs.pop("seeds", [1, 2]),
                             kwargs.pop("libraries", ["gpboost", "xgboost"]),
                             "Res", self.root / kind, project_dir=self.root, kind=kind)

    def test_default(self):
        script, manifest, _, count = self._jobs("default")
        self.assertEqual(count, 2)
        self.assertEqual(manifest.read_text().splitlines()[0], "335\t200\tgpboost")
        text = script.read_text()
        self.assertIn("run_default.py", text)
        self.assertIn('--suite_id "$SUITE_ID"', text)
        self.assertIn(str(Path("Res").resolve()) + "/Results_Default_Run", text)
        self.assertNotIn("--seed", text)

    def test_hyperband_and_smac(self):
        script, manifest, _, count = self._jobs("hyperband")
        self.assertEqual(count, 4)
        self.assertEqual(manifest.read_text().splitlines()[0], "335\t200\t1\tgpboost")
        text = script.read_text()
        self.assertIn("--only hyperband", text)
        self.assertIn("Results_Hyperband_2007", text)
        self.assertIn('--task_id "$TASK_ID"', text)
        script, manifest, _, count = self._jobs("smac")
        self.assertEqual(count, 4)
        self.assertEqual({line.split("\t")[3] for line in manifest.read_text().splitlines()}, {"gpboost", "xgboost"})
        text = script.read_text()
        self.assertIn("--only smac", text)
        self.assertIn("Results_SMAC", text)
        for needle in ("--partition=normal.4h", "--time=04:00:00", "--cpus-per-task=1",
                       "--mem-per-cpu=4G", "OMP_NUM_THREADS=1", "run_experiment_H_SMAC.py", "cd "):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
