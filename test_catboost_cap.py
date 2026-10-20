"""test_catboost_cap.py -- CatBoost: border_count <= 1024 und 24h-Partition fuer CatBoost-Jobs.
Aufruf: python test_catboost_cap.py
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from methods import ParameterOptimization, CATBOOST_MAX_BORDER_COUNT
from make_euler_jobs import generate_jobs


def _make(library, n=3000):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(n, 4), columns=list("abcd"))
    y = pd.Series(X["a"] + 0.1 * rng.randn(n))
    return ParameterOptimization(
        X=X, y=y, categorical_indicator=np.zeros(4, dtype=bool),
        try_max_depth=True, suite_id=335, seed=1, library=library)


class CatBoostCapTests(unittest.TestCase):
    def test_cap_catboost(self):
        self.assertEqual(CATBOOST_MAX_BORDER_COUNT, 1024)
        self.assertEqual(_make("catboost").max_bin_val, 1024)

    def test_small_data_unchanged(self):
        self.assertEqual(_make("catboost", n=600).max_bin_val, 600)

    def test_other_libraries_unchanged(self):
        self.assertEqual(_make("xgboost").max_bin_val, 3000)
        self.assertEqual(_make("gpboost").max_bin_val, 3000)

    def test_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks.txt"
            tasks.write_text("335 361288\n")
            for libs, part, limit in ((["catboost"], "normal.24h", "24:00:00"),
                                      (["xgboost"], "normal.4h", "04:00:00")):
                for kind in ("main", "hyperband"):
                    script = generate_jobs(tasks, [1], libs, Path(tmp) / "res",
                                           Path(tmp) / f"o_{libs[0]}_{kind}", kind=kind)[0]
                    text = script.read_text()
                    self.assertIn(f"--partition={part}", text)
                    self.assertIn(f"--time={limit}", text)


if __name__ == "__main__":
    unittest.main()
