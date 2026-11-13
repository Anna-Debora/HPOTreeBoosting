"""test_scales_runs.py -- kurze echte Laeufe von run_ablation.py fuer --scales (statt langer 270-Versuch-Tests).

2 Aufgaben (337_361276 Klassifikation, 335_361093 Regression), Seed 27225, 1 Fold, 12 Versuche, Hyperband/BOHB/DEHB
mit R = 20; jede Methode mit --l1 on und --l1 off (--paper_grid wie im Lauf paper).
  bitgleich  --scales paper mit dem neuen Code = alter Code (Commit BASE, per git archive in einen temporaeren
             Ordner) in allen Spalten ausser runtime, fuer jede Methode und beide l1-Einstellungen.
  log        --scales gpboost: alle Werte im Bereich, lambda_l1 = 0 bei --l1 off, und die Log-Skala ist zu sehen:
             lambda_l2 (und lambda_l1 bei --l1 on) liegt in der Mehrheit der Zeilen unter 1 (log-uniform auf
             (1e-8, 100): 80 %; linear auf (0, 100): 1 %); learning_rate oefter unter 0.1 als mit paper (log 67 %,
             linear 10 % der Ziehungen; die Zeilen sind ausgewaehlte Konfigurationen, daher nur der Vergleich). Zufallsgitter und deterministisches Gitter: gpboost = paper (feste Wertelisten).
Die Verteilung mit 10000 Ziehungen je Methode prueft test_scales.py. BOHB, DEHB, HEBO nur in .venv_extra.
Dauer etwa 3 Minuten mit 8 Kernen.
Aufruf: ../HPOTreeBoosting-extra/.venv_extra/bin/python -m unittest test_scales_runs -v
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = "4042c8b"            # Stand vor --scales (extra-methods)
TASKS = ((337, 361276), (335, 361093))
SEED, TRIALS, R = 27225, 12, 20
HAVE_EXTRA = all(importlib.util.find_spec(m) for m in ("hpbandster", "dehb", "hebo"))
METHODS = ("random_search", "grid_search", "tpe", "gp_bo", "SMAC", "hyperband",
           *(("bohb", "dehb", "hebo") if HAVE_EXTRA else ()))
SEARCH = ("tpe", "gp_bo", "SMAC", "hyperband", "bohb", "dehb", "hebo")
BOUNDS = {"learning_rate": (0.001, 1), "lambda_l2": (0, 100), "lambda_l1": (0, 100), "min_data_in_leaf": (1, 1000),
          "num_leaves": (2, 1024), "bagging_fraction": (0.5, 1), "feature_fraction": (0.5, 1)}
L1 = ("on", "off")


def data_ok():
    return all((HERE / "data" / f"{s}_{t}" / f"{s}_{t}_X.csv").is_file() for s, t in TASKS)


@unittest.skipUnless(data_ok(), "data/ fuer 337_361276 und 335_361093 fehlt")
class TestScalesRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        old = root / "old_code"
        old.mkdir()
        archive = subprocess.run(["git", "archive", BASE, "*.py"], cwd=HERE, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(old)], input=archive, check=True)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
        env.pop("SLURM_JOB_ID", None)
        commands = []
        for suite, task in TASKS:
            for method in METHODS:
                for l1 in L1:
                    if method == "grid_search" and l1 == "on":
                        continue      # 270 Punkte; das Gitter haengt nicht von scales ab (test_scales)
                    common = ["--method", method, "--seed", str(SEED), "--suite_id", str(suite), "--task_id",
                              str(task), "--trials", str(TRIALS), "--folds", "1", "--hyperband_R", str(R)]
                    common += ["--paper_grid"] if l1 == "off" else []
                    commands.append([sys.executable, str(old / "run_ablation.py"), *common,
                                     *(["--no_lambda_l1"] if l1 == "off" else []),
                                     "--result_folder", str(root / f"old_{l1}")])
                    for scales in ("paper", "gpboost"):
                        commands.append([sys.executable, str(HERE / "run_ablation.py"), *common, "--l1", l1,
                                         "--scales", scales, "--result_folder", str(root / f"{scales}_{l1}")])

        def run(command):
            done = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
            return command, done

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            cls.failed = [(c, d.stderr[-2000:]) for c, d in pool.map(run, commands) if d.returncode != 0]
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def result(self, folder, method, suite, task):
        return pd.read_csv(self.root / folder / method / f"seed_{SEED}" / "gpboost" / f"{suite}_{task}__num_leaves.csv")

    def cases(self):
        for suite, task in TASKS:
            for method in METHODS:
                for l1 in L1:
                    if not (method == "grid_search" and l1 == "on"):
                        yield suite, task, method, l1

    def test_runs_succeeded(self):
        self.assertEqual(self.failed, [])

    def test_paper_bit_identical_to_old_code(self):
        for suite, task, method, l1 in self.cases():
            with self.subTest(method=method, l1=l1, task=task):
                old = self.result(f"old_{l1}", method, suite, task).drop(columns="runtime")
                new = self.result(f"paper_{l1}", method, suite, task).drop(columns="runtime")
                self.assertEqual(list(old.columns), list(new.columns))
                pd.testing.assert_frame_equal(old, new, check_exact=True)

    def test_gpboost_grids_equal_paper(self):
        for suite, task, method, l1 in self.cases():
            if method in ("random_search", "grid_search"):
                with self.subTest(method=method, l1=l1, task=task):
                    pd.testing.assert_frame_equal(
                        self.result(f"gpboost_{l1}", method, suite, task).drop(columns="runtime"),
                        self.result(f"paper_{l1}", method, suite, task).drop(columns="runtime"), check_exact=True)

    def test_gpboost_log_scales(self):
        for method in SEARCH:
            if method not in METHODS:
                continue
            for l1 in L1:
                with self.subTest(method=method, l1=l1):
                    frame = pd.concat([self.result(f"gpboost_{l1}", method, s, t) for s, t in TASKS])
                    paper = pd.concat([self.result(f"paper_{l1}", method, s, t) for s, t in TASKS])
                    self.assertEqual(list(frame.columns), list(paper.columns))
                    for name, (low, high) in BOUNDS.items():
                        self.assertTrue(frame[name].between(low, high).all(), name)
                    self.assertGreater((frame["lambda_l2"] < 1).mean(), 0.5, "lambda_l2 nicht log")
                    if l1 == "off":
                        self.assertTrue((frame["lambda_l1"] == 0).all())
                    else:
                        self.assertTrue((frame["lambda_l1"] > 0).all())
                        self.assertGreater((frame["lambda_l1"] < 1).mean(), 0.5, "lambda_l1 nicht log")
                        # paper (linear) zum Vergleich: deutlich seltener unter 1 (nur ausgewaehlte Zeilen, daher
                        # keine feste Schwelle)
                        self.assertGreater((frame["lambda_l1"] < 1).mean(), (paper["lambda_l1"] < 1).mean())
            # learning_rate: die Zeilen sind ausgewaehlte Konfigurationen (Hyperband/BOHB/DEHB: Ueberlebende; TPE/SMAC
            # lenken bei 20 Runden zu grossen Lernraten), daher keine feste Schwelle, sondern der Vergleich mit paper
            # bei gleichem Seed ueber beide l1-Einstellungen (12 Laeufe mit 4-12 Zeilen: gpboost 19-62 %, paper 0-25 %)
            with self.subTest(method=method, parameter="learning_rate"):
                share = {scales: pd.concat([self.result(f"{scales}_{l1}", method, s, t) for s, t in TASKS for l1 in L1])
                         ["learning_rate"].lt(0.1).mean() for scales in ("gpboost", "paper")}
                self.assertGreater(share["gpboost"], share["paper"], f"learning_rate nicht log: {share}")


if __name__ == "__main__":
    unittest.main()
