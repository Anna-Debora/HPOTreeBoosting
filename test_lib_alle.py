"""test_lib_alle.py -- Laeufe xgb_alle und cat_alle (XGBoost, CatBoost; SUCHRAEUME_XGB_CAT.md).

  Jobs        je Lauf 345 x (1 + 20 x 9) = 62445 Jobs, alle Methoden auf allen Aufgaben, T = 135, ohne lambda_l1,
              Flags --library <lib> --l1 off --scales gpboost (Gitter + --paper_grid, Default --library <lib>)
  Repo        jobs.tsv, probe.idx, submit_<lauf>.sh = neu erzeugt
  Suchraum    CatBoost: depth [1, 10] linear statt num_leaves, ohne min_data_in_leaf, border_count <= 1024,
              Bernoulli, Early Stopping Klassifikation auf Accuracy; XGBoost: wie GPBoost (Log-Skalen), reg_alpha 0, lossguide, max_depth 0
  Gitter      deterministisch 135 Punkte (CatBoost: subsample statt min_data_in_leaf), ohne lambda_l1
  Tiefe       CatBoost: getunte Tiefe in der Ergebnisdatei und in den Test-Refits (nicht -1)
  paper       mit scales="paper" unveraendert (CatBoost: max_depth -1, kein Bernoulli)
  Infrastruktur  refill/watch/check: Pfade je Bibliothek, run_alle_job.sbatch, Speicher aus kosten.tsv
Aufruf: .venv/bin/python -m unittest test_lib_alle -v
"""
import tempfile
import unittest
from pathlib import Path

import catboost
import numpy as np
import optuna
import pandas as pd

import make_paper_jobs as mpj
import methods
import refill
import run_ablation
import SMAC_method
import watch_paper as wp
from SMAC_method import ParameterOptimizationSMAC

TASK = (1001, 920074)          # housing_price_prediction, 545 Zeilen


def make(library, scales="gpboost", suite=None, task=TASK):
    path = f"data/{task[0]}_{task[1]}/{task[0]}_{task[1]}"
    X, y = pd.read_csv(f"{path}_X.csv"), pd.read_csv(f"{path}_y.csv")
    extra = dict(scales="gpboost", lambda_l1=False) if scales == "gpboost" else {}
    return ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=np.load(f"{path}_categorical_indicator.npy"),
                                     suite_id=suite or task[0], seed=3, library=library, **extra,
                                     **run_ablation.STRATEGY)


PATCHED = [(optuna.study.Study, "optimize"), (optuna.trial.Trial, "suggest_float"), (methods, "gp_minimize"),
           (methods, "modified_grid_search_tune_parameters"), (SMAC_method, "Scenario"),
           (SMAC_method, "ConfigurationSpace")]


def restore_settings(test):
    """run_ablation.apply_settings patcht Module fuer den ganzen Prozess: nach dem Test zuruecksetzen, sonst sehen
    spaetere Tests (z. B. test_gpb_l1 mit 270 Versuchen) noch die Versuchszahl dieses Tests."""
    saved = [(owner, name, getattr(owner, name)) for owner, name in PATCHED]
    test.addCleanup(lambda: [setattr(owner, name, value) for owner, name, value in saved])


class JobsTests(unittest.TestCase):
    def test_jobs(self):
        for run, library in (("xgb_alle", "xgboost"), ("cat_alle", "catboost")):
            jobs = wp.load_jobs(run, with_files=False)
            self.assertEqual(len(jobs), 345 * (1 + 20 * 9), run)
            self.assertEqual(jobs["task"].nunique(), 345)
            self.assertEqual(set(jobs["method"]), set(mpj.GPB_ALLE_METHODS))
            self.assertEqual(set(jobs["trials"]), {135})
            base = f"--library {library} --l1 off --scales gpboost"
            for method, group in jobs.groupby("method"):
                want = (f"--library {library}" if method == "default"
                        else base + (" --paper_grid" if method == "grid_search" else ""))
                self.assertEqual(set(group["flags"]), {want}, (run, method))
                if method != "default":
                    args = run_ablation.parse_args(["--method", method, "--seed", "1", "--result_folder", "x"]
                                                   + want.split())
                    self.assertTrue(args.no_lambda_l1)
                    self.assertEqual(args.library, library)

    def test_repo_files_are_current(self):
        with tempfile.TemporaryDirectory() as out:
            for name in ("speicher_xgboost_mac.tsv", "speicher_catboost_mac.tsv"):
                Path(out, name).write_text(Path("jobs_paper", name).read_text())
            for run in ("xgb_alle", "cat_alle", "gpb_alle"):
                mpj.main_alle(out, run, check=False)
                for name in (f"{run}/jobs.tsv", f"{run}/arrays/probe.idx", f"submit_{run}.sh", "run_alle_job.sbatch"):
                    self.assertEqual(Path(out, name).read_text(), Path("jobs_paper", name).read_text(), name)

    def test_library_paths(self):
        job = pd.Series(dict(method="tpe", seed=1, suite=1, task=2, folder="R", trials=135,
                             flags="--library catboost --l1 off"))
        self.assertEqual(wp.library_of(wp.job_flags(job)), "catboost")        # Series: job.flags waere pandas-Flags
        self.assertEqual(wp.job_flags(next(pd.DataFrame([job]).itertuples())), job["flags"])
        self.assertEqual(wp.library_of("--l1 off --scales gpboost"), "gpboost")
        import check_results
        path, _, _ = check_results.result_path(job)
        self.assertEqual(path.parts[-2], "catboost")

    def test_infrastructure(self):
        for run in ("xgb_alle", "cat_alle"):
            self.assertIn(run, refill.LONGEST_FIRST)
            self.assertIn("jobs_paper/run_alle_job.sbatch", refill.sbatch_command(run, "normal.4h", 4, 1, "x", 2))
            self.assertIn(run, wp.LIBRARY_RUNS)
            self.assertIn(f"hpo-{run}", wp.JOB_NAMES)
        self.assertIn("--library", Path("jobs_paper/run_alle_job.sbatch").read_text())

    def test_library_requires_gpboost_scales(self):
        with self.assertRaises(SystemExit):
            run_ablation.parse_args(["--method", "tpe", "--seed", "1", "--result_folder", "x", "--library", "xgboost"])


class SpaceTests(unittest.TestCase):
    def test_catboost_space(self):
        obj = make("catboost")
        space, fixed = obj._extra_space()
        names = {p["name"]: p for p in space}
        self.assertNotIn("num_leaves", names)
        self.assertNotIn("min_data_in_leaf", names)
        self.assertEqual((names["max_depth"]["low"], names["max_depth"]["high"], names["max_depth"]["log"]),
                         (1, 10, False))
        for name in ("learning_rate", "lambda_l2", "max_bin"):
            self.assertTrue(names[name]["log"], name)
        self.assertLessEqual(names["max_bin"]["high"], methods.CATBOOST_MAX_BORDER_COUNT)
        self.assertEqual(fixed, {"lambda_l1": 0.0})
        self.assertNotIn("max_depth", obj.other_params)

    def test_xgboost_space_like_gpboost(self):
        xgb, gpb = make("xgboost")._extra_space(), make("gpboost")._extra_space()
        self.assertEqual(xgb, gpb)

    def test_grids(self):
        for library, expected in (("xgboost", {"learning_rate", "min_data_in_leaf", "num_leaves", "lambda_l2"}),
                                  ("catboost", {"learning_rate", "max_depth", "lambda_l2", "bagging_fraction"})):
            obj = make(library)
            obj.paper_grid = True
            grid = obj._library_grid({"learning_rate": [0.01, 0.1, 1], "min_data_in_leaf": [10, 100, 1000],
                                      "lambda_l2": [0, 1, 10], "bagging_fraction": [0.5, 1],
                                      "num_leaves": [2, 4, 8, 32, 1024]}, deterministic=True)
            self.assertEqual(set(grid), expected, library)
            self.assertEqual(int(np.prod([len(v) for v in grid.values()])), 135, library)
            random = obj._library_grid({"lambda_l1": [0, 1, 10, 100], "min_data_in_leaf": [1, 10],
                                        "num_leaves": [2, 4]}, deterministic=False)
            self.assertNotIn("lambda_l1", random)

    def test_catboost_depth_in_file_and_refits(self):
        restore_settings(self)
        run_ablation.apply_settings("tpe", 12, True, False, Path(tempfile.gettempdir()) / "smac_test", scales="gpboost")
        obj = make("catboost")
        seen, orig = [], catboost.CatBoost.__init__

        def capture(self, params=None, **kw):
            seen.append(dict(params))
            return orig(self, params=params, **kw)
        catboost.CatBoost.__init__ = capture
        try:
            train, test = next(iter(obj.splits))
            rows = run_ablation.run_fold(obj, "tpe", 12, train, test, 2150, lambda_l1=False)
        finally:
            catboost.CatBoost.__init__ = orig
        depths = [p.get("depth") for p in seen]
        self.assertEqual(depths[:12], depths[12:24])                 # Validierung = Test-Refits
        self.assertEqual(rows["max_depth"].tolist(), depths[:12])     # Datei = getunte Tiefe
        self.assertTrue(all(p.get("bootstrap_type") == "Bernoulli" for p in seen))
        self.assertGreater(len(set(depths)), 2)

    def test_catboost_early_stopping_metric(self):
        # Klassifikation: Early Stopping auf Accuracy (= binary_error wie GPBoost/XGBoost), nicht auf Logloss
        seen, orig = [], catboost.CatBoost.__init__

        def capture(self, params=None, **kw):
            seen.append(dict(params))
            return orig(self, params=params, **kw)
        catboost.CatBoost.__init__ = capture
        try:
            for scales, want in (("gpboost", "Accuracy"), ("paper", None)):
                obj = make("catboost", scales=scales, task=(1002, 930068))
                seen.clear()
                X, y = obj.X.iloc[:400], obj.y.iloc[:400]
                obj._train_model_for_validation(X[:300], y[:300], X[300:], y[300:], params={"learning_rate": 0.1},
                                                num_boost_round=30)
                self.assertEqual(seen[0].get("eval_metric"), want, scales)
            obj = make("catboost")                                   # Regression: RMSE wie bisher
            seen.clear()
            X, y = obj.X.iloc[:400], obj.y.iloc[:400]
            obj._train_model_for_validation(X[:300], y[:300], X[300:], y[300:], params={"learning_rate": 0.1},
                                            num_boost_round=30)
            self.assertNotIn("eval_metric", seen[0])
        finally:
            catboost.CatBoost.__init__ = orig

    def test_best_iter_counts_trees(self):
        # XGBoost/CatBoost zaehlen die beste Iteration ab 0, GPBoost und num_boost_round der Refits ab 1; mit
        # Accuracy-Early-Stopping ist "beste Iteration 0" haeufig -> CatBoost-Refit mit 0 Iterationen brach ab
        for library in ("xgboost", "catboost"):
            obj = make(library, task=(1002, 363700))
            X, y = obj.X, pd.Series(obj.y)
            _, best = obj._train_model_for_validation(X[:250], y[:250], X[250:], y[250:],
                                                      params={"learning_rate": 0.001}, num_boost_round=40)
            self.assertGreaterEqual(best, 1, library)
            paper = make(library, scales="paper", task=(1002, 363700))
            _, best_paper = paper._train_model_for_validation(X[:250], y[:250], X[250:], y[250:],
                                                              params={"learning_rate": 0.001}, num_boost_round=40)
            if library == "xgboost":                                  # CatBoost/paper stoppt auf Logloss
                self.assertEqual(best, best_paper + 1, library)       # paper unveraendert (ab 0)

    def test_text_columns_become_categorical(self):
        # 1002_363681: "coupon" ist Text, im categorical_indicator aber nicht markiert (Euler: 145 Jobs FAILED mit
        # "DataFrame.dtypes for data must be int, float or bool")
        task = (1002, 363681)
        raw = pd.read_csv(f"data/{task[0]}_{task[1]}/{task[0]}_{task[1]}_X.csv")
        marked = np.load(f"data/{task[0]}_{task[1]}/{task[0]}_{task[1]}_categorical_indicator.npy")
        self.assertFalse(marked[list(raw.columns).index("coupon")])
        for library in ("gpboost", "xgboost", "catboost"):
            obj = make(library, scales="gpboost" if library != "gpboost" else "paper", task=task)
            self.assertIn("coupon", obj.cat_feature_names, library)
            for column in raw.columns:
                if pd.api.types.is_numeric_dtype(raw[column]) and not marked[list(raw.columns).index(column)]:
                    self.assertTrue(raw[column].equals(obj.X[column]), (library, column))   # Zahlen unveraendert
                elif library != "catboost":
                    self.assertEqual(str(obj.X[column].dtype), "category", (library, column))
            if library == "gpboost":
                self.assertEqual(obj.X["coupon"].isna().sum(), raw["coupon"].isna().sum())

    def test_paper_scales_unchanged(self):
        obj = make("catboost", scales="paper")
        self.assertEqual(obj.other_params.get("max_depth"), -1)
        self.assertFalse(obj._depth_tuned)


if __name__ == "__main__":
    unittest.main()
