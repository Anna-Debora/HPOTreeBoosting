"""test_extra_methods.py -- BOHB, DEHB, HEBO (Lauf "extra") auf einem Mini-Datensatz und die Infrastruktur dazu.

  Rahmen     hyperband_schedule = Schleife von hyperband_method (Budgets in Aufrufreihenfolge, 11 Zeilen bei R = 2150);
             BOHB und DEHB werten genau dieselben Budgets in derselben Reihenfolge aus (Training nachgebildet)
  Methoden   echte Trainings auf 300 Zeilen, 5-10 Auswertungen bzw. kleines R: Spalten wie Hyperband, Zeilenzahl,
             Ganzzahlen, Bereiche, lambda_l1 = 0 ohne lambda_l1, kategorisches max_depth, gleicher Seed = gleiche
             Zahlen, DEHB-Zwischendateien ausserhalb von $HOME und danach geloescht, --resume pro Fold
  Infra      make_paper_jobs --run extra, refill (sbatch-Skript, "beide" unveraendert), speicher (Zuschlag, paper
             unberuehrt), make_table5 --extra (9 HPO-Methoden + Default, BOHB/DEHB wie Hyperband)
Die Methoden-Tests laufen nur in .venv_extra (hpbandster, dehb, HEBO), sonst werden sie uebersprungen.
Aufruf: .venv_extra/bin/python -m unittest test_extra_methods -v
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import methods
import run_ablation

HAVE_EXTRA = all(importlib.util.find_spec(m) for m in ("hpbandster", "dehb", "hebo"))
needs_extra = unittest.skipUnless(HAVE_EXTRA, "nur in .venv_extra (hpbandster, dehb, HEBO)")
HERE = Path(__file__).resolve().parent


def setUpModule():
    import job_io
    job_io.quiet_logging()            # keine GPBoost-Infos je Training


def mini_data(n=300, seed=0, classification=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(size=n), "c": rng.integers(0, 5, size=n),
                      "kat": rng.choice(["x", "y", "z"], size=n)})
    signal = X["a"] + 2 * X["b"] + (X["kat"] == "y")
    y = pd.Series((signal > signal.median()).astype(int) if classification else signal + rng.normal(0, .3, n))
    return X, y, np.array([False, False, False, True])


def mini(seed=27225, classification=False, **strategy):
    X, y, cat = mini_data(classification=classification)
    flags = dict(try_max_depth=False, try_num_leaves=True, joint_tuning_depth_leaves=False, try_num_iter=False)
    flags.update(strategy)
    return methods.ParameterOptimization(X=X, y=y, categorical_indicator=cat, suite_id=334 if classification else 336,
                                         seed=seed, library="gpboost", **flags)


def fold0(obj):
    train, test = next(iter(obj.splits))
    return dict(X_train_full=obj.X.iloc[train], y_train_full=obj.y.iloc[train], X_test=obj.X.iloc[test],
                y_test=obj.y.iloc[test])


def fake_training(obj):
    """Training nachbilden: Budgets in Aufrufreihenfolge mitschreiben, Score aus den Parametern (deterministisch),
    Test-Refits als Konstanten."""
    budgets = []

    def train(X_train, y_train, X_val, y_val, params, num_boost_round=1000):
        budgets.append(int(num_boost_round))
        score = (params["learning_rate"] - 0.3) ** 2 + params["num_leaves"] / 5000 + 1.0 / (num_boost_round + 1)
        return score, min(int(num_boost_round), 7)

    def test(X_train_full, y_train_full, X_test, y_test, df, best_iter=None):
        for c in ("test_score", "test_log_loss", "test_f1_score", "test_rmse"):
            df[c] = 0.5
        for c in ("current_best_test_score", "current_best_test_log_loss", "current_best_test_f1_score",
                  "current_best_test_rmse"):
            df[c] = 0.5
        test.best_iter = best_iter
        return df
    obj._train_model_for_validation = train
    obj._compute_test_scores = test
    return budgets, test


def expand(schedule):
    return [r for b in schedule for n, r in b["rungs"] for _ in range(n)]


class ScheduleTests(unittest.TestCase):
    def test_paper_numbers(self):
        schedule = methods.hyperband_schedule(2150, 2.8)
        self.assertEqual([b["s"] for b in schedule], list(range(7, -1, -1)))
        self.assertEqual(sum(n for b in schedule for n, _ in b["rungs"]), 3573)
        self.assertEqual(methods.hyperband_rows(2150, 2.8), 11)
        self.assertEqual(sorted({r for b in schedule for _, r in b["rungs"]}), [1, 4, 12, 34, 97, 274, 767, 2150])
        self.assertEqual(schedule[0]["rungs"][:2], [(1350, 1), (482, 4)])
        self.assertEqual([b["rows"] for b in schedule], [2, 1, 1, 1, 1, 2, 1, 2])

    def test_schedule_equals_hyperband_method(self):
        for R in (2150, 300, 30):
            obj = mini()
            budgets, _ = fake_training(obj)
            rows = obj.hyperband_method(**fold0(obj), R=R)
            self.assertEqual(budgets, expand(methods.hyperband_schedule(R, 2.8)), R)
            self.assertEqual(len(rows), methods.hyperband_rows(R, 2.8), R)

    def test_expected_rows(self):
        self.assertEqual(run_ablation.expected_rows("bohb", 135, False), 11)
        self.assertEqual(run_ablation.expected_rows("dehb", 135, False, 30), methods.hyperband_rows(30))
        self.assertEqual(run_ablation.expected_rows("hebo", 135, False), 135)
        self.assertIsNone(run_ablation.expected_rows("hyperband", 135, True))      # unveraendert
        self.assertEqual(run_ablation.expected_rows("tpe", 135, True), 135)

    def test_space(self):
        space, fixed = mini()._extra_space(lambda_l1=False)
        names = [p["name"] for p in space]
        self.assertEqual(names, ["learning_rate", "min_data_in_leaf", "lambda_l2", "max_bin", "bagging_fraction",
                                 "feature_fraction", "num_leaves"])
        self.assertEqual(fixed, {"lambda_l1": 0.0})
        kinds = {p["name"]: p["kind"] for p in space}
        self.assertEqual((kinds["min_data_in_leaf"], kinds["num_leaves"], kinds["learning_rate"]),
                         ("int", "int", "float"))
        self.assertIn("lambda_l1", [p["name"] for p in mini()._extra_space()[0]])
        depth = mini(try_num_leaves=False, try_max_depth=True)._extra_space()[0]
        self.assertEqual([p["choices"] for p in depth if p["name"] == "max_depth"], [list(range(1, 11))])
        obj = mini()
        obj.max_bin_val = 255                      # nur ein Wert: fest statt Suchraum
        space, fixed = obj._extra_space(lambda_l1=False)
        self.assertNotIn("max_bin", [p["name"] for p in space])
        self.assertEqual(fixed["max_bin"], 255)


@needs_extra
class FrameTests(unittest.TestCase):
    """BOHB und DEHB im Rahmen von Hyperband: dieselben Budgets in derselben Reihenfolge, dieselben Zeilen."""

    def check(self, method, R=300):
        obj = mini()
        budgets, test = fake_training(obj)
        rows = getattr(obj, method)(**fold0(obj), R=R, lambda_l1=False)
        self.assertEqual(budgets, expand(methods.hyperband_schedule(R, 2.8)))
        self.assertEqual(len(rows), methods.hyperband_rows(R, 2.8))
        self.assertEqual(list(rows.columns), methods.HB_COLUMNS)
        self.assertEqual(test.best_iter, 7)
        return obj, rows

    def test_bohb(self):
        obj, _ = self.check("bohb_method")
        self.assertGreater(obj.model_based_picks[0], 0)          # das KDE-Modell wird benutzt

    def test_dehb(self):
        self.check("dehb_method")


@needs_extra
class MethodTests(unittest.TestCase):
    """Echte Trainings, kleine Budgets (je unter 2 Minuten)."""

    def check_rows(self, rows, n, depth=None):
        self.assertEqual(len(rows), n)
        self.assertTrue((rows["lambda_l1"] == 0).all())
        for c in ("min_data_in_leaf", "num_leaves", "max_bin", "max_depth"):
            self.assertTrue(np.all(rows[c] == rows[c].round()), c)
        self.assertTrue(rows["learning_rate"].between(0.001, 1).all())
        self.assertTrue(rows["min_data_in_leaf"].between(1, 1000).all())
        self.assertTrue(rows["num_leaves"].between(2, 1024).all() if depth is None else (rows["num_leaves"] == 1024).all())
        self.assertTrue(rows["bagging_fraction"].between(0.5, 1).all())
        self.assertTrue(rows["max_bin"].between(255, 300).all())
        if depth is None:
            self.assertTrue((rows["max_depth"] == -1).all())
        else:
            self.assertTrue(rows["max_depth"].isin(depth).all())
        self.assertTrue(rows["test_score"].notna().all())
        self.assertTrue((rows["current_best_test_score"].diff().dropna() >= 0).all())

    def test_bohb(self):
        a = mini().bohb_method(**fold0(mini()), R=30, lambda_l1=False)
        b = mini().bohb_method(**fold0(mini()), R=30, lambda_l1=False)
        self.check_rows(a, methods.hyperband_rows(30))
        self.assertEqual(list(a.columns), methods.HB_COLUMNS)
        pd.testing.assert_frame_equal(a.drop(columns="runtime"), b.drop(columns="runtime"))

    def test_dehb(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"TMPDIR": tmp}):
            import job_io
            made = []
            original = job_io.temp_dir
            with mock.patch.object(job_io, "temp_dir", lambda prefix="hpo_": made.append(original(prefix)) or made[-1]):
                a = mini().dehb_method(**fold0(mini()), R=30, lambda_l1=False)
            self.assertTrue(str(made[0]).startswith(os.path.realpath(tmp)) or str(made[0]).startswith(tmp))
            self.assertFalse(made[0].exists())                     # danach geloescht
        b = mini().dehb_method(**fold0(mini()), R=30, lambda_l1=False)
        self.check_rows(a, methods.hyperband_rows(30))
        pd.testing.assert_frame_equal(a.drop(columns="runtime"), b.drop(columns="runtime"))

    def test_hebo(self):
        a = mini().hebo_method(**fold0(mini()), n_trials=8, lambda_l1=False)
        b = mini().hebo_method(**fold0(mini()), n_trials=8, lambda_l1=False)
        self.check_rows(a, 8)
        pd.testing.assert_frame_equal(a.drop(columns="runtime"), b.drop(columns="runtime"))
        self.assertGreater(a["learning_rate"].nunique(), 5)        # nicht immer derselbe Vorschlag

    def test_categorical_max_depth_and_classification(self):
        depth = list(range(1, 11))
        obj = mini(classification=True, try_num_leaves=False, try_max_depth=True)
        self.check_rows(obj.hebo_method(**fold0(obj), n_trials=5, lambda_l1=False), 5, depth)
        obj = mini(classification=True, try_num_leaves=False, try_max_depth=True)
        self.check_rows(obj.bohb_method(**fold0(obj), R=10, lambda_l1=False), methods.hyperband_rows(10), depth)
        obj = mini(classification=True, try_num_leaves=False, try_max_depth=True)
        self.check_rows(obj.dehb_method(**fold0(obj), R=10, lambda_l1=False), methods.hyperband_rows(10), depth)

    def test_lambda_l1_tuned_by_default(self):
        rows = mini().hebo_method(**fold0(mini()), n_trials=5)
        self.assertGreater(rows["lambda_l1"].max(), 0)


@needs_extra
class ResumeTests(unittest.TestCase):
    """run_ablation.py --resume pro Fold: Abbruch nach Fold 0 + halbem Fold 1 ergibt dieselben Zahlen."""

    def run_job(self, folder, method, resume=False):
        cmd = [sys.executable, str(HERE / "run_ablation.py"), "--method", method, "--seed", "27225", "--suite_id",
               "336", "--task_id", "99", "--result_folder", folder, "--folds", "2", "--trials", "5",
               "--hyperband_R", "10", "--no_lambda_l1"] + (["--resume"] if resume else [])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.work)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = self.tmp.name
        X, y, cat = mini_data()
        data = Path(self.work) / "data" / "336_99"
        data.mkdir(parents=True)
        X.to_csv(data / "336_99_X.csv", index=False)
        y.to_frame("y").to_csv(data / "336_99_y.csv", index=False)
        np.save(data / "336_99_categorical_indicator.npy", cat)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_equals_single_run(self):
        for method in ("bohb", "hebo"):
            whole, broken = f"{self.work}/A", f"{self.work}/B"
            self.run_job(whole, method)
            path = lambda root: Path(root) / method / "seed_27225" / "gpboost" / "336_99__num_leaves.csv"
            full = pd.read_csv(path(whole))
            path(broken).parent.mkdir(parents=True)
            part = pd.concat([full[full["fold"] == 0], full[full["fold"] == 1].iloc[:2]])
            part.to_csv(path(broken), index=False)
            self.run_job(broken, method, resume=True)
            pd.testing.assert_frame_equal(pd.read_csv(path(broken)).drop(columns="runtime"),
                                          full.drop(columns="runtime"))


class InfraTests(unittest.TestCase):
    def test_make_paper_jobs_extra(self):
        import make_paper_jobs as mpj
        task_list = mpj.tasks()
        sec = {(t.suite, t.task): 1.0 for t in task_list.itertuples()}
        jobs = mpj.plan(mpj.build("extra", sec, task_list))
        self.assertEqual(len(jobs), 59 * 20 * 3)
        self.assertEqual(set(jobs["method"]), {"bohb", "dehb", "hebo"})
        self.assertEqual(set(jobs["folder"]), {"Results_extra"})
        self.assertEqual(set(jobs["flags"]), {"--no_lambda_l1"})
        self.assertEqual(set(jobs["trials"]), {135})
        wave1 = jobs[jobs["welle"] == 1]
        self.assertEqual(sorted(set(wave1["seed"])), sorted(mpj.SEEDS[:5]))
        self.assertEqual(list(mpj.RUNS), ["paper", "l1_270"])                 # Standardaufruf unveraendert
        self.assertEqual(mpj.METHODS[0], "default")

    def test_refill_extra(self):
        import refill
        cmd = refill.sbatch_command("extra", "normal.4h", 4, 1.5, "x.idx", 3)
        self.assertIn("jobs_paper/run_extra_job.sbatch", cmd)
        self.assertIn("--job-name=hpo-extra", cmd)
        self.assertIn("--account=ls_math", cmd)
        self.assertIn("jobs_paper/run_paper_job.sbatch", refill.sbatch_command("paper", "normal.4h", 4, 1, "x", 3))
        self.assertEqual(refill.RUNS, ("paper", "l1_270"))                   # --run beide wie bisher

    def test_speicher_extra(self):
        import speicher
        euler = pd.DataFrame([dict(suite=336, task=1, name="t", method="tpe", jobs=10, maxrss_gb=1.0, oom=0,
                                   oom_bei=0.0),
                              dict(suite=336, task=1, name="t", method="hebo", jobs=2, maxrss_gb=5.0, oom=1,
                                   oom_bei=6.0)])
        plan = speicher.Plan(mac={}, euler=euler)
        base = speicher.Plan(mac={}, euler=euler[euler["method"] == "tpe"])
        for method in ("tpe", "gp_bo", "hyperband"):                         # paper/l1_270 unberuehrt
            self.assertEqual(plan(336, 1, method), base(336, 1, method))
        self.assertEqual(plan.source(336, 1, "bohb")[0], 1.0 + speicher.EXTRA_OVERHEAD_GB["bohb"])
        self.assertEqual(plan(336, 1, "hebo"), 8)                            # OOM bei 6 GB -> naechste Stufe

    def test_table5_extra(self):
        import make_table5 as t5
        import test_make_table5 as base
        with tempfile.TemporaryDirectory() as root:
            paper, extra = f"{root}/paper", f"{root}/extra"
            base.write_new_layout(paper, budget=3)
            base.write_new_layout(extra, budget=3, values={"bohb": (0.95, 0.4), "dehb": (0.3, 1.2),
                                                           "hebo": (0.85, 0.55)})
            # BOHB wie Hyperband: Endwert = letzte Zeile (2 Zeilen je Fold)
            bohb = Path(extra) / "bohb"
            for csv in bohb.rglob("*.csv"):
                frame = pd.read_csv(csv)
                frame = frame.groupby("fold").head(2).copy()
                frame["current_best_test_score"] = np.tile([0.2, 0.95], len(frame) // 2)
                frame.to_csv(csv, index=False)
            data = t5.load_folders(f"{paper}+{extra}")
            table, _ = t5.table5(data, 3, columns=t5.EXTRA_METHODS)
            self.assertEqual(set(table["method"]), set(t5.EXTRA_METHODS))
            ranks = table[(table["block"] == "rank") & (table["metric"] == "R2")].set_index("method")["value"]
            self.assertEqual(ranks["BOHB"], 1.0)
            self.assertEqual(ranks.max(), 10.0)                                # 9 HPO-Methoden + Default
            self.assertEqual(ranks["Default"], 10.0)
            seven, _ = t5.table5(t5.load(paper), 3)                           # 7-Methoden-Tabelle
            self.assertEqual(set(seven["method"]), set(t5.METHODS))
            seven_mixed, _ = t5.table5(data, 3)                               # Extra-Methoden aendern sie nicht
            pd.testing.assert_frame_equal(seven, seven_mixed)
            t5.main([("Paper+Extra", f"{paper}+{extra}", [3])], f"{root}/t5x", columns=t5.EXTRA_METHODS)
            out = pd.read_csv(f"{root}/t5x.csv")
            self.assertEqual(set(out["method"]), set(t5.EXTRA_METHODS))
            self.assertTrue(out.loc[out["method"] == "HEBO", "paper"].isna().all())


if __name__ == "__main__":
    unittest.main()
