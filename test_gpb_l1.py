"""test_gpb_l1.py -- Lauf "gpb_l1" (wie gpb, aber mit lambda_l1 auf der Log-Skala und 270 Versuchen).

  Jobs        make_paper_jobs --run gpb_l1: 4720 Jobs (Zufallsgitter, TPE, GP-BO, SMAC x 59 Aufgaben x 20 Seeds),
              270 Versuche, Flags --l1 on --scales gpboost, Ordner Results_gpb_l1, Probe 8 Jobs, laengste zuerst;
              jobs_paper/gpb_l1/jobs.tsv im Repo = neu erzeugt; gpb unveraendert
  n_iter 270  run_ablation.apply_settings: SMAC Scenario n_trials = 270 (Sobol-Startdesign 67 Punkte, im Lauf gpb
              33), TPE n_trials = 270 mit n_startup_trials = 10, GP-BO n_calls = 270 mit n_initial_points = 10;
              die ersten Versuche eines 270-Laufs = ein kurzer Lauf (TPE, GP-BO im echten Suchraum)
  Zeilen      check_results/run_ablation: 270 Zeilen je Fold (Zufallsgitter, TPE, GP-BO), Pfade in Results_gpb_l1
  lambda_l1   nicht mehr konstant 0, log-uniform in [1e-8, 100]: Anteil unter 1 ~80 % (TPE-Startphase, GP-BO,
              SMAC-Stichprobe und Sobol-Startdesign) statt ~1 % linear wie im alten Lauf l1_270 (dort zusaetzlich
              aus den Ergebnisdateien, falls Results_l1_270 da ist); Zufallsgitter weiter {0, 1, 10, 100}
  Infrastruktur  refill (--run gpb_l1, hpo-gpb_l1, run_paper_job.sbatch, laengste zuerst), watch_paper,
              check_results, vergleich_l1 --variante gpb
Aufruf: .venv/bin/python -m unittest test_gpb_l1 -v
"""
import glob
import inspect
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import make_paper_jobs as mpj
import methods
import refill
import run_ablation
import SMAC_method
import watch_paper as wp
from SMAC_method import smac_configspace
from test_scales import make

METHODS = ("random_search", "tpe", "gp_bo", "SMAC")
FLAGS = "--l1 on --scales gpboost"
N = 10000
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Captured(Exception):
    pass


class JobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_list = mpj.tasks()
        sec = mpj.seconds_per_trial()
        cls.jobs = mpj.plan(mpj.build("gpb_l1", sec, cls.task_list), longest_first=True)

    def test_jobs(self):
        jobs = self.jobs
        self.assertEqual(len(jobs), 4720)
        self.assertEqual(set(jobs["method"]), set(METHODS))
        self.assertNotIn("hyperband", set(jobs["method"]))
        self.assertEqual(set(jobs["folder"]), {"Results_gpb_l1"})
        self.assertEqual(set(jobs["flags"]), {FLAGS})
        self.assertEqual(set(jobs["trials"]), {270})
        self.assertEqual(jobs["task"].nunique(), 59)
        self.assertEqual(jobs["seed"].nunique(), 20)
        self.assertEqual(len(jobs.drop_duplicates(["method", "seed", "suite", "task"])), 4720)
        for welle, group in jobs.groupby("welle"):
            self.assertEqual(len(group), 1180, welle)

    def test_longest_first(self):
        ordered = self.jobs.sort_values("reihenfolge")
        self.assertEqual(sorted(ordered["reihenfolge"]), list(range(1, 4721)))
        self.assertTrue(ordered["kernstunden"].is_monotonic_decreasing)
        # gpb behaelt die alte Reihenfolge (Welle zuerst)
        gpb = mpj.plan(mpj.build("gpb", mpj.seconds_per_trial(), self.task_list))
        self.assertTrue(gpb.sort_values("reihenfolge")["welle"].is_monotonic_increasing)

    def test_cost_factors(self):
        self.assertEqual(mpj.GPB_L1_COST_FACTOR["random_search"], 1.0)
        for m in ("tpe", "gp_bo", "SMAC"):
            self.assertEqual(mpj.GPB_L1_COST_FACTOR[m], mpj.GPB_COST_FACTOR[m])
        # 270 Versuche: TPE genau doppelt so teuer wie in gpb (kein Surrogat), GP-BO/SMAC mehr als doppelt
        gpb = mpj.build("gpb", mpj.seconds_per_trial(), self.task_list).set_index(["method", "seed", "suite", "task"])
        new = self.jobs.set_index(["method", "seed", "suite", "task"])
        key = ("tpe", 27225, 336, 361076)
        self.assertAlmostEqual(new.loc[key, "kernstunden"], 2 * gpb.loc[key, "kernstunden"], delta=0.002)

    def test_repo_files_are_current(self):
        with tempfile.TemporaryDirectory() as out:
            mpj.main_gpb(out, "jobs_paper/zeitmessung_mac.tsv", run="gpb_l1")
            for name in ("gpb_l1/jobs.tsv", "gpb_l1/arrays/probe.idx"):
                self.assertEqual(Path(out, name).read_text(), Path("jobs_paper", name).read_text(), name)
            self.assertEqual(Path(out, "submit_gpb_l1.sh").read_text(), Path("jobs_paper/submit_gpb_l1.sh").read_text())
            # gpb unveraendert (byte-gleich zu den Dateien im Repo)
            mpj.main_gpb(out, "jobs_paper/zeitmessung_mac.tsv", run="gpb")
            for name in ("gpb/jobs.tsv", "gpb/arrays/probe.idx", "submit_gpb.sh"):
                self.assertEqual(Path(out, name).read_text(), Path("jobs_paper", name).read_text(), name)

    def test_probe(self):
        repo = wp.load_jobs("gpb_l1", with_files=False)
        lines = [int(x) for x in Path("jobs_paper/gpb_l1/arrays/probe.idx").read_text().split()]
        rows = repo.iloc[[i - 1 for i in lines]]
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows["method"]), set(METHODS))
        self.assertEqual(set(zip(rows["suite"], rows["task"])), set(mpj.GPB_PROBE))
        self.assertEqual(set(rows["seed"]), {27225})
        script = Path("jobs_paper/submit_gpb_l1.sh").read_text()
        self.assertIn("--job-name=hpo-gpb_l1 jobs_paper/run_paper_job.sbatch gpb_l1", script)
        self.assertIn("--array=0-7", script)
        self.assertIn("DIR=jobs_paper/gpb_l1", script)

    def test_flags(self):
        base = ["--method", "tpe", "--seed", "1", "--result_folder", "x", "--trials", "270"]
        args = run_ablation.parse_args(base + FLAGS.split())
        self.assertEqual(args.scales, "gpboost")
        self.assertFalse(args.no_lambda_l1)
        self.assertEqual(args.trials, 270)


class Iterations270Tests(unittest.TestCase):
    """apply_settings mit 270 Versuchen: was bei SMAC, TPE und GP-BO ankommt."""

    def setUp(self):
        self.saved = (optuna.trial.Trial.suggest_float, methods.gp_minimize, methods.modified_grid_search_tune_parameters,
                      SMAC_method.ConfigurationSpace, optuna.study.Study.optimize, SMAC_method.Scenario)
        self.obj = make()              # gpboost, lambda_l1 an, num_leaves

    def tearDown(self):
        (optuna.trial.Trial.suggest_float, methods.gp_minimize, methods.modified_grid_search_tune_parameters,
         SMAC_method.ConfigurationSpace, optuna.study.Study.optimize, SMAC_method.Scenario) = self.saved

    def test_smac_n_trials_and_initial_design(self):
        from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
        seen = {}
        original = SMAC_method.Scenario
        SMAC_method.Scenario = lambda *a, **kw: seen.update(kw) or original(*a, **kw)
        run_ablation.apply_settings("SMAC", 270, False, False, tempfile.mkdtemp(), scales="gpboost")
        space, fixed = self.obj._extra_space()
        self.assertEqual(fixed, {})
        cs = smac_configspace(space, fixed)
        scenario = SMAC_method.Scenario(cs, deterministic=True, n_trials=999, seed=7, name="t")   # wie run_smac
        self.assertEqual(seen["n_trials"], 270)
        self.assertEqual(scenario.n_trials, 270)
        self.assertEqual(len(list(cs.values())), 8)
        self.assertEqual(HPOFacade.get_initial_design(scenario)._n_configs, 67)    # min(80, 0.25 x 270)
        # zum Vergleich gpb: 135 Versuche, lambda_l1 als Konstante -> 33
        space0, fixed0 = make(lambda_l1=False)._extra_space()
        cs0 = smac_configspace(space0, fixed0)
        self.assertEqual(HPOFacade.get_initial_design(original(cs0, n_trials=135, seed=7))._n_configs, 33)

    def test_tpe_n_trials_and_startup(self):
        seen = {}

        def capture(study, func, n_trials=None, **kw):
            seen.update(n_trials=n_trials, sampler=study.sampler)
            raise Captured
        optuna.study.Study.optimize = capture
        run_ablation.apply_settings("tpe", 270, False, False, tempfile.mkdtemp(), scales="gpboost")
        with self.assertRaises(Captured):
            self.obj.tpe_method(self.obj.X, self.obj.y, self.obj.X, self.obj.y)
        self.assertEqual(seen["n_trials"], 270)
        self.assertIsInstance(seen["sampler"], optuna.samplers.TPESampler)
        self.assertEqual(seen["sampler"]._n_startup_trials, 10)

    def test_gp_bo_n_calls_and_initial_points(self):
        from skopt import gp_minimize
        seen = {}

        def capture(func, dimensions, **kw):
            seen.update(kw, dimensions=dimensions)
            raise Captured
        methods.gp_minimize = capture
        run_ablation.apply_settings("gp_bo", 270, False, False, tempfile.mkdtemp(), scales="gpboost")
        with self.assertRaises(Captured):
            self.obj.gp_bo_method(self.obj.X, self.obj.y, self.obj.X, self.obj.y)
        self.assertEqual(seen["n_calls"], 270)
        self.assertNotIn("n_initial_points", seen)      # skopt-Standard
        self.assertEqual(inspect.signature(gp_minimize).parameters["n_initial_points"].default, 10)
        dims = {d.name: d for d in seen["dimensions"]}
        self.assertIn("lambda_l1", dims)
        self.assertEqual(dims["lambda_l1"].prior, "log-uniform")
        self.assertEqual((dims["lambda_l1"].low, dims["lambda_l1"].high), (1e-8, 100.0))

    def test_random_grid_keeps_l1_list(self):
        seen = []

        def capture(param_grid, *a, **kw):
            seen.append((param_grid, kw))
            raise Captured
        methods.modified_grid_search_tune_parameters = capture
        run_ablation.apply_settings("random_search", 270, False, False, tempfile.mkdtemp(), scales="gpboost")
        with self.assertRaises(Captured):
            self.obj.grid_search_method(self.obj.X, self.obj.y, self.obj.X, self.obj.y, num_try_random=270)
        grid, kw = seen[0]
        self.assertEqual(grid["lambda_l1"], [0, 1, 10, 100])
        self.assertEqual(kw.get("num_try_random"), 270)

    def test_prefix_in_real_space(self):
        """Erste n Versuche eines 2n-Laufs = n-Lauf (TPE mit Startphase und Modell, GP-BO), echter Suchraum."""
        space, fixed = self.obj._extra_space()

        def score(p):
            return (np.log10(p["learning_rate"]) + 1.5) ** 2 + np.log10(p["lambda_l1"] + 1e-8) / 100 \
                + p["bagging_fraction"]

        def tpe(k):
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=7))
            study.optimize(lambda tr: score(self.obj._suggest_optuna(tr, space, fixed)), n_trials=k)
            return [tr.params for tr in study.trials]
        self.assertEqual(tpe(15), tpe(30)[:15])

        from skopt import gp_minimize
        dims = [self.obj._skopt_dimension(p) for p in space]
        names = [d.name for d in dims]

        def gp(k):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return gp_minimize(lambda x: score(dict(zip(names, x))), dims, n_calls=k, random_state=7).x_iters
        a, b = gp(12), gp(24)
        self.assertTrue(np.allclose(np.array(a, dtype=float), np.array(b[:12], dtype=float)))


class RowsTests(unittest.TestCase):
    def test_expected_rows(self):
        import check_results
        for job in wp.load_jobs("gpb_l1", with_files=False).drop_duplicates("method").itertuples():
            path, kind, rows = check_results.result_path(job)
            self.assertTrue(str(path).startswith(f"Results_gpb_l1/{job.method}/seed_{job.seed}/gpboost/"))
            self.assertEqual(kind, "hpo")
            self.assertEqual(rows, None if job.method == "SMAC" else 270, job.method)

    @unittest.skipUnless(glob.glob("Results_l1_270/tpe/seed_*/gpboost/*__num_leaves.csv"), "Results_l1_270 fehlt")
    def test_result_file_with_270_rows_is_complete(self):
        import check_results
        source = sorted(glob.glob("Results_l1_270/tpe/seed_*/gpboost/*__num_leaves.csv"))[0]
        frame = pd.read_csv(source)
        self.assertEqual(len(frame), 5 * 270)
        self.assertEqual(check_results.check_file(source, "hpo", 270)[0], "ok")
        with tempfile.TemporaryDirectory() as tmp:
            short = Path(tmp, "x.csv")
            frame[frame["fold"] < 3].to_csv(short, index=False)
            self.assertEqual(check_results.check_file(short, "hpo", 270)[0], "angefangen")


class LambdaL1DistributionTests(unittest.TestCase):
    """lambda_l1 im Lauf gpb_l1: nicht konstant 0, log-uniform in [1e-8, 100] -> ~80 % unter 1 (8 von 10
    Zehnerpotenzen); alter Lauf (linear in (0, 100)): ~1 %."""
    SHARE = 0.8

    @classmethod
    def setUpClass(cls):
        cls.obj = make()
        cls.space, cls.fixed = cls.obj._extra_space()
        old = make("paper")
        cls.old_space, _ = old._extra_space()
        cls.old = old

    def check(self, values, tolerance=0.02, name=""):
        values = np.asarray(values, dtype=float)
        self.assertGreaterEqual(values.min(), 1e-8, name)
        self.assertLessEqual(values.max(), 100.0, name)
        self.assertGreater(np.unique(values).size, len(values) // 2, name)      # nicht konstant
        self.assertAlmostEqual(np.mean(values < 1), self.SHARE, delta=tolerance, msg=name)
        # je Zehnerpotenz etwa 10 %
        counts = np.histogram(np.log10(values), bins=np.arange(-8, 3))[0] / len(values)
        self.assertLess(np.abs(counts - 0.1).max(), 0.03 if len(values) >= N else 0.1, name)
        return np.mean(values < 1)

    def optuna_values(self, obj, space, fixed):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        values = []
        for _ in range(N):
            trial = study.ask()
            values.append(obj._suggest_optuna(trial, space, fixed)["lambda_l1"])
            study.tell(trial, 0.0)
        return np.array(values)

    def test_tpe_startup(self):
        new = self.check(self.optuna_values(self.obj, self.space, self.fixed), name="TPE")
        old = np.mean(self.optuna_values(self.old, self.old_space, {}) < 1)
        self.assertLess(old, 0.02)
        self.assertGreater(new - old, 0.7)

    def test_gp_bo(self):
        from skopt.space import Space
        dims = [self.obj._skopt_dimension(p) for p in self.space]
        frame = pd.DataFrame(Space(dims).rvs(N, random_state=1), columns=[d.name for d in dims])
        self.check(frame["lambda_l1"], name="GP-BO")

    def test_smac_sample_and_sobol_design(self):
        from scipy.stats.qmc import Sobol
        from smac.utils.configspace import transform_continuous_designs
        cs = smac_configspace(self.space, self.fixed)
        cs.seed(1)
        self.check([dict(c)["lambda_l1"] for c in cs.sample_configuration(N)], name="SMAC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = Sobol(d=len(self.space), scramble=True, seed=1).random(67)      # Startdesign mit 270 Versuchen
        configs = transform_continuous_designs(design=design, origin="test", configspace=cs)
        self.check([dict(c)["lambda_l1"] for c in configs], tolerance=0.1, name="SMAC Sobol 67")

    @unittest.skipUnless(glob.glob("Results_l1_270/tpe/seed_*/gpboost/*__num_leaves.csv"), "Results_l1_270 fehlt")
    def test_old_run_was_linear(self):
        """Aus den Ergebnisdateien des alten Laufs l1_270: in der Startphase (erste 10 Versuche) lambda_l1 unter 1
        nur in ~1-2 % (linear), ueber alle Versuche hoechstens ~57 % (GP-BO landet oft auf der Grenze 0: 54 % genau
        0; TPE ~10 %, SMAC ~23 %), also unter den ~80 % der Log-Skala schon beim Ziehen."""
        for method in ("tpe", "gp_bo", "SMAC"):
            files = sorted(glob.glob(f"Results_l1_270/{method}/seed_*/gpboost/*__num_leaves.csv"))[::20]
            frame = pd.concat(pd.read_csv(f, usecols=["iter", "lambda_l1"]) for f in files)
            start = frame[frame["iter"] < 10]["lambda_l1"]
            self.assertLess(np.mean(start < 1), 0.05, method)
            self.assertLess(np.mean(frame["lambda_l1"] < 1), self.SHARE - 0.15, method)
            self.assertGreater(frame["lambda_l1"].nunique(), 100, method)

    def test_random_grid_list(self):
        rng = np.random.RandomState(7)
        values = rng.choice([0, 1, 10, 100], size=N)
        self.assertAlmostEqual(np.mean(values < 1), 0.25, delta=0.02)    # nur der Wert 0, wie im alten Lauf


class InfrastructureTests(unittest.TestCase):
    def test_refill(self):
        self.assertEqual(refill.parse_runs("gpb_l1"), ["gpb_l1"])
        self.assertEqual(refill.parse_runs("gpb,gpb_l1"), ["gpb", "gpb_l1"])
        self.assertNotIn("gpb_l1", refill.parse_runs("alle"))
        cmd = refill.sbatch_command("gpb_l1", "normal.120h", 120, 2, "x.idx", 5)
        self.assertIn("jobs_paper/run_paper_job.sbatch", cmd)
        self.assertIn("--job-name=hpo-gpb_l1", cmd)
        self.assertEqual(cmd[-2:], ["gpb_l1", "x.idx"])

    def test_refill_order_longest_first(self):
        todo = pd.DataFrame(dict(
            lauf=["gpb_l1"] * 4 + ["gpb"] * 2, lauf_nr=[1] * 4 + [0] * 2,
            welle=[1, 4, 2, 1, 1, 2], aktion=["neu", "neu", "neu", "wiederholen", "neu", "neu"],
            partition_neu=["normal.4h", "normal.120h", "normal.24h", "normal.4h", "normal.4h", "normal.4h"],
            reihenfolge=[4, 1, 2, 3, 1, 2]))
        order = refill.order(todo).index.tolist()
        # Wiederholung zuerst, dann gpb_l1 nach reihenfolge (Welle 4 / 120h vor Welle 1), dann gpb nach Welle
        self.assertEqual(order, [3, 1, 2, 0, 4, 5])
        # andere Laeufe unveraendert: Welle, dann Partition
        other = todo[todo["lauf"] == "gpb"]
        self.assertEqual(refill.order(other).index.tolist(), [4, 5])

    def test_watch_and_check(self):
        import check_results
        self.assertIn("gpb_l1", wp.RUN_ORDER)
        self.assertIn("hpo-gpb_l1", wp.JOB_NAMES)
        self.assertEqual(len(wp.load_jobs("gpb_l1", with_files=False)), 4720)
        with self.assertRaises(SystemExit):
            check_results.main(["--run", "gpb_l2"])

    def test_vergleich_variants(self):
        import vergleich_l1 as v
        self.assertEqual(v.VARIANTS["alt"]["mit"], "Results_l1_270")
        self.assertEqual(set(v.VARIANTS["alt"]["ohne"].values()), {"Results_paper"})
        gpb = v.VARIANTS["gpb"]
        self.assertEqual(gpb["mit"], "Results_gpb_l1")
        self.assertEqual(gpb["ohne"]["random_search"], "Results_paper")
        self.assertEqual({gpb["ohne"][m] for m in ("SMAC", "tpe", "gp_bo")}, {"Results_gpb"})
        self.assertEqual(gpb["out"], "vergleich_l1_gpb")
        self.assertNotEqual(gpb["out"], v.VARIANTS["alt"]["out"])
        with tempfile.TemporaryDirectory() as tmp:     # leerer Lauf mit l1: keine Paare, kein Absturz
            summary = v.run_variant("gpb", gpb["ohne"], tmp, 135, f"{tmp}/out")
            self.assertTrue(summary.empty)
            text = Path(f"{tmp}/out.md").read_text()
            self.assertIn("log-skaliert", text)
            self.assertIn("67 statt 33", text)


if __name__ == "__main__":
    unittest.main()
