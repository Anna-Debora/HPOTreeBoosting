"""test_scales.py -- Skalen der Suchraeume (--scales paper|gpboost, --l1 on|off).

  Verteilung  je Methode 10000 gezogene Werte aus dem Sampler, den die Methode benutzt (TPE und Hyperband:
              optuna suggest_* mit RandomSampler = Startphase von TPE und alle Ziehungen von Hyperband; GP-BO:
              skopt-Raum rvs; SMAC: ConfigSpace-Stichprobe und Sobol-Startdesign von SMAC; BOHB: ConfigSpace-
              Stichprobe; DEHB: Einheitswuerfel -> vector_to_configspace; HEBO: DesignSpace.sample):
              Min/Max im Bereich; log-Parameter: Anteil je Zehnerpotenz = log-Laenge des Abschnitts / log-Laenge des
              Bereichs (bei vollen Zehnerpotenzen gleich viele); lineare Parameter: 10 gleich breite Abschnitte je
              etwa 10 %.
  SMAC        Default-Konfiguration im Bereich und positiv (gpboost), Konstante lambda_l1 = 0 bei --l1 off,
              Groesse des Sobol-Startdesigns wie im Paper-Lauf.
  Struktur    paper: _extra_space linear wie vorher; gpboost: welche Parameter log sind, Grenzen 1e-8 / 1,
              max_depth; Zufallsgitter und deterministisches Gitter haengen nicht von scales ab; run_ablation --l1.
BOHB, DEHB und HEBO nur in .venv_extra (sonst uebersprungen).
Aufruf: .venv/bin/python -m unittest test_scales -v   (alles: ../HPOTreeBoosting-extra/.venv_extra/bin/python)
"""
import importlib.util
import unittest
import warnings

import numpy as np
import optuna
import pandas as pd

import methods
import run_ablation
from SMAC_method import smac_configspace

N = 10000
HAVE_EXTRA = all(importlib.util.find_spec(m) for m in ("hpbandster", "dehb", "hebo"))
needs_extra = unittest.skipUnless(HAVE_EXTRA, "nur in .venv_extra (hpbandster, dehb, HEBO)")
LOG = {"learning_rate", "lambda_l1", "lambda_l2", "num_leaves", "min_data_in_leaf", "max_bin"}
BOUNDS = {"learning_rate": (0.001, 1.0), "lambda_l1": (1e-8, 100.0), "lambda_l2": (1e-8, 100.0),
          "min_data_in_leaf": (1, 1000), "max_bin": (255, 10000), "num_leaves": (2, 1024),
          "bagging_fraction": (0.5, 1.0), "feature_fraction": (0.5, 1.0), "max_depth": (1, 10)}
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make(scales="gpboost", lambda_l1=True, strategy="num_leaves", rows=12000):
    """ParameterOptimization ohne Training (max_bin_val = 10000 bei 12000 Zeilen)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=rows), "b": rng.normal(size=rows)})
    y = pd.Series(rng.normal(size=rows))
    flags = dict(try_num_leaves=strategy == "num_leaves", try_max_depth=strategy == "max_depth",
                 joint_tuning_depth_leaves=strategy == "joint")
    return methods.ParameterOptimization(X=X, y=y, categorical_indicator=[False, False], suite_id=336, seed=7,
                                         scales=scales, lambda_l1=lambda_l1, **flags)


def edges(low, high, log):
    """Abschnittsgrenzen: log -> Zehnerpotenzen (plus Bereichsgrenzen), linear -> 10 gleich breite."""
    if not log:
        return np.linspace(low, high, 11)
    inner = [10.0 ** k for k in range(int(np.floor(np.log10(low))) + 1, int(np.ceil(np.log10(high))))]
    return np.array([low, *[e for e in inner if low < e < high], high])


class DistributionMixin:
    def check(self, name, values, integer=False):
        values = np.asarray(values, dtype=float)
        low, high = BOUNDS[name]
        log = name in LOG
        self.assertEqual(len(values), N, name)
        self.assertGreaterEqual(values.min(), low, name)
        self.assertLessEqual(values.max(), high, name)
        cuts = edges(low, high, log)
        counts = np.histogram(values, bins=cuts)[0] / N
        scale = np.log if log else (lambda v: v)
        expected = np.diff(scale(cuts)) / (scale(high) - scale(low))
        for k, (got, want) in enumerate(zip(counts, expected)):
            # Ganzzahlen: Runden verschiebt Masse an den Abschnittsgrenzen, am staerksten bei 1..9 (optuna zieht
            # log-uniform auf [low - 0.5, high + 0.5]: 1..9 bekommt log(19)/log(2001) = 39 % statt 33 %, 100..1000 30 %)
            tolerance = (0.07 if cuts[k] < 10 else 0.05) if integer else 0.02
            self.assertAlmostEqual(got, want, delta=tolerance,
                                   msg=f"{name} Abschnitt {cuts[k]:g}-{cuts[k + 1]:g}: {got:.3f} statt {want:.3f}")
        if log and name in ("learning_rate", "lambda_l1", "lambda_l2"):
            # volle Zehnerpotenzen: etwa gleich haeufig
            self.assertLess(counts.max() - counts.min(), 0.03, f"{name}: {np.round(counts, 3)}")

    def check_all(self, frame, space):
        for p in space:
            self.check(p["name"], frame[p["name"]], integer=p["kind"] == "int")


class TestSpace(unittest.TestCase):
    def test_paper_space_is_linear_and_unchanged(self):
        space, fixed = make("paper")._extra_space()
        self.assertTrue(all(not p["log"] for p in space))
        bounds = {p["name"]: (p["low"], p["high"]) for p in space}
        self.assertEqual(bounds["lambda_l1"], (0.0, 100.0))
        self.assertEqual(bounds["lambda_l2"], (0.0, 100.0))
        self.assertEqual(fixed, {})

    def test_gpboost_space(self):
        space, fixed = make()._extra_space()
        by_name = {p["name"]: p for p in space}
        self.assertEqual({n for n, p in by_name.items() if p["log"]}, LOG)
        for name, (low, high) in BOUNDS.items():
            if name in by_name:
                self.assertEqual((by_name[name]["low"], by_name[name]["high"]), (low, high), name)
        self.assertEqual(fixed, {})

    def test_l1_off(self):
        space, fixed = make(lambda_l1=False)._extra_space()
        self.assertNotIn("lambda_l1", [p["name"] for p in space])
        self.assertEqual(fixed, {"lambda_l1": 0.0})

    def test_max_depth_scales(self):
        p = {q["name"]: q for q in make(strategy="max_depth")._extra_space()[0]}["max_depth"]
        self.assertEqual((p["kind"], p["low"], p["high"], p["log"]), ("int", 1, 10, False))
        p = {q["name"]: q for q in make(strategy="joint")._extra_space()[0]}["max_depth"]
        self.assertEqual(p["kind"], "cat")   # -1 liegt auf keiner Skala

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            make("log")
        with self.assertRaises(ValueError):
            make("paper", lambda_l1=False)


class TestDistributions(DistributionMixin, unittest.TestCase):
    def setUp(self):
        self.obj = make()
        self.space, self.fixed = self.obj._extra_space()

    def optuna_frame(self, suffix):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        rows = []
        for _ in range(N):
            trial = study.ask()
            rows.append(self.obj._suggest_optuna(trial, self.space, self.fixed, suffix=suffix))
            study.tell(trial, 0.0)
        return pd.DataFrame(rows)

    def test_tpe(self):
        self.check_all(self.optuna_frame(""), self.space)

    def test_tpe_startup_trials_use_same_prior(self):
        # TPE zieht die ersten n_startup_trials mit dem RandomSampler, danach aus Parzen-Schaetzern auf derselben
        # (log-)Skala: die Verteilung sitzt in IntDistribution/FloatDistribution(log=True)
        trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1)).ask()
        self.obj._suggest_optuna(trial, self.space, self.fixed)
        for name, dist in trial.distributions.items():
            self.assertEqual(dist.log, name in LOG, name)

    def test_hyperband(self):
        frame = self.optuna_frame("_0_0")
        self.check_all(frame, self.space)

    def test_gp_bo(self):
        from skopt.space import Space
        dims = [self.obj._skopt_dimension(p) for p in self.space]
        frame = pd.DataFrame(Space(dims).rvs(N, random_state=1), columns=[d.name for d in dims])
        self.check_all(frame, self.space)

    def test_smac(self):
        cs = smac_configspace(self.space, self.fixed)
        cs.seed(1)
        frame = pd.DataFrame([dict(c) for c in cs.sample_configuration(N)])
        self.check_all(frame, self.space)

    def test_smac_sobol_initial_design(self):
        from scipy.stats.qmc import Sobol
        from smac.utils.configspace import transform_continuous_designs
        cs = smac_configspace(self.space, self.fixed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = Sobol(d=len(self.space), scramble=True, seed=1).random(N)
        configs = transform_continuous_designs(design=design, origin="test", configspace=cs)
        self.check_all(pd.DataFrame([dict(c) for c in configs]), self.space)

    @needs_extra
    def test_bohb(self):
        # hpbandster: Zufallskonfigurationen = configspace.sample_configuration, Modell auf ConfigSpace-Vektoren
        cs = self.obj._extra_configspace(self.space, 1)
        self.check_all(pd.DataFrame([dict(c) for c in cs.sample_configuration(N)]), self.space)

    @needs_extra
    def test_dehb(self):
        from dehb.optimizers.de import DE
        de = DE(cs=self.obj._extra_configspace(self.space, 1), dimensions=len(self.space), f=None)
        vectors = np.random.default_rng(1).uniform(size=(N, len(self.space)))
        self.check_all(pd.DataFrame([dict(de.vector_to_configspace(v)) for v in vectors]), self.space)

    @needs_extra
    def test_hebo(self):
        from hebo.design_space.design_space import DesignSpace
        np.random.seed(1)
        types = {("float", False): "num", ("float", True): "pow", ("int", False): "int", ("int", True): "pow_int"}
        design = DesignSpace().parse([dict(name=p["name"], type=types[(p["kind"], p["log"])], lb=p["low"],
                                           ub=p["high"]) for p in self.space])
        self.check_all(design.sample(N), self.space)

    def test_paper_tpe_stays_linear(self):
        obj = make("paper")
        space, _ = obj._extra_space()
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        rows = []
        for _ in range(N):
            trial = study.ask()
            rows.append(obj._suggest_optuna(trial, space, {}))
            study.tell(trial, 0.0)
        frame = pd.DataFrame(rows)
        for name in ("learning_rate", "lambda_l2", "lambda_l1"):
            share = np.mean(frame[name] < (frame[name].max() / 10))
            self.assertGreater(share, 0.05)   # linear: ~10 % unter einem Zehntel des Maximums
            self.assertLess(share, 0.15)


class TestSmacDefault(unittest.TestCase):
    def test_default_in_range_and_positive(self):
        for strategy in ("num_leaves", "max_depth", "joint"):
            obj = make(strategy=strategy)
            space, fixed = obj._extra_space()
            default = dict(smac_configspace(space, fixed).get_default_configuration())
            for p in space:
                value = default[p["name"]]
                if p["kind"] == "cat":
                    self.assertIn(value, p["choices"])
                    continue
                self.assertGreaterEqual(value, p["low"], p["name"])
                self.assertLessEqual(value, p["high"], p["name"])
                self.assertGreater(value, 0, p["name"])
                if p["log"]:   # geometrisches Mittel der Grenzen
                    self.assertAlmostEqual(np.log(value), (np.log(p["low"]) + np.log(p["high"])) / 2,
                                           delta=0.02 if p["kind"] == "int" else 1e-6, msg=p["name"])

    def test_default_values(self):
        space, fixed = make()._extra_space()
        default = dict(smac_configspace(space, fixed).get_default_configuration())
        self.assertAlmostEqual(default["learning_rate"], 10 ** -1.5, places=6)
        self.assertAlmostEqual(default["lambda_l2"], 1e-3, places=9)
        self.assertAlmostEqual(default["lambda_l1"], 1e-3, places=9)
        self.assertEqual(default["min_data_in_leaf"], 32)
        self.assertEqual(default["num_leaves"], 45)
        self.assertEqual(default["max_bin"], 1597)
        self.assertAlmostEqual(default["bagging_fraction"], 0.75)

    def test_l1_off_constant_and_initial_design_size(self):
        from smac import Scenario
        from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
        space, fixed = make(lambda_l1=False)._extra_space()
        cs = smac_configspace(space, fixed)
        self.assertEqual(dict(cs.get_default_configuration())["lambda_l1"], 0.0)
        # wie im Paper-Lauf: 8 Parameter inkl. Konstante -> min(80, 0.25 x 135) = 33 Sobol-Punkte
        self.assertEqual(len(list(cs.values())), 8)
        self.assertEqual(HPOFacade.get_initial_design(Scenario(cs, n_trials=135, seed=7))._n_configs, 33)


class Captured(Exception):
    pass


class TestGridsUnchanged(unittest.TestCase):
    def test_grids_do_not_depend_on_scales(self):
        grids = {}
        for scales in ("paper", "gpboost"):
            obj = make(scales)
            captured = []

            def capture(param_grid, **kw):
                captured.append(param_grid)
                raise Captured

            original = methods.modified_grid_search_tune_parameters
            methods.modified_grid_search_tune_parameters = capture
            try:
                for num_try_random in (None, 135):
                    try:
                        obj.grid_search_method(obj.X, obj.y, obj.X, obj.y, num_try_random=num_try_random)
                    except Captured:
                        pass
            finally:
                methods.modified_grid_search_tune_parameters = original
            grids[scales] = captured
        self.assertEqual(len(grids["paper"]), 2)
        self.assertEqual(grids["paper"], grids["gpboost"])


class TestRunAblationFlags(unittest.TestCase):
    def test_gpboost_l1_off_patches_only_grid(self):
        saved = (optuna.trial.Trial.suggest_float, methods.gp_minimize, methods.modified_grid_search_tune_parameters,
                 run_ablation.SMAC_method.ConfigurationSpace, optuna.study.Study.optimize,
                 run_ablation.SMAC_method.Scenario)
        try:
            run_ablation.apply_settings("tpe", 135, True, False, "/tmp/x", scales="gpboost")
            self.assertIs(optuna.trial.Trial.suggest_float, saved[0])
            self.assertIs(run_ablation.SMAC_method.ConfigurationSpace, saved[3])
            self.assertIsNot(methods.modified_grid_search_tune_parameters, saved[2])
        finally:
            (optuna.trial.Trial.suggest_float, methods.gp_minimize, methods.modified_grid_search_tune_parameters,
             run_ablation.SMAC_method.ConfigurationSpace, optuna.study.Study.optimize,
             run_ablation.SMAC_method.Scenario) = saved


if __name__ == "__main__":
    unittest.main()
