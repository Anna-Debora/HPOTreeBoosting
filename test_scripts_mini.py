"""Mini call of run_default.py and run_experiment_H_SMAC.py (abalone, one fold, few trials).

Checks the output layout: Default and Hyperband in <folder>/.../<library>/, SMAC without library folder.
"""
import argparse
import itertools
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd

import methods

SMAC_STUBBED = False
try:
    import SMAC_method
except ModuleNotFoundError:  # smac/ConfigSpace are not installed in the venv: layout-only stub
    import types
    SMAC_STUBBED = True
    for name, attrs in {
        "smac": ["Scenario", "RunHistory"],
        "smac.facade": [],
        "smac.facade.hyperparameter_optimization_facade": ["HyperparameterOptimizationFacade"],
        "ConfigSpace": ["ConfigurationSpace", "Integer", "Categorical"],
    }.items():
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, object)
        sys.modules[name] = mod
    import SMAC_method

    def _stub_run_smac(self, X_train_full, y_train_full, X_test, y_test):
        cols = ["learning_rate", "min_data_in_leaf", "max_depth", "lambda_l2", "num_leaves", "max_bin",
                "bagging_fraction", "feature_fraction", "val_score", "test_score", "test_log_loss",
                "test_f1_score", "test_rmse", "current_best_test_score", "current_best_test_log_loss",
                "current_best_test_f1_score", "current_best_test_rmse"]
        df = pd.DataFrame(0.0, index=range(12), columns=cols)
        df["try_num_leaves"] = self.try_num_leaves
        df["joint_tuning_depth_leaves"] = self.joint_tuning_depth_leaves
        df["try_num_iter"] = self.try_num_iter
        return df
    SMAC_method.ParameterOptimizationSMAC.run_smac = _stub_run_smac

SUITE, TASK, SEED = 335, 361288, 27225
LIBRARIES = ["gpboost", "xgboost"]
KEYS = ["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter"]

for cls in (methods.ParameterOptimization, SMAC_method.ParameterOptimizationSMAC):
    _orig = cls._split_data
    cls._split_data = lambda self, _orig=_orig: list(itertools.islice(_orig(self), 1))

_orig_hb = methods.ParameterOptimization.hyperband_method
methods.ParameterOptimization.hyperband_method = lambda self, *a, **kw: _orig_hb(self, *a, **{**kw, "R": 100})
_orig_scenario = SMAC_method.Scenario
SMAC_method.Scenario = lambda *a, **kw: _orig_scenario(*a, **{**kw, "n_trials": 12})

import run_default
import run_experiment_H_SMAC


def tree(root):
    return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs
                  if f.endswith(".csv"))


def main():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            os.symlink(os.path.join(ROOT, "data"), "data")
            res_default, res_h, res_smac = (os.path.join(tmp, n) for n in ("default", "hyperband", "smac"))
            for lib in LIBRARIES:
                open(f"suite_{SUITE}_tasks.txt", "w").write(f"{TASK}\n{TASK}\n")  # run_default indexes the array
                run_default.main(argparse.Namespace(task_id="7", result_folder=res_default, library=lib))
                open(f"suite_{SUITE}_tasks.txt", "w").write(f"{TASK}\n")
                run_experiment_H_SMAC.main(argparse.Namespace(
                    suite_id=str(SUITE), seed=str(SEED), result_folder_SMAC=res_smac,
                    result_folder_H=res_h, library=lib))
            print("\n===== files =====")
            for name, root in [("default", res_default), ("hyperband", res_h), ("smac", res_smac)]:
                for f in tree(root):
                    print(f"{name}/{f}")
            expect = {
                f"default/{lib}/{SUITE}_{TASK}.csv" for lib in LIBRARIES} | {
                f"hyperband/seed_{SEED}/{lib}/{SUITE}_{TASK}.csv" for lib in LIBRARIES} | {
                f"smac/seed_{SEED}/{SUITE}_{TASK}.csv"}
            got = {f"{n}/{f}" for n, r in [("default", res_default), ("hyperband", res_h), ("smac", res_smac)]
                   for f in tree(r)}
            assert got == expect, (got ^ expect)
            for name, root in [("default", res_default), ("hyperband", res_h), ("smac", res_smac)]:
                for f in tree(root):
                    df = pd.read_csv(os.path.join(root, f))
                    print(f"\n--- {name}/{f}: {len(df)} rows")
                    print(df.groupby([k for k in KEYS if k in df.columns], sort=False).size().to_string())
            print("\nOK" + (" (SMAC stubbed: smac not installed, only the file layout was checked)" if SMAC_STUBBED else ""))
        finally:
            os.chdir(old)


if __name__ == "__main__":
    main()
