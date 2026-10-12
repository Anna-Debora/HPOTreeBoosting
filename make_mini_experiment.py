"""Mini version of run_experiments.py: same four runs, few trials, one fold.

Usage: python make_mini_experiment.py [result_folder]
Writes <result_folder>/seed_27225/gpboost/335_361288.csv (temp folder by default).
"""
import itertools
import os
import sys
import tempfile

import numpy as np
import optuna
import pandas as pd

import methods
from methods import ParameterOptimization

N_RANDOM, N_TPE, N_GPBO = 10, 30, 12
SUITE, TASK, SEED, LIBRARY = 335, 361288, 27225, "gpboost"

_orig_grid = ParameterOptimization.grid_search_method
_orig_optimize = optuna.study.Study.optimize
_orig_gp_minimize = methods.gp_minimize


def _grid(self, *args, num_try_random=None, **kw):
    if num_try_random == 270:
        num_try_random = N_RANDOM
    return _orig_grid(self, *args, num_try_random=num_try_random, **kw)


def _optimize(self, func, n_trials=None, **kw):
    return _orig_optimize(self, func, n_trials=N_TPE, **kw)


def _gp_minimize(func, dims, **kw):
    kw["n_calls"] = N_GPBO
    return _orig_gp_minimize(func, dims, **kw)


ParameterOptimization.grid_search_method = _grid
optuna.study.Study.optimize = _optimize
methods.gp_minimize = _gp_minimize


CONFIGS = [  # same flag combinations and order as run_experiments.py
    dict(try_max_depth=True, try_num_leaves=False, joint_tuning_depth_leaves=False, try_num_iter=False),
    dict(try_max_depth=False, try_num_leaves=True, joint_tuning_depth_leaves=False, try_num_iter=False),
    dict(try_max_depth=False, try_num_leaves=False, joint_tuning_depth_leaves=True, try_num_iter=False),
    dict(try_max_depth=True, try_num_leaves=False, joint_tuning_depth_leaves=False, try_num_iter=True),
]


def run_config(cfg, suite=SUITE, task=TASK, seed=SEED):
    """One run_methods() call on the first fold only."""
    path = f"data/{suite}_{task}"
    X = pd.read_csv(os.path.join(path, f"{suite}_{task}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{suite}_{task}_y.csv"))
    cat = np.load(os.path.join(path, f"{suite}_{task}_categorical_indicator.npy"))
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=cat,
                                suite_id=suite, seed=seed, library=LIBRARY, **cfg)
    obj.splits = list(itertools.islice(obj.splits, 1))
    return obj.run_methods()


def main(result_folder, suite=SUITE, task=TASK, seed=SEED):
    final = pd.concat([run_config(cfg, suite, task, seed) for cfg in CONFIGS], ignore_index=True)
    final["task_id"] = task
    final["library"] = LIBRARY
    final["classification"] = 1 if suite in [334, 337] else 0

    out_dir = os.path.join(result_folder, f"seed_{seed}", LIBRARY)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{suite}_{task}.csv")
    final.to_csv(out, index=False)
    print("FILE:", out)
    print(final.groupby(["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter"],
                        sort=False).size().to_string())
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="mini_exp_"))
