"""Layout test for create_scores_dict with the run_experiments.py layout (num_iter rows in the main file).

Part 1: real mini CSVs from make_mini_experiment.py (one fold, seed 27225, two tasks).
Part 2: synthetic classification CSVs (classification=True, logloss=True).
The mini CSVs are cached in /tmp/mini_layout_cache; delete the folder to regenerate them.
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd

import make_mini_experiment as mm
import utils_plot_maker as upm

N = 12
SEED = 27225
CACHE = "/tmp/mini_layout_cache"
FLAGS = {"max_depth": (False, False, False), "num_leaves": (True, False, False),
         "joint": (False, True, False), "num_iter": (False, False, True)}
rng = np.random.default_rng(1)


def write(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def synth(methods, n, flags, fold=0):
    rows = []
    for m in methods:
        rows.append(pd.DataFrame({
            "iter": np.arange(n), "val_score": rng.random(n), "test_score": rng.random(n),
            "current_best_test_score": rng.random(n), "current_best_test_rmse": rng.random(n),
            "current_best_test_log_loss": rng.random(n),
            "try_num_leaves": flags[0], "joint_tuning_depth_leaves": flags[1], "try_num_iter": flags[2],
            "fold": fold, "method": m}))
    return pd.concat(rows, ignore_index=True)


def synth_hyperband(n_h):
    return pd.concat([synth(["hyperband"], n_h, f) for k, f in FLAGS.items() if k != "num_iter"], ignore_index=True)


def synth_smac(n):
    return pd.concat([synth(["SMAC"], n, f) for k, f in FLAGS.items() if k != "num_iter"], ignore_index=True)


def shuffled(df):
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def expected(df, method, strategy, metric):
    nl, jt, it = FLAGS[strategy]
    sel = df[(df["method"] == method) & (df["fold"] == 0) & (df["try_num_leaves"] == nl)
             & (df["joint_tuning_depth_leaves"] == jt) & (df["try_num_iter"] == it)]
    return sel.sort_values("iter")[metric].to_numpy()


def check(res, k, main, hb, smac, metric, n_h):
    for method, df in [("grid_search", main), ("random_search", main), ("tpe", main), ("gp_bo", main),
                       ("hyperband", hb), ("SMAC", smac)]:
        for j, strategy in enumerate(upm.TUNING_STRAT):
            if (method == "grid_search" and j >= 2) or (method == "hyperband" and j == 3):
                continue
            got = res[method][j, :, k, 0]
            exp = expected(df, method, strategy, metric)
            if len(exp) == 0:
                assert strategy == "num_iter" and np.isnan(got).all(), (method, strategy)
                continue
            n = n_h if method == "hyperband" else N
            assert len(exp) == n == len(got), (method, strategy, len(exp), len(got))
            np.testing.assert_allclose(got, exp, err_msg=f"{method}/{strategy}")


def make_tree(root, suite, tasks, mains, hbs, smacs, defaults, legacy=None):
    for t in tasks:
        name = f"{suite}_{t}.csv"
        write(f"{root}/Results_Default_Run/{name}", defaults[t])
        write(f"{root}/Results/seed_{SEED}/gpboost/{name}", shuffled(mains[t]))
        write(f"{root}/Results_Hyperband_2007/seed_{SEED}/gpboost/{name}", shuffled(hbs[t]))
        write(f"{root}/Results_SMAC/seed_{SEED}/gpboost/{name}", shuffled(smacs[t]))
        if legacy is not None:
            write(f"{root}/Results_NUM_ITER/seed_{SEED}/gpboost/{name}", legacy[t])


def write_task_indices(suite_tasks, other_suites):
    os.makedirs("task_indices", exist_ok=True)
    for suite, tasks in suite_tasks.items():
        np.save(f"task_indices/{suite}_task_indices.npy", np.array(tasks))
        json.dump([f"t{t}" for t in tasks], open(f"task_indices/{suite}_task_names.json", "w"))
    for suite in other_suites:
        np.save(f"task_indices/{suite}_task_indices.npy", np.array([], dtype=int))
        json.dump([], open(f"task_indices/{suite}_task_names.json", "w"))


def patch(n_tasks_attr, n_tasks):
    upm.seeds = [SEED]
    upm.NUM_SEEDS = 1
    upm.FOLDS = [0]
    upm.NUM_FOLDS = 1
    upm.NUM_ITERS = N
    setattr(upm, n_tasks_attr, n_tasks)


def real_mini_csv(task):
    out = f"{CACHE}/seed_{SEED}/gpboost/335_{task}.csv"
    if not os.path.exists(out):
        mm.N_RANDOM = mm.N_TPE = mm.N_GPBO = N
        mm.main(CACHE, 335, task, SEED)
    df = pd.read_csv(out)
    # deterministic grid has 270 rows; cut it to N so that every cell has NUM_ITERS rows
    return df[~((df["method"] == "grid_search") &
                (df.groupby(["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter", "fold"]).cumcount() >= N))]


def test_real_mini(tmp):
    tasks = [361288, 361102]
    mains = {t: real_mini_csv(t) for t in tasks}
    for t, df in mains.items():
        counts = df.groupby(["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter"]).size()
        assert (counts == N).all(), counts
        assert df["try_num_iter"].any()
    n_h = 4
    hbs = {t: synth_hyperband(n_h) for t in tasks}
    smacs = {t: synth_smac(N) for t in tasks}
    defaults = {t: synth(["default"], N, (False, False, False)) for t in tasks}
    root = f"{tmp}/real"
    make_tree(root, 335, tasks, mains, hbs, smacs, defaults)
    write_task_indices({335: tasks}, [336])
    patch("NUM_REGR_TASKS", len(tasks))
    for rmse, metric in [(False, "current_best_test_score"), (True, "current_best_test_rmse")]:
        res, names = upm.create_scores_dict(root, rmse=rmse)
        assert names == [f"t{t}" for t in tasks]
        for k, t in enumerate(tasks):
            check(res, k, mains[t], hbs[t], smacs[t], metric, n_h)
            np.testing.assert_allclose(res["Default"][:, k], defaults[t][metric].to_numpy())
        print(f"real mini CSV, rmse={rmse}: shapes " + ", ".join(f"{m}={res[m].shape}" for m in
              ["grid_search", "random_search", "hyperband", "SMAC", "Default"]))
        assert np.isnan(res["SMAC"][3]).all() and not np.isnan(res["SMAC"][:3]).any()

    # Results_NUM_ITER must be ignored when the main file already has num_iter rows
    legacy = {t: pd.concat([synth(["random_search", "tpe", "gp_bo", "SMAC"], N, FLAGS["num_iter"]).assign(
        current_best_test_score=999.0)]) for t in tasks}
    make_tree(root, 335, tasks, mains, hbs, smacs, defaults, legacy)
    res, _ = upm.create_scores_dict(root)
    assert not (res["random_search"] == 999.0).any(), "legacy num_iter file was double counted"
    assert np.isnan(res["SMAC"][3]).all()
    print("Results_NUM_ITER ignored when main file has num_iter rows: ok")

    # Without num_iter rows in the main file the legacy folder is used
    mains_old = {t: df[~df["try_num_iter"]] for t, df in mains.items()}
    legacy = {t: synth(["random_search", "tpe", "gp_bo", "SMAC"], N, (False, False, True)) for t in tasks}
    make_tree(root, 335, tasks, mains_old, hbs, smacs, defaults, legacy)
    res, _ = upm.create_scores_dict(root)
    for k, t in enumerate(tasks):
        for method in ["random_search", "tpe", "gp_bo", "SMAC"]:
            np.testing.assert_allclose(res[method][3, :, k, 0],
                                       expected(legacy[t], method, "num_iter", "current_best_test_score"))
    print("legacy Results_NUM_ITER used when main file has no num_iter rows: ok")


def test_classification(tmp):
    tasks = [501, 502]
    n_h = 5
    mains = {t: pd.concat([synth(["grid_search", "random_search", "tpe", "gp_bo"], N, FLAGS["max_depth"]),
                           synth(["grid_search", "random_search", "tpe", "gp_bo"], N, FLAGS["num_leaves"]),
                           synth(["random_search", "tpe", "gp_bo"], N, FLAGS["joint"]),
                           synth(["random_search", "tpe", "gp_bo"], N, FLAGS["num_iter"])], ignore_index=True)
             for t in tasks}
    hbs = {t: synth_hyperband(n_h) for t in tasks}
    smacs = {t: synth_smac(N) for t in tasks}
    defaults = {t: synth(["default"], N, (False, False, False)) for t in tasks}
    root = f"{tmp}/cls"
    make_tree(root, 334, tasks, mains, hbs, smacs, defaults)
    write_task_indices({334: tasks}, [337])
    patch("NUM_CLASS_TASKS", len(tasks))
    for logloss, metric in [(True, "current_best_test_log_loss"), (False, "current_best_test_score")]:
        res, names = upm.create_scores_dict(root, classification=True, logloss=logloss)
        assert res["classification"] and res["logloss"] == logloss
        for k, t in enumerate(tasks):
            check(res, k, mains[t], hbs[t], smacs[t], metric, n_h)
            np.testing.assert_allclose(res["Default"][:, k], defaults[t][metric].to_numpy())
        print(f"classification, logloss={logloss}: shapes " + ", ".join(
            f"{m}={res[m].shape}" for m in ["grid_search", "random_search", "hyperband", "SMAC", "Default"]))


if __name__ == "__main__":
    old = os.getcwd()
    saved = {a: getattr(upm, a) for a in ["seeds", "NUM_SEEDS", "FOLDS", "NUM_FOLDS", "NUM_ITERS",
                                          "NUM_REGR_TASKS", "NUM_CLASS_TASKS"]}
    real_csvs = {t: real_mini_csv(t) for t in [361288, 361102]}  # needs data/ in the project folder
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            test_real_mini(tmp)
            test_classification(tmp)
        finally:
            os.chdir(old)
            for a, v in saved.items():
                setattr(upm, a, v)
    print("ALL OK")
