"""Hyperband per library, create_scores_dict on mini script output, NaN/warning survey of the plot helpers.

Only reports; nothing is repaired. SMAC is stubbed (random values) if `smac` is not installed.
"""
import argparse
import contextlib
import copy
import itertools
import json
import os
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import optuna
import pandas as pd

import make_mini_experiment as mm
import test_scripts_mini as tsm  # first-fold patches, Hyperband R=100, SMAC stub if smac is missing
import methods
import utils_plot_maker as upm

optuna.study.Study.optimize = mm._orig_optimize  # mm patches it globally; re-enabled only for the main-file runs

SUITE, TASK, SEED, N = 335, 361288, 27225, 12
CACHE = "/tmp/mini_hblib_cache"
KEYS = ["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter"]


def say(*a):
    print("##", *a)


@contextlib.contextmanager
def mini_trials():
    optuna.study.Study.optimize = mm._optimize
    try:
        yield
    finally:
        optuna.study.Study.optimize = mm._orig_optimize


# ---------------------------------------------------------------- part 1
def part1():
    say("PART 1: hyperband_method, max_depth, abalone, fold 0, seed", SEED, "R=100")
    path = f"data/{SUITE}_{TASK}"
    X = pd.read_csv(f"{path}/{SUITE}_{TASK}_X.csv")
    y = pd.read_csv(f"{path}/{SUITE}_{TASK}_y.csv")
    cat = np.load(f"{path}/{SUITE}_{TASK}_categorical_indicator.npy")
    out = {}
    for lib in ["gpboost", "xgboost", "catboost"]:
        try:
            obj = methods.ParameterOptimization(X=X.copy(), y=y.copy(), categorical_indicator=cat, suite_id=SUITE,
                                                seed=SEED, library=lib, try_max_depth=True)
            tr, te = next(iter(obj.splits))
            out[lib] = obj.hyperband_method(X_train_full=obj.X.iloc[tr], y_train_full=obj.y.iloc[tr],
                                            X_test=obj.X.iloc[te], y_test=obj.y.iloc[te], R=100)
        except Exception as e:  # noqa: BLE001
            say(f"{lib}: {type(e).__name__}: {e}")
            continue
        d = out[lib]
        say(f"{lib}: rows={len(d)} val_score min/mean/max = {d.val_score.min():.4f}/{d.val_score.mean():.4f}/{d.val_score.max():.4f}"
            f"  test_score min/mean/max = {d.test_score.min():.4f}/{d.test_score.mean():.4f}/{d.test_score.max():.4f}")
    pcols = ["learning_rate", "min_data_in_leaf", "max_depth", "lambda_l2", "lambda_l1", "max_bin",
             "bagging_fraction", "feature_fraction"]
    for a, b in itertools.combinations(out, 2):
        da, db = out[a], out[b]
        if len(da) != len(db):
            say(f"{a} vs {b}: different row counts {len(da)} / {len(db)}")
            continue
        same_cfg = all(np.allclose(da[c].to_numpy(dtype=float), db[c].to_numpy(dtype=float), equal_nan=True) for c in pcols)
        say(f"{a} vs {b}: identical sampled configs={same_cfg}; max|val_score diff|="
            f"{np.abs(da.val_score.to_numpy() - db.val_score.to_numpy()).max():.4f}; max|test_score diff|="
            f"{np.abs(da.test_score.to_numpy() - db.test_score.to_numpy()).max():.4f}")


# ---------------------------------------------------------------- part 2
def fit_grid(df):
    """Cut/pad the deterministic grid to N rows per block so that every cell has NUM_ITERS rows."""
    parts, notes = [], set()
    for key, g in df.groupby(KEYS + ["fold"], sort=False):
        g = g.sort_values("iter")
        if key[0] == "grid_search" and len(g) != N:
            notes.add(len(g))
            if len(g) > N:
                g = g.iloc[:N]
            else:
                pad = pd.concat([g.iloc[[-1]]] * (N - len(g)), ignore_index=True)
                pad["iter"] = range(len(g), N)
                g = pd.concat([g, pad], ignore_index=True)
        parts.append(g)
    if notes:
        say(f"  grid_search had {sorted(notes)} rows per block; cut/padded (repeat last row) to {N}")
    return pd.concat(parts, ignore_index=True)


def main_csv(lib):
    out = f"{CACHE}/{lib}/seed_{SEED}/{lib}/{SUITE}_{TASK}.csv"
    if not os.path.exists(out):
        mm.N_RANDOM = mm.N_TPE = mm.N_GPBO = N
        mm.LIBRARY = lib
        with mini_trials():
            mm.main(f"{CACHE}/{lib}", SUITE, TASK, SEED)
    say(f"main file {lib}:", dict(pd.read_csv(out).groupby(["method"]).size()))
    return fit_grid(pd.read_csv(out))


def smac_stub(self, X_train_full, y_train_full, X_test, y_test):
    g = np.random.default_rng(0)
    n = 12
    test_rmse, test_score = g.uniform(2, 3, n), g.uniform(0.3, 0.6, n)
    df = pd.DataFrame({c: g.uniform(0, 1, n) for c in
                       ["learning_rate", "min_data_in_leaf", "max_depth", "lambda_l2", "num_leaves", "max_bin",
                        "bagging_fraction", "feature_fraction"]})
    df["val_score"] = g.uniform(2, 3, n)
    df["test_score"], df["test_rmse"] = test_score, test_rmse
    df["test_log_loss"] = df["test_f1_score"] = np.nan
    df["current_best_test_score"] = np.maximum.accumulate(test_score)
    df["current_best_test_rmse"] = np.minimum.accumulate(test_rmse)
    df["current_best_test_log_loss"] = df["current_best_test_f1_score"] = np.nan
    df["try_num_leaves"], df["joint_tuning_depth_leaves"], df["try_num_iter"] = (
        self.try_num_leaves, self.joint_tuning_depth_leaves, self.try_num_iter)
    return df


def build_tree(tmp, libs, mains):
    os.symlink(os.path.join(ROOT, "data"), "data")
    os.makedirs("task_indices")
    np.save(f"task_indices/{SUITE}_task_indices.npy", np.array([TASK]))
    json.dump([f"t{TASK}"], open(f"task_indices/{SUITE}_task_names.json", "w"))
    np.save("task_indices/336_task_indices.npy", np.array([], dtype=int))
    json.dump([], open("task_indices/336_task_names.json", "w"))
    root = f"{tmp}/res"
    for lib in libs:
        p = f"{root}/Results/seed_{SEED}/{lib}"
        os.makedirs(p, exist_ok=True)
        mains[lib].to_csv(f"{p}/{SUITE}_{TASK}.csv", index=False)
        open(f"suite_{SUITE}_tasks.txt", "w").write(f"{TASK}\n{TASK}\n")  # run_default indexes the array
        tsm.run_default.main(argparse.Namespace(task_id="7", result_folder=f"{root}/Results_Default_Run", library=lib))
        open(f"suite_{SUITE}_tasks.txt", "w").write(f"{TASK}\n")
        tsm.run_experiment_H_SMAC.main(argparse.Namespace(
            suite_id=str(SUITE), seed=str(SEED), result_folder_SMAC=f"{root}/Results_SMAC",
            result_folder_H=f"{root}/Results_Hyperband_2007", library=lib))
        f = f"{root}/Results_Default_Run/{lib}/{SUITE}_{TASK}.csv"  # 270 repeated rows -> N
        d = pd.read_csv(f)
        d.groupby("fold").head(N).to_csv(f, index=False)
    return root


# ---------------------------------------------------------------- part 3
def survey(label, fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, err = fn(), None
        except Exception as e:  # noqa: BLE001
            out, err = None, f"{type(e).__name__}: {e}"
    ws = sorted({f"{x.category.__name__}: {str(x.message)[:90]}" for x in w})
    return out, err, ws


def cell_state(res):
    st = {}
    for m in upm.HPO_METHODS:
        a = res[m]
        st[m] = ["NaN" if np.isnan(a[j]).all() else "part-NaN" if np.isnan(a[j]).any() else "ok"
                 for j in range(a.shape[0])]
    return st


def part3(lib, res_score, res_rmse):
    say(f"PART 3 [{lib}]")
    results = {"R2": res_score, "RMSE": res_rmse}
    for name, r in results.items():
        say(f"  raw cells {name} (strategies {upm.TUNING_STRAT}):", cell_state(r))
    for name, r, loss in [("R2", res_score, False), ("RMSE", res_rmse, True)]:
        st, err, ws = survey(f"stat {name}", lambda r=r, loss=loss: upm.get_statistics(r, 10, 90, loss=loss))
        say(f"  get_statistics {name}: {err or 'ok'}; stat1={None if st is None else np.round(st[0], 4)} "
            f"stat2={None if st is None else np.round(st[1], 4)}; warnings={ws}")
        nr, err, ws = survey(f"norm {name}", lambda r=r: upm.normalize_scores(copy.deepcopy(r), (10, 90)))
        say(f"  normalize_scores {name}: {err or 'ok'}; warnings={ws}")
        if nr is not None:
            say(f"    normalized cells: {cell_state(nr)}")
            say(f"    inf anywhere: {any(np.isinf(nr[m]).any() for m in upm.HPO_METHODS)}")
    for tuning in upm.TUNING_STRAT:
        for rnd in ["task", "seeds"]:
            for norm in [False, True]:
                _, err, ws = survey("unc", lambda: upm.get_uncertainties(copy.deepcopy(results), tuning, rnd, norm))
                say(f"  get_uncertainties tuning={tuning} randomness={rnd} normalize={norm}: {err or 'ok'}; warnings={ws}")
    ret = upm.get_uncertainties(copy.deepcopy(results), "max_depth", "seeds", False)
    say("  get_uncertainties return value:", ret)

    normed = {k: upm.normalize_scores(copy.deepcopy(v), (10, 90)) for k, v in results.items()}
    problems, n_ok = [], 0
    for tag, data in [("raw", results), ("normalized", normed)]:
        for j, tuning in enumerate(upm.TUNING_STRAT):
            methods_j = [m for m in upm.HPO_METHODS if not (m == "grid_search" and j >= 2)]
            for m in methods_j:
                for rnd in ["task", "seeds", "both"]:
                    avg, err, ws = survey("cas", lambda: upm.compute_avg_std(data, "R2", m, j, rnd))
                    if err or ws or (avg is not None and avg[1] is not None and np.isnan(avg[1]).any()):
                        problems.append(f"{tag}/{m}/{tuning}/{rnd}: {err or ''} {ws} "
                                        f"{'NaN in result' if avg and avg[1] is not None and np.isnan(avg[1]).any() else ''}")
                    else:
                        n_ok += 1
    say(f"  compute_avg_std (plot-relevant combos): {n_ok} ok, {len(problems)} with NaN/error/warning")
    for p in problems:
        say("    ", p)
    _, err, _ = survey("cas", lambda: upm.compute_avg_std(results, "R2", "grid_search", 2, "seeds"))
    say("  compute_avg_std grid_search/joint (not used by plots):", err)
    for j, tuning in enumerate(upm.TUNING_STRAT):
        hm = [m for m in upm.HPO_METHODS if not (m == "grid_search" and j >= 2) and not (m == "hyperband" and j == 3)]
        ex, err, ws = survey("ext", lambda: upm.get_extremum(normed, hm, j))
        say(f"  get_extremum {tuning} (without grid/hyperband where absent): {ex if not err else err}; warnings={ws}")
    ex, err, ws = survey("ext", lambda: upm.get_extremum(normed, ["random_search", "SMAC"], 3))
    say(f"  get_extremum num_iter with SMAC: {ex if not err else err}; warnings={ws}")


def main():
    part1()
    say("PART 2: create_scores_dict on script output")
    libs = ["gpboost", "xgboost"]
    mains = {lib: main_csv(lib) for lib in libs}
    if tsm.SMAC_STUBBED:
        tsm.SMAC_method.ParameterOptimizationSMAC.run_smac = smac_stub
    saved = {a: getattr(upm, a) for a in ["seeds", "NUM_SEEDS", "FOLDS", "NUM_FOLDS", "NUM_ITERS", "NUM_REGR_TASKS"]}
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            root = build_tree(tmp, libs, mains)
            upm.seeds, upm.NUM_SEEDS, upm.FOLDS, upm.NUM_FOLDS, upm.NUM_ITERS, upm.NUM_REGR_TASKS = [SEED], 1, [0], 1, N, 1
            for lib in libs:
                res_score, _ = upm.create_scores_dict(root, library=lib)
                res_rmse, _ = upm.create_scores_dict(root, rmse=True, library=lib)
                say(f"create_scores_dict [{lib}] shapes:",
                    {m: res_score[m].shape for m in upm.HPO_METHODS + ["Default"]})
                part3(lib, res_score, res_rmse)
        finally:
            os.chdir(ROOT)
            for a, v in saved.items():
                setattr(upm, a, v)


if __name__ == "__main__":
    main()
