"""Synthetic check of utils_plot_maker.create_scores_dict with the library folder."""
import json
import os
import tempfile

import numpy as np
import pandas as pd

import utils_plot_maker as upm

LIBRARY = "gpboost"
SEEDS = [11, 22]
SUITE = 335
TASKS = [1001, 1002]
FOLDS = 5
N = 270
N_H = 11
rng = np.random.default_rng(0)


def block(methods, n, flags):
    rows = []
    for fold in range(FOLDS):
        for m in methods:
            d = pd.DataFrame({
                "val_score": rng.random(n), "runtime": rng.random(n),
                "test_score": rng.random(n), "test_rmse": rng.random(n),
                "current_best_test_score": rng.random(n),
                "current_best_test_rmse": rng.random(n),
                "try_num_leaves": flags[0], "joint_tuning_depth_leaves": flags[1],
                "try_num_iter": flags[2], "iter": np.arange(n),
                "fold": fold, "method": m,
            })
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def full(methods, n):
    return pd.concat([
        block(methods, n, (False, False, False)),
        block(methods, n, (True, False, False)),
        block([m for m in methods if m != "grid_search"], n, (False, True, False)),
    ], ignore_index=True)


def write(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def build(root):
    os.makedirs("task_indices")
    np.save(f"task_indices/{SUITE}_task_indices.npy", np.array(TASKS))
    json.dump([f"t{t}" for t in TASKS], open(f"task_indices/{SUITE}_task_names.json", "w"))
    np.save("task_indices/336_task_indices.npy", np.array([], dtype=int))
    json.dump([], open("task_indices/336_task_names.json", "w"))

    for t in TASKS:
        name = f"{SUITE}_{t}.csv"
        d = block(["default"], N, (False, False, False))
        write(f"{root}/Results_Default_Run/{name}", d)
        for seed in SEEDS:
            write(f"{root}/Results/seed_{seed}/{LIBRARY}/{name}",
                  full(["grid_search", "random_search", "tpe", "gp_bo"], N))
            h = full(["hyperband"], N_H)
            h["param_ind"] = 0
            write(f"{root}/Results_Hyperband_2007/seed_{seed}/{LIBRARY}/{name}", h)
            write(f"{root}/Results_SMAC/seed_{seed}/{LIBRARY}/{name}", full(["SMAC"], N).assign(try_num_iter=False))
            write(f"{root}/Results_NUM_ITER/seed_{seed}/{LIBRARY}/{name}",
                  # Legacy flags: create_scores_dict expects joint=True on the num_iter rows
                  block(["random_search", "tpe", "gp_bo", "SMAC"], N, (False, False, True)))
    # create_scores_dict reads this fixed file for the Hyperband length
    h = full(["hyperband"], N_H)
    h["param_ind"] = 0
    write(f"{root}/Results_Hyperband_2007/seed_27225/{LIBRARY}/334_361110.csv", h)


def main():
    old = os.getcwd()
    upm.seeds = SEEDS
    upm.NUM_SEEDS = len(SEEDS)
    upm.NUM_REGR_TASKS = len(TASKS)
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            root = os.path.join(tmp, "res")
            build(root)
            for rmse in (True, False):
                try:
                    res, names = upm.create_scores_dict(root, rmse=rmse, library=LIBRARY)
                except ValueError as e:
                    print(f"rmse={rmse}: ValueError {e}")
                    continue
                print(f"rmse={rmse} names={names}")
                for k in ["grid_search", "random_search", "tpe", "gp_bo", "hyperband", "SMAC", "Default"]:
                    print(f"  {k:14s} {res[k].shape}")
                assert res["random_search"].shape == (4, N, len(TASKS), len(SEEDS))
                assert res["hyperband"].shape == (4, N_H, len(TASKS), len(SEEDS))
                assert res["Default"].shape == (N, len(TASKS))
            try:
                upm.create_scores_dict(root, library="catboost")
            except FileNotFoundError:
                print("library='catboost' -> FileNotFoundError (erwartet)")
        finally:
            os.chdir(old)


if __name__ == "__main__":
    main()
