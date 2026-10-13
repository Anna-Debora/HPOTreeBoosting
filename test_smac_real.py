"""Real SMAC 2.4.1 smoke run for abalone with a one-fold, 20-trial budget."""

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd

import methods
import run_experiment_H_SMAC
import SMAC_method
import utils_plot_maker as upm


ROOT = os.path.dirname(os.path.abspath(__file__))
SUITE = 335
TASK = 361288
SEED = 27225
TRIALS = 20


def _write(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


def _fill_reader_inputs(root):
    os.makedirs("task_indices", exist_ok=True)
    np.save(f"task_indices/{SUITE}_task_indices.npy", np.array([TASK]))
    with open(f"task_indices/{SUITE}_task_names.json", "w") as file:
        json.dump(["abalone"], file)
    np.save("task_indices/336_task_indices.npy", np.array([], dtype=int))
    with open("task_indices/336_task_names.json", "w") as file:
        json.dump([], file)

    metric_row = {"current_best_test_score": 0.25, "current_best_test_rmse": 2.0,
                  "current_best_test_log_loss": np.nan}
    main_rows = []
    for method in ("grid_search", "random_search", "tpe", "gp_bo"):
        for flags in ((False, False, False), (True, False, False), (False, True, False)):
            if method == "grid_search" and flags[1]:
                continue
            for iteration in range(TRIALS):
                main_rows.append({"method": method, "fold": 0, "iter": iteration,
                                  "try_num_leaves": flags[0],
                                  "joint_tuning_depth_leaves": flags[1],
                                  "try_num_iter": flags[2], **metric_row})
    _write(f"{root}/Results/seed_{SEED}/gpboost/{SUITE}_{TASK}.csv", pd.DataFrame(main_rows))

    hyperband_rows = []
    for flags in ((False, False, False), (True, False, False), (False, True, False)):
        for iteration in range(2):
            hyperband_rows.append({"method": "hyperband", "fold": 0, "iter": iteration,
                                   "try_num_leaves": flags[0],
                                   "joint_tuning_depth_leaves": flags[1],
                                   "try_num_iter": flags[2], **metric_row})
    _write(f"{root}/Results_Hyperband_2007/seed_{SEED}/gpboost/{SUITE}_{TASK}.csv",
           pd.DataFrame(hyperband_rows))
    _write(f"{root}/Results_Default_Run/gpboost/{SUITE}_{TASK}.csv",
           pd.DataFrame([{"fold": 0, **metric_row} for _ in range(TRIALS)]))


def main():
    original_scenario = SMAC_method.Scenario
    SMAC_method.Scenario = lambda *args, **kwargs: original_scenario(
        *args, **{**kwargs, "n_trials": TRIALS}
    )
    original_split = SMAC_method.ParameterOptimizationSMAC._split_data
    SMAC_method.ParameterOptimizationSMAC._split_data = lambda self: list(
        original_split(self)
    )[:1]
    methods.ParameterOptimization.run_hyperband = lambda self: pd.DataFrame(
        [{"fold": 0, "method": "hyperband", "try_num_leaves": False,
          "joint_tuning_depth_leaves": False, "try_num_iter": False,
          "current_best_test_score": 0.0}]
    )

    with tempfile.TemporaryDirectory(prefix="smac-real-") as temp:
        old_cwd = os.getcwd()
        os.chdir(temp)
        try:
            os.symlink(os.path.join(ROOT, "data"), "data")
            with open(f"suite_{SUITE}_tasks.txt", "w") as file:
                file.write(f"{TASK}\n")
            result_smac = os.path.join(temp, "Results_SMAC")
            result_h = os.path.join(temp, "Results_Hyperband")
            run_experiment_H_SMAC.main(argparse.Namespace(
                suite_id=str(SUITE), seed=str(SEED), result_folder_SMAC=result_smac,
                result_folder_H=result_h, library="gpboost"
            ))

            smac_path = os.path.join(result_smac, f"seed_{SEED}", f"{SUITE}_{TASK}.csv")
            smac_data = pd.read_csv(smac_path)
            keys = ["try_num_leaves", "joint_tuning_depth_leaves"]
            print(f"SMAC CSV: {smac_path}")
            print(f"Columns: {smac_data.columns.tolist()}")
            print("Rows per strategy:")
            print(smac_data.groupby(keys, dropna=False).size().to_string())
            print(f"val_score finite and not sentinel: "
                  f"{np.isfinite(smac_data['val_score']).all() and (smac_data['val_score'] != 1e99).all()}")
            print(f"test_score range: {smac_data['test_score'].min()} .. {smac_data['test_score'].max()}")
            print(f"runtime column present: {'runtime' in smac_data.columns}")

            _fill_reader_inputs(temp)
            upm.seeds = [SEED]
            upm.NUM_SEEDS = 1
            upm.NUM_REGR_TASKS = 1
            upm.FOLDS = [0]
            upm.NUM_FOLDS = 1
            upm.NUM_ITERS = TRIALS
            scores, names = upm.create_scores_dict(temp, rmse=False)
            print(f"create_scores_dict names: {names}")
            print(f"create_scores_dict SMAC shape: {scores['SMAC'].shape}")
            print(f"create_scores_dict finite SMAC rows by strategy: "
                  f"{np.isfinite(scores['SMAC']).sum(axis=1).ravel().tolist()}")
            assert len(smac_data) == 3 * TRIALS
            assert np.isfinite(smac_data["val_score"]).all()
            assert names == ["abalone"]
            assert scores["SMAC"].shape == (4, TRIALS, 1, 1)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    main()