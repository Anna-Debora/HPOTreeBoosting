"""Run Hyperband and SMAC experiments over the tasks of a suite.

By default this iterates over all tasks listed in suite_<id>_tasks.txt and runs
both Hyperband and SMAC. Use --task_id to run one task only and --only to run
just one of the two methods (e.g. one Slurm job per task/seed/library).
"""

import argparse
import os

import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
from SMAC_method import ParameterOptimizationSMAC

# (try_num_leaves, try_max_depth, joint_tuning_depth_leaves); the first strategy creates the CSV, the others append.
STRATEGIES = (
    dict(try_num_leaves=False, try_max_depth=True, joint_tuning_depth_leaves=False),
    dict(try_num_leaves=True, try_max_depth=False, joint_tuning_depth_leaves=False),
    dict(try_num_leaves=False, try_max_depth=False, joint_tuning_depth_leaves=True),
)


def _save(frame, task_id, suite_id, path, first):
    frame["task_id"] = task_id
    frame["classification"] = 1 if suite_id in [334, 337] else 0
    if first:
        frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False, mode='a', header=False)


def main(args):
    only = getattr(args, "only", None)
    single_task = getattr(args, "task_id", None)
    run_hyperband = only in (None, "hyperband")
    run_smac = only in (None, "smac")
    seed_folder = f"seed_{args.seed}"
    if run_smac:
        os.makedirs(args.result_folder_SMAC, exist_ok=True)
        os.makedirs(os.path.join(args.result_folder_SMAC, seed_folder, args.library), exist_ok=True)
    if run_hyperband:
        os.makedirs(args.result_folder_H, exist_ok=True)
        os.makedirs(os.path.join(args.result_folder_H, seed_folder, args.library), exist_ok=True)
    seed = int(args.seed)
    suite_id = int(args.suite_id)
    if single_task is not None:
        tasks = [single_task]
    else:
        tasks = open(f"suite_{args.suite_id}_tasks.txt", "r")
    for task_id in tasks:
        task_id = str(task_id).strip()
        path = f'data/{suite_id}_{task_id}'
        X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
        y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
        categorical_indicator = np.load(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"))
        file_path_SMAC = f"seed_{args.seed}/{args.library}/{args.suite_id}_{task_id}.csv"
        file_path_H = f"seed_{args.seed}/{args.library}/{args.suite_id}_{task_id}.csv"
        if run_hyperband:
            for index, flags in enumerate(STRATEGIES):
                obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id,
                                            try_num_iter=False, seed=seed, library=args.library, **flags)
                _save(obj.run_hyperband(), task_id, suite_id,
                      os.path.join(args.result_folder_H, file_path_H), index == 0)
        if run_smac:
            for index, flags in enumerate(STRATEGIES):
                obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id,
                                                try_num_iter=False, seed=seed, library=args.library, **flags)
                _save(obj.method_smac(), task_id, suite_id,
                      os.path.join(args.result_folder_SMAC, file_path_SMAC), index == 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--seed')
    parser.add_argument('--task_id', default=None, help="run only this task instead of all tasks in suite_<id>_tasks.txt")
    parser.add_argument('--only', choices=('hyperband', 'smac'), default=None)
    parser.add_argument('--result_folder_SMAC')
    parser.add_argument('--result_folder_H')
    parser.add_argument('--library', default='gpboost')
    args = parser.parse_args()
    main(args)
