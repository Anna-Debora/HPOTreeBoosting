"""Experiment runner for all HPO methods.

This script loads preprocessed task data and runs the different tuning
strategies implemented in `methods.ParameterOptimization`. Results are
written to the folder specified by `--result_folder`.
"""

import argparse
import os

# import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
from task_types import is_classification


RESULT_COLUMNS = (
    "iter", "learning_rate", "min_data_in_leaf", "max_depth", "lambda_l2",
    "lambda_l1", "num_leaves", "max_bin", "bagging_fraction", "feature_fraction",
    "val_score", "runtime", "test_score", "test_log_loss", "test_f1_score",
    "test_rmse", "current_best_test_score", "current_best_test_log_loss",
    "current_best_test_f1_score", "current_best_test_rmse", "try_num_leaves",
    "joint_tuning_depth_leaves", "try_num_iter", "fold", "method", "task_id",
    "library", "classification", "n_iter",
)



def main(args):                                  
    os.makedirs(args.result_folder, exist_ok=True)
    strategy = getattr(args, "strategy", None)
    strategy_configs = {
        "max_depth": dict(try_max_depth=True, try_num_leaves=False,
                           joint_tuning_depth_leaves=False, try_num_iter=False),
        "num_leaves": dict(try_max_depth=False, try_num_leaves=True,
                            joint_tuning_depth_leaves=False, try_num_iter=False),
        "joint": dict(try_max_depth=False, try_num_leaves=False,
                      joint_tuning_depth_leaves=True, try_num_iter=False),
        "num_iter": dict(try_max_depth=True, try_num_leaves=False,
                         joint_tuning_depth_leaves=False, try_num_iter=True),
    }
    if strategy is not None and strategy not in strategy_configs:
        raise ValueError(f"Unknown strategy: {strategy}")
    seed_folder = f"seed_{args.seed}"
    library_folder = os.path.join(seed_folder, args.library)
    os.makedirs(os.path.join(args.result_folder, library_folder), exist_ok=True)
    filename = f"{args.suite_id}_{args.task_id}"
    if strategy is not None:
        filename += f"__{strategy}"
    file_path = os.path.join(library_folder, f"{filename}.csv")
    path = f"data/{args.suite_id}_{args.task_id}"

    # Read the data from local files
    X = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{args.task_id}_categorical_indicator.npy"))

    # benchmark_suite = openml.study.get_suite(args.suite_id)  # obtain the benchmark suite
    # task = openml.tasks.get_task(args.task_id)  # download the OpenML task
    # dataset = task.get_dataset()
    # X, y, categorical_indicator, attribute_names = dataset.get_data(
    #     dataset_format="dataframe", target=dataset.default_target_attribute
    # )

    suite_id = int(args.suite_id)
    task_id = int(args.task_id)
    seed = int(args.seed)

    strategies = [strategy] if strategy is not None else list(strategy_configs)
    output_path = os.path.join(args.result_folder, file_path)
    first_chunk = True
    for strategy_name in strategies:
        obj = ParameterOptimization(
            X=X, y=y, categorical_indicator=categorical_indicator,
            suite_id=suite_id, seed=seed, library=args.library,
            **strategy_configs[strategy_name],
        )
        splits = list(obj.splits)
        for fold_index, split in enumerate(splits):
            obj.splits = [split]
            fold_results = obj.run_methods()
            fold_results["fold"] = fold_index
            fold_results["task_id"] = task_id
            fold_results["library"] = args.library
            fold_results["classification"] = 1 if is_classification(suite_id) else 0
            extra_columns = [column for column in fold_results.columns if column not in RESULT_COLUMNS]
            fold_results = fold_results.reindex(columns=[*RESULT_COLUMNS, *extra_columns])
            fold_results.to_csv(
                output_path, index=False, mode="w" if first_chunk else "a",
                header=first_chunk,
            )
            first_chunk = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--task_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    parser.add_argument('--library', default='gpboost')
    parser.add_argument('--strategy', choices=('max_depth', 'num_leaves', 'joint', 'num_iter'))
    args = parser.parse_args()
    main(args)
