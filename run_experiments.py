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



def main(args):                                  
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    library_folder = os.path.join(seed_folder, args.library)
    os.makedirs(os.path.join(args.result_folder, library_folder), exist_ok=True)
    file_path = os.path.join(library_folder, f"{args.suite_id}_{args.task_id}.csv")
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

    # Run the experiment for the current task
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator,
                                try_max_depth=True,
                                try_num_leaves=False,
                                joint_tuning_depth_leaves=False,
                                try_num_iter= False,
                                suite_id=suite_id, seed=seed, library=args.library)
    final_results_md = obj.run_methods()

    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator,
                                try_max_depth=False,
                                try_num_leaves=True,
                                joint_tuning_depth_leaves=False,
                                try_num_iter= False,
                                suite_id=suite_id, seed=seed, library=args.library)
    final_results_nl = obj.run_methods()
    
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator,
                                try_max_depth=False,
                                try_num_leaves=False,
                                joint_tuning_depth_leaves=True,
                                try_num_iter= False,
                                suite_id=suite_id, seed=seed, library=args.library)
    final_results_joint = obj.run_methods()
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator,
                                try_max_depth=True,
                                try_num_leaves=False,
                                joint_tuning_depth_leaves=False,
                                try_num_iter= True,
                                suite_id=suite_id, seed=seed, library=args.library)
    final_results_iter = obj.run_methods()
    # Format the DataFrame
    final_results = pd.concat([final_results_md, final_results_nl,final_results_joint,final_results_iter], ignore_index=True)
    final_results["task_id"] = task_id
    final_results["library"] = args.library
    final_results["classification"] = 1 if suite_id in [334, 337] else 0

    # Save the results
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--task_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    parser.add_argument('--library', default='gpboost')
    args = parser.parse_args()
    main(args)
