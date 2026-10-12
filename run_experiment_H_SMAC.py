"""Run Hyperband and SMAC experiments over all tasks in a suite.

This runner iterates over tasks listed in the suite file and runs both
Hyperband and SMAC experiments, saving results to the provided folders.
"""

import argparse
import os

import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
from SMAC_method import ParameterOptimizationSMAC

def main(args):
    os.makedirs(args.result_folder_SMAC, exist_ok=True)
    os.makedirs(args.result_folder_H, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder_SMAC, seed_folder), exist_ok=True)
    os.makedirs(os.path.join(args.result_folder_H, seed_folder, args.library), exist_ok=True)
    seed = int(args.seed)
    suite_id = int(args.suite_id)
    #task_id = int(args.task_id)
    tasks = open(f"suite_{args.suite_id}_tasks.txt","r")
    for task_id in tasks:
        task_id = str(task_id).strip()
        path = f'data/{suite_id}_{task_id}'
        X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
        y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
        categorical_indicator = np.load(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"))
        file_path_SMAC = f"seed_{args.seed}/{args.suite_id}_{task_id}.csv"
        file_path_H = f"seed_{args.seed}/{args.library}/{args.suite_id}_{task_id}.csv"
        # Run the experiment for the current task
        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, 
                                    try_num_leaves=False,try_max_depth = True, joint_tuning_depth_leaves=False,try_num_iter=False, seed=seed, library=args.library)
        final_results_H_md = obj.run_hyperband()
        final_results_H_md["task_id"] = task_id
        final_results_H_md["classification"] = 1 if suite_id in [334, 337] else 0
        final_results_H_md.to_csv(os.path.join(args.result_folder_H, file_path_H), index=False)
        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id,
                                        try_num_leaves=True,try_max_depth=False,try_num_iter=False, joint_tuning_depth_leaves=False,seed=seed, library=args.library)
        final_results_H_nl = obj.run_hyperband()
        final_results_H_nl["task_id"] = task_id
        final_results_H_nl["classification"] = 1 if suite_id in [334, 337] else 0
        final_results_H_nl.to_csv(os.path.join(args.result_folder_H, file_path_H), index=False, mode='a',header=False)
        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id,
                                        try_num_leaves=False,try_max_depth=False,try_num_iter=False, joint_tuning_depth_leaves=True,seed=seed, library=args.library)
        final_results_H_joint = obj.run_hyperband()
        final_results_H_joint["task_id"] = task_id
        final_results_H_joint["classification"] = 1 if suite_id in [334, 337] else 0
        final_results_H_joint.to_csv(os.path.join(args.result_folder_H, file_path_H), index=False, mode='a',header=False)
        obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, 
                                    try_num_leaves=False,try_max_depth = True, joint_tuning_depth_leaves=False,try_num_iter=False, seed=seed)
        final_results_SMAC_md = obj.method_smac()
        final_results_SMAC_md["task_id"] = task_id
        final_results_SMAC_md["classification"] = 1 if suite_id in [334, 337] else 0
        final_results_SMAC_md.to_csv(os.path.join(args.result_folder_SMAC, file_path_SMAC), index=False)
        obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id,
                                        try_num_leaves=True,try_max_depth=False,try_num_iter=False, joint_tuning_depth_leaves=False,seed=seed)
        final_results_SMAC_nl = obj.method_smac()
        final_results_SMAC_nl["task_id"] = task_id
        final_results_SMAC_nl["classification"] = 1 if suite_id in [334, 337] else 0
        final_results_SMAC_nl.to_csv(os.path.join(args.result_folder_SMAC, file_path_SMAC), index=False, mode='a',header=False)
        obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id,
                                        try_num_leaves=False,try_max_depth=False,try_num_iter=False, joint_tuning_depth_leaves=True,seed=seed)
        final_results_SMAC_joint = obj.method_smac()
        final_results_SMAC_joint["task_id"] = task_id
        final_results_SMAC_joint["classification"] = 1 if suite_id in [334, 337] else 0
        final_results_SMAC_joint.to_csv(os.path.join(args.result_folder_SMAC, file_path_SMAC), index=False, mode='a',header=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder_SMAC')
    parser.add_argument('--result_folder_H')
    parser.add_argument('--library', default='gpboost')  # Hyperband only; SMAC is GPBoost-only
    args = parser.parse_args()
    main(args)