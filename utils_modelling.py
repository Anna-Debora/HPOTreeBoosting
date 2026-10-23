"""Dataset assembly and GPBoost helper functions.

Helpers to read experiment results, assemble the analysis dataset, split
into features/targets, and save GPBoost model summaries.
"""

import io
import sys
from typing import Optional

import gpboost as gpb
import numpy as np
import pandas as pd
from task_types import suites_for



# random.seed(42)
# seeds = random.sample(range(1000, 1000000), 20)
# seeds.sort()
seeds = [27225, 32244, 34326, 92161, 99246, 108473, 117739, 147316, 235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 777646, 778572, 936518]
SEED = seeds[12]


def read_and_preprocess_dataset_onlyOneTask():
    df = pd.DataFrame()

    data = pd.read_csv(f"OneTaskTest/seed_27225/335_361102.csv")
    df = pd.concat([df, data], ignore_index=True)

            # Save the full dataset
            # df.to_csv("final_dataset.csv", index=False)

    df.drop(columns=["iter", "val_score", "test_log_loss", "test_f1_score", "test_rmse", "current_best_test_score", "current_best_test_log_loss", "current_best_test_f1_score", "current_best_test_rmse", "try_num_leaves", "joint_tuning_depth_leaves", "fold", "method", "classification"], inplace=True)

    return df

def read_and_preprocess_dataset(classification=False):
    df = pd.DataFrame()

    if classification:
        suites = suites_for(True)

    else:
        suites = suites_for(False)

    for suite_id in suites:
        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            data1 = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results/seed_{SEED}/{suite_id}_{task_id}.csv")
            data2 = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_SMAC/seed_{SEED}/{suite_id}_{task_id}.csv")
            data3 = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_Hyperband_2007/seed_{SEED}/{suite_id}_{task_id}.csv")
            #data4 = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_NUM_ITER/seed_{SEED}/{suite_id}_{task_id}.csv")
            df = pd.concat([df, data1,data2,data3], ignore_index=True) #add ,data4

            # Save the full dataset
            # df.to_csv("final_dataset.csv", index=False)

    df.drop(columns=["iter", "val_score", "test_log_loss", "test_f1_score", "test_rmse", "current_best_test_score", "current_best_test_log_loss", "current_best_test_f1_score", 
                     "current_best_test_rmse", "try_num_leaves", "try_num_iter","joint_tuning_depth_leaves","fold", "method", "classification","param_ind"], inplace=True)

    return df


def split_dataset(df):
    # Split the dataset into group data, features and target
    X = df.drop(columns=["task_id", "test_score"])
    y = df["test_score"]
    group_data = df["task_id"]

    # Handle very small target values
    # min = np.finfo(np.float32).min
    # y = np.clip(y, min, 1)
    # y = np.clip(y, -1, 1)

    return X, y, group_data


def save_gpboost_summary(gp_model: gpb.GPModel, file_path: Optional[str] = 'gp_model_summary.txt') -> None:
    """
    Saves the summary of a GPBoost model to a text file.

    Parameters:
    gp_model: gpb.GPModel
        The GPBoost model whose summary you want to save.
    file_path: Optional[str]
        The path where the summary will be saved. Default is 'gp_model_summary.txt'.

    Returns:
    None
    """
    # Capture the summary output
    output = io.StringIO()
    sys.stdout = output

    # Generate the summary
    gp_model.summary()

    # Restore original stdout
    sys.stdout = sys.__stdout__

    # Write the summary to a text file
    with open(file_path, 'w') as f:
        f.write(output.getvalue())

    print(f"Summary saved to {file_path}")

# Example usage:
# gp_model = gpb.GPModel(...)
# save_gpboost_summary(gp_model, 'my_gp_model_summary.txt')
