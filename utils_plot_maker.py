"""Plotting helpers used by `plot_maker.py`.

Contains functions to build publication-ready figures, normalize scores,
compute uncertainty bounds and save plots in the repository layout.
"""

import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from functools import reduce
from matplotlib.gridspec import GridSpec
from typing import Union
seeds = [27225, 34326,92161, 99246, 108473, 117739,  235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 936518,32244,147316, 777646, 778572]

#Adjust Font Size
TICKSIZE = 14
FONTSIZE = 14
TITLESIZE = 18
SUBTITLESIZE = 16
ALPHA = 0.6
ALPHA_FILL = 0.1
#Adjust lines in plot
avg_linewidth = 2.25
linewidth_2 = 1.5
linewidth_grid = 0.5
alpha_grid = 0.3
# Calculate figure size for A4 page width with quadratic subplots
# A4 width ≈ 8.27 inches, target ~7.5 inches for nice fit with margins
WIDTH = 7.5
HEIGHT = 7.5
NUM_SEEDS = len(seeds)
NUM_TASKS = 59
NUM_REGR_TASKS = 36
NUM_CLASS_TASKS = 23
NUM_ITERS = 270
START = 90
NUM_ITERS_H = 11
START_HYPERBAND = int(START/NUM_ITERS*NUM_ITERS_H)+1
NUM_METRIC = 4
METRIC = [r"$\mathrm{R}^2$", "RMSE","Accuracy", "Log Loss"]
HPO_METHODS_SUBS = ['random_search', 'tpe', 'gp_bo','hyperband', 'SMAC']
HPO_METHODS_SUBS_NAMES = ['Random Grid', 'TPE', 'GP-BO','Hyperband', 'SMAC']
HPO_METHODS_SUBS2 = ['random_search', 'tpe', 'gp_bo', 'SMAC']
HPO_METHODS_SUBS_NAMES2 = ['Random Grid Search', 'TPE', 'GP-BO', 'SMAC']
HPO_METHODS = ['grid_search','random_search', 'tpe', 'gp_bo','hyperband', 'SMAC']
HPO_METHODS_NAMES = ['Deterministic Grid','Random Grid', 'TPE', 'GP-BO','Hyperband', 'SMAC']
TUNING_STRAT = ['num_leaves','max_depth','joint','num_iter']
NAME_TUNING_STRAT = ['Num Leaves','Max Depth','Joint','Num Iter']
TUNING_STRAT_SUBS = ['num_leaves','max_depth','joint']
NAME_TUNING_STRAT_SUBS = ['Num Leaves','Max Depth','Joint']
NUM_TUNING_STRAT = len(TUNING_STRAT)
NUM_TUNING_STRAT_GS = 2
NUM_METHODS = len(HPO_METHODS)
NUM_METHODS_SUBS = len(HPO_METHODS_SUBS)
RANDOMNESS = ['both','seeds','tasks']
FOLDS = [0, 1, 2, 3, 4]
NUM_FOLDS = len(FOLDS)
MARKERS = ["o", "s", "^", "D", "*", "v", "P", "X"] 
def create_scores_dict(filepath, classification=False, rmse=False, logloss = False):
    """
    Creates a dictionary containing scores/results for various hyperparameter optimization (HPO) methods
    across multiple tasks, seeds, folds, and tuning strategies. Supports regression and classification tasks,
    and can return RMSE, log loss, or custom scores depending on the arguments.
    Parameters
    ----------
    filepath: string
        Filepath of different folders with results.
    classification : bool, optional
        If True, processes classification tasks. If False, processes regression tasks.
    rmse : bool, optional
        If True, returns RMSE scores (only for regression tasks).
    logloss : bool, optional
        If True, returns log loss scores (only for classification tasks).
    Returns
    -------
    res : dict
        Dictionary containing score arrays for each HPO method and configuration. Keys include:
        - 'grid_search', 'random_search', 'tpe', 'gp_bo', 'hyperband', 'SMAC', 'Default'
        - 'classification', 'rmse', 'logloss', 'normalized'
        Each method contains a numpy array of scores with dimensions corresponding to
        tuning strategies, iterations, tasks, seeds, and folds.
    names : list of str
        List of task names corresponding to the scores.
    Raises
    ------
    ValueError
        If invalid combinations of arguments are provided (e.g., logloss=True with classification=False),
        or if NaN values are encountered in the score data.
    Notes
    -----
    - Reads results from multiple CSV files for each method, seed, and task.
    - Aggregates and averages scores over folds.
    - Handles special cases for different tuning strategies and methods.
    - Assumes existence of several global variables and file paths.
    """
    if logloss and not classification:
        raise ValueError("'logloss' can only be True if 'classification' is also True.")
    if rmse and classification:
        raise ValueError("'rmse' can only be True if 'classification' is False.")
    if classification:
        NUM_TASKS = NUM_CLASS_TASKS
        suites = [334, 337]
    else:
        NUM_TASKS = NUM_REGR_TASKS
        suites = [335, 336] 
    #get here the number of iterations for hyperband
    data_H = pd.read_csv(f"{filepath}/Results_Hyperband_2007/seed_27225/334_361110.csv")    
    NUM_ITERS_H = len(data_H.loc[(data_H['fold'] == 0) & (data_H['joint_tuning_depth_leaves'] == True)])
    if NUM_ITERS_H > NUM_ITERS:
                    ValueError('There are too much iterations in the Hyperband method.')
    START_HYPERBAND = int(START / NUM_ITERS * NUM_ITERS_H)
    grid_search = np.zeros((NUM_TUNING_STRAT_GS,NUM_ITERS,NUM_TASKS,NUM_SEEDS,NUM_FOLDS))
    random_search = np.zeros((NUM_TUNING_STRAT,NUM_ITERS,NUM_TASKS,NUM_SEEDS,NUM_FOLDS))
    tpe = np.zeros((NUM_TUNING_STRAT,NUM_ITERS,NUM_TASKS,NUM_SEEDS,NUM_FOLDS))
    gp_bo = np.zeros((NUM_TUNING_STRAT,NUM_ITERS,NUM_TASKS,NUM_SEEDS,NUM_FOLDS))
    hyperband = np.zeros((NUM_TUNING_STRAT,NUM_ITERS_H,NUM_TASKS,NUM_SEEDS,NUM_FOLDS))
    SMAC = np.zeros((NUM_TUNING_STRAT,NUM_ITERS,NUM_TASKS,NUM_SEEDS,NUM_FOLDS))
    default = np.zeros((NUM_ITERS, NUM_TASKS, NUM_FOLDS))
    res = {
        'grid_search': grid_search,
        'random_search': random_search,
        'tpe': tpe,
        'gp_bo': gp_bo,
        'hyperband': hyperband,
        'SMAC': SMAC,
        'Default': default,
        'classification':classification,
        'rmse':rmse,
        'logloss': logloss,
        'normalized': False
    }
    names = []
    k=0
    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:            
            for l, seed in enumerate(seeds):
                if l == 0:
                    data_default = pd.read_csv(f"{filepath}/Results_Default_Run/{suite_id}_{task_id}.csv")
                data = pd.read_csv(f"{filepath}/Results/seed_{seed}/{suite_id}_{task_id}.csv")
                data_H = pd.read_csv(f"{filepath}/Results_Hyperband_2007/seed_{seed}/{suite_id}_{task_id}.csv")
                data_H.drop(columns=['param_ind','try_num_iter'], inplace=True)
                data_SMAC = pd.read_csv(f"{filepath}/Results_SMAC/seed_{seed}/{suite_id}_{task_id}.csv")
                data_SMAC.drop(columns=['try_num_iter'], inplace=True)
                data_NUM_ITER = pd.read_csv(f"{filepath}/Results_NUM_ITER/seed_{seed}/{suite_id}_{task_id}.csv") 
                #get the len of the data of all folds for one TUNING_STRAT strategy
                LEN_DATA = len(data.loc[(data['try_num_leaves'])])
                LEN_DATA_SMAC = len(data_SMAC.loc[(data_SMAC['try_num_leaves'])])
                LEN_DATA_H = len(data_H.loc[(data_H['try_num_leaves'])])
                LEN_DATA_NUM_ITER = len(data_NUM_ITER.loc[data_NUM_ITER['try_num_iter']]) 
                data = pd.concat([data, data_H, data_SMAC,data_NUM_ITER], ignore_index=True) 
               
                if rmse:
                    if l==0:
                        default = data_default['current_best_test_rmse'].reset_index(drop=True)
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_rmse'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    try_num_iter = data.loc[data['try_num_iter'] == True, 'current_best_test_rmse'].reset_index(drop=True) 
                    df_default = pd.DataFrame({'default_rmse': default})
                    df =pd.DataFrame({'try_max_depth_rmse': try_max_depth, 'try_num_leaves_rmse': try_num_leaves}) 
                    df_joint =pd.DataFrame({'try_joint_rmse': try_joint,'try_num_iter_rmse': try_num_iter})
                    df_iter = pd.DataFrame({'try_num_iter_rmse': try_num_iter})
                    if l == 0:
                        df_default['method'] = data_default['fold']
                        df_default['fold'] = data_default['fold']
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    df_joint['method'] = pd.concat([data.loc[2*LEN_DATA:2*LEN_DATA + (NUM_ITERS*(NUM_METHODS-3)*NUM_FOLDS)-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)
                    df_joint['fold'] = pd.concat([data.loc[2*LEN_DATA:2*LEN_DATA + (NUM_ITERS*(NUM_METHODS-3)*NUM_FOLDS)-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    df_iter['method'] = pd.concat([data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'method']], ignore_index = True) 
                    df_iter['fold'] = pd.concat([data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        skip = False
                        for j,tuning in enumerate(TUNING_STRAT):
                            if (method == 'grid_search' and j == 2) or (method == 'hyperband' and j==3):
                                skip = True
                            for m in FOLDS:
                                if j == 0 and l == 0:
                                    res['Default'][:,k,m] = df_default.loc[(df_default['fold'] == m),'default_rmse'].values
                                if skip:
                                    break
                                if tuning != 'joint' and tuning != 'num_iter':
                                    if df.loc[(df['method'] == method) & (df['fold'] == m), f'try_{tuning}_rmse'].isnull().any():
                                        raise ValueError(f'NaN in Suite {suite_id}, task {task_id} and seed {seed}')
                                    res[method][j,:,k,l,m] = df.loc[(df['method'] == method) & (df['fold'] == m), f'try_{tuning}_rmse'].values
                                elif tuning == 'num_iter':
                                    res[method][j,:,k,l,m] = df_iter.loc[(df_iter['method'] == method) & (df_iter['fold'] == m), f'try_{tuning}_rmse'].values
                                else:
                                    if df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_rmse'].isnull().any():
                                        raise ValueError(f'NaN in Suite {suite_id}, task {task_id} and seed {seed}')
                                    res[method][j,:,k,l,m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_rmse'].values
                            if skip:
                                break
                        if skip:
                            continue
                    
                elif logloss:
                    if l==0:
                        default = data_default['current_best_test_log_loss'].reset_index(drop=True)
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_log_loss'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_log_loss'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_log_loss'].reset_index(drop=True)
                    try_num_iter = data.loc[data['try_num_iter'] == True, 'current_best_test_log_loss'].reset_index(drop=True) 
                    df_default = pd.DataFrame({'default_logloss': default})
                    df =pd.DataFrame({'try_max_depth_log_loss': try_max_depth, 'try_num_leaves_log_loss': try_num_leaves}) 
                    df_joint =pd.DataFrame({'try_joint_log_loss': try_joint})
                    df_iter = pd.DataFrame({'try_num_iter_log_loss': try_num_iter})
                    if l == 0:
                        df_default['method'] = data_default['fold']
                        df_default['fold'] = data_default['fold']
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    df_joint['method'] = pd.concat([data.loc[2*LEN_DATA:2*LEN_DATA + (NUM_ITERS*(NUM_METHODS-3)*NUM_FOLDS)-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True) 
                    df_joint['fold'] = pd.concat([data.loc[2*LEN_DATA:2*LEN_DATA + (NUM_ITERS*(NUM_METHODS-3)*NUM_FOLDS)-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    df_iter['method'] = pd.concat([data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'method']], ignore_index = True) 
                    df_iter['fold'] = pd.concat([data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        skip = False
                        for j,tuning in enumerate(TUNING_STRAT):
                            if (method == 'grid_search' and j == 2) or (method == 'hyperband' and j==3):
                                skip = True
                            for m in FOLDS:
                                if j == 0 and l ==0:
                                    res['Default'][:,k,m] = df_default.loc[(df_default['fold'] == m),'default_logloss'].values
                                if skip:
                                    break
                                if tuning != 'joint' and tuning != 'num_iter':
                                    if df.loc[(df['method'] == method) & (df['fold'] == m), f'try_{tuning}_log_loss'].isnull().any():
                                        raise ValueError(f'NaN in Suite {suite_id}, task {task_id} and seed {seed}')
                                    res[method][j,:,k,l,m] = df.loc[(df['method'] == method) & (df['fold'] == m), f'try_{tuning}_log_loss'].values
                                elif tuning == 'num_iter':
                                    res[method][j,:,k,l,m] = df_iter.loc[(df_iter['method'] == method) & (df_iter['fold'] == m), f'try_{tuning}_log_loss'].values
                                else:
                                    if df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_log_loss'].isnull().any():
                                        raise ValueError(f'NaN in Suite {suite_id}, task {task_id} and seed {seed}')
                                    res[method][j,:,k,l,m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_log_loss'].values
                            if skip:
                                break
                        if skip:
                            continue
                else:
                    if l == 0:
                        default = data_default['current_best_test_score'].reset_index(drop=True)                   
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_score'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    try_num_iter = data.loc[data['try_num_iter'] == True, 'current_best_test_score'].reset_index(drop=True)
                    df_default = pd.DataFrame({'default_score': default})
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_score': try_num_leaves})
                    df_joint =pd.DataFrame({'try_joint_score': try_joint})
                    df_iter = pd.DataFrame({'try_num_iter_score': try_num_iter})#, 'try_num_iter_score': try_num_iter})
                    if l == 0:
                        df_default['method'] = data_default['fold']
                        df_default['fold'] = data_default['fold']
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    df_joint['method'] = pd.concat([data.loc[2*LEN_DATA:2*LEN_DATA + (NUM_ITERS*(NUM_METHODS-3)*NUM_FOLDS)-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method'],data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'method']], ignore_index = True) 
                    df_joint['fold'] = pd.concat([data.loc[2*LEN_DATA:2*LEN_DATA + (NUM_ITERS*(NUM_METHODS-3)*NUM_FOLDS)-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold'],data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'fold']], ignore_index = True)
                    df_iter['method'] = pd.concat([data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'method']], ignore_index = True) 
                    df_iter['fold'] = pd.concat([data_NUM_ITER.loc[0:LEN_DATA_NUM_ITER-1,'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        skip = False
                        for j,tuning in enumerate(TUNING_STRAT):
                            if (method == 'grid_search' and j == 2) or (method == 'hyperband' and j==3):
                                skip = True
                            for m in FOLDS:
                                if j == 0 and l == 0:
                                    res['Default'][:,k,m] = df_default.loc[(df_default['fold'] == m),'default_score'].values
                                if skip:
                                    break
                                if tuning != 'joint' and tuning != 'num_iter':
                                    if df.loc[(df['method'] == method) & (df['fold'] == m), f'try_{tuning}_score'].isnull().any():
                                        raise ValueError(f'NaN in Suite {suite_id}, task {task_id} and seed {seed}')
                                    res[method][j,:,k,l,m] = df.loc[(df['method'] == method) & (df['fold'] == m), f'try_{tuning}_score'].values
                                elif tuning == 'num_iter':
                                    res[method][j,:,k,l,m] = df_iter.loc[(df_iter['method'] == method) & (df_iter['fold'] == m), f'try_{tuning}_score'].values
                                else:
                                    if df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_score'].isnull().any():
                                        print(method, m, tuning, df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_score'])
                                        raise ValueError(f'NaN in Suite {suite_id}, task {task_id} and seed {seed}')
                                    res[method][j,:,k,l,m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), f'try_{tuning}_score'].values
                            if skip:
                                break
                        if skip:
                            continue
            k += 1
    res['Default'] = np.mean(res['Default'],axis=-1)
    for i, method in enumerate(HPO_METHODS):
        res[method] = np.mean(res[method],axis=-1)
    return res, names
def get_statistics(result_dict,percentile_score, percentile_loss, loss = False, debug = False):
    """
    Compute per-task summary statistics across all HPO methods, tuning strategies,
    iterations and seeds.

    Produces two arrays per task:
    - statistic_1: the task-level maximum (or percentile for losses)
    - statistic_2: the task-level 10%-quantile (or minimum for losses)

    Parameters
    ----------
    result_dict : dict
        Results dictionary containing arrays for each HPO method and a 'Default' entry.
    percentile_score : float
        Percentile to use for statistic_2 when computing scores (e.g. 10).
    percentile_loss : float
        Percentile to use for statistic_1 when computing losses.
    loss : bool, optional
        If True, treat the metric as a loss (smaller is better).

    Returns
    -------
    statistic_1, statistic_2 : np.ndarray
        Arrays of length NUM_TASKS with the computed statistics per task.
    """
    if result_dict['classification']:        
        NUM_TASKS = NUM_CLASS_TASKS
        if debug:
            print(f'Classification is true')
    else:
        NUM_TASKS = NUM_REGR_TASKS
        if debug:
            print(f'Classification is false')
    
    # Initialize arrays to store all scores for each task
    all_task_metrics = [[] for _ in range(NUM_TASKS)]
    
    # Collect all scores for each task
    for _, method in enumerate(HPO_METHODS):
        method_array = result_dict[method]            
        # Other methods have shape (NUM_TUNING_STRAT, NUM_ITERS, NUM_TASKS, NUM_SEEDS)
        # or (NUM_TUNING_STRAT, NUM_ITERS_H, NUM_TASKS, NUM_SEEDS) for hyperband
        if method == 'hyperband':
            num_tuning = method_array.shape[0]-1
        else:
            num_tuning = method_array.shape[0]
        for tuning_idx in range(num_tuning):
            for task_idx in range(NUM_TASKS):
                default_metrics = result_dict['Default'][:,task_idx].flatten()
                # Get all scores for this task across all iterations and seeds
                task_metrics = method_array[tuning_idx, :, task_idx, :].flatten()
                # Remove any NaN or infinite values
                task_metrics = task_metrics[np.isfinite(task_metrics)]
                all_task_metrics[task_idx].extend(default_metrics)
                all_task_metrics[task_idx].extend(task_metrics)
    
    # Calculate statistics for each task
    statistic_1 = np.zeros(NUM_TASKS)
    statistic_2 = np.zeros(NUM_TASKS)
    
    for task_idx in range(NUM_TASKS):
        if loss:
            if len(all_task_metrics[task_idx]) > 0:
                scores_array = np.array(all_task_metrics[task_idx])
                statistic_1[task_idx] = np.percentile(scores_array,percentile_loss)
                statistic_2[task_idx] = np.min(scores_array)
                if debug:
                    check_where_statistic_2_belongs(result_dict, statistic_2[task_idx],task_idx)
            else:
                statistic_1[task_idx] = np.nan
                statistic_2[task_idx] = np.nan
        else:
            if len(all_task_metrics[task_idx]) > 0:
                scores_array = np.array(all_task_metrics[task_idx])
                statistic_1[task_idx] = np.max(scores_array)
                statistic_2[task_idx] = np.percentile(scores_array, percentile_score)
                if debug:
                    check_where_statistic_2_belongs(result_dict, statistic_2[task_idx],task_idx)
                    print(f'scaling in normalization {statistic_1[task_idx]-statistic_2[task_idx]}')               
            else:
                statistic_1[task_idx] = np.nan
                statistic_2[task_idx] = np.nan
    return statistic_1, statistic_2

def normalize_scores(result_dict, percentiles):
    """
    Normalize result scores using per-task statistics.

    The function computes task-level statistics via `get_statistics` and scales
    each method and the default accordingly. For losses the values are clipped
    at 1, for scores they are clipped at 0.

    Parameters
    ----------
    result_dict : dict
        Result dictionary for a metric containing arrays for each method.
    percentiles : tuple
        Pair (percentile_score, percentile_loss) passed to get_statistics.

    Returns
    -------
    result_dict : dict
        The same dictionary with arrays normalized in-place.
    """
    if percentiles is not None:
        percentile_score = percentiles[0]
        percentile_loss = percentiles[1]
    if not result_dict['normalized']:
        # print('Is the metric a loss', {(result_dict['rmse'] or result_dict['logloss'])})
        statistic1, statistic2 = get_statistics(result_dict,percentile_score,percentile_loss,loss=(result_dict['rmse'] or result_dict['logloss']))

        for i, method in enumerate(HPO_METHODS):
                if (result_dict['rmse'] or result_dict['logloss']):
                    # print('We use normalization for a loss.')
                    # print('This is SMAC:', (result_dict['SMAC']))
                    # print(f'This is normalized SMAC:', (result_dict['SMAC']-statistic2[np.newaxis,np.newaxis, :, np.newaxis])/(statistic1[np.newaxis,np.newaxis, :, np.newaxis]-statistic2[np.newaxis,np.newaxis, :, np.newaxis]))
                    # temp = (result_dict['SMAC']-statistic2[np.newaxis,np.newaxis, :, np.newaxis])/(statistic1[np.newaxis,np.newaxis, :, np.newaxis]-statistic2[np.newaxis,np.newaxis, :, np.newaxis])
                    # print(f'Number of values greater than 1', np.sum(temp>=1), 'and the shape', temp.shape, 'and the ratio', np.sum(temp>=1)/(4*135*temp.shape[2]*20))
                    result_dict[method] = np.minimum((result_dict[method]-statistic2[np.newaxis,np.newaxis, :, np.newaxis])/(statistic1[np.newaxis,np.newaxis, :, np.newaxis]-statistic2[np.newaxis,np.newaxis, :, np.newaxis]),1)
                    # print(f'this is the shift term {statistic2} and the scalig {statistic1-statistic2}.')
                    # print(f'This is the shape of the output {result_dict[method].shape}')
                    if i == 0:
                        result_dict['Default'] = np.minimum((result_dict['Default']-statistic2[np.newaxis, :])/(statistic1[np.newaxis, :]-statistic2[np.newaxis, :]),1)
                else:
                    print('We use normalization for a score.')
                    result_dict[method] = np.maximum((result_dict[method]-statistic2[np.newaxis,np.newaxis, :, np.newaxis])/(statistic1[np.newaxis,np.newaxis, :, np.newaxis]-statistic2[np.newaxis,np.newaxis, :, np.newaxis]),0)
                    #print(f'this is the shift term {statistic2} and the scalig {statistic1-statistic2}.')
                    if i == 0:
                        result_dict['Default'] = np.maximum((result_dict['Default']-statistic2[np.newaxis,:])/(statistic1[np.newaxis,:]-statistic2[np.newaxis,:]),0)
        result_dict['normalized'] = True
    else: 
        print('Data is not normalized because it is already normalized or we do not want it to be normalized')
    return result_dict
def get_uncertainties(result_dict, TUNING, randomness = None, normalize = True):
    """
    Compute average standard deviations across tasks for each HPO method.

    This helper computes per-method `avg_std_<method>` entries inside the
    returned dictionary. It supports both 'task' and 'seeds' randomness modes
    and can operate on normalized data.

    Parameters
    ----------
    result_dict : dict
        Results dictionary containing arrays for each metric and method.
    TUNING : str
        Tuning strategy name (one of TUNING_STRAT).
    randomness : {'task','seeds', None}
        Whether to compute uncertainties across tasks or seeds.
    normalize : bool
        Whether to normalize the input results before computing uncertainties.

    Returns
    -------
    res : dict
        Deep copy of the results dictionary augmented with avg_std entries.
    """
    res = copy.deepcopy(result_dict)
    if normalize:
        for metric, result in res.items():
            res[metric] = normalize_scores(result)
    index_tuning = TUNING_STRAT.index(TUNING)
    #define bool to check wether to consider grid_search in plot
    add_grid_search = not (index_tuning > TUNING_STRAT.index('max_depth'))
    add_hyperband = not (index_tuning > TUNING_STRAT.index('joint'))
    if not add_grid_search:
        for metric in res:
            del res[metric]['grid_search']
        hpo_methods = HPO_METHODS_SUBS
        if not add_hyperband:
            for metric in res:
                del res[metric]['hyperband']
            hpo_methods = HPO_METHODS_SUBS2
    if add_grid_search and add_hyperband:
        hpo_methods = HPO_METHODS
    for metric in res:
        for method in hpo_methods:
            if randomness == 'task':
                m = len(res[metric][method][0,0,:,0])
                print(f'This is the method {method}, metric {metric} and m {m}. This is the shape of the results {res[metric][method].shape} and after mean taken over axis = 3 {np.mean(res[metric][method],axis=3).shape}')
                res[metric][f'avg_std_{method}'] = np.std(np.mean(res[metric][method],axis=3)[index_tuning,:,:],axis=1)
                # --- Analysis of avg_std values for similarity and evolution ---
                avg_std_vals = res[metric][f'avg_std_{method}']
                unique_vals, counts = np.unique(np.round(avg_std_vals, 8), return_counts=True)
                print(f"[ANALYSIS] {method} {metric}: Number of unique avg_std values: {len(unique_vals)}")
                print(f"[ANALYSIS] {method} {metric}: Most common value: {unique_vals[np.argmax(counts)]} (count: {np.max(counts)})")
                print(f"[ANALYSIS] {method} {metric}: All unique values and counts: {list(zip(unique_vals, counts))}")
                print(f"[ANALYSIS] {method} {metric}: Evolution (first 10): {avg_std_vals[:10]}")
                print(f"[ANALYSIS] {method} {metric}: Difference (first and last): {avg_std_vals[0]-avg_std_vals[-1]}")
            elif randomness == 'seeds':
                m = len(res[metric][method][0,0,0,:])
                print(f'This is the method {method} and m {m}. This is the shape of the results {res[metric][method].shape}')
                res[metric][f'avg_std_{method}'] = np.std(np.mean(res[metric][method],axis=2)[index_tuning,:,:],axis=1)
                # --- Analysis of avg_std values for similarity and evolution ---
                avg_std_vals = res[metric][f'avg_std_{method}']
                unique_vals, counts = np.unique(np.round(avg_std_vals, 8), return_counts=True)
                print(f"[ANALYSIS] {method} {metric}: Number of unique avg_std values: {len(unique_vals)}")
                print(f"[ANALYSIS] {method} {metric}: Most common value: {unique_vals[np.argmax(counts)]} (count: {np.max(counts)})")
                print(f"[ANALYSIS] {method} {metric}: All unique values and counts: {list(zip(unique_vals, counts))}")
                print(f"[ANALYSIS] {method} {metric}: Evolution (first 10): {avg_std_vals[:10]}")
                print(f"[ANALYSIS] {method} {metric}: Difference (first and last): {avg_std_vals[0]-avg_std_vals[-1]}")
            else:
                res[metric]['avg_std'] = None
def get_nested_shape(arr, level=0):
    """
    Recursively determines and prints the shape of a nested list or numpy array.

    Parameters
    ----------
    arr : list or numpy.ndarray
        The input object whose nested shape is to be determined.
    level : int, optional
        The current recursion depth, used for indentation in printed output (default is 0).

    Returns
    -------
    tuple
        A tuple representing the shape of the nested structure.

    Notes
    -----
    - For numpy arrays, the shape is obtained directly from `arr.shape`.
    - For lists, the function recursively inspects the first element to determine deeper levels.
    - For empty lists, returns a tuple with a single zero.
    - For other types, returns an empty tuple.

    Examples
    --------
    >>> import numpy as np
    >>> get_nested_shape([[1, 2], [3, 4]])
    list of length: 2
      list of length: 2
        type: <class 'int'>
    (2, 2)
    >>> get_nested_shape(np.array([[1, 2], [3, 4]]))
    ndarray shape: (2, 2)
    (2, 2)
    """
    if isinstance(arr, np.ndarray):
        print("  " * level + f"ndarray shape: {arr.shape}")
        return arr.shape
    elif isinstance(arr, list):
        print("  " * level + f"list of length: {len(arr)}")
        if len(arr) > 0:
            return (len(arr),) + tuple(get_nested_shape(arr[0], level + 1))
        else:
            return (0,)
    else:
        print("  " * level + f"type: {type(arr)}")
        return ()

def set_plot_theme(number):
    """
    Sets the Seaborn theme and returns a color palette for publication-quality plots.
    Parameters
    ----------
    number : int
        Number of colors to include in the palette.
    Returns
    -------
    palette : list
        List of color codes for plotting.
    """
    # # Set Seaborn theme for publication quality
    sns.set_theme(context="paper", style="white")
    
    # # Publication-ready colorblind-friendly palette
    # # Based on Wong (2011) "Points of view: Color blindness" and Tol's schemes
    # publication_colors = [
    #     '#1f77b4',  # Blue - primary, most distinguishable
    #     '#ff7f0e',  # Orange - high contrast with blue
    #     '#2ca02c',  # Green - good for success/positive
    #     '#d62728',  # Red - attention-grabbing, for emphasis
    #     '#9467bd',  # Purple - good contrast
    #     '#8c564b',  # Brown - earthy, professional
    #     '#e377c2',  # Pink - distinctive
    #     '#17becf'   # Cyan - light but distinguishable
    # ]
    palette = sns.color_palette('muted', number)
    
    return palette

def check_where_statistic_2_belongs(result_dict, value,task_idx):
    """
    Checks which HPO method and tuning strategy contains a specific value for a given task index.
    Parameters
    ----------
    result_dict : dict
        Dictionary of results for each HPO method.
    value : float
        Value to search for in the results.
    task_idx : int
        Index of the task to check.
    """
    for method in HPO_METHODS:
        for j in range(result_dict[method].shape[0]):
            if value in result_dict[method][j, -1, task_idx, :]:
                print(f"Found statistic_2 value in {method} with Tuning {TUNING_STRAT[j]}")
def get_limit(res, metric, hpo_method, start,iterations, m,randomness =None, debug = False):
    """
    Calculates y-axis limits for plots based on results and standard deviations.
    Parameters
    ----------
    res : dict
        Results dictionary.
    metric : str
        Metric to plot (e.g., 'R2', 'Accuracy').
    hpo_method : list
        List of HPO methods to include.
    start : int
        Starting iteration index.
    iterations : array-like
        Iteration values for x-axis.
    m : int
        Number of samples for standard deviation calculation.
    randomness: str optional
        Randomness type ('task', 'seeds', 'both').
    debug : bool, optional
        If True, prints debug information.
    
    Returns
    -------
    y_min : float
        Minimum y-axis value.
    y_max : float
        Maximum y-axis value.
    """
    # Enhanced zoom logic: focus on most important parts of the plots
    # Clip the values before appending
    all_y_max = []
    all_y_min = []
    all_x = []
    for method in hpo_method:
        if randomness is not None:
            clipped_y_max =res[metric][method][start:] + (2*res[metric][f'avg_std_{method}'][start:])/(np.sqrt(m))
            clipped_y_min = res[metric][method][start:] - (2*res[metric][f'avg_std_{method}'][start:])/(np.sqrt(m))
        else:
            clipped_y_max =res[metric][method][start:] 
            clipped_y_min = res[metric][method][start:]
        all_y_max.append(clipped_y_max)
        all_y_min.append(clipped_y_min)
        all_x.append(iterations)
    all_y_max_flat = np.concatenate(all_y_max)
    all_y_min_flat = np.concatenate(all_y_min)
        
    
    y_min = np.min(all_y_min_flat)
    y_max = np.max(all_y_max_flat)
    if debug:
        print(f'This is the y_min limit {y_min} and the y_max {y_max}')
    return y_min, y_max
def get_extremum(res, hpo_methods,index_tuning, debug = False):
    """
    Finds the extremum (max or min) value for each metric across HPO methods at a given tuning index.
    Parameters
    ----------
    res : dict
        Results dictionary.
    hpo_methods : list
        List of HPO methods to check.
    index_tuning : int
        Index of the tuning strategy.
    Returns
    -------
    extremum : dict
        Dictionary of extremum values for each metric.
    """
    temp_res = copy.deepcopy(res)
    if debug:
        print(temp_res['R2']['grid_search'].shape)
    extremum = {}
    for metric in temp_res:
        if debug:
            print(f'this is the metric {metric}--------\n')
        if metric in ['R2','Accuracy']:
            temp_max = -float(np.inf)
            for method_iterate in hpo_methods:
                if debug:
                    print(f'This is the method {method_iterate} and the shape of the array {temp_res[metric][method_iterate].shape}')
                method_best = np.mean(temp_res[metric][method_iterate],axis = (2,3))[index_tuning,-1]
                if  method_best> temp_max:
                    temp_max = method_best
            extremum[metric] = temp_max
        else:
            temp_min = float(np.inf)
            for method_iterate in hpo_methods:
                method_best = np.mean(temp_res[metric][method_iterate],axis = (2,3))[index_tuning,-1]
                if  method_best< temp_min:
                    temp_min = method_best
            extremum[metric] = temp_min
    return extremum
        
def print_default(default_value,metric, y_min, y_max, palette, ax, k, leg_default):
    """
    Plots a horizontal line for the default value on the axis if within y-limits.
    Parameters
    ----------
    default_value : float
        Value to plot.
    metric : str
        Metric name.
    y_min : float
        Minimum y-axis value.
    y_max : float
        Maximum y-axis value.
    palette : list
        Color palette for plotting.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    k : int
        Index for legend.
    Returns
    -------
    leg_default : int
        Legend index for default line.
    """
    if default_value <= y_max and default_value>=y_min:
            ax.axhline(y=default_value, color=palette[-1], linestyle='-.', linewidth=avg_linewidth, label='Default')
            leg_default = k
    else:
        if metric in ['R2', 'Accuracy']:
            # Bottom right annotation for R2 and Accuracy
            ax.text(
                0.98, 0.02,
                f"Default = {default_value:.3g}",
                ha='right', va='bottom',
                fontsize=FONTSIZE, color='black',#palette[-1],
                transform=ax.transAxes
            )
        else:
            # Bottom left annotation for RMSE and Logloss
            ax.text(
                0.98, 0.98,
                f"Default = {default_value:.3g}",
                ha='right', va='top',
                fontsize=FONTSIZE, color='black',#palette[-1],
                transform=ax.transAxes
            )
    return leg_default
def order_legend(handles1, labels1):
    """
    Orders legend handles and labels alphabetically by label.
    Parameters
    ----------
    handles1 : list
        List of legend handles.
    labels1 : list
        List of legend labels.
    Returns
    -------
    tuple
        (ordered_handles, ordered_labels)
    """
    unique_legend_elements = {}

    
    for handle, label in zip(handles1,labels1):
        unique_legend_elements[label] = handle
        print("infinite loop")
    sorted_items = sorted(unique_legend_elements.items(), key=lambda x: x[0])
    keys, values = zip(*sorted_items)
    keys = list(keys)
    values = list(values)

    return keys, values
def save_plot(plt,task, randomness, relative_comparison, normalize, test, 
              TUNING = None, metric = None, suffix = None, percentiles = None):
    """
    Saves the plot to the appropriate folder and filename based on task and options.
    Parameters
    ----------
    plt : matplotlib.pyplot
        Plot object to save.
    task : str
        Type of plot ('compare_method', 'compare_tuning', 'by_task').
    randomness : str
        Randomness type ('task', 'seeds', 'both').
    confidence_interval : bool
        Whether to include confidence intervals.
    relative_comparison : bool
        Whether to use relative comparison.
    normalize : bool
        Whether to normalize results.
    test : bool
        Whether this is a test plot.
    TUNING : str, optional
        Tuning strategy name.
    metric : str, optional
        Metric name.
    suffix : str, optional
        Suffix for filename.
    percentiles : tuple, optional
        Percentiles for test plots.
    """
    if randomness == 'task':
        folder = 'main'
    elif randomness == 'seeds' or randomness == 'both':
        folder = 'appendix'
        
    if task == "compare_method":
        file = f'compare_methods_{TUNING}_{randomness}'
        if not normalize:
            file += 'not_normalized'
        if relative_comparison:
            folder = 'appendix'
            file += 'relative'
    elif task == "compare_tuning":
        file = f"compare_tuning_settings_{randomness}"
        if not normalize:
            file += '_not_normalized'
    elif task == "by_task":
        file = f'compare_methods_bytask_{TUNING}_{metric}'
        if metric in ['R2', 'RMSE']:
            file += '_regr'
        else:
            file += '_class'
        if TUNING == 'num_leaves':
            folder = 'main'
        else:
            folder = 'appendix'
        if randomness is not None:
            file += f'_{randomness}'
        if not normalize:
            file += f'_not_normalized'
        
        # Add suffix for multiple plots
        file += suffix
    if test:
        plt.suptitle(f"Score Quantile {percentiles[0]} Loss Quantile {percentiles[1]}")
        file += f"{percentiles[0]}_{percentiles[1]}"
        plt.savefig(f'Test/{file}.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'plots_pub/{folder}/{file}.pdf', bbox_inches='tight')
    plt.close()


def compute_avg_std(res, metric, method, index_tuning, randomness, relative=False, extremum=None, per_task = False, debug = False):
    """
    Computes the standard deviation of the average metric values across either tasks or seeds, 
    optionally relative to an extremum value.
    Args:
        res (dict): Dictionary containing metric results, indexed by metric and method.
        metric (str): The metric key to use from the results dictionary.
        method (str): The method key to use from the results dictionary.
        index_tuning (int or list): Index or indices for selecting tuning data.
        randomness (str): Specifies the axis of randomness, either 'task', 'seeds' or 'both'.
        relative (bool, optional): If True, computes values relative to the provided extremum. Defaults to False.
        extremum (dict, optional): Dictionary of extremum values for normalization. Required if relative is True.
        per_task (bool, default = False): if we need std for every task over seeds
        debug (bool, default = False): true enables print of some values to debug
    Returns:
        tuple:
            m (int or None): The number of tasks or seeds, depending on the randomness axis.
            avg (np.ndarray or None): The computed standard deviation of the average metric values.
    """
    if per_task:
        if randomness == 'task':
            ValueError("Randomness task and per task std is not possible.")
        else:
            avg = np.std(res[metric][method] / (extremum[metric] if relative and extremum is not None else 1), axis=3)[index_tuning,:,:]
            m = res[metric][method].shape[3]
            if debug:
                print(f"this is the {avg} if relative is {relative}")
                print(f"randomness {randomness} and {m}")   
    else:
        if randomness == 'task':
            arr = np.mean(res[metric][method] / (extremum[metric] if relative and extremum is not None else 1), axis=3)
            avg = np.std(arr[index_tuning,:,:], axis=1)
            m = res[metric][method].shape[2]
            if debug:
                print(f"{metric}, {method}: this is the arr {arr[index_tuning,-1,:]} with shape {arr.shape} and the avg {avg} if relative is {relative}")
                print(f"randomness {randomness} and {m}")   
        elif randomness == 'seeds':
            arr = np.mean(res[metric][method] / (extremum[metric] if relative and extremum is not None else 1), axis=2)
            avg = np.std(arr[index_tuning,:,:], axis=1)
            m = res[metric][method].shape[3]
            if debug:
                print(f"this is the {avg} if relative is {relative}")
                print(f"randomness {randomness} and {m}")   
        elif randomness == 'both':
            avg = np.std(res[metric][method] / (extremum[metric] if relative and extremum is not None else 1), axis=(2,3))[index_tuning,:]
            m = res[metric][method].shape[2]*res[metric][method].shape[3]
            if debug:
                print(f"randomness {randomness} and {m}")
        else:
            avg = None
            m = None
        if debug:
            with open("plots_pub/Test/diff_uncertainties.txt", "a") as f:
                diff_metric = avg[0]/np.sqrt(m) - avg[-1]/np.sqrt(m)
                f.write(f"Randomness {randomness} and scaling {m}, {metric} with {method} diff: {diff_metric}\n")
    return m, avg

def plot_confidence_interval(ax, iterations, y, avg_std, m, color, alpha_fill=ALPHA_FILL, alpha_line=ALPHA, linewidths=(linewidth_2,linewidth_2)):
    """
    Plots a confidence interval around a mean curve on the given matplotlib axis.
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        iterations (array-like): The x-values (e.g., iteration numbers).
        y (array-like): The mean values corresponding to each iteration.
        avg_std (float or array-like): The average standard deviation(s) for the confidence interval.
        m (int): The number of samples used to compute the mean and standard deviation.
        color (str or tuple): The color for the confidence interval and lines.
        alpha_fill (float, optional): Transparency for the filled confidence interval. Default is ALPHA_FILL.
        alpha_line (float, optional): Transparency for the boundary lines. Default is ALPHA.
        linewidths (tuple, optional): Line widths for the lower and upper boundary lines. Default is (linewidth_2, linewidth_2).
    Returns:
        None
    """
    lower = y - (2*avg_std)/np.sqrt(m)
    upper = y + (2*avg_std)/np.sqrt(m)
    print(f'uncertainty {(2*avg_std)/np.sqrt(m)}')
    ax.fill_between(iterations, lower, upper, color=color, alpha=alpha_fill)
    ax.plot(iterations, lower, linestyle='--', color=color, alpha=alpha_line, linewidth=linewidths[0])
    ax.plot(iterations, upper, linestyle='--', color=color, alpha=alpha_line, linewidth=linewidths[1])

def style_hyperparameter_results(df: pd.DataFrame):
    """
    Generates a LaTeX table with conditional formatting from a pandas DataFrame.

    This function manually adds LaTeX color commands to the DataFrame content
    before converting it to a LaTeX string.

    Args:
        df: The pandas DataFrame to convert.

    Returns:
        A string containing the full LaTeX table code.
    """
    # Add LaTeX color commands to the numerical data
    # Create a copy to modify
    df_latex = df.copy()
    for col in df_latex.columns:
        if '(norm)' in col or '(rel)' in col:
            # Get min/max from the original, unmodified data
            v_min = df[col].min()
            v_max = df[col].max()
            
            # Use floating point numbers for precision
            reverse = ('RMSE' in col or 'Log Loss' in col)
            
            # Apply a continuous color gradient based on value
            for row in df_latex.index:
                
                value = df.loc[row, col]
                if reverse:
                    if value == v_min:
                        df_latex.loc[row, col] =  f"\\textbf{{{value:.3f}}}"
                    else:
                        df_latex.loc[row, col] = f"{value:.3f}"
                else:
                    if value == v_max:
                        df_latex.loc[row, col] =  f"\\textbf{{{value:.3f}}}"
                    else:
                        df_latex.loc[row, col] = f"{value:.3f}"

        elif '(rank)' in col:
            v_min = df[col].min()
            for row in df_latex.index:
                value = df.loc[row, col]
                if v_min == value:
                    df_latex.loc[row, col] =  f"\\textbf{{{value:.3f}}}"
                else:
                    df_latex.loc[row, col] = f"{value:.3f}"


    df_latex = df_latex.transpose()
    # Use to_latex to generate the table. `escape=False` is crucial.
    latex_string = df_latex.to_latex(
        escape=False,
        column_format='l' + 'c' * len(df_latex.columns),
        caption='Compare Methods with Num Leaves Tuning',
        label='tab:table_compare_methods_num_leaves'
    )
    string = latex_string
    return string