"""Some functions copied from Ioana Iacobici https://github.com/iiacobici and modified by Floris Koster https://github.com/fl0risk"""
import json

import numpy as np
import pandas as pd
from task_types import suites_for
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns



seeds = [27225, 34326,92161, 99246, 108473, 117739,  235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 936518,32244,147316, 777646, 778572]

NUM_SEEDS = len(seeds)
NUM_TASKS = 59
NUM_REGR_TASKS = 36
NUM_CLASS_TASKS = 23
NUM_ITERS = 135
METHODS = ['grid_search', 'random_search', 'tpe', 'gp_bo']
METHODS_JOINT = ['random_search', 'tpe', 'gp_bo']
NAME_METHODS = ['Fixed grid search', 'Random grid search', 'TPE', 'GP-BO']
NAME_METHODS_JOINT = ['Random grid search', 'TPE', 'GP-BO']
NAME_TUNING = ['Num Leaves','Max Depth','Joint']
NUM_TUNING = len(NAME_TUNING)
NUM_METHODS = len(METHODS)
RANDOMNESS = ['both','seeds','tasks']
NUM_METHODS_JOINT = len(METHODS_JOINT)
FOLDS = [0, 1, 2, 3, 4]
NUM_FOLDS = len(FOLDS)
MARKERS = ["o", "*", "^", "s"]


def set_plot_theme():
    # Set Seaborn theme
    sns.set_theme(context="paper", style="white")
    palette = ['lime', 'darkorange', 'fuchsia', 'deepskyblue']

    return palette


def create_scores_dict(classification=False, rmse=False):
    if classification:
        NUM_TASKS = NUM_CLASS_TASKS
        suites = suites_for(True)
    
    else:
        NUM_TASKS = NUM_REGR_TASKS
        suites = suites_for(False) 
    
    scores_NL = np.zeros((NUM_METHODS, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    scores_MD = np.zeros((NUM_METHODS, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    scores_J = np.zeros((NUM_METHODS_JOINT, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    k = 0
    names = []
    
    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                #print('Seed',seed, 'Suite_id', suite_id, 'Task_id', task_id)
                data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results/seed_{seed}/{suite_id}_{task_id}.csv")
                #data['current_best_test_score'] = data['current_best_test_score'].clip(0, 1)
                if rmse:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_rmse'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_rmse': try_max_depth, 'try_num_leaves_rmse': try_num_leaves})
                    df_joint = pd.DataFrame({'try_joint_rmse': try_joint})
                    df['method'] = data.loc[0:df.shape[0]-1, 'method']
                    df['fold'] = data.loc[0:df.shape[0]-1, 'fold']
                    df_joint['method'] = data.loc[(data['method']!= 'grid_search'), 'method'].reset_index(drop=True)
                    df_joint['fold'] = data.loc[(data['method']!= 'grid_search'), 'fold'].reset_index(drop=True)
                    for i, method in enumerate(METHODS):
                        for m in FOLDS:
                            scores_NL[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_rmse'].values
                            scores_MD[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_rmse'].values
                    for i, method in enumerate(METHODS_JOINT):
                        for m in FOLDS:
                            scores_J[i, :, k, l, m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), 'try_joint_rmse'].values
                else:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_score'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_score': try_num_leaves})
                    df_joint = pd.DataFrame({'try_joint_score': try_joint})
                    df['method'] = data.loc[0:df.shape[0]-1, 'method']
                    df['fold'] = data.loc[0:df.shape[0]-1, 'fold']
                    df_joint['method'] = data.loc[(data['method']!= 'grid_search'), 'method'].reset_index(drop=True)
                    df_joint['fold'] = data.loc[(data['method']!= 'grid_search'), 'fold'].reset_index(drop=True)                    
                    for i, method in enumerate(METHODS):
                        for m in FOLDS:
                            scores_NL[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_score'].values
                            scores_MD[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_score'].values
                    for i, method in enumerate(METHODS_JOINT):
                        for m in FOLDS:
                            scores_J[i, :, k, l, m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), 'try_joint_score'].values
            k += 1
    aggregated_scores_NL = np.mean(scores_NL, axis=-1) #take mean over folds
    aggregated_scores_MD = np.mean(scores_MD, axis=-1) #take mean over folds
    aggregated_scores_J = np.mean(scores_J, axis=-1) #take mean over folds
    if not rmse :
        aggregated_scores_NL = aggregated_scores_NL.clip(0,1)
        aggregated_scores_MD = aggregated_scores_MD.clip(0,1)
        aggregated_scores_J = aggregated_scores_J.clip(0,1)
    return aggregated_scores_NL, aggregated_scores_MD, aggregated_scores_J, names
def create_scores_dict_joint_tuning(classification=False, rmse=False):
    if classification:
        NUM_TASKS = NUM_CLASS_TASKS
        suites = suites_for(True)
    
    else:
        NUM_TASKS = NUM_REGR_TASKS
        suites = suites_for(False) 

    scores_J = np.zeros((NUM_METHODS_JOINT, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    k = 0
    names = []
    
    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                #print('Seed',seed, 'Suite_id', suite_id, 'Task_id', task_id)
                data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Result_Task_2/seed_{seed}/{suite_id}_{task_id}.csv")
                #data['current_best_test_score'] = data['current_best_test_score'].clip(0, 1)
                if rmse:
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    df_joint = pd.DataFrame({'try_joint_rmse': try_joint})
                    df_joint['method'] = data.loc[(data['method']!= 'grid_search'), 'method'].reset_index(drop=True)
                    df_joint['fold'] = data.loc[(data['method']!= 'grid_search'), 'fold'].reset_index(drop=True)
                    for i, method in enumerate(METHODS_JOINT):
                        for m in FOLDS:
                            scores_J[i, :, k, l, m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), 'try_joint_rmse'].values
                else:
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    df_joint = pd.DataFrame({'try_joint_score': try_joint})
                    df_joint['method'] = data.loc[(data['method']!= 'grid_search'), 'method'].reset_index(drop=True)
                    df_joint['fold'] = data.loc[(data['method']!= 'grid_search'), 'fold'].reset_index(drop=True)                                    
                    for i, method in enumerate(METHODS_JOINT):
                        for m in FOLDS:
                            scores_J[i, :, k, l, m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), 'try_joint_score'].values
            k += 1
    aggregated_scores_J = np.mean(scores_J, axis=-1) #take mean over folds
    if not rmse :
        aggregated_scores_J = aggregated_scores_J.clip(0,1)
    return aggregated_scores_J, names
def normalize_scores(scores, adtm=False):
    if adtm: 
        max = np.max(scores, axis=(0, 1, 3)) #gets maximum for each task
        min = np.percentile(scores, 10, axis=(0, 1, 3))

        norm_scores = (scores - min[np.newaxis, np.newaxis, :, np.newaxis]) / (max[np.newaxis, np.newaxis, :, np.newaxis] - min[np.newaxis, np.newaxis, :, np.newaxis])
        norm_scores = np.clip(norm_scores, 0, 1)

    else:
        mean_for_norm = np.mean(scores, axis=(0, 1, 3))
        std_for_norm = np.std(scores, axis=(0, 1, 3))

        norm_scores = (scores - mean_for_norm[np.newaxis, np.newaxis, :, np.newaxis]) / std_for_norm[np.newaxis, np.newaxis, :, np.newaxis]

    return norm_scores


def plot_scores_per_task_tuning_methods(scores, METHOD,names, classification=False, adtm=False, confidence_interval=False, rmse=False):
    name_tuning = NAME_TUNING
    if METHOD == 'grid_search':
        name_tuning = ['num_leaves','max_depth']
    if rmse:
        title = 'Average RMSE'

    else:
        title = 'Average score'

    if adtm:
        title += ' (ADTM)'

    if classification:
        num_tasks = NUM_CLASS_TASKS
        title += ' per iteration for each classification task'

    else:
        num_tasks = NUM_REGR_TASKS
        title += ' per iteration for each regression task'
    title += f' using the hyperparameter selction method: {METHOD}'
    num_cols = 4
    num_rows = (num_tasks + num_cols - 1) // num_cols
    method_index = METHODS.index(METHOD)
    palette = set_plot_theme()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    mean_norm_scores = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    std_norm_scores  = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    lower_lim  = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    upper_lim  = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    helper = [name =='joint' for name in name_tuning] #used since for joint tuning there is one method less
    for i in range(len(name_tuning)):
        scores[i] = normalize_scores(scores[i], adtm)
        mean_norm_scores[i] = np.mean(scores[i], axis=-1)
        std_norm_scores[i] = np.std(scores[i], axis=-1)
        if confidence_interval:
            lower_lim[i] = np.percentile(scores[i], 5, axis=-1)
            upper_lim[i] = np.percentile(scores[i], 95, axis=-1)
    for i in range(len(name_tuning)):
        for k in range(num_tasks):
            ax = axes[k]
            ax.plot(
                np.arange(NUM_ITERS),
                mean_norm_scores[i][method_index , :, k],
                label=name_tuning[i],
                color=palette[i],
                marker=MARKERS[i],
                markersize=14,
                linewidth=2.5,
                markevery=20
            )
            if confidence_interval:
                ax.fill_between(np.arange(NUM_ITERS), lower_lim[i][method_index , :, k], upper_lim[i][method_index , :, k], hatch='/', alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), lower_lim[i][method_index , :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), upper_lim[i][method_index , :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

            else:
                ax.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :, k]
                                    - std_norm_scores[i][method_index , :, k], mean_norm_scores[i][method_index , :, k] + std_norm_scores[i][method_index , :, k], alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :, k]
                            - std_norm_scores[i][method_index , :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :, k]
                            + std_norm_scores[i][method_index , :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

            if len(names[k]) > 24:
                names[k] = names[k][:24]

            if not classification:
                if k == 3:
                    names[k] = names[k][:21]
                elif k == 9:
                    names[k] = names[k][:16]

            ax.set_title(names[k], fontsize=28)

            # Customize grid
            ax.grid(True, color='lightgray', linewidth=0.5)

            # Hide x labels and tick labels for all but the bottom row
            if k < (num_rows - 1) * num_cols:
                if not (classification and k == 19):
                    ax.set_xticklabels([])

            # Hide y labels and tick labels for all but the leftmost column
            if k % num_cols != 0:
                    ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_edgecolor('dimgray')  # Set the desired color here
                spine.set_linewidth(1)      # Optionally, adjust the thickness

            # Increase the size of remaining tick labels
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            ax.set_xlim(0, NUM_ITERS - 1)
            if adtm:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(-1, 1)

    # Remove any empty subplots
    for a in range(num_tasks, num_rows * num_cols):
        fig.delaxes(axes[a])

    fig.suptitle(title, fontsize=32)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=len(name_tuning), fontsize=30, bbox_to_anchor=(0.5, 0.959)) if classification else fig.legend(lines, labels, loc='upper center', ncol=len(METHODS), fontsize=30, bbox_to_anchor=(0.5, 0.968))#0.953
    fig.text(0.5, 0.045, 'Iteration', ha='center', va='center', fontsize=30)
    fig.text(0.04, 0.5, 'Average test score', ha='center', va='center', rotation='vertical', fontsize=30)

    plt.tight_layout(rect=[0, 0, 1, 0.85]) 
    plt.subplots_adjust(top=0.894, bottom=0.08, left=0.1, hspace=0.2, wspace=0.2) if classification else plt.subplots_adjust(top=0.924, bottom=0.07, left=0.1, hspace=0.2, wspace=0.2)#top=0.907
    file = "scores_per_task"
    file += "_classification" if classification else "_regression"
    if adtm:
        file += "_adtm"

    if confidence_interval:
        file += "_confidence_interval"

    if rmse:
        file += "_rmse"
    file += f"_{METHOD}"

    plt.savefig(f"plots/{file}.png")
    plt.show()

def plot_scores_aggregated_tasks_per_tuning_method(scores, METHOD, classification=False, adtm=False, confidence_interval=False, randomness=0, rmse=False):
    name_tuning = NAME_TUNING
    if classification:
        num_tasks = NUM_CLASS_TASKS
    else: 
        num_tasks = NUM_REGR_TASKS
    if METHOD == 'grid_search':
        raise ValueError("Aggregated plots are not available for the 'grid_search' method.")
    if rmse:
        title = 'Average RMSE'

    else:
        title = 'Aggregated score'

    # if adtm:
    #     title += ' (ADTM)'
    
    # if classification:
    #     title += ' for classification tasks'

    # else:
    #     title += ' for regression tasks'
    # title += f' using the hyperparameter selction method: {METHOD}'
    palette = set_plot_theme()
    plt.figure(figsize=(12, 8))
    method_index = METHODS_JOINT.index(METHOD)
    
    norm_scores = [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    mean_seeds = [np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks)) for _ in range(len(name_tuning))]
    mean_tasks = [np.zeros((NUM_METHODS_JOINT,NUM_ITERS,NUM_SEEDS))for _ in range(len(name_tuning))]
    avg_var_across_tasks = [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    avg_var_across_seeds = [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    mean_norm_scores =  [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    std_norm_scores  =  [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    lower_lim  =  [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    upper_lim  =  [np.zeros((NUM_METHODS_JOINT,NUM_ITERS)) for _ in range(len(name_tuning))]
    
    for i in range(len(name_tuning)):
        if i==0 or i ==1:
            norm_scores[i] = normalize_scores(scores[i][1:,:,:,:], adtm)
        else:
            norm_scores[i] = normalize_scores(scores[i], adtm)
        mean_norm_scores[i] = np.mean(norm_scores[i], axis=(2,3))
        std_norm_scores[i] = np.std(norm_scores[i], axis=(2,3))
        if confidence_interval:
            lower_lim[i] = np.percentile(norm_scores[i], 5, axis=(2,3))
            upper_lim[i] = np.percentile(norm_scores[i], 95, axis=(2,3))

    for i in range(len(name_tuning)):
        plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index, :], label=NAME_TUNING[i], color=palette[i], marker=MARKERS[i], markersize=14, linewidth=2.5, markevery=15)
        #print(i, method_index , NAME_TUNING[i])
        if randomness == 1:       # Randomness due to the seeds
            mean_tasks[i] = np.mean(norm_scores[i], axis=2)
            avg_var_across_tasks[i] = np.std(mean_tasks[i], axis=-1)

            plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] - avg_var_across_tasks[i][method_index , :], mean_norm_scores[i][method_index , :] + avg_var_across_tasks[i][method_index , :], alpha=0.2, color=palette[i])
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] - avg_var_across_tasks[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] + avg_var_across_tasks[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
        
        # Randomness due to the tasks
        elif randomness == 2:
            mean_seeds[i] = np.mean(norm_scores[i], axis=-1)
            avg_var_across_seeds[i] = np.std(mean_seeds[i], axis=-1)
            plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] - avg_var_across_seeds[i][method_index , :], mean_norm_scores[i][method_index , :] + avg_var_across_seeds[i][method_index , :], alpha=0.2, color=palette[i])
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] - avg_var_across_seeds[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] + avg_var_across_seeds[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

        else:
            if confidence_interval:
                plt.fill_between(np.arange(NUM_ITERS), lower_lim[i][method_index , :], upper_lim[i][method_index , :], hatch='/', alpha=0.2, color=palette[i])
                plt.plot(np.arange(NUM_ITERS), lower_lim[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(np.arange(NUM_ITERS), upper_lim[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            
            else:
                plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] - std_norm_scores[i][method_index , :], mean_norm_scores[i][method_index , :] + std_norm_scores[i][method_index , :], alpha=0.2, color=palette[i])
                plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] - std_norm_scores[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index , :] + std_norm_scores[i][method_index , :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('dimgray')  # Set the desired color here
        spine.set_linewidth(1)      # Optionally, adjust the thickness
    
    # Customize grid
    plt.gca().grid(True, color='lightgray', linewidth=0.5)

    # Set the y-axis to have at most 5 ticks
    plt.gca().tick_params(axis='both', which='major', labelsize=28)
    plt.gca().xaxis.set_major_locator(MaxNLocator(6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.title(title, fontsize=24)
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Average aggregated test score', fontsize=30)
    if rmse:
        plt.legend(loc="center left", ncol=1, bbox_to_anchor=(0.413, 0.8), fontsize=30)

    else:
        plt.legend(loc="center left", ncol=1, bbox_to_anchor=(0.413, 0.2), fontsize=30)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(METHODS), fontsize=30)

    plt.xlim(0, NUM_ITERS - 1)
    if adtm:
        plt.ylim(0, 1)

    else:
        plt.ylim(-0.5, 1)
    
    # # Create a table with the information as one row below the title
    # table_data = [
    #     ['Task', 'Regression' if classification == False else 'Classification', 
    #      "Method", 'GP-Boost' if METHOD == 'gp_bo' else 'TPE' if METHOD == 'tpe' else 'Random Search', 
    #      "Randomness", "Seeds" if randomness == 1 else "Tasks" if randomness == 2 else "Both", 
    #      "ADTM", "Yes" if adtm else "No"]
    # ]
    # plt.subplots_adjust(bottom=0.2) 
    # table = plt.table(cellText=table_data, colWidths=[0.15] * len(table_data[0]), loc='bottom', cellLoc='center', bbox=[0, -0.3, 1, 0.1])
    
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # for (row,col), cell in table.get_celld().items():
    #     cell.set_height(0.05)
    #     if col  % 2 == 0:
    #         cell.get_text().set_fontweight('bold') 
    #         cell.set_facecolor('lightgray')
    #     else:
    #         cell.set_facecolor('white')
    #         cell.set_edgecolor('black')
    file = "agg_scores"
    file += "_classification" if classification else "_regression"
    if adtm:
        file += "_adtm"

    if confidence_interval:
        file += "_confidence_interval"

    if randomness == 0:
        file += "_total_randomness"

    elif randomness == 1:
        file += "_randomness_seeds"

    elif randomness == 2:
        file += "_randomness_tasks"

    if rmse:
        file += "_rmse"
    file += f"_{METHOD}"

    plt.savefig(f"plots/{file}.png")
    plt.show()


def aggregated_tasks(scores,METHOD,confidence_interval=False):
    'Scores need to be of the form [SCORES_REG, SCORES_CLASS, RMSE]'
    method_ind = METHODS_JOINT.index(METHOD)
    palette = set_plot_theme()
    norm_scores = [[np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    
    mean_norm_scores = [[np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    std_norm_scores = [[np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    lower_lim = [[np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    upper_lim =  [[np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    mean_seeds = [[np.zeros((NUM_ITERS,NUM_TASKS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    mean_tasks = [[np.zeros((NUM_ITERS,NUM_SEEDS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    avg_var_across_tasks = [[np.zeros((NUM_METHODS,NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    avg_var_across_seeds = [[np.zeros((NUM_METHODS,NUM_ITERS)) for _ in range(len(NAME_TUNING))] for _ in range(len(scores))]
    
    for j in range(len(scores)): #scores = [scores_reg, scores_class, RMSE]
        for i in range(len(NAME_TUNING)):
            #norm_scores[j][i] = normalize_scores(scores[j][i],  False)
            if i==0 or i ==1:
                norm_scores[j][i] = normalize_scores(scores[j][i][1:,:,:,:], False if j==2 else True)
            else:
                norm_scores[j][i] = normalize_scores(scores[j][i],  False if j==2 else True)
            mean_norm_scores[j][i] = np.mean(norm_scores[j][i], axis=(2,3))

            std_norm_scores[j][i] = np.std(norm_scores[j][i], axis=(2,3))
            if confidence_interval:
                lower_lim[j][i] = np.percentile(norm_scores[j][i], 5, axis=(2,3))
                upper_lim[j][i] = np.percentile(norm_scores[j][i], 95, axis=(2,3))
        
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
            
    for randomness in range(len(RANDOMNESS)):
        for j in range(len(scores)):
                ax = axes[randomness * 3 + j]  # Select the appropriate subplot
                for i in range(len(NAME_TUNING)):
                    ax.plot(
                        np.arange(NUM_ITERS),
                        mean_norm_scores[j][i][method_ind,:],
                        color=palette[i],
                        marker=MARKERS[i],
                        label=NAME_TUNING[i],
                        markersize=14,
                        linewidth=2.5,
                        markevery=15
                    )
                    #print(i,j, NAME_TUNING[i])
                    if randomness == 1:  # Randomness due to the seeds
                        mean_tasks[j][i] = np.mean(norm_scores[j][i], axis=1)
                        avg_var_across_tasks[j][i] = np.std(mean_tasks[j][i], axis=-1)

                        ax.fill_between(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[j][i][method_ind , :] - avg_var_across_tasks[j][i][method_ind , :],
                            mean_norm_scores[j][i][method_ind , :] + avg_var_across_tasks[j][i][method_ind , :],
                            alpha=0.2,
                            color=palette[i]
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[j][i][method_ind , :] - avg_var_across_tasks[j][i][method_ind , :],
                            linestyle='--',
                            color=palette[i],
                            alpha=0.6,
                            linewidth=2.5
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[j][i][method_ind , :] + avg_var_across_tasks[j][i][method_ind , :],
                            linestyle='--',
                            color=palette[i],
                            alpha=0.6,
                            linewidth=2.5
                        )

                    elif randomness == 2:  # Randomness due to the tasks
                        mean_seeds[j][i] = np.mean(norm_scores[j][i], axis=2)
                        avg_var_across_seeds[j][i] = np.std(mean_seeds[j][i], axis=-1)

                        ax.fill_between(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[j][i][method_ind , :] - avg_var_across_seeds[j][i][method_ind , :],
                            mean_norm_scores[j][i][method_ind , :] + avg_var_across_seeds[j][i][method_ind , :],
                            alpha=0.2,
                            color=palette[i]
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[j][i][method_ind , :] - avg_var_across_seeds[j][i][method_ind , :],
                            linestyle='--',
                            color=palette[i],
                            alpha=0.6,
                            linewidth=2.5
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[j][i][method_ind , :] + avg_var_across_seeds[j][i][method_ind , :],
                            linestyle='--',
                            color=palette[i],
                            alpha=0.6,
                            linewidth=2.5
                        )

                    else:
                        if confidence_interval:
                            ax.fill_between(
                                np.arange(NUM_ITERS),
                                lower_lim[j][i][method_ind,:],
                                upper_lim[j][i][method_ind,:],
                                hatch='/',
                                alpha=0.2,
                                color=palette[i]
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                lower_lim[j][i][method_ind,:],
                                linestyle='--',
                                color=palette[i],
                                alpha=0.6,
                                linewidth=2.5
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                upper_lim[j][i][method_ind,:],
                                linestyle='--',
                                color=palette[i],
                                alpha=0.6,
                                linewidth=2.5
                            )
                        else:
                            ax.fill_between(
                                np.arange(NUM_ITERS),
                                mean_norm_scores[j][i][method_ind,:] - std_norm_scores[j][i][method_ind,:],
                                mean_norm_scores[j][i][method_ind,:] + std_norm_scores[j][i][method_ind,:],
                                alpha=0.2,
                                color=palette[i]
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                mean_norm_scores[j][i][method_ind,:] - std_norm_scores[j][i][method_ind,:],
                                linestyle='--',
                                color=palette[i],
                                alpha=0.6,
                                linewidth=2.5
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                mean_norm_scores[j][i][method_ind,:] + std_norm_scores[j][i][method_ind,:],
                                linestyle='--',
                                color=palette[i],
                                alpha=0.6,
                                linewidth=2.5
                            )

                for spine in ax.spines.values():
                    spine.set_edgecolor('dimgray')  # Set the desired color here
                    spine.set_linewidth(1)  # Optionally, adjust the thickness

                # Customize grid
                ax.grid(True, color='lightgray', linewidth=0.5)

                # Set the y-axis to have at most 5 ticks
                ax.tick_params(axis='both', which='major', labelsize=28)
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                title = 'Regression' if j == 0 else 'Classification' if j == 1 else 'RMSE in Regression'
                
                title += f' and Randomness: '
                title += 'Total' if randomness == 0 else 'Seeds' if randomness == 1 else 'Task'
                ax.set_title(title, fontsize=18)
                ax.set_xlim(0, NUM_ITERS - 1)
                if j == 0 or j == 1:
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(-0.5, 1)
                    
                for i, ax in enumerate(axes):
                    if i % 3 == 0:  
                        ax.set_ylabel('Average aggregated test score', fontsize=14)
                    if i // 3 == 2: 
                        ax.set_xlabel('Iteration', fontsize=14)
    lines, labels = axes[0].get_legend_handles_labels()                        
    fig.legend(lines, labels, loc='lower center', ncol=len(NAME_METHODS_JOINT), fontsize=16, bbox_to_anchor=(0.5, -0.025))
    bigtitle = 'Random Search' if METHOD == 'random_search' else 'TPE' if METHOD == 'tpe' else 'GP-Boost'
    if confidence_interval:
        bigtitle += ' with Confindence Interval'
    plt.suptitle(bigtitle, fontsize=30,y=1.001)
    plt.tight_layout()
    plt.savefig(f"plots/agg_scores_{METHOD}.png",bbox_inches='tight')
    plt.show()

def compare_method(scores,classification = False, RMSE = False, confidence_interval = False):
    'Scores need to be of the form [NL, MD, JOINT]'
    if classification and RMSE:
        ValueError('Classification and RMSE not possible')
    if classification:
        num_tasks = NUM_CLASS_TASKS
    else:
        num_tasks = NUM_REGR_TASKS
    palette = set_plot_theme()
    norm_scores = [np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    
    mean_norm_scores = [np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    std_norm_scores = [np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    lower_lim = [np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    upper_lim =  [np.zeros((NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    mean_seeds = [np.zeros((NUM_ITERS,num_tasks)) for _ in range(len(NAME_TUNING))] 
    mean_tasks = [np.zeros((NUM_ITERS,NUM_SEEDS)) for _ in range(len(NAME_TUNING))] 
    avg_var_across_tasks = [np.zeros((NUM_METHODS,NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    avg_var_across_seeds = [np.zeros((NUM_METHODS,NUM_ITERS)) for _ in range(len(NAME_TUNING))]
    
    
    for i in range(len(NAME_TUNING)):
        if i==0 or i ==1:
            norm_scores[i] = normalize_scores(scores[i][1:,:,:,:],not RMSE)
        else:
            norm_scores[i] = normalize_scores(scores[i],  not RMSE)
        mean_norm_scores[i] = np.mean(norm_scores[i], axis=(2,3))

        std_norm_scores[i] = np.std(norm_scores[i], axis=(2,3))
        if confidence_interval:
            lower_lim[i] = np.percentile(norm_scores[i], 5, axis=(2,3))
            upper_lim[i] = np.percentile(norm_scores[i], 95, axis=(2,3))
    
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
            
    for randomness in range(len(RANDOMNESS)):
        for i in range(len(NAME_TUNING)):
                ax = axes[randomness * 3 + i]  # Select the appropriate subplot
                for method_ind in range(len(METHODS_JOINT)):
                    ax.plot(
                        np.arange(NUM_ITERS),
                        mean_norm_scores[i][method_ind,:],
                        color=palette[method_ind],
                        marker=MARKERS[method_ind],
                        label=NAME_METHODS_JOINT[method_ind],
                        markersize=14,
                        linewidth=2.5,
                        markevery=15
                    )
                    #print(i,j, NAME_TUNING[i])
                    if randomness == 1:  # Randomness due to the seeds
                        mean_tasks[i] = np.mean(norm_scores[i], axis=2)
                        avg_var_across_tasks[i] = np.std(mean_tasks[i], axis=-1)

                        ax.fill_between(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[i][method_ind , :] - avg_var_across_tasks[i][method_ind , :],
                            mean_norm_scores[i][method_ind , :] + avg_var_across_tasks[i][method_ind , :],
                            alpha=0.2,
                            color=palette[method_ind]
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[i][method_ind , :] - avg_var_across_tasks[i][method_ind , :],
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[i][method_ind , :] + avg_var_across_tasks[i][method_ind , :],
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )

                    elif randomness == 2:  # Randomness due to the tasks
                        mean_seeds[i] = np.mean(norm_scores[i], axis=2)
                        avg_var_across_seeds[i] = np.std(mean_seeds[i], axis=-1)

                        ax.fill_between(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[i][method_ind , :] - avg_var_across_seeds[i][method_ind , :],
                            mean_norm_scores[i][method_ind , :] + avg_var_across_seeds[i][method_ind , :],
                            alpha=0.2,
                            color=palette[method_ind]
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[i][method_ind , :] - avg_var_across_seeds[i][method_ind , :],
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )
                        ax.plot(
                            np.arange(NUM_ITERS),
                            mean_norm_scores[i][method_ind , :] + avg_var_across_seeds[i][method_ind , :],
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )

                    else:
                        if confidence_interval:
                            ax.fill_between(
                                np.arange(NUM_ITERS),
                                lower_lim[i][method_ind,:],
                                upper_lim[i][method_ind,:],
                                hatch='/',
                                alpha=0.2,
                                color=palette[method_ind]
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                lower_lim[i][method_ind,:],
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                upper_lim[i][method_ind,:],
                                linestyle='--',
                                color=palette[i],
                                alpha=0.6,
                                linewidth=2.5
                            )
                        else:
                            ax.fill_between(
                                np.arange(NUM_ITERS),
                                mean_norm_scores[i][method_ind,:] - std_norm_scores[i][method_ind,:],
                                mean_norm_scores[i][method_ind,:] + std_norm_scores[i][method_ind,:],
                                alpha=0.2,
                                color=palette[method_ind]
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                mean_norm_scores[i][method_ind,:] - std_norm_scores[i][method_ind,:],
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )
                            ax.plot(
                                np.arange(NUM_ITERS),
                                mean_norm_scores[i][method_ind,:] + std_norm_scores[i][method_ind,:],
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )

                for spine in ax.spines.values():
                    spine.set_edgecolor('dimgray')  # Set the desired color here
                    spine.set_linewidth(1)  # Optionally, adjust the thickness

                # Customize grid
                ax.grid(True, color='lightgray', linewidth=0.5)

                # Set the y-axis to have at most 5 ticks
                ax.tick_params(axis='both', which='major', labelsize=28)
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                title = 'Number Leaves' if i == 0 else 'Max Depth' if i == 1 else 'Joint Tuning'
                
                title += f' and Randomness: '
                title += 'Total' if randomness == 0 else 'Seeds' if randomness == 1 else 'Task'
                ax.set_title(title, fontsize=18)
                ax.set_xlim(0, NUM_ITERS - 1)
                if not RMSE:
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(-0.5, 1)
                    
                for i, ax in enumerate(axes):
                    if i % 3 == 0:  
                        ax.set_ylabel('Average aggregated test score', fontsize=14)
                    if i // 3 == 2: 
                        ax.set_xlabel('Iteration', fontsize=14)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=len(NAME_METHODS_JOINT), fontsize=16, bbox_to_anchor=(0.5, -0.025)) if classification else fig.legend(lines, labels, loc='lower center', ncol=len(METHODS), fontsize=16, bbox_to_anchor=(0.5, -0.025))
    bigtitle = 'Comparison of Methods'
    if RMSE:
        bigtitle += ' using RMSE'
    if classification:
        bigtitle += ' for Classification Tasks'
    if confidence_interval:
        bigtitle += ' with Confindence Interval'
    plt.suptitle(bigtitle, fontsize=30, y=1.001)
    plt.tight_layout()
    file = 'compare_methods'
    if RMSE:
        file += '_RMSE'
    if classification:
        file += '_classficiation'
    if confidence_interval:
        file += '_confindence_interval'
    plt.savefig(f'plots/{file}.png', bbox_inches='tight')
    plt.show()