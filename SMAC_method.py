"""SMAC-based hyperparameter tuning wrappers.

This module mirrors the functionality in `methods.py` but uses the SMAC
optimizer for hyperparameter search. It provides `ParameterOptimizationSMAC`
which exposes the same high-level interface used in the experiment runners.

Author: Floris Koster (https://github.com/fl0risk)
"""
import os
import random
import re
import copy
import gpboost as gpb
import numpy as np
import optuna
import smac
from ConfigSpace import ConfigurationSpace, Integer, Categorical
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario
from functools import partial #used to equip _train_model_for_validation function with inputs
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from skopt import gp_minimize
# from skopt.space import Real, Integer, Categorical
# from skopt.utils import use_named_args

from utils import modified_grid_search_tune_parameters, modify,_get_param_combination,truncate,tune_pars_TPE_algorithm_optuna,runhistory_to_dataframe



class ParameterOptimizationSMAC:
    """This class is used to perform hyperparameter tuning using the proposed methods and to evaluate the model obtained using the best hyperparameters."""

    def __init__(self, X, y, categorical_indicator, suite_id, test_size=0.2, val_size=0.2, try_num_leaves=False, seed=42,joint_tuning_depth_leaves = False,try_num_iter = False, 
                hyperband = False, try_max_depth = True):
        if  (try_max_depth and try_num_leaves) or (try_num_leaves and joint_tuning_depth_leaves) or (try_max_depth and joint_tuning_depth_leaves):
            raise ValueError("You can only perform num_leaves, max_depth or joint_tuning at the same time.")
        self.seed = seed
        self.fixed_seeds = self._generate_local_seeds()

        X, y = self._subsample_data(X, y)

        # Check if the target variable is a pandas Series or a numpy array
        if isinstance(y, np.ndarray):    #np.ndarray is the type of an object created with np.array()
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()  # Convert (n_samples, 1) to (n_samples,)

            y = pd.Series(y)

        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()

        self.X = X
        self.y = y
        self.categorical_indicator = categorical_indicator
        self.test_size = test_size
        self.val_size = val_size
        self.suite_id = suite_id
        #All options for tuning
        self.try_num_leaves = try_num_leaves
        self.joint_tuning_depth_leaves = joint_tuning_depth_leaves
        self.try_num_iter = try_num_iter
        self.try_max_depth = try_max_depth
        self.max_bin_val = np.min([self.X.shape[0], 10000])
        self.X = self._clean_column_names(self.X)

        self._create_other_params()
        self._preprocess_features()

        self.splits = self._split_data()
        self.df_trials = None #variable to store results from objective in hyperband method
        

    def method_smac(self, ):
        """This function runs all hyperparameter tuning methods on the 5 folds and returns the results in a DataFrame."""
        # Iterate through the 5 folds
        for fold, (full_train_index, test_index) in enumerate(self.splits):
            X_train_full, X_test = self.X.iloc[full_train_index], self.X.iloc[test_index]
            y_train_full, y_test = self.y.iloc[full_train_index], self.y.iloc[test_index]
            trials_SMAC = self.run_smac(X_train_full=X_train_full, y_train_full=y_train_full,
                    X_test=X_test, y_test=y_test)
            
            trials = eval(f'trials_SMAC')
            trials['fold'] = fold
            trials['method'] = 'SMAC'

            if fold == 0:
                final_results = trials
            else:
                final_results = pd.concat([final_results, trials])
            
        final_results.reset_index(inplace=True)
        final_results.rename(columns={"index": "iter"}, inplace=True)
        return final_results
    
    def run_smac(self,  X_train_full, y_train_full, X_test, y_test):
        
        self.min_score = float('inf')
        
        # Split the full training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.val_size, random_state=self.fixed_seeds[2]
        )
        param = ConfigurationSpace(
            space = {
            'learning_rate': (0.001, 1.0),
            'min_data_in_leaf': (1, 1000),
            'lambda_l2':  (0.0, 100.0),
            'max_bin': (255, int(self.max_bin_val)),
            'bagging_fraction': (0.5, 1.0),
            'feature_fraction': (0.5, 1.0)
        })
        if self.try_num_leaves:
            num_leaves = Integer(name='num_leaves',bounds = (2,1024))
            param.add([num_leaves])
        if self.joint_tuning_depth_leaves:
                num_leaves = Integer(name='num_leaves',bounds = (2,1024))
                max_depth = Categorical(name='max_depth',items=[-1,1,2,3,4,5,6,7,8,9,10])
                param.add([num_leaves,max_depth])
        if self.try_max_depth:
                max_depth = Categorical(name='max_depth',items=[-1,1,2,3,4,5,6,7,8,9,10])
                param.add([max_depth])
        if self.try_num_iter:
                n_iter = Integer(name='n_iter', bounds = (1,1000))
                param.add([n_iter])
        def objective_opt(params, seed = self.seed):
            
            score, best_iter = self._train_model_for_validation(
                X_train, y_train, X_val, y_val, 
                params
            )
            # Get the best number of iterations
            if score < self.min_score:
                self.min_score = score
                self.best_iter = best_iter
            
            return score
        
        scenario = Scenario(param,deterministic=True, n_trials=270, seed = self.seed)
        smac = HPOFacade(scenario,objective_opt,overwrite=True)
        #incumbent = smac.optimize()
        _ = smac.optimize()
        rh = smac.runhistory
        configuration = rh.get_configs()
        trials = {}
        for k in range(len(configuration)):
            trials[k] = {'params':dict(configuration[k]), 'score': rh.get_cost(configuration[k])}
        #print('This are the trials',trials)
        df_trials = self._convert_dict_to_df(trials)
        #print('Best iteration:',self.best_iter)
        df_trials = self._compute_test_scores(
            X_train_full=X_train_full, y_train_full=y_train_full, X_test=X_test, y_test=y_test,
            df=df_trials, best_iter=self.best_iter
        )

        # Add the 'try_num_leaves', 'joint_tuning_depth_leaves and try_num_iter columns to the DataFrame
        df_trials['try_num_leaves'] = self.try_num_leaves
        df_trials['joint_tuning_depth_leaves'] = self.joint_tuning_depth_leaves
        df_trials['try_num_iter'] = self.try_num_iter
        #reoder order of columns
        #print('Here are the columns', df)
        df_trials = df_trials[['learning_rate','min_data_in_leaf','max_depth','lambda_l2','num_leaves','max_bin',	'bagging_fraction','feature_fraction','val_score','test_score','test_log_loss','test_f1_score','test_rmse',	'current_best_test_score','current_best_test_log_loss',	'current_best_test_f1_score','current_best_test_rmse','try_num_leaves',	'joint_tuning_depth_leaves','try_num_iter']]
        return df_trials

    def _generate_local_seeds(self):
        """This function generates the local seeds for the current task."""
        random.seed(42)
        seeds = random.sample(range(1000, 1000000), 3)

        return seeds
    

    def _subsample_data(self, X, y):
        """This function subsamples the data if the dataset is too large."""
        if X.shape[0] > 100000:
            X = X.sample(n=100000, random_state=self.fixed_seeds[0])
            y = y.loc[X.index]

        return X, y
    

    def _clean_column_names(self, df):
        """This function cleans the column names of a DataFrame by replacing special JSON characters with underscores."""
        # Define a regular expression pattern to match special characters
        pattern = re.compile(r'[\{\}\[\]\"\:\\,]')
 
        df.columns = [pattern.sub('_', col) for col in df.columns]

        return df
    
    def _create_other_params(self):
        """This function creates the 'other_params' dictionary based on the task."""

        self.other_params = {'verbose': -1}

        # Adjust the 'other_params' dictionary based on the task
        if self.try_num_leaves:
            self.other_params['max_depth'] = -1     #{'verbose': -1, 'max_depth': -1}
        elif not self.joint_tuning_depth_leaves:
            self.other_params['num_leaves'] = 2**10     #{'verbose': -1, 'num_leaves': 2**10}
        # Set the objective and metric functions based on the given task
        if self.suite_id in [334, 337]:
            self.other_params['objective'] = 'binary_logit'
            self.other_params['metric'] = 'binary_error'

        else:
            self.other_params['objective'] = 'regression'
            self.other_params['metric'] = 'rmse'
    

    def _preprocess_features(self):
        """This function converts the categorical features of the data into numeric types."""
        # Extract the categorical columns in the DataFrame
        categorical_columns = self.X.columns[self.categorical_indicator]

        # Fit the OneHotEncoder
        enc = OneHotEncoder()
        encoded_columns = enc.fit_transform(self.X[categorical_columns])

        # Format the DataFrame with the encoded columns
        encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=enc.get_feature_names_out(categorical_columns))
        self.X.reset_index(drop=True, inplace=True)
        self.X = pd.concat([self.X.drop(columns=categorical_columns), encoded_df], axis=1)

        # Convert the target variable into numeric type
        if self.suite_id in [334, 337]:
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)

            # Ensure the target variable is a pandas Series
            self.y = pd.Series(self.y)


    def _split_data(self):
        """This function splits the data into 5 folds used for training and testing iteratively."""
        kf = KFold(n_splits=5, shuffle=True, random_state=self.fixed_seeds[1])

        return kf.split(self.X)



    def _train_model_for_validation(self, X_train, y_train, X_val, y_val, params, num_boost_round: int = 1000) -> float: 
        """This function performs the model training and evaluation and returns the prediction accuracy based on the validation set."""
        params_dict = params.get_dictionary()
        params_dict.update(self.other_params)
        train_set = gpb.Dataset(X_train, label=y_train) 
        valid_set = gpb.Dataset(X_val, label=y_val)
        #print(f'Here we go!! {params_dict}')
        # Train the model
        if self.try_num_iter:
            bst = gpb.train(
                params=params_dict, train_set=train_set, valid_sets=[valid_set],
                verbose_eval=False 
            )
        else:
            bst = gpb.train(
                params=params_dict, train_set=train_set, num_boost_round=num_boost_round,
                valid_sets=[valid_set], early_stopping_rounds=20,
                verbose_eval=False
            )
        y_pred = bst.predict(data=X_val, pred_latent=False) #pred_latent = FALSE => response variable is predicted

        # Get the best number of iterations
        best_iter = bst.best_iteration

        # Evaluate the model based on the respective task
        if self.suite_id in [334, 337]:
            y_pred = (y_pred > 0.5).astype(int)
            score = 1 - accuracy_score(y_val, y_pred)

        else:
            score = root_mean_squared_error(y_val, y_pred)
        #print(f'This is the score {score}')
        return score, best_iter
    

    def _compute_test_scores(self, X_train_full, y_train_full, X_test, y_test, df, best_iter = None) -> float:
        """This function trains the model on the full training set and evaluates it on the test set, adding the 'test_score' column to the corresponding DataFrame."""
        test_scores = []
        test_log_loss = []
        test_f1_scores = []
        test_rmse = []
        #df.to_csv('Results/dataframe.csv', index=False)
        for _, row in df.iterrows():
            params_copy = row.drop(['val_score']).to_dict()
            # Ensure the correct types for specific parameters
            params_copy = {key: int(value) if key in ['min_data_in_leaf', 'max_depth', 'num_leaves', 'max_bin','n_iter'] else value for key, value in params_copy.items()} 
            #print("----------This is a row", row)
            #print("----------Here are the params_copy",params_copy)
            train_set_full = gpb.Dataset(X_train_full, label=y_train_full)
    
            # Train the model
            if self.try_num_iter:
                bst = gpb.train(
                    params=params_copy, train_set=train_set_full,
                    verbose_eval=False
                )
            else:
                bst = gpb.train(
                    params=params_copy, train_set=train_set_full, num_boost_round=best_iter,
                    verbose_eval=False
                )
            y_pred = bst.predict(data=X_test, pred_latent=False)

            # Evaluate the model based on the respective task
            if self.suite_id in [334, 337]:
                log_loss_score = log_loss(y_test, y_pred)
                y_pred = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_test, y_pred)
                f_1_score = f1_score(y_test, y_pred)
                rmse = np.nan

            else:
                log_loss_score = np.nan
                f_1_score = np.nan
                score = r2_score(y_test, y_pred, force_finite=True)
                rmse = root_mean_squared_error(y_test, y_pred)


            test_scores.append(score)
            test_log_loss.append(log_loss_score)
            test_f1_scores.append(f_1_score)
            test_rmse.append(rmse)

        # Add the 'test_score' and the 'current_best_test_score' columns to the DataFrame
        df['test_score'] = test_scores
        df['test_log_loss'] = test_log_loss
        df['test_f1_score'] = test_f1_scores
        df['test_rmse'] = test_rmse
        df['current_best_test_score'] = df['test_score'].cummax()
        df['current_best_test_log_loss'] = df['test_log_loss'].cummin()
        df['current_best_test_f1_score'] = df['test_f1_score'].cummax()
        df['current_best_test_rmse'] = df['test_rmse'].cummin()

        return df


    def _convert_dict_to_df(self, dict_params):
        """This function converts the dictionary outputted by the 'modified_grid_search_tune_parameters' function into a DataFrame."""
        # Normalize the dictionary to ensure all parameters are present in each element
        normalized_data = [{**v['params'], 'val_score': v['score']} for v in dict_params.values()]

        # Format the DataFrame
        df = pd.DataFrame(normalized_data)
        if self.try_num_leaves:
            df['max_depth'] = -1
        elif not self.joint_tuning_depth_leaves:
            df['num_leaves'] = 2**10
        return df
    def _create_df(self,param_grid, param_ind, scores):
        """This function creates a dataframe from the dictionary which are created in the 'hyperband' function."""
        # Normalize the dictionary to ensure all parameters are present in each element
        normalized_data = [{**_get_param_combination(ind,param_grid), 'val_score': scores[ind]} for ind in param_ind]

        # Format the DataFrame
        df = pd.DataFrame(normalized_data)
        if self.try_num_leaves:
            df['max_depth'] = -1
        elif not self.joint_tuning_depth_leaves:
            df['num_leaves'] = 2**10
        return df

    def _modify_df_for_tpe(self, df):
        """This function modifies the DataFrame for the TPE method."""
        # Drop unnecessary columns from the DataFrame with the trials
        df.drop(columns=['number', 'datetime_start', 'datetime_complete', 'duration', 'state'], inplace=True)
        if self.try_num_leaves and not(self.try_num_iter):
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin', 'min_data_in_leaf', 'num_leaves']
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves and not(self.try_num_iter):
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin','max_depth', 'min_data_in_leaf', 'num_leaves']
        elif self.try_max_depth and not(self.try_num_iter):
            df.columns = ['val_score','bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin', 'max_depth', 'min_data_in_leaf']
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_max_depth:
            df.columns = ['val_score','bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin', 'max_depth', 'min_data_in_leaf','n_iter','num_leaves']
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_num_leaves:
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin', 'min_data_in_leaf', 'n_iter','num_leaves']
            df['max_depth'] = -1
        elif self.try_num_iter and self.joint_tuning_depth_leaves:
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin','max_depth', 'min_data_in_leaf','n_iter','num_leaves']
        return df
    

    def _convert_array_to_df(self, x_iters, func_vals):
        """This function converts the arrays from the GP-BO trials into a DataFrame."""

        if self.try_num_leaves and not(self.try_num_iter):
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction', 'num_leaves'])
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves and not(self.try_num_iter):
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf','lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction','max_depth', 'num_leaves'])
        elif self.try_max_depth and not(self.try_num_iter):
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'max_depth', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction'])
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_max_depth:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'max_depth', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction','n_iter'])
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_num_leaves:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction', 'num_leaves','n_iter'])
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves and self.try_num_iter:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf','lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction','max_depth', 'num_leaves','n_iter'])
        # Convert the 'score' array into a column for the DataFrame
        df['val_score'] = func_vals

        return df
    def _generate_random_grid_seach_combinations(self,param_grid,num_try_random=None):
        grid_size, param_grid = modify(param_grid)
        if num_try_random is not None:
            if num_try_random > grid_size:
                raise ValueError("num_try_random is larger than the number of all possible combinations of parameters in param_grid ")
            
            try_param_combs = np.random.RandomState(self.seed).choice(a=grid_size, size=num_try_random, replace=False)

        all_combinations = {}
        for param_comb_number in try_param_combs:
            param_comb = _get_param_combination(param_comb_number=param_comb_number, param_grid=param_grid)
            all_combinations[param_comb_number] = {'params': param_comb, 'score': np.nan}

        return all_combinations
    
    def _generate_hyperband_combinations(self,param_grid,num_try_random=None):
        grid_size, param_grid = modify(param_grid)
        if num_try_random is not None:
            if num_try_random > grid_size:
                raise ValueError("num_try_random is larger than the number of all possible combinations of parameters in param_grid ")
            
            try_param_combs = np.random.RandomState(self.seed).choice(a=grid_size, size=num_try_random, replace=False)

        return try_param_combs
    
