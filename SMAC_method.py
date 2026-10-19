"""SMAC-based hyperparameter tuning wrapper.

`ParameterOptimizationSMAC` inherits data handling, training (`_train_and_predict`
for gpboost/xgboost/catboost) and test evaluation from `methods.ParameterOptimization`
and only adds the SMAC search (`method_smac`, `run_smac`).

Author: Floris Koster (https://github.com/fl0risk)
"""
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from ConfigSpace import ConfigurationSpace, Integer, Categorical
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from methods import ParameterOptimization


def _configuration_key(configuration):
    values = configuration.get_dictionary() if hasattr(configuration, 'get_dictionary') else configuration
    return tuple(sorted(values.items()))


def _align_config_runtimes(configurations, runtime_by_configuration):
    aligned_runtimes = []
    for configuration in configurations:
        key = _configuration_key(configuration)
        durations = runtime_by_configuration.get(key)
        if not durations:
            raise RuntimeError(f'No objective runtime recorded for SMAC configuration {key}')
        aligned_runtimes.append(sum(durations))
    return aligned_runtimes



class ParameterOptimizationSMAC(ParameterOptimization):
    """SMAC search on top of ParameterOptimization; the constructor (incl. `library`) is inherited."""

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
            'lambda_l1':  (0.0, 100.0),
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
                max_depth = Categorical(name='max_depth',items=[1,2,3,4,5,6,7,8,9,10])
                param.add([max_depth])
        if self.try_num_iter:
                n_iter = Integer(name='n_iter', bounds = (1,1000))
                param.add([n_iter])
        runtime_by_configuration = {}

        def objective_opt(params, seed = self.seed):
            
            runtime_start = time.perf_counter()
            try:
                score, best_iter = self._train_model_for_validation(
                    X_train, y_train, X_val, y_val,
                    params.get_dictionary()
                )
            finally:
                runtime_by_configuration.setdefault(_configuration_key(params), []).append(
                    time.perf_counter() - runtime_start
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
        runtimes = _align_config_runtimes(configuration, runtime_by_configuration)
        trials = {}
        for k in range(len(configuration)):
            trials[k] = {'params':dict(configuration[k]), 'score': rh.get_cost(configuration[k])}
        #print('This are the trials',trials)
        df_trials = self._convert_dict_to_df(trials)
        df_trials.insert(df_trials.columns.get_loc('val_score') + 1, 'runtime', runtimes)
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
        df_trials = df_trials[['learning_rate','min_data_in_leaf','max_depth','lambda_l1','lambda_l2','num_leaves','max_bin',	'bagging_fraction','feature_fraction','val_score','runtime','test_score','test_log_loss','test_f1_score','test_rmse',	'current_best_test_score','current_best_test_log_loss',	'current_best_test_f1_score','current_best_test_rmse','try_num_leaves',	'joint_tuning_depth_leaves','try_num_iter']]
        return df_trials

    # Older copies of the base-class versions, kept until reviewed (not used by SMAC; they shadow the base methods).
    def _modify_df_for_tpe(self, df):
        """This function modifies the DataFrame for the TPE method."""
        # Drop unnecessary columns from the DataFrame with the trials
        df.drop(columns=['number', 'datetime_start', 'datetime_complete', 'duration', 'state'], inplace=True)
        if self.try_num_leaves and not(self.try_num_iter):
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate', 'max_bin', 'min_data_in_leaf', 'num_leaves']
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves and not(self.try_num_iter):
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate', 'max_bin','max_depth', 'min_data_in_leaf', 'num_leaves']
        elif self.try_max_depth and not(self.try_num_iter):
            df.columns = ['val_score','bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate', 'max_bin', 'max_depth', 'min_data_in_leaf']
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_max_depth:
            df.columns = ['val_score','bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate', 'max_bin', 'max_depth', 'min_data_in_leaf','n_iter','num_leaves']
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_num_leaves:
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate', 'max_bin', 'min_data_in_leaf', 'n_iter','num_leaves']
            df['max_depth'] = -1
        elif self.try_num_iter and self.joint_tuning_depth_leaves:
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate', 'max_bin','max_depth', 'min_data_in_leaf','n_iter','num_leaves']
        return df
    

    def _convert_array_to_df(self, x_iters, func_vals):
        """This function converts the arrays from the GP-BO trials into a DataFrame."""

        if self.try_num_leaves and not(self.try_num_iter):
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'lambda_l1', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction', 'num_leaves'])
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves and not(self.try_num_iter):
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf','lambda_l1', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction','max_depth', 'num_leaves'])
        elif self.try_max_depth and not(self.try_num_iter):
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'max_depth', 'lambda_l1', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction'])
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_max_depth:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'max_depth', 'lambda_l1', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction','n_iter'])
            df['num_leaves'] = 2**10
        elif self.try_num_iter and self.try_num_leaves:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'lambda_l1', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction', 'num_leaves','n_iter'])
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves and self.try_num_iter:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf','lambda_l1', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction','max_depth', 'num_leaves','n_iter'])
        # Convert the 'score' array into a column for the DataFrame
        df['val_score'] = func_vals

        return df
