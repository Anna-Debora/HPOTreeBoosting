"""General utilities used across the project.

Includes helper functions for grid search manipulation, formatting, and
wrappers around GPBoost tuning utilities adapted from Fabio Sigrist's
implementation.
"""

import copy
import time
import numpy as np
import pandas as pd
from pandas import Series as pd_Series
from pandas import DataFrame as pd_DataFrame
from pandas.api.types import is_sparse as is_dtype_sparse
from gpboost import cv, Dataset
import optuna


def is_numeric(obj):
    """Check whether object is a number or not, include numpy number, etc."""
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: invalid literal
        return False
    

def is_1d_list(data):
    """Check whether data is a 1-D list."""
    return isinstance(data, list) and (not data or is_numeric(data[0]))


def _get_bad_pandas_dtypes(dtypes):
    pandas_dtype_mapper = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                           'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                           'uint32': 'int', 'uint64': 'int', 'bool': 'int',
                           'float16': 'float', 'float32': 'float', 'float64': 'float'}
    bad_indices = [i for i, dtype in enumerate(dtypes) if (dtype.name not in pandas_dtype_mapper
                                                           and (not is_dtype_sparse(dtype)
                                                                or dtype.subtype.name not in pandas_dtype_mapper))]
    return bad_indices


def _get_bad_pandas_dtypes_int(dtypes):
    pandas_dtype_mapper = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                           'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                           'uint32': 'int', 'uint64': 'int'}
    bad_indices = [i for i, dtype in enumerate(dtypes) if (dtype.name not in pandas_dtype_mapper
                                                           and (not is_dtype_sparse(dtype)
                                                                or dtype.subtype.name not in pandas_dtype_mapper))]
    return bad_indices


def _format_check_1D_data(data, data_name="data", check_data_type=True, check_must_be_int=False, convert_to_type=None):
    if not isinstance(data, (np.ndarray, pd_Series, pd_DataFrame)) and not is_1d_list(data):
        raise ValueError(
            data_name + " needs to be either 1-D pandas.DataFrame, pandas.Series, numpy.ndarray or a 1-D list")
    if not isinstance(data, list):
        if len(data.shape) != 1 or data.shape[0] < 1:
            raise ValueError(data_name + " needs to be 1 dimensional and it must be non empty ")
    if isinstance(data, pd_DataFrame):
        if check_data_type:
            if check_must_be_int:
                if _get_bad_pandas_dtypes_int([data.dtypes]):
                    raise ValueError(data_name + ': DataFrame.dtypes must be int')
            else:
                if _get_bad_pandas_dtypes([data.dtypes]):
                    raise ValueError(data_name + ': DataFrame.dtypes must be int, float or bool')
        data = np.ravel(data.values)
    elif isinstance(data, pd_Series):
        if check_data_type:
            if check_must_be_int:
                if _get_bad_pandas_dtypes_int([data.dtypes]):
                    raise ValueError(data_name + ': Series.dtypes must be int')
            else:
                if _get_bad_pandas_dtypes([data.dtypes]):
                    raise ValueError(data_name + ': Series.dtypes must be int, float or bool')
        data = data.values
    elif isinstance(data, np.ndarray):
        if check_must_be_int:
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError(data_name + ': must be of integer type')
    elif is_1d_list(data):
        data = np.array(data)
    if convert_to_type is not None:
        if data.dtype != convert_to_type:
            data = data.astype(convert_to_type)
    return data


def _get_grid_size(param_grid):
    """Determine total number of parameter combinations on a grid

    Parameters
    ----------
    param_grid : dict
        Parameter grid

    Returns
    -------
    grid_size : int
        Parameter grid size

    :Authors:
        Fabio Sigrist
    """
    grid_size = 1
    for param in param_grid:
        grid_size = grid_size * len(param_grid[param])
    return (grid_size)


def _get_param_combination(param_comb_number, param_grid):
    """Select parameter combination from a grid of parameters

    Parameters
    ----------
    param_comb_number : int
        Index number of parameter combination on parameter grid that should be returned (counting starts at 0).
    param_grid : dict
        Parameter grid

    Returns
    -------
    param_comb : dict
        Parameter combination

    :Authors:
        Fabio Sigrist
    """
    param_comb = {}
    nk = param_comb_number
    for param in param_grid:
        ind_p = int(nk % len(param_grid[param]))
        param_comb[param] = param_grid[param][ind_p]
        nk = (nk - ind_p) / len(param_grid[param])
    return (param_comb)


def modified_grid_search_tune_parameters(param_grid, train_set, params=None, num_try_random=None,
                                num_boost_round=100, gp_model=None,
                                line_search_step_length=False,
                                use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
                                folds=None, nfold=5, stratified=False, shuffle=True,
                                metric=None, fobj=None, feval=None, init_model=None,
                                feature_name='auto', categorical_feature='auto',
                                early_stopping_rounds=None, fpreproc=None,
                                verbose_eval=1, seed=0, callbacks=None, metrics=None,
                                measure_runtime=False):
    """Function that allows for choosing tuning parameters from a grid in a determinstic or random way using cross validation or validation data sets.

    Parameters
    ----------
    param_grid : dict
        Candidate parameters defining the grid over which a search is done.
        See https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#tuning-parameters--hyperparameters-for-the-tree-boosting-part
    train_set : Dataset
        Data to be trained on.
    params : dict, optional (default=None)
        Other parameters not included in param_grid.
    num_try_random : int, optional (default=None)
        Number of random trial on parameter grid. If none, a deterministic search is done
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    gp_model : GPModel or None, optional (default=None)
        GPModel object for the GPBoost algorithm
    line_search_step_length : bool, optional (default=False)
        If True, a line search is done to find the optimal step length for every boosting update
        (see, e.g., Friedman 2001). This is then multiplied by the 'learning_rate'.
        Applies only to the GPBoost algorithm
    use_gp_model_for_validation : bool, optional (default=True)
        If True, the 'gp_model' (Gaussian process and/or random effects) is also used (in addition to the tree model)
        for calculating predictions on the validation data. If False, the 'gp_model' (random effects part) is ignored
        for making predictions and only the tree ensemble is used for making predictions for calculating the validation / test error.
    train_gp_model_cov_pars : bool, optional (default=True)
        If True, the covariance parameters of the 'gp_model' (Gaussian process and/or random effects) are estimated
        in every boosting iterations, otherwise the 'gp_model' parameters are not estimated. In the latter case, you
        need to either estimate them beforehand or provide values via the 'init_cov_pars' parameter when creating
        the 'gp_model'
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=False)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metric : string, list of strings or None, optional (default=None)
        Evaluation metric to be monitored when doing CV and parameter tuning.
        If not None, the metric in ``params`` will be overridden.
        Non-exhaustive list of supported metrics: "test_neg_log_likelihood", "mse", "rmse", "mae",
        "auc", "average_precision", "binary_logloss", "binary_error"
        See https://gpboost.readthedocs.io/en/latest/Parameters.html#metric-parameters
        for a complete list of valid metrics.
    fobj : callable or None, optional (default=None)
        Customized objective function. Only for independent boosting.
        The GPBoost algorithm currently does not support this.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) for each sample point.

        For binary task, the preds is margin.
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable, list of callable functions or None, optional (default=None)
        Customized evaluation function.
        If more than one evaluation function is provided, only the first evaluation function will be used to choose tuning parameters
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
            train_data : Dataset
                The training dataset.
            eval_name : string
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For binary task, the preds is probability of positive class (or margin in case of specified ``fobj``).
        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metric`` to the string ``"None"``.
    init_model : string, Booster or None, optional (default=None)
        Filename of GPBoost model or Booster instance used for continue training.
    feature_name : list of strings or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of strings or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping.
        CV score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue.
        Requires at least one metric. If there's more than one, will check all of them.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True`` in ``params``.
        Last entry in evaluation history is the one from the best iteration.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : int or None, optional (default=1)
        Whether to display information on the progress of tuning parameter choice.
        If None or 0, verbose is of.
        If = 1, summary progress information is displayed for every parameter combination.
        If >= 2, detailed progress is displayed at every boosting stage for every parameter combination.
    seed : int, optional (default=0)
        Seed used to generate folds and random grid search (passed to numpy.random.seed).
    callbacks : list of callables or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    metrics : string, list of strings or None, discontinued (default=None)
        This is discontinued. Use the renamed equivalent argument 'metric' instead

    Returns
    -------
    return : dict
        Dictionary with the best parameter combination and score
        The dictionary has the following format:
        {'best_params': best_params, 'best_num_boost_round': best_num_boost_round, 'best_score': best_score}

    :Authors:
        Fabio Sigrist
    """
    # if metrics is not None:
    #     raise GPBoostError("The argument 'metrics' is discontinued. "
    #                     "Use the renamed equivalent argument 'metric' instead")
    # Check correct format
    if not isinstance(param_grid, dict):
        raise ValueError("param_grid needs to be a dict")
    
    if verbose_eval is None:
        verbose_eval = 0

    else:
        if not isinstance(verbose_eval, int):
            raise ValueError("verbose_eval needs to be int")
        
    if params is None:
        params = {}

    else:
        params = copy.deepcopy(params)

    param_grid = copy.deepcopy(param_grid)
    for param in param_grid:
        if is_numeric(param_grid[param]):
            param_grid[param] = [param_grid[param]]

        param_grid[param] = _format_check_1D_data(param_grid[param],
                                                  data_name=param, check_data_type=False,
                                                  check_must_be_int=False, convert_to_type=None)
        
    higher_better = False

    if metric is not None:
        if isinstance(metric, str):
            metric = [metric]

        if metric[0].startswith(('auc', 'ndcg@', 'map@', 'average_precision')):
            higher_better = True

    elif feval is not None:
        if callable(feval):
            feval = [feval]

        PH1, PH2, higher_better = feval[0](np.array([0]), Dataset(np.array([0]), np.array([0])))

    # Determine combinations of parameter values that should be tried out
    grid_size = _get_grid_size(param_grid)

    if num_try_random is not None:
        if num_try_random > grid_size:
            raise ValueError("num_try_random is larger than the number of all possible combinations of parameters in param_grid ")
        
        try_param_combs = np.random.RandomState(seed).choice(a=grid_size, size=num_try_random, replace=False)
        if verbose_eval >= 1:
            print("Starting random grid search with " + str(num_try_random) + " trials out of " + str(
                grid_size) + " parameter combinations ")
            
    else:
        try_param_combs = range(grid_size)
        if verbose_eval >= 1:
            print("Starting deterministic grid search with " + str(grid_size) + " parameter combinations ")

    if verbose_eval < 2:
        verbose_eval_cv = False

    else:
        verbose_eval_cv = True

    best_score = 1e99
    current_score = 1e99

    if higher_better:
        best_score = -1e99
        current_score = -1e99

    best_params = {}
    best_num_boost_round = num_boost_round
    counter_num_comb = 1

    all_combinations = {}

    if 'max_bin' in param_grid:
        if train_set.handle is not None:
            raise ValueError("'train_set' cannot be constructed already when 'max_bin' is in 'param_grid' ")
        
        else:
            train_set_not_constructed = copy.deepcopy(train_set)

    for param_comb_number in try_param_combs:
        param_comb = _get_param_combination(param_comb_number=param_comb_number, param_grid=param_grid)

        for param in param_comb:
            params[param] = param_comb[param]

        if verbose_eval >= 1:
            print("Trying parameter combination " + str(counter_num_comb) +
                  " of " + str(len(try_param_combs)) + ": " + str(param_comb))
            
        if 'max_bin' in param_grid:
            train_set = copy.deepcopy(train_set_not_constructed)

        current_score_is_better = False
        current_runtime = None

        try:
            runtime_start = time.perf_counter() if measure_runtime else None
            try:
                cvbst = cv(params=params, train_set=train_set, num_boost_round=num_boost_round, gp_model=gp_model,
                           use_gp_model_for_validation=use_gp_model_for_validation,
                           train_gp_model_cov_pars=train_gp_model_cov_pars,
                           folds=folds, nfold=nfold, stratified=stratified, shuffle=shuffle,
                           metric=metric, fobj=fobj, feval=feval, init_model=init_model,
                           feature_name=feature_name, categorical_feature=categorical_feature,
                           early_stopping_rounds=early_stopping_rounds, fpreproc=fpreproc,
                           verbose_eval=verbose_eval_cv, seed=seed, callbacks=callbacks,
                           eval_train_metric=False, return_cvbooster=False)
            finally:
                if measure_runtime:
                    current_runtime = time.perf_counter() - runtime_start
            
            if higher_better:
                current_score = np.max(cvbst[next(iter(cvbst))])
                if current_score > best_score:
                    current_score_is_better = True

            else:
                current_score = np.min(cvbst[next(iter(cvbst))])
                if current_score < best_score:
                    current_score_is_better = True

        except Exception as err: # Note: this is typically not called anymore since gpv.cv() now already contains a tryCatch statement
            if verbose_eval < 1:
                print("Error for parameter combination " + str(counter_num_comb) +
                      " of " + str(len(try_param_combs)) + ": " + str(param_comb))
                
        all_combinations[param_comb_number] = {'params': param_comb, 'score': current_score}
        if measure_runtime:
            all_combinations[param_comb_number]['runtime'] = current_runtime

        if current_score_is_better:
            best_score = current_score
            best_params = param_comb
            if higher_better:
                best_num_boost_round = np.argmax(cvbst[next(iter(cvbst))]) + 1

            else:
                best_num_boost_round = np.argmin(cvbst[next(iter(cvbst))]) + 1

            if verbose_eval >= 1:
                metric_name = list(cvbst.keys())[0]
                metric_name = metric_name.split('-mean', 1)[0]
                print("***** New best test score ("+metric_name+" = " + str(best_score) +
                      ") found for the following parameter combination:")
                best_params_print = copy.deepcopy(best_params)
                best_params_print['num_boost_round'] = best_num_boost_round
                print(best_params_print)

        counter_num_comb = counter_num_comb + 1

    return {'best_params': best_params, 'best_iter': best_num_boost_round, 'best_score': best_score, 'all_combinations': all_combinations}
def modify(param_grid):
    '''Helper Function used in methods.py'''
    for param in param_grid:
        if is_numeric(param_grid[param]):
            param_grid[param] = [param_grid[param]]

        param_grid[param] = _format_check_1D_data(param_grid[param],
                                                data_name=param, check_data_type=False,
                                                check_must_be_int=False, convert_to_type=None)
        
    
    # Determine combinations of parameter values that should be tried out
    grid_size = _get_grid_size(param_grid)

    return grid_size, param_grid


def tune_pars_TPE_algorithm_optuna(search_space, n_trials, X, y, gp_model = None,
                                max_num_boost_round=1000, early_stopping_rounds=None,
                                metric=None, folds=None, nfold=5,
                                cv_seed=0, tpe_seed=0,
                                params=None, verbose_train=0, verbose_eval=1,
                                use_gp_model_for_validation=True, train_gp_model_cov_pars=True, feval=None,
                                categorical_feature='auto'):
    """Function for choosing tuning parameters using the TPE (Tree-structured Parzen Estimator) algorithm implemented in optuna"""
    if not isinstance(search_space, dict):
        raise ValueError("'search_space' must be a dictionary")
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("'n_trials' must be a positive integer")
    
    if params is None:
        params = {}
    else:
        params = copy.deepcopy(params)
    search_space = copy.deepcopy(search_space)
    metric_higher_better = False
    if metric is not None:
        if isinstance(metric, str):
            metric = [metric]
        if metric[0].startswith(('auc', 'ndcg@', 'map@', 'average_precision')):
            metric_higher_better = True
    elif feval is not None:
        if callable(feval):
            feval = [feval]
        PH1, PH2, metric_higher_better = feval[0](np.array([0]), Dataset(np.array([0]), np.array([0])))
    best_score = -1e99 if metric_higher_better else 1e99
    best_iter = -1
    verbose_eval_cv = verbose_eval >= 2

    def objective_opt(trial):
        nonlocal best_score, best_iter
        """Objective function for tuning parameter search with Optuna."""
        # Parse parameters
        params_loc = {}
        for param in search_space:
            if len(search_space[param]) != 2:
                raise ValueError(f"search_space['{param}'] must have length 2")
            if param in ['learning_rate', 'shrinkage_rate',
                         'min_gain_to_split', 'min_split_gain',
                         'min_sum_hessian_in_leaf', 'min_sum_hessian_per_leaf', 'min_sum_hessian', 'min_hessian', 'min_child_weight']:
                params_loc[param] = trial.suggest_float(param, search_space[param][0], search_space[param][1], log=True)
            elif param in ['lambda_l2', 'reg_lambda', 'lambda',
                           'lambda_l1', 'reg_alpha',
                           'bagging_fraction', 'sub_row', 'subsample', 'bagging',
                           'feature_fraction', 'sub_feature', 'colsample_bytree',
                           'cat_l2',
                           'cat_smooth']:
                params_loc[param] = trial.suggest_float(param, search_space[param][0], search_space[param][1], log=False)
            elif param in ['num_leaves', 'num_leaf', 'max_leaves', 'max_leaf',
                           'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples',
                           'max_bin']:
                params_loc[param] = trial.suggest_int(param, search_space[param][0], search_space[param][1], log=True)
            elif param in ['max_depth']:
                params_loc[param] = trial.suggest_int(param, search_space[param][0], search_space[param][1], log=False)
            elif param in ['line_search_step_length']:
                params_loc[param] = trial.suggest_categorical(param, [search_space[param][0], search_space[param][1]])
            else: 
                raise ValueError(f"Unknown parameter '{param}'")
        params_loc.update({'verbose': verbose_train})
        params_loc.update(params)

        # Train the model
        data_bst = Dataset(data=X, label=y)
        cvbst = cv(params=params_loc, train_set=data_bst, gp_model=gp_model, 
                   use_gp_model_for_validation=use_gp_model_for_validation,  
                   train_gp_model_cov_pars=train_gp_model_cov_pars,
                   num_boost_round=max_num_boost_round, 
                   early_stopping_rounds=early_stopping_rounds,
                   folds=folds, nfold=nfold, verbose_eval=verbose_eval_cv, show_stdv=False, 
                   seed=cv_seed, metric=metric, feval=feval,
                   categorical_feature=categorical_feature)
        if trial.should_prune():
            raise optuna.TrialPruned()
        metric_name = list(cvbst.keys())[0]
        best_score_trial = np.min(cvbst[metric_name])
        best_iter_trial = np.argmin(cvbst[metric_name]) + 1
        if metric_higher_better:
            best_score_trial = np.max(cvbst[metric_name])
            best_iter_trial = np.argmax(cvbst[metric_name]) + 1

        # Save the best number of iterations
        found_better_combination = False
        if metric_higher_better:
            if best_score_trial > best_score:
                found_better_combination = True
        else:
            if best_score_trial < best_score:
                found_better_combination = True
        if found_better_combination:
            best_score = best_score_trial
            best_iter = best_iter_trial

        return best_score_trial
    
    direction = 'maximize' if metric_higher_better else 'minimize'
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=tpe_seed),pruner=optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource=max_num_boost_round, reduction_factor=3
    ))
    study.optimize(objective_opt, n_trials=n_trials)
    return {'best_params': study.best_trial.params, 'best_iter': best_iter, 'best_score': study.best_trial.values}

def truncate(param_config,scores,k):
    '''Function that takes a set of param combinations and their corresponding scores
    and returns the top k performing combinations'''
    sorted_ind = sorted(scores.items(), key = lambda item: item[1])[:k]
    param_config= {param_ind: param_config[param_ind] for param_ind, _ in sorted_ind}
    scores= {param_ind: scores[param_ind] for param_ind, _ in sorted_ind}
    return param_config, scores



def runhistory_to_dataframe(run_history):
    rows = []
    for config in run_history.get_all_configs():
        row = {param.name: param for param in config.values()}
        rows.append(row)
    return pd.DataFrame(rows)