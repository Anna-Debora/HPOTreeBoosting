"""Compare original and Hessian-scaled XGBoost TPE on eye_movements."""
from __future__ import annotations

import contextlib
import io
import os
import traceback

import numpy as np
import optuna
import pandas as pd
from methods import ParameterOptimization


SUITE_ID = 337
TASK_ID = 361070
SEED = 27225
TPE_TRIALS = 30
CONSTANT_THRESHOLD = 0.49

_original_optimize = optuna.study.Study.optimize
_original_train_and_predict = ParameterOptimization._train_and_predict


def _limited_optimize(self, func, *args, **kwargs):
    kwargs["n_trials"] = TPE_TRIALS
    return _original_optimize(self, func, *args, **kwargs)


def _scaled_train_and_predict(
    self, library, X_train, y_train, X_predict, train_options,
    X_valid=None, y_valid=None,
):
    options = dict(train_options)
    params = options.get("params")
    if library == "xgboost" and self.suite_id in (334, 337) and params is not None:
        params = params.copy()
        if "min_data_in_leaf" in params:
            params["min_data_in_leaf"] = 0.25 * params["min_data_in_leaf"]
        options["params"] = params
    return _original_train_and_predict(
        self, library, X_train, y_train, X_predict, options,
        X_valid=X_valid, y_valid=y_valid,
    )


def _load_first_fold():
    data_path = f"data/{SUITE_ID}_{TASK_ID}"
    X = pd.read_csv(os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_X.csv"))
    y = pd.read_csv(os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_y.csv"))
    categorical_indicator = np.load(
        os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_categorical_indicator.npy")
    )
    obj = ParameterOptimization(
        X=X,
        y=y,
        categorical_indicator=categorical_indicator,
        suite_id=SUITE_ID,
        seed=SEED,
        library="xgboost",
        try_max_depth=True,
        try_num_leaves=False,
        joint_tuning_depth_leaves=False,
        try_num_iter=False,
    )
    train_indices, test_indices = next(iter(obj.splits))
    return obj, train_indices, test_indices


def _run_tpe(obj, train_indices, test_indices):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return obj.tpe_method(
            X_train_full=obj.X.iloc[train_indices],
            y_train_full=obj.y.iloc[train_indices],
            X_test=obj.X.iloc[test_indices],
            y_test=obj.y.iloc[test_indices],
        )


def _summarize(label, result):
    val_score = pd.to_numeric(result["val_score"], errors="coerce")
    test_score = pd.to_numeric(result["test_score"], errors="coerce")
    constant_mask = val_score >= CONSTANT_THRESHOLD
    constant_values = pd.to_numeric(
        result.loc[constant_mask, "min_data_in_leaf"], errors="coerce"
    )
    best_validation_index = val_score.idxmin()
    print(f"{label}: Trials={len(result)}")
    print(
        f"  val_score >= {CONSTANT_THRESHOLD}: "
        f"{int(constant_mask.sum())}/{len(result)} "
        f"({constant_mask.mean():.1%}); min_data_in_leaf range="
        f"{constant_values.min() if not constant_values.empty else 'n/a'}.."
        f"{constant_values.max() if not constant_values.empty else 'n/a'}"
    )
    print(
        f"  best val_score={val_score.min():.6g}; "
        f"test_score at best-val trial={test_score.loc[best_validation_index]:.6g}; "
        f"best test_score overall={test_score.max():.6g}"
    )


def main():
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    optuna.study.Study.optimize = _limited_optimize
    obj, train_indices, test_indices = _load_first_fold()

    try:
        original_result = _run_tpe(obj, train_indices, test_indices)
        ParameterOptimization._train_and_predict = _scaled_train_and_predict
        patched_result = _run_tpe(obj, train_indices, test_indices)
    except Exception:
        lines = traceback.format_exc().strip().splitlines()
        print("ERROR:", " | ".join(lines[-8:]))
        return
    finally:
        ParameterOptimization._train_and_predict = _original_train_and_predict

    print(f"eye_movements | suite={SUITE_ID} | task={TASK_ID} | first fold | seed={SEED}")
    _summarize("Original min_child_weight", original_result)
    _summarize("Scaled classification min_child_weight", patched_result)


if __name__ == "__main__":
    main()
