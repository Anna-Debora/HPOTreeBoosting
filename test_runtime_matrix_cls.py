"""Classification runtime and score checks for the benchmark matrix."""
from __future__ import annotations

import contextlib
import io
import os
import traceback

import methods
import numpy as np
import optuna
import pandas as pd
from methods import ParameterOptimization


SUITE_ID = 337
TASK_ID = 361070
SEED = 27225
RANDOM_TRIALS = 10
TPE_TRIALS = 30
GP_BO_TRIALS = 12
LIBRARIES = ("gpboost", "xgboost", "catboost")
STRATEGIES = {
    "max_depth": {"try_max_depth": True},
    "num_leaves": {"try_num_leaves": True, "try_max_depth": False},
    "joint": {"joint_tuning_depth_leaves": True, "try_max_depth": False},
    "num_iter": {
        "try_num_iter": True,
        "try_max_depth": True,
        "try_num_leaves": False,
        "joint_tuning_depth_leaves": False,
    },
}

_original_optimize = optuna.study.Study.optimize
_original_gp_minimize = methods.gp_minimize
_original_grid_search = methods.modified_grid_search_tune_parameters


def _limited_optimize(self, func, *args, **kwargs):
    kwargs["n_trials"] = TPE_TRIALS
    return _original_optimize(self, func, *args, **kwargs)


def _limited_gp_minimize(func, dimensions, *args, **kwargs):
    kwargs["n_calls"] = GP_BO_TRIALS
    return _original_gp_minimize(func, dimensions, *args, **kwargs)


def _reduced_deterministic_grid(**kwargs):
    kwargs["param_grid"] = {
        "learning_rate": [0.1],
        "min_data_in_leaf": [10],
        "max_depth": [1, 2],
        "lambda_l2": [0],
        "bagging_fraction": [1],
    }
    return _original_grid_search(**kwargs)


optuna.study.Study.optimize = _limited_optimize
methods.gp_minimize = _limited_gp_minimize


def _trial_count(method_name: str) -> int:
    return {
        "Random": RANDOM_TRIALS,
        "TPE": TPE_TRIALS,
        "GP-BO": GP_BO_TRIALS,
        "Grid (det.)": 2,
    }[method_name]


def _format_number(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{value:.6g}"


def _score_range(result, column):
    values = pd.to_numeric(result.get(column, pd.Series(dtype=float)), errors="coerce")
    if values.empty:
        return "FEHLT"
    return f"{_format_number(values.min())} / {_format_number(values.max())}"


def _run_case(X, y, categorical_indicator, library, strategy, flags, method_name):
    obj = ParameterOptimization(
        X=X.copy(),
        y=y.copy(),
        categorical_indicator=categorical_indicator.copy(),
        suite_id=SUITE_ID,
        seed=SEED,
        library=library,
        **flags,
    )
    train_indices, test_indices = next(iter(obj.splits))
    X_train_full = obj.X.iloc[train_indices]
    y_train_full = obj.y.iloc[train_indices]
    X_test = obj.X.iloc[test_indices]
    y_test = obj.y.iloc[test_indices]

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if method_name in ("Random", "Grid (det.)"):
            result = obj.grid_search_method(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
                num_try_random=(RANDOM_TRIALS if method_name == "Random" else None),
            )
        elif method_name == "TPE":
            result = obj.tpe_method(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
            )
        else:
            result = obj.gp_bo_method(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
            )
    return result


def main():
    data_path = f"data/{SUITE_ID}_{TASK_ID}"
    X = pd.read_csv(os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_X.csv"))
    y = pd.read_csv(os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_y.csv"))
    categorical_indicator = np.load(
        os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_categorical_indicator.npy")
    )

    cases = [
        (library, strategy, method_name)
        for library in LIBRARIES
        for strategy in STRATEGIES
        for method_name in ("Random", "TPE", "GP-BO")
    ]
    cases.append(("gpboost", "max_depth", "Grid (det.)"))

    rows = []
    anomalies = []
    for library, strategy, method_name in cases:
        flags = STRATEGIES[strategy]
        expected = _trial_count(method_name)
        try:
            if method_name == "Grid (det.)":
                methods.modified_grid_search_tune_parameters = _reduced_deterministic_grid
            else:
                methods.modified_grid_search_tune_parameters = _original_grid_search
            result = _run_case(
                X, y, categorical_indicator, library, strategy, flags, method_name
            )
            row_count = len(result)
            columns = list(result.columns)
            runtime_after_score = (
                "ja"
                if "val_score" in columns
                and "runtime" in columns
                and columns.index("runtime") == columns.index("val_score") + 1
                else "nein"
            )
            runtimes = pd.to_numeric(result.get("runtime", pd.Series(dtype=float)), errors="coerce")
            runtime_mean = runtimes.mean() if not runtimes.empty else np.nan
            log_loss = pd.to_numeric(result.get("test_log_loss", pd.Series(dtype=float)), errors="coerce")
            test_score = pd.to_numeric(result.get("test_score", pd.Series(dtype=float)), errors="coerce")
            test_f1 = pd.to_numeric(result.get("test_f1_score", pd.Series(dtype=float)), errors="coerce")
            case_warnings = []

            if row_count != expected:
                case_warnings.append(f"Zeilen {row_count}, erwartet {expected}")
            if runtime_after_score == "nein":
                case_warnings.append("runtime nicht direkt nach val_score")
            if runtimes.isna().any():
                case_warnings.append("NaN in runtime")
            if pd.notna(runtime_mean) and runtime_mean < 0.001:
                case_warnings.append("runtime-Mittel < 0.001")
            if test_score.empty or test_f1.empty or log_loss.empty:
                case_warnings.append("Klassifikationsmetriken fehlen")
            if not test_score.empty:
                if (test_score > 0.95).any():
                    case_warnings.append("test_score > 0.95")
                if (test_score < 0.45).any():
                    case_warnings.append("test_score < 0.45")
            if not log_loss.empty:
                if (~np.isfinite(log_loss)).any():
                    case_warnings.append("test_log_loss NaN/inf")
                if (log_loss > 0.7).any():
                    case_warnings.append("test_log_loss > 0.7")

            catboost_probability = "-"
            if library == "catboost":
                catboost_probability = (
                    "Wahrscheinlichkeitspfad / Log-Loss endlich"
                    if not log_loss.empty and np.isfinite(log_loss).all()
                    else "Wahrscheinlichkeitspfad / Log-Loss nicht endlich"
                )

            if case_warnings:
                anomalies.append(
                    f"{library} | {strategy} | {method_name}: " + "; ".join(case_warnings)
                )
            rows.append(
                {
                    "library": library,
                    "strategie": strategy,
                    "methode": method_name,
                    "Zeilen": row_count,
                    "runtime direkt nach val_score?": runtime_after_score,
                    "val_score min/max": _score_range(result, "val_score"),
                    "runtime mean": _format_number(runtime_mean),
                    "test_score min/max": _score_range(result, "test_score"),
                    "test_log_loss min/max": _score_range(result, "test_log_loss"),
                    "test_f1_score min/max": _score_range(result, "test_f1_score"),
                    "CatBoost-Probas / Log-Loss": catboost_probability,
                    "Auffälligkeit": "; ".join(case_warnings) or "-",
                }
            )
        except Exception:
            trace = traceback.format_exc().strip().splitlines()
            tail = " | ".join(trace[-8:])
            anomalies.append(f"{library} | {strategy} | {method_name}: FEHLER; {tail}")
            rows.append(
                {
                    "library": library,
                    "strategie": strategy,
                    "methode": method_name,
                    "Zeilen": "FEHLER",
                    "runtime direkt nach val_score?": "nein",
                    "val_score min/max": "-",
                    "runtime mean": "-",
                    "test_score min/max": "-",
                    "test_log_loss min/max": "-",
                    "test_f1_score min/max": "-",
                    "CatBoost-Probas / Log-Loss": "-",
                    "Auffälligkeit": "Fehler; Traceback-Ende siehe Liste",
                }
            )

    print(f"DATENSATZ: eye_movements | Suite {SUITE_ID} | Task {TASK_ID} | {len(X)} Zeilen")
    print("TABELLE")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nAUFFÄLLIGKEITEN")
    if anomalies:
        for item in anomalies:
            print(f"- {item}")
    else:
        print("- Keine nach den definierten Klassifikationsschwellen.")
    print("\nCATBOOST-HINWEIS")
    print("- Methodenpfad fordert prediction_type='Probability'; Endlichkeit wird pro Ergebnis anhand test_log_loss geprüft.")


if __name__ == "__main__":
    main()
