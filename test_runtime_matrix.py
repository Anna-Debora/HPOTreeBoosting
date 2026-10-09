"""Runtime-column checks across supported libraries and tuning strategies."""
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


SUITE_ID = 335
TASK_ID = 361288
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


def _limited_optimize(self, func, *args, **kwargs):
    kwargs["n_trials"] = TPE_TRIALS
    return _original_optimize(self, func, *args, **kwargs)


def _limited_gp_minimize(func, dimensions, *args, **kwargs):
    kwargs["n_calls"] = GP_BO_TRIALS
    return _original_gp_minimize(func, dimensions, *args, **kwargs)


optuna.study.Study.optimize = _limited_optimize
methods.gp_minimize = _limited_gp_minimize


def _trial_count(method_name: str) -> int:
    return {
        "Random": RANDOM_TRIALS,
        "TPE": TPE_TRIALS,
        "GP-BO": GP_BO_TRIALS,
        "Grid (det.)": 270,
    }[method_name]


def _format_number(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{value:.6g}"


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
        (library, "num_iter", method_name)
        for library in LIBRARIES
        for method_name in ("Random", "TPE", "GP-BO")
    ]

    rows = []
    anomalies = []
    for library, strategy, method_name in cases:
        flags = STRATEGIES[strategy]
        expected = _trial_count(method_name)
        try:
            result = _run_case(
                X, y, categorical_indicator, library, strategy, flags, method_name
            )
            row_count = len(result)
            columns = list(result.columns)
            has_n_iter = "n_iter" in columns
            has_max_depth = "max_depth" in columns
            runtime_after_score = (
                "ja"
                if "val_score" in columns
                and "runtime" in columns
                and columns.index("runtime") == columns.index("val_score") + 1
                else "nein"
            )
            values = pd.to_numeric(result.get("val_score", pd.Series(dtype=float)), errors="coerce")
            runtimes = pd.to_numeric(result.get("runtime", pd.Series(dtype=float)), errors="coerce")
            score_min = values.min() if not values.empty else np.nan
            score_max = values.max() if not values.empty else np.nan
            runtime_mean = runtimes.mean() if not runtimes.empty else np.nan
            case_warnings = []

            if values.eq(1e99).any():
                case_warnings.append("val_score == 1e+99")
            if pd.notna(runtime_mean) and runtime_mean < 0.001:
                case_warnings.append("runtime-Mittel < 0.001")
            if runtimes.isna().any():
                case_warnings.append("NaN in runtime")
            if row_count != expected:
                case_warnings.append(f"Zeilen {row_count}, erwartet {expected}")
            if runtime_after_score == "nein":
                case_warnings.append("runtime nicht direkt nach val_score")
            if not has_n_iter:
                case_warnings.append("Spalte n_iter fehlt")
            if not has_max_depth:
                case_warnings.append("Spalte max_depth fehlt")

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
                    "n_iter vorhanden?": "ja" if has_n_iter else "nein",
                    "max_depth vorhanden?": "ja" if has_max_depth else "nein",
                    "val_score min/max": f"{_format_number(score_min)} / {_format_number(score_max)}",
                    "runtime mean": _format_number(runtime_mean),
                    "Auffälligkeit": "; ".join(case_warnings) or "-",
                }
            )
        except Exception:
            error_lines = traceback.format_exc().strip().splitlines()[-6:]
            summary = " | ".join(error_lines)
            anomalies.append(f"{library} | {strategy} | {method_name}: FEHLER; {summary}")
            rows.append(
                {
                    "library": library,
                    "strategie": strategy,
                    "methode": method_name,
                    "Zeilen": "FEHLER",
                    "runtime direkt nach val_score?": "nein",
                    "n_iter vorhanden?": "nein",
                    "max_depth vorhanden?": "nein",
                    "val_score min/max": "-",
                    "runtime mean": "-",
                    "Auffälligkeit": "Fehler; Traceback-Ende siehe Auffälligkeiten",
                }
            )

    table = pd.DataFrame(rows)
    print("TABELLE")
    print(table.to_string(index=False))
    print("\nAUSGELASSEN")
    print(
        "- try_num_iter kombiniert mit try_max_depth, try_num_leaves oder "
        "joint_tuning_depth_leaves: nicht als eigene Strategie getestet."
    )
    print(
        "- Mehr als eine der Optionen try_max_depth, try_num_leaves und "
        "joint_tuning_depth_leaves zugleich: __init__ weist dies zurück."
    )
    print("- Nur num_iter-Fälle ausgeführt; andere Strategien und Grid ausgelassen.")
    print("\nAUFFÄLLIGKEITEN")
    if anomalies:
        for anomaly in anomalies:
            print(f"- {anomaly}")
    else:
        print("- Keine Warnungen aus den definierten Prüfungen.")
if __name__ == "__main__":
    main()
