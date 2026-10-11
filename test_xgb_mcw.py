"""Compare XGBoost min_child_weight behavior on classification and regression."""
from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from methods import ParameterOptimization


SEED = 27225
MIN_CHILD_WEIGHTS = (1, 5, 20, 50, 100, 300, 1000)
TASKS = (
    ("eye_movements", 337, 361070),
    ("abalone", 335, 361288),
)


def _load_project_fold(dataset_name: str, suite_id: int, task_id: int):
    data_path = f"data/{suite_id}_{task_id}"
    X = pd.read_csv(os.path.join(data_path, f"{suite_id}_{task_id}_X.csv"))
    y = pd.read_csv(os.path.join(data_path, f"{suite_id}_{task_id}_y.csv"))
    categorical_indicator = np.load(
        os.path.join(data_path, f"{suite_id}_{task_id}_categorical_indicator.npy")
    )

    obj = ParameterOptimization(
        X=X,
        y=y,
        categorical_indicator=categorical_indicator,
        suite_id=suite_id,
        seed=SEED,
        library="xgboost",
        try_max_depth=True,
    )
    full_train_indices, test_indices = next(iter(obj.splits))
    X_full_train = obj.X.iloc[full_train_indices]
    y_full_train = obj.y.iloc[full_train_indices]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full_train,
        y_full_train,
        test_size=obj.val_size,
        random_state=obj.fixed_seeds[2],
    )
    return dataset_name, suite_id in (334, 337), X_train, X_valid, y_train, y_valid


def _run_dataset(dataset_name, is_classification, X_train, X_valid, y_train, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    objective = "binary:logistic" if is_classification else "reg:squarederror"
    eval_metric = "error" if is_classification else "rmse"
    max_hessian_sum = 0.25 * len(y_train) if is_classification else float("nan")

    rows = []
    failures = []
    for min_child_weight in MIN_CHILD_WEIGHTS:
        try:
            model = xgb.train(
                {
                    "objective": objective,
                    "eval_metric": eval_metric,
                    "tree_method": "hist",
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "min_child_weight": min_child_weight,
                    "seed": SEED,
                    "verbosity": 0,
                },
                dtrain,
                num_boost_round=1000,
                evals=[(dvalid, "validation")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )
            predictions = model.predict(dvalid)
            constant_prediction = np.unique(predictions).size == 1
            if is_classification:
                predicted_labels = (predictions >= 0.5).astype(int)
                validation_error = 1 - accuracy_score(y_valid, predicted_labels)
                metric_label = "1-Accuracy"
            else:
                validation_error = root_mean_squared_error(y_valid, predictions)
                metric_label = "RMSE"

            rows.append(
                {
                    "Datensatz": dataset_name,
                    "min_child_weight": min_child_weight,
                    "Train-Zeilen": len(y_train),
                    "max. Hessian-Summe": (
                        f"{max_hessian_sum:.2f}" if is_classification else "n/a"
                    ),
                    "Validierungsmetrik": metric_label,
                    "Validierungsfehler": validation_error,
                    "Bäume": model.num_boosted_rounds(),
                    "Konstant vorhergesagt?": "ja" if constant_prediction else "nein",
                }
            )
        except Exception:
            trace = traceback.format_exc().strip().splitlines()
            failures.append(
                f"{dataset_name}, min_child_weight={min_child_weight}: "
                + " | ".join(trace[-8:])
            )

    return rows, failures


def main():
    all_rows = []
    all_failures = []
    for task in TASKS:
        dataset = _load_project_fold(*task)
        rows, failures = _run_dataset(*dataset)
        all_rows.extend(rows)
        all_failures.extend(failures)

    print("ERGEBNISSE")
    if all_rows:
        print(pd.DataFrame(all_rows).to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nFEHLER")
    if all_failures:
        for failure in all_failures:
            print(f"- {failure}")
    else:
        print("- Keine.")


if __name__ == "__main__":
    main()
