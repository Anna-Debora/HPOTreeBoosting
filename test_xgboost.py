"""test_xgboost.py

Testet den NEUEN XGBoost-Pfad in _train_and_predict direkt.
Ruft die Weiche einmal mit library="gpboost" und einmal mit library="xgboost"
auf denselben Daten auf und vergleicht, ob beide sinnvolle Vorhersagen liefern.

Laeuft auf dem kleinen abalone-Datensatz (Regression).

Aufruf:  python test_xgboost.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from methods import ParameterOptimization

suite_id = 335
task_id = 361288   # abalone (Regression, klein)

path = f"data/{suite_id}_{task_id}"
X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
categorical_indicator = np.load(
    os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy")
)

obj = ParameterOptimization(
    X=X, y=y, categorical_indicator=categorical_indicator,
    try_max_depth=True, suite_id=suite_id, seed=27225,
)

# Einen Fold nehmen und daraus Trainings-/Validierungsdaten bauen
splits = list(obj.splits)
full_train_index, test_index = splits[0]
X_train = obj.X.iloc[full_train_index].reset_index(drop=True)
y_train = obj.y.iloc[full_train_index].reset_index(drop=True)
X_val = obj.X.iloc[test_index].reset_index(drop=True)
y_val = obj.y.iloc[test_index].reset_index(drop=True)

# Ein einfacher, fester Satz Hyperparameter (GPBoost-Namen -- die Weiche uebersetzt fuer XGBoost)
params = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "lambda_l1": 0.0,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
}

train_options = {
    "params": params,
    "num_boost_round": 100,
    "early_stopping_rounds": 20,
    "verbose_eval": False,
}

print("=== Test 1: GPBoost ===")
gpb_params = dict(params)
gpb_params.update({"verbose": -1, "objective": "regression", "metric": "rmse", "num_leaves": 1024})
gpb_options = dict(train_options)
gpb_options["params"] = gpb_params
y_pred_gpb, best_iter_gpb = obj._train_and_predict(
    library="gpboost", X_train=X_train, y_train=y_train,
    X_predict=X_val, X_valid=X_val, y_valid=y_val, train_options=gpb_options,
)
rmse_gpb = root_mean_squared_error(y_val, y_pred_gpb)
print(f"GPBoost  RMSE: {rmse_gpb:.4f}   best_iter: {best_iter_gpb}")

print("\n=== Test 2: XGBoost ===")
xgb_params = dict(params)
xgb_params.update({"objective": "regression", "metric": "rmse"})
xgb_options = dict(train_options)
xgb_options["params"] = xgb_params
y_pred_xgb, best_iter_xgb = obj._train_and_predict(
    library="xgboost", X_train=X_train, y_train=y_train,
    X_predict=X_val, X_valid=X_val, y_valid=y_val, train_options=xgb_options,
)
rmse_xgb = root_mean_squared_error(y_val, y_pred_xgb)
print(f"XGBoost  RMSE: {rmse_xgb:.4f}   best_iter: {best_iter_xgb}")

print("\n--- FERTIG ---")
print("Beide RMSE sollten in einer aehnlichen, sinnvollen Groessenordnung liegen")
print("(fuer abalone grob im Bereich 2.1 - 2.3). Sehr unterschiedliche oder")
print("unsinnige Werte (z.B. 0 oder riesig) wuerden auf ein Problem hindeuten.")
