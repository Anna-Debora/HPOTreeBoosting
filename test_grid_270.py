"""test_grid_270.py

Testet die DETERMINISTISCHE Grid-Suche nach der 270-Aenderung.
Prueft zwei Dinge:
  1. Kommen 270 Konfigurationen pro Fold heraus (statt 135)?
  2. Variiert bagging_fraction wirklich (Werte 0.5 und 1)?

Laeuft auf dem kleinen abalone-Datensatz, damit es schnell geht.

Aufruf:  python test_grid_270.py
"""

import os
import numpy as np
import pandas as pd

from methods import ParameterOptimization

suite_id = 335
task_id = 361288   # abalone (klein, schnell)

path = f"data/{suite_id}_{task_id}"
X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
categorical_indicator = np.load(
    os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy")
)

print("Starte Test der deterministischen Grid-Suche (Ziel: 270 pro Fold) ...")

obj = ParameterOptimization(
    X=X, y=y, categorical_indicator=categorical_indicator,
    try_max_depth=True, try_num_leaves=False,
    joint_tuning_depth_leaves=False, try_num_iter=False,
    suite_id=suite_id, seed=27225,
)

# Nur EINEN Fold, damit es schnell geht: wir rufen grid_search_method direkt auf.
splits = list(obj.splits)
full_train_index, test_index = splits[0]
X_train_full = obj.X.iloc[full_train_index]
X_test = obj.X.iloc[test_index]
y_train_full = obj.y.iloc[full_train_index]
y_test = obj.y.iloc[test_index]

# num_try_random NICHT setzen -> deterministische Grid-Suche
results = obj.grid_search_method(
    X_train_full=X_train_full, y_train_full=y_train_full,
    X_test=X_test, y_test=y_test,
)

print("\n--- FERTIG ---")
print(f"Anzahl Konfigurationen (ein Fold): {len(results)}")
print("   -> Erwartet: 270")

print("\nWelche bagging_fraction-Werte kommen vor?")
print(sorted(results['bagging_fraction'].unique()))
print("   -> Erwartet: [0.5, 1.0]")

print("\nWelche lambda_l1-Werte kommen vor?")
print(sorted(results['lambda_l1'].unique()))
