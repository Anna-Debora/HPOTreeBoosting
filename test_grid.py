"""test_grid.py

Schneller Test NUR der Grid-Search-Methode, um die lambda_l1-Aenderung zu pruefen.
Laedt den schon heruntergeladenen Datensatz und laesst nur run_random_search laufen
(das ist die Grid-Methode mit num_try_random=135).

Aufruf:  python test_grid.py
"""

import os
import numpy as np
import pandas as pd

from methods import ParameterOptimization

suite_id = 335
task_id = 361102

path = f"data/{suite_id}_{task_id}"
X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
categorical_indicator = np.load(
    os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy")
)

print("Starte Grid-Search-Test mit lambda_l1 ...")

obj = ParameterOptimization(
    X=X, y=y, categorical_indicator=categorical_indicator,
    try_max_depth=True, try_num_leaves=False,
    joint_tuning_depth_leaves=False, try_num_iter=False,
    suite_id=suite_id, seed=27225,
)

results = obj.run_random_search()

print("\n--- FERTIG ---")
print("Spalten der Ergebnis-Tabelle:")
print(list(results.columns))
print(f"\nAnzahl Zeilen: {len(results)}")
print("\nErste paar Zeilen (nur lambda-Spalten):")
lambda_cols = [c for c in results.columns if "lambda" in c]
print(results[lambda_cols].head())
