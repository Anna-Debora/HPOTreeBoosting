"""test_classification.py

Testet den CatBoost-Klassifikations-Fix (Wahrscheinlichkeiten statt Klassen)
mit einer TPE-Suche auf eye_movements (Suite 337, Task 361070).

Aufruf:  python test_classification.py
"""

import os
import numpy as np
import pandas as pd

from methods import ParameterOptimization

suite_id = 337
task_id = 361070   # eye_movements (binaer, 7608 Zeilen)

path = f"data/{suite_id}_{task_id}"
X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
categorical_indicator = np.load(
    os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy")
)

print(f"X: {X.shape}, y-Klassen: {sorted(y.iloc[:, 0].unique())}")

obj = ParameterOptimization(
    X=X.copy(), y=y.copy(), categorical_indicator=categorical_indicator,
    try_max_depth=True, suite_id=suite_id, seed=27225,
    library="catboost",
)
results = obj.run_tpe()

print(f"\nZeilen: {len(results)}")
score_cols = [c for c in results.columns if "score" in c or "loss" in c]
print("Score-Spalten:", score_cols)
print(results[score_cols].describe().loc[["min", "mean", "max"]])

print("\n--- FERTIG ---")
print("Erwartet: kein Fehler, Accuracy grob 0.6 - 0.7 (nicht 0.5, nicht nahe 1.0).")