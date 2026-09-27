"""test_library_switch.py

Testet, ob eine komplette HPO-Suche (hier: TPE) sowohl mit GPBoost als auch
mit XGBoost durchlaeuft -- gesteuert ueber das neue library-Argument.

Laeuft auf dem kleinen abalone-Datensatz.

Aufruf:  python test_library_switch.py
"""

import os
import numpy as np
import pandas as pd

from methods import ParameterOptimization

suite_id = 335
task_id = 361288   # abalone (klein)

path = f"data/{suite_id}_{task_id}"
X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
categorical_indicator = np.load(
    os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy")
)


def run_one(library):
    print(f"\n=== TPE-Suche mit library='{library}' ===")
    obj = ParameterOptimization(
        X=X.copy(), y=y.copy(), categorical_indicator=categorical_indicator,
        try_max_depth=True, suite_id=suite_id, seed=27225,
        library=library,
    )
    results = obj.run_tpe()
    print(f"  Zeilen: {len(results)}")
    print(f"  bester test_score (R2): {results['test_score'].max():.4f}")
    return results


res_gpb = run_one("gpboost")
res_xgb = run_one("xgboost")

print("\n--- FERTIG ---")
print("Beide Suchen sollten ohne Fehler durchlaufen und einen plausiblen")
print("R2 (fuer abalone grob 0.5 - 0.57) liefern.")
