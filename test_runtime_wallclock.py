"""Test: GP-BO-Laufzeit ist Wall-Clock inklusive Surrogat.

Synthetische Daten, GP-BO auf 15 Aufrufe gekuerzt (nur in diesem Prozess).
Aufruf: python test_runtime_wallclock.py
"""
import time

import numpy as np
import pandas as pd

import methods
from methods import ParameterOptimization

N_CALLS = 15
_orig = methods.gp_minimize
methods.gp_minimize = lambda f, dims, **kw: _orig(f, dims, **{**kw, "n_calls": N_CALLS})

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
y = pd.DataFrame({"y": X["a"] + rng.normal(scale=0.2, size=400)})
obj = ParameterOptimization(X=X, y=y, categorical_indicator=np.zeros(4, dtype=bool), suite_id=335,
                            seed=1, library="xgboost", try_max_depth=True)

start = time.perf_counter()
res = obj.run_gp_bo()
total_wall = time.perf_counter() - start

assert len(res) == 5 * N_CALLS, len(res)
# Ein Fold = die letzten N_CALLS Zeilen; nur dort ist die Trainingszeit noch im Objekt gespeichert
runtime = res["runtime"].to_numpy()[-N_CALLS:]
objective = np.array(obj._gp_objective_runtimes)
assert len(objective) == N_CALLS, len(objective)
assert (runtime >= objective - 1e-9).all()
surrogate = runtime - objective
print(f"Letzter Fold: Wall-Clock (CSV) {runtime.sum():.2f}s | nur Training {objective.sum():.2f}s | Surrogat-Anteil {surrogate.sum():.2f}s")
all_runtime = res["runtime"].sum()
print(f"Alle Folds: Summe runtime {all_runtime:.2f}s | gesamte Methodendauer inkl. Test-Refits {total_wall:.2f}s")
assert all_runtime <= total_wall + 1e-6
assert surrogate.sum() > 0
print("ALL OK")
