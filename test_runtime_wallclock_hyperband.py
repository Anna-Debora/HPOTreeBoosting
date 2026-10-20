"""Test: Hyperband-Laufzeit ist Wall-Clock des ganzen Brackets (nicht nur Training).

Synthetische Daten, kleines R. Aufruf: python test_runtime_wallclock_hyperband.py
"""
import time

import numpy as np
import pandas as pd

from methods import ParameterOptimization

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
y = pd.DataFrame({"y": X["a"] + rng.normal(scale=0.2, size=300)})
obj = ParameterOptimization(X=X, y=y, categorical_indicator=np.zeros(4, dtype=bool), suite_id=335,
                            seed=1, library="xgboost", try_max_depth=True)

train_seconds = 0.0
orig = obj._train_model_for_validation


def timed(*a, **k):
    global train_seconds
    t0 = time.perf_counter()
    try:
        return orig(*a, **k)
    finally:
        train_seconds += time.perf_counter() - t0


obj._train_model_for_validation = timed
full_train, test = next(iter(obj.splits))
t0 = time.perf_counter()
res = obj.hyperband_method(obj.X.iloc[full_train], obj.y.iloc[full_train], obj.X.iloc[test], obj.y.iloc[test], R=30)
total = time.perf_counter() - t0
rt = res["runtime"].sum()
print(f"Summe runtime (CSV): {rt:.3f}s | nur Training: {train_seconds:.3f}s | Gesamtdauer Methode: {total:.3f}s")
assert rt >= train_seconds * 0.99
assert rt <= total + 1e-6
print("ALL OK")
