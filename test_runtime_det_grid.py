"""test_runtime_grid.py: prueft die runtime-Spalte bei Grid/Random (abalone)."""
import os
import numpy as np
import pandas as pd
from methods import ParameterOptimization

suite_id, task_id = 335, 361288
path = f"data/{suite_id}_{task_id}"
X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
cat = np.load(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"))

for name, kw in [("max_depth", dict(try_max_depth=True))]:
    print(f"\n=== Det. Grid, {name} ===")
    obj = ParameterOptimization(X=X.copy(), y=y.copy(), categorical_indicator=cat,
                                suite_id=suite_id, seed=27225, library="gpboost", **kw)
    fold0 = next(iter(obj.splits))
    tr, te = fold0
    res = obj.grid_search_method(
        X_train_full=obj.X.iloc[tr], y_train_full=obj.y.iloc[tr],
        X_test=obj.X.iloc[te], y_test=obj.y.iloc[te],
    )
    print("Zeilen:", len(res))
    print("Alle Spalten:", list(res.columns))
    print("try_num_iter:", res["try_num_iter"].unique())
    print("n_iter vorhanden:", "n_iter" in res.columns)
    print("val_score:", res["val_score"].describe()[["min", "mean", "max"]].round(4).to_dict())
    print(res["runtime"].describe()[["min", "mean", "max"]])