"""Test: CatBoost bekommt Kategorien nativ (kein One-Hot), GPBoost/XGBoost weiter One-Hot.

Synthetische Daten mit 3 kategorialen Spalten (eine mit fehlenden Werten, eine mit 200 Stufen).
Aufruf: python test_catboost_categorical.py
"""
import numpy as np
import pandas as pd

from methods import ParameterOptimization

rng = np.random.default_rng(0)
n = 600
X = pd.DataFrame({
    "num1": rng.normal(size=n),
    "num2": rng.normal(size=n),
    "cat_small": rng.choice(["a", "b", "c"], n),
    "cat_missing": rng.choice(["x", "y", None], n),
    "cat_many": rng.integers(0, 200, n).astype(str),
})
signal = (X["cat_small"] == "a") * 1.5 + (X["cat_missing"] == "x") * 1.0 + X["num1"]
y = pd.DataFrame({"y": signal + rng.normal(scale=0.3, size=n)})
indicator = np.array([False, False, True, True, True])


def build(library, suite_id=335, target=y):
    return ParameterOptimization(
        X=X.copy(), y=target.copy(), categorical_indicator=indicator, try_max_depth=True,
        suite_id=suite_id, seed=1, library=library,
    )


cat = build("catboost")
gpb = build("gpboost")
assert list(cat.X.columns) == list(X.columns), cat.X.columns.tolist()
assert cat.cat_feature_names == ["cat_small", "cat_missing", "cat_many"]
assert cat.X["cat_missing"].isin(["x", "y", "__missing__"]).all()
print(cat.X.shape, gpb.X.shape)
assert cat.X.shape[1] == 5 and gpb.X.shape[1] > 100
assert gpb.cat_feature_names == []
print("Spalten: catboost", cat.X.shape[1], "| gpboost (One-Hot)", gpb.X.shape[1])

for name, obj in (("catboost", cat), ("gpboost", gpb)):
    full_train, test = next(iter(obj.splits))
    pred, _ = obj._train_and_predict(
        library=name, X_train=obj.X.iloc[full_train], y_train=obj.y.iloc[full_train],
        X_predict=obj.X.iloc[test], X_valid=None, y_valid=None,
        train_options={"params": {"learning_rate": 0.1, "max_depth": 3, "min_data_in_leaf": 5},
                       "num_boost_round": 50, "verbose_eval": False},
    ) if name == "catboost" else (None, None)
    if name == "catboost":
        r2 = 1 - ((obj.y.iloc[test].values - pred) ** 2).sum() / ((obj.y.iloc[test].values - obj.y.iloc[test].values.mean()) ** 2).sum()
        print(f"catboost R2 (nativ): {r2:.3f}")
        assert r2 > 0.5

# Klassifikation (Suite 337): Probability-Pfad mit Kategorien
yc = pd.DataFrame({"y": (signal > signal.median()).astype(int)})
cc = build("catboost", suite_id=337, target=yc)
full_train, test = next(iter(cc.splits))
pred, _ = cc._train_and_predict(
    library="catboost", X_train=cc.X.iloc[full_train], y_train=cc.y.iloc[full_train],
    X_predict=cc.X.iloc[test], X_valid=None, y_valid=None,
    train_options={"params": {"learning_rate": 0.1, "max_depth": 3}, "num_boost_round": 50, "verbose_eval": False},
)
acc = ((pred > 0.5) == cc.y.iloc[test].values).mean()
print(f"catboost Accuracy (nativ): {acc:.3f}")
assert acc > 0.7

# Validierungspfad (X_valid gesetzt)
tr, va = full_train[:350], full_train[350:]
pred, _ = cc._train_and_predict(
    library="catboost", X_train=cc.X.iloc[tr], y_train=cc.y.iloc[tr], X_predict=cc.X.iloc[va],
    X_valid=cc.X.iloc[va], y_valid=cc.y.iloc[va],
    train_options={"params": {"learning_rate": 0.1, "max_depth": 3}, "num_boost_round": 50,
                   "early_stopping_rounds": 10, "verbose_eval": False},
)
assert len(pred) == len(va)
print("ALL OK")
