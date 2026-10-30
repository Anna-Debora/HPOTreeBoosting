"""Test: alle Bibliotheken behandeln Kategorien nativ (kein One-Hot mehr, auch nicht fuer GPBoost/XGBoost).

GPBoost und XGBoost bekommen kategoriale Spalten als pandas-Typ "category" (GPBoost erkennt sie ueber
categorical_feature='auto', XGBoost ueber enable_categorical=True), CatBoost als Strings ueber cat_features.
Geprueft wird:
  1. synthetische Daten (Regression und Klassifikation; fehlende Werte, 200 Stufen, eine Stufe nur im Test):
     Spaltenzahl bleibt gleich, richtiger Typ, die Modelle lernen das Signal aus den Kategorien,
     ein kurzer Zufallssuche-Lauf (grid_search_method) laeuft durch
  2. echte Daten mit Kategorien: compas-two-years (334, 361286, Klassifikation) und abalone
     (335, 361288, Regression), jeweils mit einer kuenstlichen Stufe, die nur im Testteil vorkommt
     (uebersprungen, falls data/ fehlt)
Aufruf: python test_native_categorical.py
"""
import os

import numpy as np
import pandas as pd

from methods import ParameterOptimization

LIBRARIES = ("gpboost", "xgboost", "catboost")
OPTIONS = {"params": {"learning_rate": 0.1, "max_depth": 3, "min_data_in_leaf": 5},
           "num_boost_round": 80, "verbose_eval": False}

rng = np.random.default_rng(0)
n = 800
X = pd.DataFrame({
    "num1": rng.normal(size=n),
    "num2": rng.normal(size=n),
    "cat_small": rng.choice(["a", "b", "c"], n),
    "cat_missing": rng.choice(["x", "y", None], n),
    "cat_many": rng.integers(0, 200, n).astype(str),
    "cat_numeric": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], n),   # Kategorien als Zahlen (wie oft in BCCO/OpenML)
})
signal = ((X["cat_small"] == "a") * 2.0 + (X["cat_missing"] == "x") * 1.0
          + X["cat_numeric"].isin([2.0, 5.0]) * 1.5 + 0.5 * X["num1"])
y_reg = pd.DataFrame({"y": signal + rng.normal(scale=0.3, size=n)})
y_cls = pd.DataFrame({"y": (signal > signal.median()).astype(int)})
indicator = np.array([False, False, True, True, True, True])
cat_columns = ["cat_small", "cat_missing", "cat_many", "cat_numeric"]


def build(library, X_in, y_in, ind, suite_id):
    return ParameterOptimization(X=X_in.copy(), y=y_in.copy(), categorical_indicator=ind, try_max_depth=True,
                                 suite_id=suite_id, seed=1, library=library)


def first_fold(obj):
    full_train, test = next(iter(obj.splits))
    return full_train, test


def inject_test_only_level(obj, test, column, level="NUR_IM_TEST"):
    """Setzt in 10 Testzeilen eine Stufe, die in keiner Trainingszeile vorkommt."""
    rows = test[:10]
    if obj.library == "catboost":
        obj.X.loc[rows, column] = level
    else:
        obj.X[column] = obj.X[column].cat.add_categories([level])
        obj.X.loc[rows, column] = level
    return rows


def fit_predict(obj, train, predict, options=OPTIONS):
    options = dict(options, params={**obj.other_params, **options["params"]})   # Zielfunktion wie im echten Lauf
    pred, _ = obj._train_and_predict(library=obj.library, X_train=obj.X.iloc[train], y_train=obj.y.iloc[train],
                                     X_predict=obj.X.iloc[predict], train_options=options)
    return np.asarray(pred)


def r2(y_true, pred):
    y_true = np.asarray(y_true)
    return 1 - ((y_true - pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


# 1. synthetische Daten
for library in LIBRARIES:
    obj = build(library, X, y_reg, indicator, suite_id=335)
    assert list(obj.X.columns) == list(X.columns), (library, obj.X.columns.tolist())   # kein One-Hot
    assert obj.cat_feature_names == cat_columns, obj.cat_feature_names
    for column in cat_columns:
        if library == "catboost":
            assert obj.X[column].map(type).eq(str).all(), (library, column)
        else:
            assert isinstance(obj.X[column].dtype, pd.CategoricalDtype), (library, column, obj.X[column].dtype)
    if library != "catboost":
        assert obj.X["cat_missing"].isna().any()                     # fehlende Werte bleiben NaN
        assert obj.X["cat_many"].cat.categories.size == 200
    for column in ("num1", "num2"):
        assert pd.api.types.is_float_dtype(obj.X[column]), (library, column)

    train, test = first_fold(obj)
    rows = inject_test_only_level(obj, test, "cat_small")
    assert "NUR_IM_TEST" not in set(obj.X["cat_small"].iloc[train].astype(str))
    pred = fit_predict(obj, train, test)
    assert len(pred) == len(test) and np.isfinite(pred).all(), library
    score = r2(obj.y.iloc[test], pred)
    print(f"{library:8s} Regression R2 = {score:.3f} (synthetisch, Stufe nur im Test in {len(rows)} Zeilen)")
    assert score > 0.7, (library, score)

    obj_c = build(library, X, y_cls, indicator, suite_id=337)
    train, test = first_fold(obj_c)
    inject_test_only_level(obj_c, test, "cat_numeric")
    pred = fit_predict(obj_c, train, test)
    assert np.all((pred >= 0) & (pred <= 1)), library
    acc = ((pred > 0.5) == obj_c.y.iloc[test].values).mean()
    print(f"{library:8s} Klassifikation Accuracy = {acc:.3f}")
    assert acc > 0.75, (library, acc)

    # ganzer Tuning-Pfad (innere Validierung, Early Stopping, Testauswertung) mit 2 Zufallskombinationen
    obj_t = build(library, X, y_reg, indicator, suite_id=335)
    train, test = first_fold(obj_t)
    trials = obj_t.grid_search_method(X_train_full=obj_t.X.iloc[train], y_train_full=obj_t.y.iloc[train],
                                      X_test=obj_t.X.iloc[test], y_test=obj_t.y.iloc[test], num_try_random=2)
    assert len(trials) == 2 and trials["test_score"].notna().all(), (library, trials)
    print(f"{library:8s} grid_search_method (2 Kombinationen): Test-R2 {trials['test_score'].round(3).tolist()}")

# die Bibliotheken splitten wirklich nach Kategorien (nicht nach den Codes als Zahlen)
import gpboost
import xgboost
obj = build("gpboost", X, y_reg, indicator, suite_id=335)
bst = gpboost.train(params={"objective": "regression", "verbose": -1}, num_boost_round=5,
                    train_set=gpboost.Dataset(obj.X, label=obj.y))
assert bst.pandas_categorical is not None
assert "==" in {node.get("decision_type") for tree in bst.dump_model()["tree_info"] for node in [tree["tree_structure"]]}
bst = xgboost.train({"tree_method": "hist"}, xgboost.DMatrix(obj.X, label=obj.y, enable_categorical=True), 5)
assert [t for t, c in zip(bst.feature_types, obj.X.columns) if c in cat_columns] == ["c"] * 4, bst.feature_types
assert any(":{" in tree for tree in bst.get_dump()), "XGBoost: kein Kategorien-Split"
print("gpboost/xgboost: Kategorien-Splits vorhanden")

# 2. echte Daten
REAL = [(334, 361286, "compas-two-years"), (335, 361288, "abalone")]
for suite_id, task_id, name in REAL:
    path = f"data/{suite_id}_{task_id}/{suite_id}_{task_id}"
    if not os.path.exists(f"{path}_X.csv"):
        print(f"uebersprungen: {name} ({path}_X.csv fehlt)")
        continue
    X_real = pd.read_csv(f"{path}_X.csv")
    y_real = pd.read_csv(f"{path}_y.csv")
    ind_real = np.load(f"{path}_categorical_indicator.npy")
    for library in LIBRARIES:
        obj = build(library, X_real, y_real, ind_real, suite_id)
        assert obj.X.shape[1] == X_real.shape[1], (name, library, obj.X.shape)
        column = obj.cat_feature_names[0]
        train, test = first_fold(obj)
        inject_test_only_level(obj, test, column)
        pred = fit_predict(obj, train, test, {"params": {"learning_rate": 0.1}, "num_boost_round": 100,
                                              "verbose_eval": False})
        assert np.isfinite(pred).all(), (name, library)
        if suite_id == 334:
            score = ((pred > 0.5) == obj.y.iloc[test].values).mean()
            print(f"{library:8s} {name}: Accuracy {score:.3f} ({len(obj.cat_feature_names)} kategoriale Spalten)")
            assert score > 0.6, (name, library, score)
        else:
            score = r2(obj.y.iloc[test], pred)
            print(f"{library:8s} {name}: R2 {score:.3f} ({len(obj.cat_feature_names)} kategoriale Spalten)")
            assert score > 0.3, (name, library, score)

print("ALL OK")
