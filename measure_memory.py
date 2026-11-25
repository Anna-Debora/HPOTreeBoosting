"""Spitzenspeicher je Aufgabe fuer den teuersten Versuch des Suchraums (num_leaves = 1024, max_bin = min(Zeilen,
10'000)), gemessen auf dem Mac: Daten ganz laden, auf 100'000 Zeilen unterabtasten, GPBoost auf 80 % der Zeilen
(Refit auf dem ganzen Trainingsteil), 3 Runden, 1 Thread. LightGBM/GPBoost legt fuer jedes Blatt ein Histogramm an:
num_leaves x Summe der Bins x 16 Byte, bei vielen stetigen Merkmalen (MiniBooNE, jannis, topo_2_1) mehrere GB.
Die Messung ist eine Untergrenze fuer den echten Job (dort kommen Python-Objekte, Testvorhersagen, Surrogate dazu);
make_paper_jobs.memory_gb() schlaegt deshalb Reserve auf.

Aufruf:  python measure_memory.py [--out jobs_paper/speicher_mac.tsv] [--only 337_361068 ...]
         python measure_memory.py --tasks jobs_paper/aufgaben_neu.tsv --out jobs_paper/speicher_neu_mac.tsv
         (--tasks: andere Aufgabenliste mit Spalten suite, task, name; Standard jobs_paper/aufgaben.tsv)
         python measure_memory.py --library catboost --tasks jobs_paper/aufgaben_alle.tsv --out jobs_paper/speicher_catboost_mac.tsv
         (--library xgboost: max_leaves 1024, lossguide, max_bin wie oben; catboost: depth 10, border_count
          min(Zeilen, 1024), Kategorien nativ als Strings; jeweils 3 Runden, 1 Thread, Suchraeume SUCHRAEUME_XGB_CAT.md)
Jede Aufgabe laeuft in einem eigenen Prozess (ru_maxrss gilt pro Prozess).
"""
import argparse
import resource
import subprocess
import sys

import pandas as pd

LEAVES = 1024


def one_library(name, library, X, y):
    """Teuerster Versuch des Suchraums fuer XGBoost bzw. CatBoost (Kategorien nativ wie methods.py)."""
    import numpy as np
    cat = np.load(f"data/{name}/{name}_categorical_indicator.npy").astype(bool)
    columns = list(X.columns[cat])
    n = int(len(X) * 0.8)
    y = pd.Series(pd.factorize(y)[0] if y.dtype == object else y.to_numpy(), dtype=float)
    if library == "xgboost":
        import xgboost as xgb
        for column in columns:
            X[column] = X[column].astype(str).where(X[column].notna()).astype("category")
        data = xgb.DMatrix(X.iloc[:n], label=y.iloc[:n], enable_categorical=True)
        xgb.train({"tree_method": "hist", "grow_policy": "lossguide", "max_leaves": LEAVES, "max_depth": 0,
                   "max_bin": int(min(len(X), 10000)), "min_child_weight": 0.25, "nthread": 1}, data, num_boost_round=3)
    else:
        import catboost
        for column in columns:
            X[column] = X[column].astype(object).where(X[column].notna(), "__missing__").astype(str)
        pool = catboost.Pool(X.iloc[:n], label=y.iloc[:n], cat_features=columns or None)
        catboost.CatBoost({"iterations": 3, "depth": 10, "border_count": int(min(len(X), 1024)), "thread_count": 1,
                           "loss_function": "RMSE", "verbose": False, "allow_writing_files": False}).fit(pool)


def one(name, library="gpboost"):
    import gpboost as gpb
    import numpy as np
    X = pd.read_csv(f"data/{name}/{name}_X.csv")
    y = pd.read_csv(f"data/{name}/{name}_y.csv").iloc[:, 0]
    if len(X) > 100000:
        X = X.sample(100000, random_state=0)
        y = y.loc[X.index]
    if library != "gpboost":
        one_library(name, library, X, y)
        scale = 1 if sys.platform == "darwin" else 1024
        print(f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9:.3f}")
        return
    for column in X.columns[X.dtypes == object]:
        X[column] = X[column].astype("category").cat.codes
    n = int(len(X) * 0.8)
    max_bin = int(min(len(X), 10000))
    y = pd.Series(pd.factorize(y)[0] if y.dtype == object else y.to_numpy(), dtype=float)
    data = gpb.Dataset(X.iloc[:n].astype(float), label=y.iloc[:n], params={"max_bin": max_bin, "verbose": -1})
    gpb.train({"num_leaves": LEAVES, "max_bin": max_bin, "verbose": -1, "min_data_in_leaf": 1, "num_threads": 1,
               "objective": "regression"}, data, num_boost_round=3)
    scale = 1 if sys.platform == "darwin" else 1024          # ru_maxrss: Bytes auf macOS, KB auf Linux
    print(f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9:.3f}")


def main(out, only, task_file="jobs_paper/aufgaben.tsv", library="gpboost"):
    tasks = pd.read_csv(task_file, sep="\t")
    rows = []
    for t in tasks.itertuples():
        name = f"{t.suite}_{t.task}"
        if only and name not in only:
            continue
        result = subprocess.run([sys.executable, __file__, "--one", name, "--library", library], capture_output=True,
                                text=True)
        peak = float(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else float("nan")
        rows.append(dict(suite=t.suite, task=t.task, name=t.name, peak_gb=peak))
        print(f"{name} {t.name}: {peak:.2f} GB", flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jobs_paper/speicher_mac.tsv")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--one")
    parser.add_argument("--tasks", default="jobs_paper/aufgaben.tsv", help="Aufgabenliste (suite, task, name)")
    parser.add_argument("--library", choices=("gpboost", "xgboost", "catboost"), default="gpboost")
    args = parser.parse_args()
    if args.one:
        one(args.one, args.library)
    else:
        main(args.out, args.only, args.tasks, args.library)
