"""Spitzenspeicher je Aufgabe fuer den teuersten Versuch des Suchraums (num_leaves = 1024, max_bin = min(Zeilen,
10'000)), gemessen auf dem Mac: Daten ganz laden, auf 100'000 Zeilen unterabtasten, GPBoost auf 80 % der Zeilen
(Refit auf dem ganzen Trainingsteil), 3 Runden, 1 Thread. LightGBM/GPBoost legt fuer jedes Blatt ein Histogramm an:
num_leaves x Summe der Bins x 16 Byte, bei vielen stetigen Merkmalen (MiniBooNE, jannis, topo_2_1) mehrere GB.
Die Messung ist eine Untergrenze fuer den echten Job (dort kommen Python-Objekte, Testvorhersagen, Surrogate dazu);
make_paper_jobs.memory_gb() schlaegt deshalb Reserve auf.

Aufruf:  python measure_memory.py [--out jobs_paper/speicher_mac.tsv] [--only 337_361068 ...]
Jede Aufgabe laeuft in einem eigenen Prozess (ru_maxrss gilt pro Prozess).
"""
import argparse
import resource
import subprocess
import sys

import pandas as pd

LEAVES = 1024


def one(name):
    import gpboost as gpb
    import numpy as np
    X = pd.read_csv(f"data/{name}/{name}_X.csv")
    y = pd.read_csv(f"data/{name}/{name}_y.csv").iloc[:, 0]
    if len(X) > 100000:
        X = X.sample(100000, random_state=0)
        y = y.loc[X.index]
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


def main(out, only):
    tasks = pd.read_csv("jobs_paper/aufgaben.tsv", sep="\t")
    rows = []
    for t in tasks.itertuples():
        name = f"{t.suite}_{t.task}"
        if only and name not in only:
            continue
        result = subprocess.run([sys.executable, __file__, "--one", name], capture_output=True, text=True)
        peak = float(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else float("nan")
        rows.append(dict(suite=t.suite, task=t.task, name=t.name, peak_gb=peak))
        print(f"{name} {t.name}: {peak:.2f} GB", flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jobs_paper/speicher_mac.tsv")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--one")
    args = parser.parse_args()
    if args.one:
        one(args.one)
    else:
        main(args.out, args.only)
