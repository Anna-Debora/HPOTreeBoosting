"""Prueft fuer Aufgaben aus stdin ("<suite> <task>" je Zeile), dass die Daten so geladen werden wie beim Training
(methods.ParameterOptimization, GPBoost): kategoriale Spalten als pandas "category", gleich viele Merkmale wie in
der Datei (kein One-Hot). Exit-Code 1, wenn eine Aufgabe nicht passt.

Aufruf:  echo "336 361076" | python check_native_encoding.py
         tail -n +2 jobs_paper/aufgaben.tsv | cut -f1,2 | python check_native_encoding.py
"""
import sys

import numpy as np
import pandas as pd

from methods import ParameterOptimization


def check(suite, task):
    stem = f"data/{suite}_{task}/{suite}_{task}"
    X, y = pd.read_csv(f"{stem}_X.csv"), pd.read_csv(f"{stem}_y.csv")
    indicator = np.load(f"{stem}_categorical_indicator.npy")
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=indicator, suite_id=suite, library="gpboost")
    n_cat = int(np.sum(indicator))
    as_category = sum(str(obj.X[c].dtype) == "category" for c in obj.cat_feature_names)
    ok = as_category == n_cat and obj.X.shape[1] == X.shape[1]
    print(f"  {'OK    ' if ok else 'FEHLER'} {suite}_{task}: {obj.X.shape[0]} Zeilen, {obj.X.shape[1]} Merkmale, "
          f"davon {as_category}/{n_cat} kategorial (category)", flush=True)
    return ok


if __name__ == "__main__":
    results = [check(*map(int, line.split()[:2])) for line in sys.stdin if line.strip()]
    print(f"{sum(results)} von {len(results)} Aufgaben in Ordnung")
    sys.exit(0 if all(results) else 1)
