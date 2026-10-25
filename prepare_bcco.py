"""Laedt BCCO (Hugging Face: stable-ai/bcco_reg und stable-ai/bcco_cls), verarbeitet die Datensaetze wie
Grinsztajn et al. (siehe preprocess_tabular.py) und legt sie wie die OpenML-Aufgaben ab:
    data/{suite}_{task}/{suite}_{task}_X.csv, _y.csv, _categorical_indicator.npy
Train- und Testdatei werden zusammengefuegt (die Aufteilung macht unser Code selbst in 5 Folds).
Die Zielspalte heisst 'target' (sonst wird die letzte Spalte genommen, Name steht im Bericht). Spalten vom Typ Text/Kategorie gelten als kategorial,
ausserdem ganzzahlige Spalten mit 3 bis 20 verschiedenen Werten (BCCO speichert Kategorien oft als Zahlen und gibt keine Typen an).
Kuenstlich erzeugte Datensaetze (ARTIFICIAL) werden wie bei Grinsztajn uebersprungen, ebenso die in
excluded_duplicates.csv eingetragenen Duplikate alter Aufgaben (sonst wuerden sie wieder registriert).
BCCO hat keine OpenML-IDs: wir vergeben task_id = 900000 + laufende Nummer (alphabetisch pro Repo,
Regression ab 900000, Klassifikation ab 910000).
Datensaetze, deren Name schon in den alten Suiten oder in new_tasks.csv vorkommt, werden uebersprungen.

Aufruf:  python prepare_bcco.py            (alle)
         python prepare_bcco.py --limit 3  (nur die ersten 3 je Repo zum Testen)
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from preprocess_tabular import MAX_LEVELS, preprocess, selection_problem
from task_registry import register_tasks

REPOS = {"stable-ai/bcco_reg": (1001, 900000, "bcco_reg"), "stable-ai/bcco_cls": (1002, 910000, "bcco_cls")}
ARTIFICIAL = {"2dplanes"}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_categorical(column):
    """Text/Kategorie, oder ganzzahlig mit 3..MAX_LEVELS Werten (2 Werte macht preprocess ohnehin kategorial)."""
    if not pd.api.types.is_numeric_dtype(column):
        return True
    values = column.dropna()
    return len(values) > 0 and bool(np.all(np.mod(values, 1) == 0)) and 3 <= values.nunique() <= MAX_LEVELS


def known_names():
    names = set()
    for suite in (334, 335, 336, 337):
        path = f"task_indices/{suite}_task_names.json"
        if os.path.exists(path):
            names |= {norm(n) for n in json.load(open(path))}
    if os.path.exists("new_tasks.csv"):
        names |= {norm(n) for n in pd.read_csv("new_tasks.csv")["name"]}
    return names


def excluded_names():
    """(suite_id, normierter Name) aus excluded_duplicates.csv."""
    if not os.path.exists("excluded_duplicates.csv"):
        return set()
    excluded = pd.read_csv("excluded_duplicates.csv")
    return {(int(s), norm(n)) for s, n in zip(excluded["suite_id"], excluded["name"])}


def main(limit=None, balance=True):
    from huggingface_hub import hf_hub_download, list_repo_files
    skip = known_names()
    excluded = excluded_names()
    report, registered = [], {1001: [], 1002: []}
    for repo, (suite, first_id, folder) in REPOS.items():
        files = list_repo_files(repo, repo_type="dataset")
        names = sorted({f.split("/")[1] for f in files if f.startswith(folder + "/") and f.endswith(".csv")})
        if limit:
            names = names[:limit]
        for index, name in enumerate(names):
            task_id = first_id + index
            print(f"[{suite}_{task_id}] {name} ...", flush=True)
            info = {"suite_id": suite, "task_id": task_id, "name": name, "source": repo}
            try:
                if norm(name) in skip:
                    info["status"] = "UEBERSPRUNGEN: Name schon vorhanden (Duplikat)"
                elif (suite, norm(name)) in excluded:
                    info["status"] = "UEBERSPRUNGEN: in excluded_duplicates.csv ausgeschlossen"
                elif name in ARTIFICIAL:
                    info["status"] = "UEBERSPRUNGEN: kuenstlich erzeugte Daten"
                else:
                    parts = [pd.read_csv(hf_hub_download(repo, f"{folder}/{name}/{name}_{kind}.csv", repo_type="dataset"))
                             for kind in ("train", "test")]
                    frame = pd.concat(parts, ignore_index=True)
                    # Zielspalte: 'target'; falls es sie nicht gibt (z. B. bei bcco_cls), die letzte Spalte
                    target_col = "target" if "target" in frame.columns else frame.columns[-1]
                    info["target_col"] = target_col
                    y = frame.pop(target_col)
                    categorical = [is_categorical(frame[c]) for c in frame.columns]
                    X, y, indicator, details = preprocess(frame, y, categorical, classification=(suite == 1002),
                                                          balance=balance)
                    info.update(details)
                    problem = "nach der Bereinigung bleibt nichts uebrig" if (X.shape[1] == 0 or len(X) == 0) \
                        else selection_problem(details)
                    if problem:
                        info["status"] = f"AUSGESCHIEDEN: {problem}"
                    else:
                        path = f"data/{suite}_{task_id}"
                        os.makedirs(path, exist_ok=True)
                        X.to_csv(f"{path}/{suite}_{task_id}_X.csv", index=False)
                        y.rename("target").to_csv(f"{path}/{suite}_{task_id}_y.csv", index=False)
                        np.save(f"{path}/{suite}_{task_id}_categorical_indicator.npy", indicator)
                        info["status"] = "ok"
                        registered[suite].append((task_id, name))
            except Exception as error:
                info["status"] = f"FEHLER: {error}"
            print("   ", info.get("status"), {k: info[k] for k in ("target_col", "rows_before", "rows_after", "cols_after") if k in info},
                  flush=True)
            report.append(info)
    pd.DataFrame(report).to_csv("preprocessing_report_bcco.csv", index=False)
    for suite, tasks in registered.items():
        if tasks:
            register_tasks(suite, tasks)
    ok = sum(len(v) for v in registered.values())
    print(f"\nfertig: {ok} BCCO-Aufgaben vorbereitet; Bericht: preprocessing_report_bcco.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()
    main(args.limit, balance=not args.no_balance)
