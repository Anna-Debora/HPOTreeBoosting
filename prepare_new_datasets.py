"""Laedt die neuen Aufgaben aus new_tasks.csv von OpenML, verarbeitet sie wie Grinsztajn et al. (inkl. Auswahlkriterien: >= 4 Merkmale, Merkmale/Zeilen < 1/10;
statt >= 3000 Zeilen nur die technische Untergrenze aus preprocess_tabular.py)
und legt sie im Format ab, das run_experiments.py erwartet:
    data/{suite}_{task}/{suite}_{task}_X.csv, _y.csv, _categorical_indicator.npy
Ausserdem: task_indices/{suite}_task_indices.npy und _task_names.json (fuer Bericht und Plots)
und preprocessing_report.csv (was pro Aufgabe weggefallen ist).
Aufgaben aus excluded_duplicates.csv (Duplikate alter Aufgaben) werden uebersprungen und nicht registriert.

Aufruf:  python prepare_new_datasets.py new_tasks.csv            (alle)
         python prepare_new_datasets.py new_tasks.csv --limit 3   (nur die ersten 3 zum Testen)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from preprocess_tabular import preprocess, selection_problem
from task_registry import register_tasks


def excluded_tasks():
    """(suite_id, task_id) aus excluded_duplicates.csv."""
    if not os.path.exists("excluded_duplicates.csv"):
        return set()
    excluded = pd.read_csv("excluded_duplicates.csv")
    return {(int(s), int(t)) for s, t in zip(excluded["suite_id"], excluded["task_id"])}


def main(tasks_csv, limit=None, balance=True):
    import openml
    tasks = pd.read_csv(tasks_csv)
    excluded = excluded_tasks()
    if limit:
        tasks = tasks.head(limit)
    os.makedirs("task_indices", exist_ok=True)
    report, ok_rows = [], []
    for _, row in tasks.iterrows():
        suite, task_id, name = int(row["suite_id"]), int(row["task_id"]), row["name"]
        print(f"[{suite}_{task_id}] {name} ...", flush=True)
        if (suite, task_id) in excluded:
            info = {"suite_id": suite, "task_id": task_id, "name": name,
                    "status": "UEBERSPRUNGEN: in excluded_duplicates.csv ausgeschlossen"}
            print("   ", info, flush=True)
            report.append(info)
            continue
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            dataset = task.get_dataset()
            X, y, cat, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            X, y, indicator, info = preprocess(X, y, cat, classification=(suite == 1002), balance=balance)
            problem = "nach der Bereinigung bleibt nichts uebrig" if (X.shape[1] == 0 or len(X) == 0) \
                else selection_problem(info)
            if problem:
                info.update(suite_id=suite, task_id=task_id, name=name, status=f"AUSGESCHIEDEN: {problem}")
                print("   ", info, flush=True)
                report.append(info)
                continue
            path = f"data/{suite}_{task_id}"
            os.makedirs(path, exist_ok=True)
            X.to_csv(f"{path}/{suite}_{task_id}_X.csv", index=False)
            y.rename("target").to_csv(f"{path}/{suite}_{task_id}_y.csv", index=False)
            np.save(f"{path}/{suite}_{task_id}_categorical_indicator.npy", indicator)
            info.update(suite_id=suite, task_id=task_id, name=name, status="ok")
            ok_rows.append((suite, task_id, name))
        except Exception as error:  # eine kaputte Aufgabe soll den Rest nicht stoppen
            info = {"suite_id": suite, "task_id": task_id, "name": name, "status": f"FEHLER: {error}"}
        print("   ", info, flush=True)
        report.append(info)
    pd.DataFrame(report).to_csv("preprocessing_report.csv", index=False)
    for suite in (1001, 1002):
        chosen = [(t, n) for s, t, n in ok_rows if s == suite]
        if chosen:
            register_tasks(suite, chosen)
    print(f"\nfertig: {len(ok_rows)} von {len(tasks)} Aufgaben vorbereitet; Bericht: preprocessing_report.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-balance", action="store_true",
                        help="Klassen nicht auf gleiche Zeilenzahl kuerzen (Standard: kuerzen, wie Grinsztajn)")
    args = parser.parse_args()
    main(args.tasks_csv, args.limit, balance=not args.no_balance)
