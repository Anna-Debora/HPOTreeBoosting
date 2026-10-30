"""Anteil doppelter Zeilen pro neuer Aufgabe (Suiten 1001/1002, vorbereitete Daten aus data/).

Doppelt = Zeile (Merkmale und Ziel) kommt weiter oben im Datensatz schon genau so vor. Aufgaben mit einem
Anteil ueber THRESHOLD werden ausgegeben; sie gehoeren in fraglich.csv ("viele doppelte Zeilen").
Mit --update-fraglich werden fehlende solche Aufgaben dort angehaengt (bestehende Eintraege bleiben).

Aufruf:  python duplicate_rows_report.py [--update-fraglich]
Ausgabe: duplicate_rows_report.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

THRESHOLD = 0.3


def main(update_fraglich=False, out="duplicate_rows_report.csv"):
    rows = []
    for suite in (1001, 1002):
        ids = np.load(f"task_indices/{suite}_task_indices.npy").tolist()
        names = json.load(open(f"task_indices/{suite}_task_names.json"))
        for task_id, name in zip(ids, names):
            path = f"data/{suite}_{task_id}/{suite}_{task_id}"
            frame = pd.concat([pd.read_csv(f"{path}_X.csv"), pd.read_csv(f"{path}_y.csv")], axis=1)
            duplicates = int(frame.duplicated().sum())
            rows.append({"suite_id": suite, "task_id": task_id, "name": name, "rows": len(frame),
                         "doppelte_zeilen": duplicates, "anteil": round(duplicates / len(frame), 4)})
    table = pd.DataFrame(rows).sort_values("anteil", ascending=False)
    table.to_csv(out, index=False)
    many = table[table["anteil"] > THRESHOLD]
    print(f"{len(many)} von {len(table)} Aufgaben mit mehr als {THRESHOLD:.0%} doppelten Zeilen:")
    print(many.to_string(index=False))
    if update_fraglich:
        fraglich = pd.read_csv("fraglich.csv")
        known = set(zip(fraglich["suite_id"], fraglich["task_id"]))
        new = [{"suite_id": r.suite_id, "task_id": r.task_id, "name": r.name,
                "grund": f"viele doppelte Zeilen ({r.anteil:.0%}), Fabio fragen"}
               for r in many.itertuples() if (r.suite_id, r.task_id) not in known]
        if new:
            pd.concat([fraglich, pd.DataFrame(new)], ignore_index=True).to_csv("fraglich.csv", index=False)
        print(f"{len(new)} Aufgaben in fraglich.csv eingetragen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-fraglich", action="store_true")
    args = parser.parse_args()
    main(args.update_fraglich)
