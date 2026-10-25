"""Erzeugt die Task-Liste der NEUEN Aufgaben aus union.csv.

Regeln: kein albert (der grosse TabZilla-Albert ist dieselbe Datenquelle wie der alte), Mehrklassen-Aufgaben
sind dabei (sie werden spaeter wie bei Grinsztajn auf die zwei haeufigsten Klassen reduziert), Duplikate (gleicher normalisierter Name + Aufgabentyp) nur einmal, nichts, was schon in
den alten Suiten 334-337 steckt. Regression -> Pseudo-Suite 1001, Klassifikation -> 1002.

Aufruf:  python make_new_task_list.py union.csv new_tasks.csv
"""
import re
import sys

import pandas as pd

EXCLUDE_NAMES = {"albert"}
NEW_REGRESSION_SUITE, NEW_CLASSIFICATION_SUITE = 1001, 1002


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def main(src, out):
    df = pd.read_csv(src)
    df = df[~df["dataset"].astype(str).str.startswith("FEHLER")].copy()
    df["key"] = df["dataset"].map(norm)
    df["typ"] = df["task_type"].fillna("")
    n_input = len(df)

    rows, dropped = [], {"alt": 0, "albert": 0, "mehrklassen": 0}
    for (key, typ), grp in df.groupby(["key", "typ"], sort=True):
        if grp["quellen"].str.contains("alt").any():
            dropped["alt"] += 1
            continue
        if key in EXCLUDE_NAMES:
            dropped["albert"] += 1
            continue
        if typ != "regression" and not (grp["n_classes"] >= 2).any():
            dropped["mehrklassen"] += 1   # Klassifikation mit weniger als 2 Klassen (sollte nicht vorkommen)
            continue
        # Vertreter: bevorzugt TabArena (457), sonst TabZilla (379)
        pairs = []
        for _, r in grp.iterrows():
            for t in str(r["tasks"]).split():
                suite, task = t.split(":")
                pairs.append((0 if suite == "457" else 1, int(suite), int(task), r))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))
        _, suite, task, r = pairs[0]
        rows.append({
            "suite_id": NEW_REGRESSION_SUITE if typ == "regression" else NEW_CLASSIFICATION_SUITE,
            "task_id": task, "name": r["dataset"], "source_suite": suite,
            "all_sources": "+".join(sorted(set("+".join(grp["quellen"]).split("+")))),
            "n_classes": r["n_classes"], "n_rows": r["n_rows"], "n_features": r["n_features"],
            "n_categorical": r["n_categorical"], "max_cardinality": r["max_cardinality"],
        })
    result = pd.DataFrame(rows).sort_values(["suite_id", "name"])
    result.to_csv(out, index=False)
    print(f"Zeilen in union.csv: {n_input}")
    print(f"verworfen: schon alt: {dropped['alt']}, albert: {dropped['albert']}, ungueltig: {dropped['mehrklassen']}")
    print(f"neue Aufgaben: {len(result)} (Regression {int((result.suite_id == NEW_REGRESSION_SUITE).sum())}, "
          f"Klassifikation {int((result.suite_id == NEW_CLASSIFICATION_SUITE).sum())}, "
          f"davon mit mehr als 2 Klassen {int((result.n_classes > 2).sum())})")
    print(result["all_sources"].value_counts().to_string())
    print("Geschrieben:", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
