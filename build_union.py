"""Baut die Vereinigung aller Benchmark-Quellen und entfernt Duplikate (ueber die OpenML-Datensatz-ID).

Quellen (OpenML-Suiten): 334-337 = bisherige 59 Tasks, 457 = TabArena v0.1, 379 = TabZilla.
Ausgabe: union.csv (eine Zeile je Datensatz) und eine Zusammenfassung auf dem Bildschirm.
brauchbar = Regression oder binaere Klassifikation (mehr kann der Code noch nicht).

Aufruf:  python build_union.py union.csv
"""
import sys

import pandas as pd

SUITES = {334: "alt", 335: "alt", 336: "alt", 337: "alt", 457: "TabArena", 379: "TabZilla"}


def summarize(X, cat, y, task_type):
    cols = [c for c, is_cat in zip(X.columns, cat) if is_cat]
    card = [X[c].nunique(dropna=True) for c in cols]
    t = task_type.lower()
    return {
        "n_rows": X.shape[0], "n_features": X.shape[1], "n_categorical": len(cols),
        "max_cardinality": max(card) if card else 0, "total_onehot_columns": int(sum(card)),
        "task_type": "regression" if "regression" in t else "classification",
        "n_classes": int(y.nunique()) if "classif" in t else 0,
    }


def main(out):
    import openml
    seen = {}  # dataset_id -> Zeile
    for suite_id, source in SUITES.items():
        suite = openml.study.get_suite(suite_id)
        for task_id in suite.tasks:
            try:
                task = openml.tasks.get_task(task_id, download_splits=False)
                ds = task.get_dataset()
                key = ds.dataset_id
                if key in seen:
                    seen[key]["quellen"].add(source)
                    seen[key]["tasks"].append(f"{suite_id}:{task_id}")
                    continue
                X, y, cat, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
                row = summarize(X, cat, y, str(task.task_type))
                row.update(dataset_id=key, dataset=ds.name, quellen={source}, tasks=[f"{suite_id}:{task_id}"])
            except Exception as e:
                row = {"dataset_id": f"task{task_id}", "dataset": f"FEHLER: {e}", "quellen": {source},
                       "tasks": [f"{suite_id}:{task_id}"]}
                key = row["dataset_id"]
                seen[key] = row
                print(row, flush=True)
                continue
            seen[key] = row
            print(row["dataset"], row["task_type"], row["n_classes"], sorted(row["quellen"]), flush=True)

    df = pd.DataFrame(seen.values())
    df["brauchbar"] = (df["task_type"] == "regression") | (df["n_classes"] == 2)
    df["quellen"] = df["quellen"].apply(lambda s: "+".join(sorted(s)))
    df["tasks"] = df["tasks"].apply(lambda t: " ".join(t))
    df.to_csv(out, index=False)

    ok = df[df["task_type"].notna()]
    print("\n===== Zusammenfassung =====")
    print("Datensaetze insgesamt (ohne Duplikate):", len(ok))
    print("davon brauchbar (Regression oder binaer):", int(ok["brauchbar"].sum()))
    print("Verteilung nach Quellen (alle):")
    print(ok["quellen"].value_counts().to_string())
    print("Verteilung nach Quellen (nur brauchbar):")
    print(ok[ok["brauchbar"]]["quellen"].value_counts().to_string())
    big = ok[ok["brauchbar"] & (ok["max_cardinality"] > 50)]
    print("brauchbar, aber Merkmal mit > 50 Kategorien:")
    print(big[["dataset", "max_cardinality", "total_onehot_columns", "n_rows"]].to_string(index=False))
    dup_names = ok[ok.duplicated("dataset", keep=False)]
    if len(dup_names):
        print("Gleicher Name, verschiedene OpenML-ID (pruefen!):")
        print(dup_names[["dataset", "dataset_id", "quellen"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
