"""Zaehlt pro Datensatz einer OpenML-Suite, wie viele Merkmale kategorisch sind.

Aufruf:  python check_categoricals.py 457 out_457.csv
(neu: zusaetzlich Aufgabentyp und Klassenzahl)
(457 = Suite-ID, z. B. TabArena v0.1; zweites Argument = Ausgabedatei)
"""
import sys
import pandas as pd


def summarize(X, categorical_indicator, y=None, task_type=""):
    cat_cols = [c for c, is_cat in zip(X.columns, categorical_indicator) if is_cat]
    n_feat = X.shape[1]
    card = [X[c].nunique(dropna=True) for c in cat_cols]
    return {
        "n_rows": X.shape[0],
        "n_features": n_feat,
        "n_categorical": len(cat_cols),
        "share_categorical": round(len(cat_cols) / n_feat, 3) if n_feat else 0.0,
        "max_cardinality": max(card) if card else 0,
        "total_onehot_columns": int(sum(card)),
        "task_type": task_type,
        "n_classes": int(y.nunique()) if (y is not None and "classif" in task_type.lower()) else 0,
    }


def main(suite_id, out):
    import openml
    suite = openml.study.get_suite(suite_id)
    rows = []
    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            ds = task.get_dataset()
            X, y, cat, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
            r = summarize(X, cat, y, str(task.task_type))
            r.update(task_id=task_id, dataset=ds.name)
        except Exception as e:  # ein defekter Task soll den Rest nicht stoppen
            r = {"task_id": task_id, "dataset": f"FEHLER: {e}"}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    ok = df.dropna(subset=["n_features"])
    print("\nDatensaetze:", len(ok))
    print("ohne kategorische Merkmale:", int((ok.n_categorical == 0).sum()))
    print("Anteil kategorisch > 50 %:", int((ok.share_categorical > 0.5).sum()))
    print("max. Kardinalitaet > 50:", int((ok.max_cardinality > 50).sum()))
    print("Regression:", int((ok.task_type.str.lower().str.contains("regression")).sum()))
    print("Klassifikation binaer:", int(((ok.n_classes == 2)).sum()))
    print("Klassifikation mehr als 2 Klassen:", int((ok.n_classes > 2).sum()))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
