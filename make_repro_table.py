"""Ergebnistabelle des Reproduktions-Testlaufs im Aufbau der Paper-Ergebnisse pro Datensatz.

Das Paper (Koster & Sigrist, arXiv:2602.05786v3) zeigt die Ergebnisse pro Methode und Datensatz nur als
Grafiken: Abbildungen 5/6 (R2), 7/8 (RMSE), 9 (Accuracy) und 10 (Log Loss), Testwert des bisher besten
Versuchs als Funktion der Versuche, Mittel ueber 5 Folds und 20 Seeds. Table 5 ist ueber alle Datensaetze
gemittelt. Diese Tabelle nimmt den Endwert dieser Kurven (t = 135) und stellt ihn als Zahlentabelle dar:
  Zeilen  = Datensaetze des Testlaufs, getrennt nach Kennzahl (wie die Abbildungen)
  Spalten = die sieben Methoden des Papers mit dessen Namen und in der Reihenfolge von Table 5
  Zelle   = unser Wert / Paper-Wert (Differenz wir - Paper) und Anzahl Seeds in unserem Mittelwert
Unser Wert: Mittel ueber die 5 Folds, dann ueber die Seeds. Ein Seed zaehlt nur, wenn alle 5 Folds das Budget
erreicht haben (sonst Lauf noch offen oder Zeitlimit). Deterministisches Gitter: Gitterende (bei uns 270,
im Paper 135), wie in compare_with_paper.py. Hyperband lief im Testlauf nicht mit (Spalte leer).
z wie in compare_with_paper.py: Abweichung geteilt durch die aus dem 95-%-Band des Papers erwartete Streuung.

Aufruf:  python make_repro_table.py --results Results_repro [--t 135] [--out repro_table]
Ausgabe: repro_table.csv (eine Zeile je Aufgabe, Kennzahl und Methode) und repro_table.md
"""
import argparse

import numpy as np
import pandas as pd

from compare_with_paper import GRID_END, METRICS, _markdown, load_results

METHODS = ["Default", "Deterministic Grid", "GP-BO", "Hyperband", "Random Grid", "SMAC", "TPE"]  # Table 5
NOT_RUN = {"Hyperband": "nicht im Testlauf"}
OPEN = "Lauf noch offen oder Zeitlimit"
FOLDS = 5


def repro_tasks(jobs="jobs_repro/jobs.tsv"):
    tasks = pd.read_csv(jobs, sep="\t", header=None, names=["art", "suite_id", "task_id", "seed", "dataset"])
    return tasks.drop_duplicates(["suite_id", "task_id"])[["suite_id", "task_id", "dataset"]]


def our_values(raw, t):
    """Je (Aufgabe, Methode): Mittel ueber vollstaendige Seeds beim Budget; dazu Anzahl Seeds und Anmerkung."""
    rows = []
    if raw.empty:
        return pd.DataFrame(columns=["suite_id", "task_id", "method"])
    for (suite, task, method), group in raw.groupby(["suite_id", "task_id", "method"]):
        budget = GRID_END if method == "Deterministic Grid" else t
        if method == "Default":
            budget = min(budget, group["t"].max())
        at_t = group[group["t"] == budget]
        folds = at_t.groupby("seed")["fold"].nunique()
        complete = folds[folds >= FOLDS].index
        all_seeds = group["seed"].unique()
        per_seed = at_t[at_t["seed"].isin(complete)].groupby("seed")[
            ["best_test_score", "best_test_rmse", "best_test_log_loss"]].mean()
        note = ""
        if len(complete) < len(all_seeds):
            incomplete = [s for s in all_seeds if s not in complete]
            note = f"{OPEN} (Seed {', '.join(str(s) for s in incomplete)} unvollstaendig)"
        rows.append({"suite_id": suite, "task_id": task, "method": method, "budget_wir": budget,
                     "seeds": len(complete), "anmerkung": note, **per_seed.mean().to_dict()})
    return pd.DataFrame(rows)


def build_table(raw, paper, tasks, t):
    ours = our_values(raw, t).set_index(["suite_id", "task_id", "method"])
    rows = []
    for _, task in tasks.iterrows():
        classification = task["suite_id"] in (334, 337)
        for metric, column, higher_better in METRICS[classification]:
            for method in METHODS:
                ref = paper[(paper["metric"] == metric) & (paper["dataset"] == task["dataset"]) &
                            (paper["method"] == method) & (paper["t"] == t) & ~paper["doppelt"]]
                ref = ref.iloc[0] if len(ref) else None
                key = (task["suite_id"], task["task_id"], method)
                our = ours.loc[key] if key in ours.index else None
                seeds = int(our["seeds"]) if our is not None else 0
                value = our[f"best_{column}"] if our is not None and seeds else np.nan
                if method in NOT_RUN:
                    note = NOT_RUN[method]
                elif our is None:
                    note = OPEN
                else:
                    note = our["anmerkung"]
                paper_value = ref["value"] if ref is not None else np.nan
                z = np.nan
                if ref is not None and np.isfinite(value) and method != "Default":
                    seed_sd = (ref["ci_high"] - ref["ci_low"]) / 3.92 * np.sqrt(20)
                    if seed_sd > 0:
                        z = (value - paper_value) / (seed_sd * np.sqrt(1 / seeds + 1 / 20))
                rows.append({"suite_id": task["suite_id"], "task_id": task["task_id"], "dataset": task["dataset"],
                             "metric": metric, "method": method, "wir": value, "paper": paper_value,
                             "differenz": value - paper_value,
                             "rel_differenz": (value - paper_value) / abs(paper_value) if paper_value else np.nan,
                             "paper_ci_low": ref["ci_low"] if ref is not None else np.nan,
                             "paper_ci_high": ref["ci_high"] if ref is not None else np.nan,
                             "z": z, "seeds": seeds,
                             "budget_wir": GRID_END if method == "Deterministic Grid" else t,
                             "anmerkung": note if ref is not None else (note or "kein Paper-Wert")})
    return pd.DataFrame(rows)


def _cell(row, notes):
    if not np.isfinite(row["wir"]):
        mark = notes.setdefault(row["anmerkung"], "*" * (len(notes) + 1))
        paper = f"{row['paper']:.4f}" if np.isfinite(row["paper"]) else "–"
        return f"– / {paper} {mark}"
    text = f"{row['wir']:.4f} / {row['paper']:.4f} ({row['differenz']:+.4f}) n={row['seeds']}"
    if row["anmerkung"]:
        text += " " + notes.setdefault(row["anmerkung"], "*" * (len(notes) + 1))
    return text


def write_markdown(table, t, path):
    notes = {}
    lines = [f"# Testlauf vs. Paper: Endwert der Kurven aus Abb. 5–10 (t = {t})", "",
             "Zelle: wir / Paper (wir − Paper), n = Seeds in unserem Mittelwert (Paper: 20 Seeds). "
             "Mittel über 5 Folds und Seeds. Deterministic Grid: Gitterende (wir 270, Paper 135). "
             "Default: ein Lauf ohne Seed.", ""]
    for metric, figures in (("R2", "Abb. 5/6"), ("RMSE", "Abb. 7/8"), ("Accuracy", "Abb. 9"),
                            ("Log Loss", "Abb. 10")):
        part = table[table["metric"] == metric]
        if part.empty:
            continue
        wide = pd.DataFrame({"Data set": part["dataset"].unique()})
        for method in METHODS:
            cells = {r["dataset"]: _cell(r, notes) for _, r in part[part["method"] == method].iterrows()}
            wide[method] = wide["Data set"].map(cells)
        lines += [f"## {metric} ({figures})", "", _markdown(wide), ""]
    if notes:
        lines += ["Anmerkungen: " + "; ".join(f"{mark} {text}" for text, mark in notes.items()), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(results, t, out, reference="paper_reference_values.csv", jobs="jobs_repro/jobs.tsv"):
    paper = pd.read_csv(reference)
    raw = load_results(results)
    table = build_table(raw, paper, repro_tasks(jobs), t)
    table.to_csv(f"{out}.csv", index=False)
    write_markdown(table, t, f"{out}.md")
    print(open(f"{out}.md").read())
    print(f"Geschrieben: {out}.csv, {out}.md")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Ordner mit Results/, Results_SMAC/, Results_Default_Run/")
    parser.add_argument("--t", type=int, default=135)
    parser.add_argument("--out", default="repro_table")
    args = parser.parse_args()
    main(args.results, args.t, args.out)
