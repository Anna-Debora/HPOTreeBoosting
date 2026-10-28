"""Legt die Ursachen-Laeufe A/B (run_ablation.py) neben den Testlauf und das Paper: wine_quality, R2 bei t = 135.

Spalten je Methode (Zufallsgitter, TPE, SMAC):
  Paper           Kurve aus Abb. 5/6 bei t = 135 (paper_reference_values.csv), dazu das 95-%-Band
  A               135 Versuche, ohne lambda_l1 (Paper-Einstellung)
  B               135 Versuche, mit lambda_l1
  Testlauf t=135  270 Versuche, mit lambda_l1, gelesen bei t = 135 (Results_repro)
  Testlauf t=270  derselbe Lauf am Ende
Wert = current_best_test_score bei t, Mittel ueber 5 Folds, dann ueber die Seeds; n = Seeds mit allen 5 Folds.
Lesart: A ~ Paper und B ~ Testlauf -> lambda_l1 ist die Ursache; A ~ B ~ Paper -> die 270 Versuche
(best_iter aus allen Versuchen, SMAC-Startdesign); A deutlich unter dem Paper -> noch etwas anderes
(z. B. Paketversionen).

Aufruf:  python compare_ablation.py [--ablation Results_ablation] [--repro Results_repro] [--out ablation_vergleich]
Ausgabe: ablation_vergleich.csv und ablation_vergleich.md
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from compare_with_paper import _markdown

SUITE, TASK, NAME, T = 336, 361076, "wine_quality", 135
METHODS = {"random_search": "Random Grid", "tpe": "TPE", "SMAC": "SMAC"}
FOLDS = 5


def at_t(frame, t):
    """Mittel ueber Folds je Seed, dann ueber Seeds mit allen Folds; (Wert, Anzahl Seeds)."""
    if frame.empty:
        return np.nan, 0
    frame = frame.assign(t=frame.groupby(["seed", "fold"]).cumcount() + 1)
    rows = frame[frame["t"] == t]
    complete = rows.groupby("seed")["fold"].nunique()
    complete = complete[complete >= FOLDS].index
    per_seed = rows[rows["seed"].isin(complete)].groupby("seed")["current_best_test_score"].mean()
    return (per_seed.mean() if len(per_seed) else np.nan), len(per_seed)


def read(pattern, method=None):
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        frame["seed"] = int(path.split("seed_")[1].split(os.sep)[0])
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame[frame["method"] == method] if method and not frame.empty else frame


def build(ablation, repro, reference="paper_reference_values.csv"):
    paper = pd.read_csv(reference)
    paper = paper[(paper["dataset"] == NAME) & (paper["metric"] == "R2") & (paper["t"] == T) & ~paper["doppelt"]]
    name = f"{SUITE}_{TASK}"
    rows = []
    for method, label in METHODS.items():
        ref = paper[paper["method"] == label].iloc[0]
        row = {"method": label, "paper": ref["value"], "paper_ci_low": ref["ci_low"], "paper_ci_high": ref["ci_high"]}
        for run in ("A", "B"):
            row[run], row[f"n_{run}"] = at_t(read(os.path.join(ablation, run, method, "seed_*", "gpboost",
                                                                f"{name}__num_leaves.csv")), T)
        if method == "SMAC":
            test = read(os.path.join(repro, "Results_SMAC", "seed_*", "gpboost", f"{name}.csv"))
        else:
            test = read(os.path.join(repro, "Results", "seed_*", "gpboost", f"{name}__num_leaves.csv"), method)
        row["testlauf_t135"], row["n_testlauf"] = at_t(test, T)
        row["testlauf_t270"], _ = at_t(test, 270)
        rows.append(row)
    return pd.DataFrame(rows)


def main(ablation, repro, out):
    table = build(ablation, repro)
    table.to_csv(f"{out}.csv", index=False)
    show = pd.DataFrame({
        "Methode": table["method"],
        "Paper (Band)": [f"{r.paper:.4f} ({r.paper_ci_low:.4f}–{r.paper_ci_high:.4f})" for r in table.itertuples()],
        "A: 135, ohne l1": [f"{r.A:.4f} n={r.n_A}" if r.n_A else "–" for r in table.itertuples()],
        "B: 135, mit l1": [f"{r.B:.4f} n={r.n_B}" if r.n_B else "–" for r in table.itertuples()],
        "Testlauf t=135": [f"{r.testlauf_t135:.4f} n={r.n_testlauf}" for r in table.itertuples()],
        "Testlauf t=270": [f"{r.testlauf_t270:.4f}" for r in table.itertuples()],
    })
    text = (f"# Ursachen-Laeufe wine_quality: R² bei t = {T}\n\n{_markdown(show)}\n\n"
            "n = Seeds mit allen 5 Folds; – = noch keine vollstaendigen Ergebnisse.\n")
    with open(f"{out}.md", "w") as f:
        f.write(text)
    print(text)
    print(f"Geschrieben: {out}.csv, {out}.md")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="Results_ablation")
    parser.add_argument("--repro", default="Results_repro")
    parser.add_argument("--out", default="ablation_vergleich")
    args = parser.parse_args()
    main(args.ablation, args.repro, args.out)
