"""Legt die Ursachen-Laeufe A/B (run_ablation.py) neben den Testlauf und das Paper: wine_quality, R2 und RMSE.

Zeilen: alle Methoden des Papers (Default, Deterministic Grid, GP-BO, Hyperband, Random Grid, SMAC, TPE).
Spalten je Kennzahl:
  Paper           Kurve aus Abb. 5/6 (R2) bzw. 7/8 (RMSE) bei t = 135 (paper_reference_values.csv), dazu das Band
  A               135 Versuche, ohne lambda_l1 (Paper-Einstellung); Gitter: 135er-Gitter des Paper-Codes
  B               135 Versuche, mit lambda_l1
  Testlauf t=135  270 Versuche, mit lambda_l1, gelesen bei t = 135 (Results_repro)
  Testlauf t=270  derselbe Lauf am Ende
Wert = current_best_test_score (R2) bzw. current_best_test_rmse bei t, Mittel ueber 5 Folds, dann ueber die Seeds;
n = Seeds mit allen 5 Folds. Differenz = Wert - Paper; "im Band" = innerhalb des Paper-Bands, erweitert um TOL
(die Bandbreite von Default und Gitter ist im Paper praktisch 0, die Werte stammen aus den Kurven des PDFs).

Methoden, die nicht von lambda_l1 oder der Versuchszahl abhaengen, laufen nicht mehrfach (siehe run_ablation.py):
  Default             params = {} (methods.default_method), eine Zeile je Fold, nur wiederholt -> A = B = Testlauf
  Deterministic Grid  heutiges Gitter ohne lambda_l1, 270 Punkte, keine Versuchszahl -> B = Testlauf (Gitterende);
                      A = eigener Lauf mit dem 135er-Gitter des Papers (--paper_grid)
  Hyperband           R und factor fest, 11 Zeilen je Fold, keine Versuchszahl -> Testlauf = B; das Paper legt die
                      11 Zeilen auf t = 1 ... 135, daher Wert bei "t = 135" und "t = 270" = letzte Zeile
Lesart: A ~ Paper und B ~ Testlauf -> lambda_l1 ist die Ursache; A ~ B ~ Paper -> die 270 Versuche
(best_iter aus allen Versuchen, SMAC-Startdesign); A deutlich neben dem Paper -> noch etwas anderes
(z. B. Paketversionen).

Aufruf:  python compare_ablation.py [--ablation Results_ablation] [--repro Results_repro] [--out ablation_vergleich]
Ausgabe: <out>.csv, <out>.md und ablation_alle_methoden.pdf (eine Seite je Kennzahl)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from compare_with_paper import _markdown

SUITE, TASK, NAME, T = 336, 361076, "wine_quality", 135
METHODS = {"random_search": "Random Grid", "tpe": "TPE", "SMAC": "SMAC"}   # Abbildungen (make_ablation_figures.py)
ALL_METHODS = {"default": "Default", "grid_search": "Deterministic Grid", "gp_bo": "GP-BO", "hyperband": "Hyperband",
               "random_search": "Random Grid", "SMAC": "SMAC", "tpe": "TPE"}
# Kennzahl -> (Spalte, Abbildung im Paper)
METRICS = {"R2": ("current_best_test_score", "Abb. 5/6"), "RMSE": ("current_best_test_rmse", "Abb. 7/8")}
COLUMNS = {"A": "A: 135, ohne l1", "B": "B: 135, mit l1", "testlauf_t135": "Testlauf t=135",
           "testlauf_t270": "Testlauf t=270"}
NOTES = {"default": "keine Abhängigkeit von lambda_l1 oder Versuchszahl: A = B = Testlauf (ein Lauf)",
         "grid_search": "A = 135er-Gitter des Papers; B = Testlauf am Gitterende (Gitter ohne lambda_l1, 270 Punkte)",
         "hyperband": "11 Zeilen je Fold, Wert = letzte Zeile; Testlauf = B (keine Versuchszahl)"}
FOLDS = 5
TOL = 5e-4
GREEN, RED = "#d5f0d5", "#f6d5d5"


def at_t(frame, t, column="current_best_test_score"):
    """Mittel ueber Folds je Seed, dann ueber Seeds mit allen Folds; (Wert, Anzahl Seeds). t=None: letzte Zeile."""
    if frame.empty:
        return np.nan, 0
    frame = frame.assign(t=frame.groupby(["seed", "fold"]).cumcount() + 1)
    rows = frame[frame["t"] == t] if t is not None else frame.groupby(["seed", "fold"]).tail(1)
    complete = rows.groupby("seed")["fold"].nunique()
    complete = complete[complete >= FOLDS].index
    per_seed = rows[rows["seed"].isin(complete)].groupby("seed")[column].mean()
    return (per_seed.mean() if len(per_seed) else np.nan), len(per_seed)


def read(pattern, method=None):
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        frame["seed"] = int(path.split("seed_")[1].split(os.sep)[0]) if "seed_" in path else 0
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame[frame["method"] == method] if method and not frame.empty else frame


def sources(ablation, repro, method):
    """{Spalte: (DataFrame, t)} fuer A, B, Testlauf t=135, Testlauf t=270."""
    name = f"{SUITE}_{TASK}"
    run = {r: read(os.path.join(ablation, r, method, "seed_*", "gpboost", f"{name}__num_leaves.csv")) for r in "AB"}
    if method == "default":
        test = read(os.path.join(repro, "Results_Default_Run", "gpboost", f"{name}.csv"))
        return {c: (test, T) for c in COLUMNS}
    if method == "SMAC":
        test = read(os.path.join(repro, "Results_SMAC", "seed_*", "gpboost", f"{name}.csv"))
    else:
        test = read(os.path.join(repro, "Results", "seed_*", "gpboost", f"{name}__num_leaves.csv"), method)
    if method == "hyperband":
        return {"A": (run["A"], None), "B": (run["B"], None), "testlauf_t135": (run["B"], None),
                "testlauf_t270": (run["B"], None)}
    if method == "grid_search":
        return {"A": (run["A"], T), "B": (test, 270), "testlauf_t135": (test, T), "testlauf_t270": (test, 270)}
    return {"A": (run["A"], T), "B": (run["B"], T), "testlauf_t135": (test, T), "testlauf_t270": (test, 270)}


def build(ablation, repro, reference="paper_reference_values.csv"):
    paper = pd.read_csv(reference)
    paper = paper[(paper["dataset"] == NAME) & (paper["t"] == T) & ~paper["doppelt"]]
    rows = []
    for metric, (column, _) in METRICS.items():
        for method, label in ALL_METHODS.items():
            ref = paper[(paper["metric"] == metric) & (paper["method"] == label)].iloc[0]
            row = {"metric": metric, "method": label, "paper": ref["value"], "paper_ci_low": ref["ci_low"],
                   "paper_ci_high": ref["ci_high"], "anmerkung": NOTES.get(method, "")}
            for key, (frame, t) in sources(ablation, repro, method).items():
                value, n = at_t(frame, t, column)
                row[key], row[f"n_{key}"] = value, n
                row[f"diff_{key}"] = value - ref["value"]
                row[f"im_band_{key}"] = bool(ref["ci_low"] - TOL <= value <= ref["ci_high"] + TOL) \
                    if np.isfinite(value) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _cell(row, key, mark=True):
    if not np.isfinite(row[key]):
        return "–"
    sign = ("🟢 " if row[f"im_band_{key}"] else "🔴 ") if mark else ""
    return f"{sign}{row[key]:.4f} ({row[f'diff_{key}']:+.4f}) n={row[f'n_{key}']}"


def write_markdown(table, path):
    text = f"# Ursachen-Läufe {NAME}: alle Methoden bei t = {T}\n"
    for metric, (_, figure) in METRICS.items():
        part = table[table["metric"] == metric]
        show = pd.DataFrame({"Methode": part["method"],
                             "Paper (Band)": [f"{r.paper:.4f} ({r.paper_ci_low:.4f}–{r.paper_ci_high:.4f})"
                                              for r in part.itertuples()],
                             **{label: [_cell(r, key) for _, r in part.iterrows()] for key, label in COLUMNS.items()}})
        text += f"\n## {metric} (Paper: {figure})\n\n{_markdown(show)}\n"
    text += (f"\nWert (Differenz zum Paper), n = Seeds mit allen 5 Folds; 🟢 = im Paper-Band (± {TOL}), 🔴 = ausserhalb; "
             "– = noch keine vollständigen Ergebnisse. Das Band des Papers gilt für 20 Seeds, unsere Werte haben "
             "2 Seeds und streuen stärker.\n\n" + "\n".join(f"- {ALL_METHODS[m]}: {n}" for m, n in NOTES.items()) + "\n")
    with open(path, "w") as f:
        f.write(text)
    return text


def write_pdf(table, path):
    with PdfPages(path) as pdf:
        for metric, (_, figure) in METRICS.items():
            part = table[table["metric"] == metric].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(16.5, 6.5))
            ax.axis("off")
            cells = [[r["method"], f"{r['paper']:.4f}\n({r['paper_ci_low']:.4f}–{r['paper_ci_high']:.4f})"]
                     + [_cell(r, key, mark=False).replace(" (", "\n(") for key in COLUMNS] for _, r in part.iterrows()]
            tab = ax.table(cellText=cells, colLabels=["Methode", "Paper (Band)", *COLUMNS.values()], loc="upper center",
                           cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(11)
            tab.scale(1, 3.0)
            for (r, c), cell in tab.get_celld().items():
                if r == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#eef1f4")
                elif c >= 2:
                    band = part.loc[r - 1, f"im_band_{list(COLUMNS)[c - 2]}"]
                    if band is True or band is False or band in (0, 1):
                        cell.set_facecolor(GREEN if band else RED)
                elif c == 0:
                    cell.set_text_props(ha="left")
            notes = "\n".join(f"• {ALL_METHODS[m]}: {n}" for m, n in NOTES.items())
            ax.text(0.0, 0.02, f"Wert (Differenz zum Paper), n = Seeds mit allen 5 Folds; grün = im Paper-Band "
                    f"(± {TOL}), rot = ausserhalb, – = fehlt. Band des Papers für 20 Seeds, unsere Werte 2 Seeds.\n"
                    + notes, transform=ax.transAxes, fontsize=10, va="bottom")
            fig.suptitle(f"{NAME}: {metric} bei t = {T}, Ursachen-Läufe A/B, Testlauf und Paper ({figure})",
                         fontsize=15)
            pdf.savefig(fig)
            plt.close(fig)


def main(ablation, repro, out, pdf="ablation_alle_methoden.pdf"):
    table = build(ablation, repro)
    table.to_csv(f"{out}.csv", index=False)
    text = write_markdown(table, f"{out}.md")
    write_pdf(table, pdf)
    print(text)
    print(f"Geschrieben: {out}.csv, {out}.md, {pdf}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="Results_ablation")
    parser.add_argument("--repro", default="Results_repro")
    parser.add_argument("--out", default="ablation_vergleich")
    args = parser.parse_args()
    main(args.ablation, args.repro, args.out)
