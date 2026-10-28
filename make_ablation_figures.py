"""Kurven der Ursachen-Laeufe A/B (run_ablation.py) auf wine_quality im Stil der Abbildungen 5 bis 10 des Papers.

Eine Seite je Methode (Random Grid, TPE, SMAC), vier Kurven:
  A          135 Versuche, ohne lambda_l1 (Results_ablation/A)
  B          135 Versuche, mit lambda_l1 (Results_ablation/B)
  Testlauf   270 Versuche, mit lambda_l1 (Results_repro), gezeigt bis t = 135
  Paper      Wert der Kurve aus Abb. 5/6 bei t = 135 als waagrechte Linie, Band = ci_low ... ci_high
Kurve = current_best_test_score (R2 des bisher besten Versuchs nach Validierung) ueber t = 1 ... 135, Mittel ueber
5 Folds und dann ueber die Seeds; Band = Mittel +- 2 * Seed-Streuung / sqrt(Seeds) (plot_confidence_interval).
Nur Seeds mit allen 5 Folds und mindestens 135 Versuchen je Fold.

Aufruf:  python make_ablation_figures.py [--ablation Results_ablation] [--repro Results_repro]
Ausgabe: ablation_abbildungen.pdf (eine Seite je Methode)
"""
import argparse
import contextlib
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from compare_ablation import FOLDS, METHODS, NAME, SUITE, T, TASK, read
from plot_maker import FONTSIZE, SUBTITLESIZE, TICKSIZE, WIDTH, alpha_grid, avg_linewidth, linewidth_grid
from utils_plot_maker import plot_confidence_interval, set_plot_theme

COLUMN = "current_best_test_score"
# Lauf -> (Beschriftung, Index in der muted-Palette)
RUNS = {"A": ("A: 135 Versuche, ohne lambda_l1", 0), "B": ("B: 135 Versuche, mit lambda_l1", 1),
        "test": ("Testlauf: 270 Versuche, mit lambda_l1", 2)}
PAPER_COLOR = 3


def per_seed_curves(frame):
    """Array (Seeds x T): current_best ueber t = 1 ... T, Mittel ueber die Folds; nur vollstaendige Seeds."""
    curves = []
    for _, by_seed in frame.groupby("seed"):
        folds = [g[COLUMN].to_numpy()[:T] for _, g in by_seed.groupby("fold")]
        if len(folds) >= FOLDS and min(len(f) for f in folds) >= T:
            curves.append(np.mean(folds, axis=0))
    return np.array(curves)


def load(ablation, repro, method):
    name = f"{SUITE}_{TASK}"
    runs = {run: read(os.path.join(ablation, run, method, "seed_*", "gpboost", f"{name}__num_leaves.csv"))
            for run in ("A", "B")}
    if method == "SMAC":
        runs["test"] = read(os.path.join(repro, "Results_SMAC", "seed_*", "gpboost", f"{name}.csv"))
    else:
        runs["test"] = read(os.path.join(repro, "Results", "seed_*", "gpboost", f"{name}__num_leaves.csv"), method)
    return {run: per_seed_curves(frame) if not frame.empty else np.empty((0, T)) for run, frame in runs.items()}


def method_page(label, curves, paper, palette):
    fig, ax = plt.subplots(figsize=(WIDTH * 1.3, WIDTH * 1.3))
    x = np.arange(1, T + 1)
    for run, (text, color_index) in RUNS.items():
        lines = curves[run]
        if not len(lines):
            print(f"fehlt: {label} {run}")
            continue
        color = palette[color_index]
        mean = lines.mean(axis=0)
        ax.plot(x, mean, color=color, linewidth=avg_linewidth,
                label=f"{text} (n = {len(lines)}; t = {T}: {mean[-1]:.4f})")
        if len(lines) > 1:
            with contextlib.redirect_stdout(io.StringIO()):   # plot_confidence_interval druckt die Breiten
                plot_confidence_interval(ax, x, mean, lines.std(axis=0), len(lines), color=color)
    color = palette[PAPER_COLOR]
    ax.axhspan(paper["ci_low"], paper["ci_high"], color=color, alpha=0.2, linewidth=0)
    ax.axhline(paper["value"], color=color, linestyle="--", linewidth=avg_linewidth,
               label=f"Paper bei t = {T}: {paper['value']:.4f} ({paper['ci_low']:.4f}–{paper['ci_high']:.4f})")
    for spine in ax.spines.values():
        spine.set_edgecolor("dimgray")
        spine.set_linewidth(1)
    ax.grid(True, color="lightgray", linewidth=linewidth_grid, alpha=alpha_grid)
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE)
    ax.set_xlim(1, T)
    ax.set_xticks([1, 15, 30, 45, 60, 75, 90, 105, 120, 135])
    # y-Achse ab t = 15, damit die steilen ersten Versuche die Unterschiede am Ende nicht zusammendruecken
    values = [c.mean(axis=0)[14:] for c in curves.values() if len(c)] + [[paper["ci_low"], paper["ci_high"]]]
    low, high = min(np.min(v) for v in values), max(np.max(v) for v in values)
    pad = (high - low) * 0.05
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Anzahl Versuche t", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\mathrm{R}^2$", fontsize=FONTSIZE)
    ax.set_title(f"{label} – {NAME} (Ursachen-Läufe A/B)", fontsize=SUBTITLESIZE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=FONTSIZE - 3, frameon=True, fancybox=True)
    plt.tight_layout()
    return fig


def main(ablation, repro, out="ablation_abbildungen.pdf", reference="paper_reference_values.csv"):
    palette = set_plot_theme(10)
    paper = pd.read_csv(reference)
    paper = paper[(paper["dataset"] == NAME) & (paper["metric"] == "R2") & (paper["t"] == T) & ~paper["doppelt"]]
    with PdfPages(out) as pdf:
        for method, label in METHODS.items():
            fig = method_page(label, load(ablation, repro, method), paper[paper["method"] == label].iloc[0], palette)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="Results_ablation")
    parser.add_argument("--repro", default="Results_repro")
    args = parser.parse_args()
    main(args.ablation, args.repro)
