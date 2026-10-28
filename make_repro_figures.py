"""Abbildungen 5 bis 10 des Papers (Koster & Sigrist, arXiv:2602.05786v3) fuer die 7 Datensaetze des
Reproduktions-Testlaufs, mit unseren Daten aus Results_repro, und ein Vergleichs-PDF mit Tabelle.

Wie im Paper (plot_maker._plot_single_figure):
  Kurve      Testwert des bisher besten Versuchs (current_best) ueber die Versuche t = 45 ... 135, Mittel ueber
             5 Folds und dann ueber die Seeds; Band = Mittel +- 2 * Seed-Streuung / sqrt(Seeds), gestrichelt
  Kennzahl   R2 (Abb. 5/6), RMSE (Abb. 7/8), Accuracy (Abb. 9), Log Loss (Abb. 10)
  Aussehen   seaborn "muted" in der Reihenfolge von HPO_METHODS (Farben im Paper-PDF nachgeprueft), Default als
             strichpunktierte Linie oder "Default = ..." ausserhalb der Achse, 4 Spalten, Titel = Datensatz,
             y-Achse aus den Kurven mit 5 % Rand, Legende alphabetisch unten
Unterschiede: Deterministisches Gitter = unsere Versuche mit bagging_fraction = 1; das ist genau das 135er-Gitter
des Papers in derselben Reihenfolge (unser Gitter hat zusaetzlich bagging_fraction = 0.5, 270 Punkte).
Hyperband lief im Testlauf nicht (nur in der Legende). 2 statt 20 Seeds. Methoden, bei denen nicht beide Seeds
alle 5 Folds haben (sulfur: Zeitlimit), fehlen in der Kurve; das Feld ist dann rot markiert.

Aufruf:  python make_repro_figures.py [--results Results_repro] [--out repro_figures]
Ausgabe: <out>/abb_<kennzahl>.png, repro_abbildungen.pdf (eine Seite je Abbildung),
         repro_vergleich.pdf (eine Seite je Datensatz: Tabelle links, unsere Kurven rechts)
"""
import argparse
import contextlib
import glob
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from compare_with_paper import METHOD_NAMES, load_results
from make_repro_table import build_table, repro_tasks
from plot_maker import (FONTSIZE, HEIGHT, HPO_METHODS, HPO_METHODS_NAMES, SUBTITLESIZE, TICKSIZE, WIDTH,
                        alpha_grid, avg_linewidth, linewidth_grid)
from utils_plot_maker import plot_confidence_interval, print_default, set_plot_theme

START, END, FOLDS = 45, 135, 5
# Kennzahl -> (Spalte, hoeher ist besser, Abbildung im Paper)
METRICS = {"R2": ("test_score", True, "Abb. 5/6"), "RMSE": ("test_rmse", False, "Abb. 7/8"),
           "Accuracy": ("test_score", True, "Abb. 9"), "Log Loss": ("test_log_loss", False, "Abb. 10")}
LABELS = {"R2": r"$\mathrm{R}^2$", "RMSE": "RMSE", "Accuracy": "Accuracy", "Log Loss": "Log Loss"}
MISSING_COLOR = "#c0392b"


def _read(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        frame["seed"] = int(path.split("seed_")[1].split(os.sep)[0]) if "seed_" in path else -1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_curves(results, suite, task):
    """{methode: Array (Seeds x END) je Kennzahl-Spalte} mit current_best ueber t = 1 ... END; nur Seeds mit 5 Folds."""
    name = f"{suite}_{task}"
    main = _read(os.path.join(results, "Results", "seed_*", "gpboost", f"{name}__num_leaves.csv"))
    smac = _read(os.path.join(results, "Results_SMAC", "seed_*", "gpboost", f"{name}.csv"))
    default = _read(os.path.join(results, "Results_Default_Run", "gpboost", f"{name}.csv"))
    curves, missing = {}, {}
    frames = [(m, main[main["method"] == m]) for m in ("grid_search", "random_search", "tpe", "gp_bo")] \
        if not main.empty else []
    frames.append(("SMAC", smac))
    for method, frame in frames:
        if frame.empty:
            missing[method] = "keine Ergebnisse"
            continue
        if method == "grid_search":
            frame = frame[frame["bagging_fraction"] == 1]   # = Gitter des Papers (135 Punkte)
        per_seed = {column: [] for column in ("test_score", "test_rmse", "test_log_loss")}
        incomplete = []
        for seed, by_seed in frame.groupby("seed"):
            folds = [g.head(END) for _, g in by_seed.groupby("fold")]
            if len(folds) < FOLDS or min(len(g) for g in folds) < END:
                incomplete.append(f"{seed}: {len(folds)}/{FOLDS} Folds")
                continue
            for column in per_seed:
                best = [g[column].cummax() if column == "test_score" else g[column].cummin() for g in folds]
                per_seed[column].append(np.mean([b.to_numpy() for b in best], axis=0))
        if incomplete:
            missing[method] = "Zeitlimit / unvollstaendig (Seed " + "; ".join(incomplete) + ")"
        if per_seed["test_score"]:
            curves[method] = {c: np.array(v) for c, v in per_seed.items()}
    default_values = {}
    if not default.empty:
        for column in ("test_score", "test_rmse", "test_log_loss"):
            default_values[column] = default.groupby("fold")[column].first().mean()
    return curves, missing, default_values


def draw_panel(ax, metric, curves, missing, default_values, title, palette):
    """Ein Feld wie in plot_maker._plot_single_figure."""
    column = METRICS[metric][0]
    x = np.arange(START, END + 1)
    lines = {m: c[column][:, START - 1:END] for m, c in curves.items()}
    values = np.concatenate([v.mean(axis=0) for v in lines.values()]) if lines else np.array([])
    values = values[np.isfinite(values)]
    if len(values):
        low, high = values.min(), values.max()
        pad = (high - low) * 0.05 if high > low else 0.1
        low, high = low - pad, high + pad
    else:
        low, high = (0, 1)
    for method in HPO_METHODS:
        if method not in lines:
            continue
        color = palette[HPO_METHODS.index(method)]
        mean = lines[method].mean(axis=0)
        ax.plot(x, np.clip(mean, low, high), color=color, label=HPO_METHODS_NAMES[HPO_METHODS.index(method)],
                linewidth=avg_linewidth)
        if len(lines[method]) > 1:
            with contextlib.redirect_stdout(io.StringIO()):   # plot_confidence_interval druckt die Breiten
                plot_confidence_interval(ax, x, mean, lines[method].std(axis=0), len(lines[method]), color=color)
    for spine in ax.spines.values():
        spine.set_edgecolor("dimgray")
        spine.set_linewidth(1)
    ax.grid(True, color="lightgray", linewidth=linewidth_grid, alpha=alpha_grid)
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE)
    ax.set_ylim(low, high)
    if column in default_values:
        print_default(default_values[column], metric, low, high, palette, ax, 1, 0)
    ax.set_title(title, fontsize=SUBTITLESIZE)
    ax.set_xticks(np.linspace(START, END, 7))
    ax.set_xticklabels([str(int(t)) for t in np.linspace(START, END, 7)])
    ax.set_xlim(START, END)
    ax.set_box_aspect(1)
    if missing:
        names = ", ".join(METHOD_NAMES[m] for m in missing)
        for spine in ax.spines.values():
            spine.set_edgecolor(MISSING_COLOR)
            spine.set_linewidth(3)
        ax.set_title(f"{title}\nFEHLT: {names}\n(Lauf noch offen oder Zeitlimit)", fontsize=FONTSIZE - 4,
                     color=MISSING_COLOR, fontweight="bold")


def legend_entries(palette):
    """Alle Methoden des Papers (alphabetisch, spaltenweise wie im Paper); Hyperband nur als Eintrag."""
    entries = [(HPO_METHODS_NAMES[i], Line2D([], [], color=palette[i], linewidth=avg_linewidth))
               for i, m in enumerate(HPO_METHODS) if m != "hyperband"]
    entries.append(("Hyperband (nicht im Testlauf)", Line2D([], [], color=palette[HPO_METHODS.index("hyperband")],
                                                             linewidth=avg_linewidth, alpha=0.35)))
    entries.append(("Default", Line2D([], [], color=palette[-1], linestyle="-.", linewidth=avg_linewidth)))
    entries.sort(key=lambda e: e[0])
    ncol = 3
    rows = (len(entries) + ncol - 1) // ncol
    ordered = [entries[r * ncol + c] for c in range(ncol) for r in range(rows) if r * ncol + c < len(entries)]
    return [e[1] for e in ordered], [e[0] for e in ordered], ncol


def paper_figure(metric, tasks, data, palette):
    num_cols = 4
    num_rows = -(-len(tasks) // num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(WIDTH * 2, num_rows * 4.4 + 2.0))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, task) in zip(axes, tasks.iterrows()):
        curves, missing, default = data[(task["suite_id"], task["task_id"])]
        draw_panel(ax, metric, curves, missing, default, task["dataset"], palette)
    for ax in axes[len(tasks):]:
        ax.set_visible(False)
    lines, labels, ncol = legend_entries(palette)
    fig.suptitle(f"{LABELS[metric]} as a function of the number of trials per data set "
                 f"(wie {METRICS[metric][2]}; Testlauf, 2 Seeds)", fontsize=SUBTITLESIZE)
    plt.tight_layout(pad=1.0, rect=(0, 1.6 / (num_rows * 4.4 + 2.0), 1, 1 - 0.5 / (num_rows * 4.4 + 2.0)))
    fig.legend(lines, labels, loc="lower center", ncol=ncol, fontsize=FONTSIZE, bbox_to_anchor=(0.5, 0.01),
               frameon=True, fancybox=True, bbox_transform=fig.transFigure)
    return fig


def _fmt(row):
    if not np.isfinite(row["wir"]):
        return ("–", f"{row['paper']:.4f}" if np.isfinite(row["paper"]) else "–", "–")
    return (f"{row['wir']:.4f}", f"{row['paper']:.4f}", f"{row['differenz']:+.4f}")


def comparison_page(task, table, data, palette):
    metrics = list(table["metric"].unique())
    fig = plt.figure(figsize=(16.5, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), left=0.03, right=0.98, top=0.88, bottom=0.17)
    ax_table = fig.add_subplot(grid[:, 0])
    ax_table.axis("off")
    methods = list(table["method"].unique())
    cells, colors = [], []
    for method in methods:
        row, color = [method], []
        seeds = ""
        for metric in metrics:
            r = table[(table["metric"] == metric) & (table["method"] == method)].iloc[0]
            row += _fmt(r)
            seeds = str(r["seeds"]) if np.isfinite(r["wir"]) else "–"
            note = r["anmerkung"] if isinstance(r["anmerkung"], str) else ""
        row.append(seeds)
        cells.append(row)
        colors.append(MISSING_COLOR if note.startswith("Lauf noch offen") else "black")
    header = ["Methode"] + [f"{LABELS[m]} {h}" for m in metrics for h in ("wir", "Paper", "Diff.")] + ["n"]
    tab = ax_table.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1, 2.0)
    tab.auto_set_column_width([0])
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eef1f4")
        elif c == 0:
            cell.set_text_props(ha="left")
        if r > 0 and colors[r - 1] == MISSING_COLOR:
            cell.set_text_props(color=MISSING_COLOR)
    notes = sorted({n for n in table["anmerkung"].dropna() if n})
    ax_table.text(0.0, 0.08, "wir / Paper bei t = 135 (Deterministic Grid: Gitterende), Mittel über 5 Folds und Seeds; "
                  "n = Seeds.\n" + "\n".join(f"• {n}" for n in notes), transform=ax_table.transAxes, fontsize=10,
                  va="top", color=MISSING_COLOR if any(n.startswith("Lauf noch offen") for n in notes) else "black")
    curves, missing, default = data[(task["suite_id"], task["task_id"])]
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(grid[i, 1])
        draw_panel(ax, metric, curves, missing, default, LABELS[metric], palette)
        ax.title.set_fontsize(FONTSIZE)
        ax.tick_params(labelsize=TICKSIZE - 3)
    lines, labels, ncol = legend_entries(palette)
    fig.legend(lines, labels, loc="lower center", ncol=4, fontsize=FONTSIZE - 3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"{task['dataset']} (Suite {task['suite_id']}, Task {task['task_id']})", fontsize=SUBTITLESIZE + 2)
    return fig


def main(results, out):
    palette = set_plot_theme(len(HPO_METHODS_NAMES) + 1)
    tasks = repro_tasks()
    data = {(t["suite_id"], t["task_id"]): load_curves(results, t["suite_id"], t["task_id"]) for _, t in tasks.iterrows()}
    for key, (_, missing, _) in data.items():
        for method, reason in missing.items():
            print(f"fehlt: {key} {METHOD_NAMES[method]}: {reason}")
    os.makedirs(out, exist_ok=True)
    classification = tasks["suite_id"].isin((334, 337))
    with PdfPages("repro_abbildungen.pdf") as pdf:
        for metric in METRICS:
            subset = tasks[classification] if metric in ("Accuracy", "Log Loss") else tasks[~classification]
            fig = paper_figure(metric, subset, data, palette)
            fig.savefig(os.path.join(out, f"abb_{metric.replace(' ', '_').lower()}.png"), dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
    table = build_table(load_results(results), pd.read_csv("paper_reference_values.csv"), tasks, END)
    with PdfPages("repro_vergleich.pdf") as pdf:
        for _, task in tasks.iterrows():
            part = table[(table["suite_id"] == task["suite_id"]) & (table["task_id"] == task["task_id"])]
            fig = comparison_page(task, part, data, palette)
            fig.savefig(os.path.join(out, f"vergleich_{task['dataset']}.png"), dpi=120)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Geschrieben: {out}/*.png, repro_abbildungen.pdf, repro_vergleich.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="Results_repro")
    parser.add_argument("--out", default="repro_figures")
    args = parser.parse_args()
    main(args.results, args.out)
