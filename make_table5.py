"""Table 5 des Papers (Koster & Sigrist, arXiv:2602.05786v3, S. 7) aus unseren Ergebnissen: normalisierte
Scores, relative Differenzen zum besten Score und Raenge, gemittelt ueber die Aufgaben der Suiten 334-337,
Strategie num_leaves, im selben Format wie das Paper und als Differenz zu paper_table5.csv.

Definitionen (Paper Abschnitt 2.5 und Paper-Code a19b4ff: utils_plot_maker.get_statistics/normalize_scores,
plot_maker.create_table; Begruendung in REPRO_PLAN.md, Abschnitt "Table 5"):
  s[k,m,l,t]  Testwert des bisher besten Versuchs (current_best_*) nach t Versuchen, Mittel ueber 5 Folds;
              k Aufgabe, m Methode, l Seed. Hyperband: seine Ergebniszeilen (Rungs), Endwert = letzte Zeile.
              Default: ein Wert pro Aufgabe (kein Seed, keine Versuche).
  Budget T    Endwert bei t = T (Paper: T = 135). Deterministisches Gitter: Zeile T des Gitters.
  norm        pro Aufgabe ADTM: Pool P_k = alle s[k,m,l,t] mit t <= T der sechs HPO-Methoden (alle Seeds) plus
              der Default-Wert, so oft wie im Paper-Code (T-mal je HPO-Methode, --pool code; ohne Default:
              --pool text). R2/Accuracy: (s - Q10(P_k)) / (max P_k - Q10(P_k)), unten bei 0 abgeschnitten.
              RMSE/Log Loss: (s - min P_k) / (Q90(P_k) - min P_k), oben bei 1 abgeschnitten.
              Tabelle: Mittel ueber Aufgaben und Seeds des normalisierten Endwerts.
  rel         pro Aufgabe: s[k,m] = Mittel ueber Seeds (roh, nicht normalisiert), best_k = bester Wert ueber die
              Methoden, |s[k,m] - best_k| / |best_k|; Tabelle: Mittel ueber Aufgaben.
  rank        pro Aufgabe Rang 1 (bester) bis 7 von s[k,m] unter den Methoden; Tabelle: Mittel ueber Aufgaben.
              Gleichstand wie im Paper-Code: stabile Sortierung in der Reihenfolge der Spalten, Default zuletzt.
Nur Aufgaben, fuer die alle vorhandenen Methoden einen Wert haben, gehen in rel und rank ein; ein Seed zaehlt
nur mit allen 5 Folds und mindestens T Versuchen.

Ergebnis-Layouts (beide werden erkannt):
  neu (run_ablation.py, run_default.py):  <root>/<methode>/seed_<seed>/gpboost/<suite>_<task>__num_leaves.csv
                                          <root>/default/gpboost/<suite>_<task>.csv
  Testlauf (make_repro_jobs.py):          <root>/Results/seed_*/gpboost/*__num_leaves.csv, Results_SMAC/...,
                                          Results_Default_Run/gpboost/*.csv

Zwischenstand waehrend der Laeufe:
  --seeds 1-5     nur diese Seeds (Positionen in der Seed-Liste des Papers, utils_modelling.seeds; z. B. 1-5,
                  1-10, 3,7; Zahlen > 20 gelten als Seed-Werte). Default zaehlt immer.
  --partial       nur Aufgaben, fuer die jede Methode alle verlangten Seeds vollstaendig hat (--seeds, sonst alle
                  Seeds, die im Ordner vorkommen); Methoden ohne eine einzige vollstaendige Aufgabe fallen weg. Im
                  Kopf steht, wie viele Aufgaben, Seeds und Methoden eingeflossen sind.

Aufruf:  python make_table5.py                                   # Paper=Results_paper:135, L1_270=Results_l1_270:135,270
         python make_table5.py --seeds 1-5 --partial --out table5_welle1
         python make_table5.py --run Testlauf=Results_repro:135 --out table5_testlauf
Ausgabe: <out>.csv (alle Zahlen, lang), <out>.md (Tabellen wie im Paper und Differenzen), <out>.pdf,
         <out>_datensaetze.csv und .md (je Aufgabe x Methode: Test-Score R2/Accuracy bei T, Mittel ueber die
         vorhandenen Seeds, mit Zahl der Seeds)
"""
import argparse
import glob
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from task_types import is_classification

METHODS = ["Default", "Deterministic Grid", "GP-BO", "Hyperband", "Random Grid", "SMAC", "TPE"]   # Spalten Table 5
HPO = [m for m in METHODS if m != "Default"]
NAMES = {"default": "Default", "grid_search": "Deterministic Grid", "gp_bo": "GP-BO", "hyperband": "Hyperband",
         "random_search": "Random Grid", "SMAC": "SMAC", "tpe": "TPE"}
# Kennzahl: (Spalte, Aufgabenart Klassifikation?, hoeher besser?)
METRICS = {"R2": ("current_best_test_score", False, True), "RMSE": ("current_best_test_rmse", False, False),
           "Accuracy": ("current_best_test_score", True, True), "Log Loss": ("current_best_test_log_loss", True, False)}
BLOCKS = ["norm", "rel", "rank"]
SUITES = (334, 335, 336, 337)
FOLDS = 5
FILE = re.compile(r"(\d{3})_(\d+)(?:__num_leaves)?\.csv$")


def old_tasks():
    """(suite, task) -> Name fuer die 59 Aufgaben der Suiten 334-337."""
    import json
    names = {}
    for suite in SUITES:
        ids = np.load(f"task_indices/{suite}_task_indices.npy")
        with open(f"task_indices/{suite}_task_names.json") as handle:
            names.update({(suite, int(t)): n for t, n in zip(ids, json.load(handle))})
    return names


def load(root):
    """Lange Tabelle: suite, task, method, seed, fold, t und die current_best-Spalten."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        match = FILE.search(os.path.basename(path))
        if not match or "smac3_output" in path or int(match.group(1)) not in SUITES:
            continue
        frame = pd.read_csv(path)
        if "method" not in frame or "fold" not in frame:
            continue
        if "try_num_leaves" in frame:
            frame = frame[frame["try_num_leaves"].astype(str).isin(["True", "1", "1.0"])]
        seed = re.search(r"seed_(\d+)", path)
        frame = frame.assign(suite=int(match.group(1)), task=int(match.group(2)), datei=path,
                             seed=int(seed.group(1)) if seed else -1,
                             method=frame["method"].map(NAMES).fillna(frame["method"]))
        frame["t"] = frame.groupby(["fold", "method"]).cumcount() + 1
        frames.append(frame.reindex(columns=["suite", "task", "method", "seed", "fold", "t", "datei",
                                             *sorted({c for c, _, _ in METRICS.values()})]))
    if not frames:
        return pd.DataFrame(columns=["suite", "task", "method", "seed", "fold", "t"])
    data = pd.concat(frames, ignore_index=True)
    data.loc[data["method"] == "Default", "seed"] = -1
    return data.drop_duplicates(subset=["suite", "task", "method", "seed", "fold", "t"], keep="first")


def parse_seeds(spec):
    """'1-5' -> die ersten fuenf Paper-Seeds; '3,7' Positionen; Zahlen > 20 sind Seed-Werte."""
    from utils_modelling import seeds as paper_seeds
    out = []
    for part in str(spec).split(","):
        start, _, end = part.strip().partition("-")
        for n in range(int(start), int(end or start) + 1):
            out.append(paper_seeds[n - 1] if 1 <= n <= len(paper_seeds) else n)
            if n > len(paper_seeds) and end:
                raise ValueError(f"Bereich {part}: nur Positionen 1-{len(paper_seeds)}")
    return list(dict.fromkeys(out))


def select_seeds(data, seeds):
    """Nur diese Seeds (Default mit Seed -1 bleibt)."""
    if seeds is None:
        return data
    return data[(data["method"] == "Default") | data["seed"].isin(list(seeds))]


def curves(data, column, budget):
    """{(suite, task): {method: array (Seeds x Versuche)}} mit Fold-Mittel; Versuche bis zum Budget.
    Seeds ohne alle 5 Folds oder mit zu wenigen Versuchen fallen weg (gezaehlt in skipped)."""
    out, skipped = {}, []
    for (suite, task, method, seed), group in data.groupby(["suite", "task", "method", "seed"]):
        wide = group.pivot_table(index="t", columns="fold", values=column, aggfunc="first")
        if wide.shape[1] < FOLDS or wide.isna().all(axis=None):
            skipped.append((suite, task, method, seed, f"{wide.shape[1]} Folds"))
            continue
        wide = wide.dropna()
        if method == "Default":
            curve = wide.mean(axis=1).to_numpy()[:1]
        elif method == "Hyperband":
            curve = wide.mean(axis=1).to_numpy()
        else:
            if len(wide) < budget:
                skipped.append((suite, task, method, seed, f"{len(wide)} < {budget} Versuche"))
                continue
            curve = wide.mean(axis=1).to_numpy()[:budget]
        out.setdefault((suite, task), {}).setdefault(method, []).append(curve)
    return {key: {m: np.vstack(c) for m, c in methods.items()} for key, methods in out.items()}, skipped


def normalize(task_curves, higher, budget, pool="code", quantiles=(10, 90)):
    """ADTM-Normalisierung einer Aufgabe; gibt {method: normalisierte Endwerte je Seed} zurueck."""
    values = [c.ravel() for m, c in task_curves.items() if m != "Default"]
    if pool == "code" and "Default" in task_curves:
        n_hpo = sum(m != "Default" for m in task_curves)
        values.append(np.repeat(task_curves["Default"].ravel()[0], budget * n_hpo))
    values = np.concatenate(values)
    values = values[np.isfinite(values)]
    if higher:
        low, high = np.percentile(values, quantiles[0]), values.max()
        scale = lambda x: np.maximum((x - low) / (high - low), 0)
    else:
        low, high = values.min(), np.percentile(values, quantiles[1])
        scale = lambda x: np.minimum((x - low) / (high - low), 1)
    return {m: scale(c[:, -1]) for m, c in task_curves.items()}


def ranks(final, higher):
    """Raenge 1..n wie plot_maker.create_table: stabile Sortierung, Spaltenreihenfolge, Default zuletzt."""
    order = [m for m in HPO if m in final] + (["Default"] if "Default" in final else [])
    values = np.array([final[m] for m in order])
    position = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    rank = len(order) - position if higher else position + 1
    return dict(zip(order, rank.astype(float)))


def complete_subset(task_curves, n_seeds):
    """--partial: (Methoden, Aufgaben), bei denen jede Methode n_seeds Seeds hat (Default: einen Wert).
    Methoden ohne eine vollstaendige Aufgabe fallen weg."""
    full = lambda m, c: m == "Default" or c.shape[0] >= n_seeds
    methods = [m for m in METHODS if any(m in c and full(m, c[m]) for c in task_curves.values())]
    if not set(methods) - {"Default"}:
        return [], {}
    tasks = {k: {m: c[m] for m in methods} for k, c in task_curves.items()
             if all(m in c and full(m, c[m]) for m in methods)}
    return methods, tasks


def hpo_seeds(data):
    return sorted(set(data.loc[data["method"] != "Default", "seed"]))


def table5(data, budget, pool="code", quantiles=(10, 90), partial=False, seeds=None):
    """Eine Zeile je (block, metric, method) mit Wert, Zahl der Aufgaben und Seeds.
    partial: nur Aufgaben, in denen jede Methode alle Seeds (seeds, sonst alle vorkommenden) vollstaendig hat."""
    rows, notes = [], []
    data = select_seeds(data, seeds)
    wanted = len(seeds) if seeds is not None else len(hpo_seeds(data))
    for metric, (column, classification, higher) in METRICS.items():
        part = data[[is_classification(s) == classification for s in data["suite"]]]
        if part.empty or column not in part:
            continue
        task_curves, skipped = curves(part, column, budget)
        notes += [(metric, *s) for s in skipped]
        if partial:
            present, complete = complete_subset(task_curves, wanted)
            if not complete:
                continue
        else:
            present = sorted({m for c in task_curves.values() for m in c}, key=METHODS.index)
            complete = {k: c for k, c in task_curves.items() if all(m in c for m in present)}
        norm, rel, rank = {m: [] for m in present}, {m: [] for m in present}, {m: [] for m in present}
        for key, c in complete.items():
            for m, values in normalize(c, higher, budget, pool, quantiles).items():
                norm[m].append(values.mean())
            final = {m: c[m][:, -1].mean() for m in present}
            best = max(final.values()) if higher else min(final.values())
            for m in present:
                rel[m].append(abs(final[m] - best) / abs(best) if best != 0 else 0.0)
            for m, r in ranks(final, higher).items():
                rank[m].append(r)
        seeds = {m: min((complete[k][m].shape[0] for k in complete), default=0) for m in present}
        for block, values in (("norm", norm), ("rel", rel), ("rank", rank)):
            for m in present:
                rows.append({"block": block, "metric": metric, "method": m, "value": np.mean(values[m]) if values[m]
                             else np.nan, "tasks": len(complete), "tasks_with_data": len(task_curves),
                             "min_seeds": seeds[m]})
    columns = ["block", "metric", "method", "value", "tasks", "tasks_with_data", "min_seeds"]
    return pd.DataFrame(rows, columns=columns), notes


def per_dataset(data, budget, names=None, seeds=None):
    """Je Aufgabe x Methode: Test-Score (R2 bzw. Accuracy) bei T, Mittel ueber die vorhandenen vollstaendigen
    Seeds; Hyperband: Endwert, Default: sein Wert."""
    names = names or {}
    rows = []
    data = select_seeds(data, seeds)
    if data.empty or "current_best_test_score" not in data:
        return pd.DataFrame(columns=["suite", "task", "name", "kennzahl", "method", "score", "seeds"])
    task_curves, _ = curves(data, "current_best_test_score", budget)
    for (suite, task), c in sorted(task_curves.items()):
        for m in [m for m in METHODS if m in c]:
            rows.append(dict(suite=suite, task=task, name=names.get((suite, task), ""),
                             kennzahl="Accuracy" if is_classification(suite) else "R2", method=m,
                             score=float(c[m][:, -1].mean()), seeds=0 if m == "Default" else c[m].shape[0]))
    return pd.DataFrame(rows)


def dataset_markdown(frame, title):
    if frame.empty:
        return f"## {title}\n\nKeine vollstaendigen Kurven.\n"
    lines = [f"## {title}", "", "Test-Score (R² bzw. Accuracy, höher ist besser) bei T, Mittel über die vorhandenen "
             "Seeds (Spalte Seeds: kleinste Zahl über die HPO-Methoden; fett: bester Wert der Zeile).", "",
             "| Aufgabe | | Seeds | " + " | ".join(METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
    for (suite, task, name, metric), group in frame.groupby(["suite", "task", "name", "kennzahl"], sort=False):
        values = dict(zip(group["method"], group["score"]))
        hpo = group.loc[group["method"] != "Default", "seeds"]
        best = max(values.values())
        cells = [("**%.4f**" if values[m] == best else "%.4f") % values[m] if m in values else "–" for m in METHODS]
        lines.append(f"| {name or task} | {metric} | {int(hpo.min()) if len(hpo) else 0} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def coverage_text(table, data, seeds, partial):
    """Kopfzeile: wie viele Aufgaben, Seeds und Methoden eingeflossen sind."""
    used = select_seeds(data, seeds)
    seed_list = list(seeds) if seeds is not None else hpo_seeds(used)
    tasks = table.dropna(subset=["value"]).groupby("metric")["tasks"].max()
    n_reg = int(tasks.get("R2", tasks.get("RMSE", 0)))
    n_cls = int(tasks.get("Accuracy", tasks.get("Log Loss", 0)))
    methods = [m for m in METHODS if m in set(table.dropna(subset=["value"])["method"])]
    mode = "TEILAUSWERTUNG (--partial)" if partial else "Auswertung"
    return (f"{mode}: {n_reg + n_cls} von 59 Aufgaben ({n_reg} Regression, {n_cls} Klassifikation), "
            f"{len(seed_list)} Seeds ({', '.join(map(str, seed_list))}), {len(methods)} Methoden "
            f"({', '.join(methods)})")


def paper_table(path="paper_table5.csv"):
    paper = pd.read_csv(path).melt(id_vars=["block", "metric"], var_name="method", value_name="paper")
    return paper


def compare(ours, paper):
    table = paper.merge(ours, on=["block", "metric", "method"], how="left")
    table["diff"] = table["value"] - table["paper"]
    return table


def wide(table, column):
    frame = table.pivot_table(index=["block", "metric"], columns="method", values=column, aggfunc="first")
    order = [(b, m) for b in BLOCKS for m in METRICS]
    return frame.reindex(index=[i for i in order if i in frame.index], columns=METHODS)


def _fmt(value, sign=False):
    return "–" if pd.isna(value) else (f"{value:+.3f}" if sign else f"{value:.3f}")


def markdown(table, title):
    lines = [f"## {title}", ""]
    coverage = table.dropna(subset=["value"]).groupby("metric")[["tasks", "min_seeds"]].agg(["min", "max"])
    for caption, column, sign in (("Unsere Werte", "value", False), ("Paper (Table 5)", "paper", False),
                                  ("Differenz (wir − Paper)", "diff", True)):
        frame = wide(table, column)
        lines += [f"**{caption}**", "", "| | | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for (block, metric), row in frame.iterrows():
            label = {"norm": "norm", "rel": "Rel. diff.", "rank": "Rank"}[block]
            lines.append(f"| {label} | {metric} | " + " | ".join(_fmt(v, sign) for v in row) + " |")
        lines.append("")
    if not coverage.empty:
        lines.append("Abdeckung: " + "; ".join(
            f"{metric}: {int(r[('tasks', 'max')])} Aufgaben, mind. {int(r[('min_seeds', 'min')])} Seeds je Methode"
            for metric, r in coverage.iterrows()))
        lines.append("")
    return "\n".join(lines)


def pdf_page(pdf, table, title):
    fig, axes = plt.subplots(2, 1, figsize=(13, 15), gridspec_kw={"height_ratios": [1.6, 1]})
    for ax, column, caption in ((axes[0], "value", "Unsere Werte (in Klammern: Paper)"),
                                (axes[1], "diff", "Differenz wir − Paper")):
        ax.axis("off")
        ours, paper = wide(table, "value"), wide(table, "paper")
        frame = wide(table, column)
        cells, colors = [], []
        for (block, metric), row in frame.iterrows():
            label = {"norm": "norm", "rel": "Rel. diff.", "rank": "Rank"}[block]
            if column == "value":
                cells.append([label, metric] + [f"{_fmt(row[m])}\n({_fmt(paper.loc[(block, metric), m])})"
                                                for m in METHODS])
            else:
                cells.append([label, metric] + [_fmt(row[m], sign=True) for m in METHODS])
            colors.append(["white", "white"] + [_color(block, metric, row[m], column) for m in METHODS])
        tab = ax.table(cellText=cells, cellColours=colors, colLabels=["", "", *[m.replace(" ", "\n") for m in METHODS]],
                       loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 2.4 if column == "value" else 1.6)
        for (r, c), cell in tab.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold")
        ax.set_title(caption, fontsize=12)
    coverage = table.dropna(subset=["value"]).groupby("metric")["tasks"].max().to_dict()
    fig.suptitle(f"{title}\nAufgaben mit allen Methoden: " + ", ".join(f"{k} {int(v)}" for k, v in coverage.items()),
                 fontsize=13)
    pdf.savefig(fig)
    plt.close(fig)


def _color(block, metric, diff, column):
    """Differenz einfaerben: gruen = besser als das Paper, rot = schlechter (nur norm, auf der Differenzseite)."""
    if column != "diff" or pd.isna(diff) or block == "rank":
        return "white"
    higher = METRICS[metric][2] if block == "norm" else False
    better = diff > 0 if higher else diff < 0
    return "#e3f2e1" if better else "#f8e1e1"


def parse_run(spec):
    """'Name=Ordner:135,270' -> (Name, Ordner, [135, 270])."""
    name, rest = spec.split("=", 1)
    folder, _, budgets = rest.partition(":")
    return name, folder, [int(b) for b in (budgets or "135").split(",")]


def main(runs, out, pool="code", quantiles=(10, 90), paper_path="paper_table5.csv", seeds=None, partial=False):
    paper = paper_table(paper_path)
    tasks = old_tasks()
    long, sections, datasets, dataset_sections = [], [], [], []
    with PdfPages(f"{out}.pdf", keep_empty=False) as pdf:
        for name, folder, budgets in runs:
            data = load(folder)
            if data.empty:
                sections.append(f"## {name}\n\nKeine Ergebnisse in {folder}.\n")
                print(f"{name}: keine Ergebnisse in {folder}")
                continue
            unknown = {(s, t) for s, t in zip(data["suite"], data["task"])} - set(tasks)
            for budget in budgets:
                ours, notes = table5(data, budget, pool, quantiles, partial, seeds)
                table = compare(ours, paper)
                title = f"{name} ({folder}), Budget T = {budget}"
                head = coverage_text(table, data, seeds, partial)
                table.insert(0, "run", name)
                table.insert(1, "budget", budget)
                long.append(table)
                per_task = per_dataset(data, budget, tasks, seeds)
                datasets.append(per_task.assign(run=name, budget=budget))
                dataset_sections.append(dataset_markdown(per_task, title))
                # Kopfzeile direkt unter dem Titel
                sections.append(markdown(table, title).replace("\n\n", f"\n\n**{head}**\n\n", 1) + (
                    f"Nicht gewertet (unvollstaendig): {len(notes)} Kurven, z. B. {notes[:3]}\n\n" if notes else "")
                    + (f"Aufgaben ausserhalb der 59: {sorted(unknown)[:5]}\n\n" if unknown else ""))
                print(f"\n=== {title} ===\n{head}")
                if table["value"].isna().all():
                    print("Noch keine Aufgabe mit allen verlangten Seeds vollstaendig (siehe watch_paper.py); "
                          f"Werte je Datensatz trotzdem in {out}_datensaetze.*")
                    continue
                pdf_page(pdf, table, "\n".join(textwrap.fill(x, 110) for x in (title, head)))
                print(wide(table, "value").round(3).to_string())
                print("Differenz zum Paper:")
                print(wide(table, "diff").round(3).to_string())
    if long:
        pd.concat(long, ignore_index=True).to_csv(f"{out}.csv", index=False)
    if datasets:
        pd.concat(datasets, ignore_index=True).to_csv(f"{out}_datensaetze.csv", index=False)
    choice = (f"Seeds: {', '.join(map(str, seeds))}. " if seeds is not None else "") + \
        ("Nur vollständige Aufgaben und Methoden (--partial). " if partial else "")
    header = (f"# Table 5: unsere Läufe vs. Paper\n\nNormalisierung: Pool `{pool}`, Quantile {quantiles}. {choice}"
              "Definitionen: siehe make_table5.py und REPRO_PLAN.md.\n\n")
    with open(f"{out}.md", "w") as handle:
        handle.write(header + "\n".join(sections))
    with open(f"{out}_datensaetze.md", "w") as handle:
        handle.write(f"# Ergebnisse je Datensatz\n\n{choice}\n\n" + "\n".join(dataset_sections))
    written = [f for f in (f"{out}.csv", f"{out}.md", f"{out}.pdf", f"{out}_datensaetze.csv", f"{out}_datensaetze.md")
               if os.path.exists(f)]
    print(f"\nGeschrieben: {', '.join(written)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=parse_run,
                        help="Name=Ordner:Budget[,Budget]; mehrfach. Standard: Paper=Results_paper:135 und "
                             "L1_270=Results_l1_270:135,270")
    parser.add_argument("--out", default="table5")
    parser.add_argument("--pool", choices=("code", "text"), default="code",
                        help="code: Default im Pool wie im Paper-Code (Standard); text: nur die HPO-Methoden")
    parser.add_argument("--quantiles", type=float, nargs=2, default=(10, 90))
    parser.add_argument("--seeds", type=parse_seeds, help="z. B. 1-5: nur die ersten fuenf Paper-Seeds")
    parser.add_argument("--partial", action="store_true",
                        help="nur Aufgaben und Methoden mit allen (verlangten) Seeds vollstaendig")
    args = parser.parse_args()
    main(args.run or [parse_run("Paper=Results_paper:135"), parse_run("L1_270=Results_l1_270:135,270")],
         args.out, args.pool, tuple(args.quantiles), seeds=args.seeds, partial=args.partial)
