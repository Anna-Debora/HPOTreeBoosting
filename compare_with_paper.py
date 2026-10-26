"""Vergleicht die Ergebnisse des Reproduktions-Testlaufs (siehe make_repro_jobs.py) mit den Werten aus dem
Paper (Koster & Sigrist, Abbildungen 5 bis 10, ausgelesen mit extract_paper_figures.py nach
paper_reference_values.csv).

Was verglichen wird, pro Aufgabe, Methode und Kennzahl (Regression: R2 und RMSE, Klassifikation: Accuracy
und Log Loss) beim Budget t (Standard 135 = Budget des Papers):
  unser Wert   = Testwert des bisher besten Versuchs nach t Versuchen ("current_best"), gemittelt ueber die
                 5 Folds und danach ueber unsere Seeds (wie die Kurven im Paper, dort 20 Seeds)
  Paper-Wert   = Kurve im Paper bei t
  Seed-Streuung eines einzelnen Laufs = (obere - untere Grenze des 95-%-Bandes im Paper) / 3.92 * sqrt(20)
  z            = (unser Wert - Paper-Wert) / (Seed-Streuung * sqrt(1/unsere_Seeds + 1/20))
                 |z| <= 2: im Rahmen des Zufalls; |z| > 2: auffaellig (fuer Default und das deterministische
                 Gitter gibt es keine Seed-Streuung, dort nur die relative Abweichung)
Sonderfaelle:
  - Deterministisches Gitter: im Paper 135 Konfigurationen, bei uns 270 (zusaetzlich bagging_fraction).
    Verglichen wird jeweils das Ende des Gitters (Paper t=135 mit unserem t=270).
  - Default: konstant, keine Seeds.
Ausserdem pro Aufgabe die Rangfolge der Methoden (Spearman-Korrelation unserer Rangfolge mit der des Papers)
und die mittleren Raenge ueber alle Aufgaben.

Erwartete Unterschiede (der Code ist nicht mehr genau der des Papers): lambda_l1 ist zusaetzlich im
Suchraum, Zufallsgitter und deterministisches Gitter sind groesser (270), GPBoost 1.7.4, und die
Zufallszahlen haengen von der Laenge der Laeufe ab. Kleine systematische Abweichungen sind also normal;
gross waeren z. B. ein anderer Gewinner auf vielen Aufgaben oder Abweichungen weit ueber der Seed-Streuung.

Aufruf:  python compare_with_paper.py --results Results_repro [--t 135] [--out repro_vergleich]
Ausgabe: repro_vergleich.csv (alle Zahlen) und repro_vergleich.md (Zusammenfassung zum Lesen)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

METHOD_NAMES = {"grid_search": "Deterministic Grid", "random_search": "Random Grid", "tpe": "TPE",
                "gp_bo": "GP-BO", "SMAC": "SMAC", "default": "Default"}
METRICS = {False: [("R2", "test_score", True), ("RMSE", "test_rmse", False)],
           True: [("Accuracy", "test_score", True), ("Log Loss", "test_log_loss", False)]}
GRID_END = 270


def _read(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        frame["datei"] = path
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_results(results, library="gpboost", strategy="num_leaves"):
    """Eine Zeile je (Aufgabe, Seed, Fold, Methode, Versuch t) mit den current_best-Werten."""
    main = _read(os.path.join(results, "Results", "seed_*", library, f"*__{strategy}.csv"))
    smac = _read(os.path.join(results, "Results_SMAC", "seed_*", library, "*.csv"))
    default = _read(os.path.join(results, "Results_Default_Run", library, "*.csv"))
    parts = []
    for frame, best in ((main, True), (smac, True), (default, False)):
        if frame.empty:
            continue
        frame = frame.copy()
        if "try_num_leaves" in frame and strategy == "num_leaves":
            frame = frame[frame["try_num_leaves"].astype(str).isin(["True", "1", "1.0"])]
        name = frame["datei"].str.extract(r"(\d+)_(\d+)(?:__\w+)?\.csv$")
        frame["suite_id"], frame["task_id"] = name[0].astype(int), name[1].astype(int)
        seed = frame["datei"].str.extract(r"seed_(\d+)")[0]
        frame["seed"] = pd.to_numeric(seed, errors="coerce").fillna(-1).astype(int)
        frame["method"] = frame["method"].map(METHOD_NAMES).fillna(frame["method"])
        frame["t"] = frame.groupby(["datei", "fold", "method"]).cumcount() + 1
        for column in ("test_score", "test_rmse", "test_log_loss"):
            source = f"current_best_{column}" if best and f"current_best_{column}" in frame else column
            frame[f"best_{column}"] = frame[source] if source in frame else np.nan
        parts.append(frame[["suite_id", "task_id", "seed", "fold", "method", "t",
                            "best_test_score", "best_test_rmse", "best_test_log_loss"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def our_values(results, t):
    """Mittel ueber Folds, dann ueber Seeds, beim Budget t (deterministisches Gitter: Gitterende)."""
    rows = []
    for (suite, task, method), group in results.groupby(["suite_id", "task_id", "method"]):
        budget = min(GRID_END, group["t"].max()) if method == "Deterministic Grid" else min(t, group["t"].max())
        at_t = group[group["t"] == budget]
        per_seed = at_t.groupby("seed")[["best_test_score", "best_test_rmse", "best_test_log_loss"]].mean()
        rows.append({"suite_id": suite, "task_id": task, "method": method, "budget": budget,
                     "seeds": int((per_seed.index >= 0).sum()) or 1, "folds": int(at_t["fold"].nunique()),
                     **per_seed.mean().to_dict()})
    return pd.DataFrame(rows)


def compare(ours, paper, tasks, t):
    rows = []
    for _, our in ours.iterrows():
        info = tasks.loc[(our["suite_id"], our["task_id"])]
        classification = info["suite_id"] in (334, 337)
        for metric, column, higher_better in METRICS[classification]:
            ref = paper[(paper["metric"] == metric) & (paper["dataset"] == info["name"]) &
                        (paper["method"] == our["method"]) & (paper["t"] == t) & ~paper["doppelt"]]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            value = our[f"best_{column}"]
            seed_sd = (ref["ci_high"] - ref["ci_low"]) / 3.92 * np.sqrt(20)
            z = (value - ref["value"]) / (seed_sd * np.sqrt(1 / our["seeds"] + 1 / 20)) if seed_sd > 0 else np.nan
            rows.append({"suite_id": our["suite_id"], "task_id": our["task_id"], "dataset": info["name"],
                         "metric": metric, "method": our["method"], "budget_wir": our["budget"],
                         "seeds_wir": our["seeds"], "folds_wir": our["folds"], "wir": value, "paper": ref["value"],
                         "differenz": value - ref["value"],
                         "rel_differenz": (value - ref["value"]) / abs(ref["value"]) if ref["value"] else np.nan,
                         "besser_als_paper": (value > ref["value"]) == higher_better,
                         "seed_streuung": seed_sd, "z": z,
                         "urteil": ("auffaellig" if abs(z) > 2 else "ok") if np.isfinite(z) else
                                   ("auffaellig" if abs((value - ref["value"]) / ref["value"]) > 0.02 else "ok")})
    return pd.DataFrame(rows)


def rank_summary(table):
    """Spearman-Korrelation der Methoden-Rangfolge (wir vs. Paper) je Aufgabe und Kennzahl, mittlere Raenge."""
    rows, ranks = [], []
    for (dataset, metric), group in table.groupby(["dataset", "metric"]):
        if len(group) < 3:
            continue
        higher = metric in ("R2", "Accuracy")
        ours = group["wir"].rank(ascending=not higher)
        paper = group["paper"].rank(ascending=not higher)
        rows.append({"dataset": dataset, "metric": metric, "methoden": len(group),
                     "spearman": float(np.corrcoef(ours, paper)[0, 1]),
                     "bester_wir": group.loc[ours.idxmin(), "method"],
                     "bester_paper": group.loc[paper.idxmin(), "method"]})
        ranks += [{"metric": metric, "method": m, "rang_wir": a, "rang_paper": b}
                  for m, a, b in zip(group["method"], ours, paper)]
    mean_ranks = pd.DataFrame(ranks).groupby(["metric", "method"])[["rang_wir", "rang_paper"]].mean()
    return pd.DataFrame(rows), mean_ranks


def _markdown(frame, index=False):
    """Kleine Markdown-Tabelle (ohne Zusatzpaket)."""
    frame = frame.reset_index() if index else frame
    cells = [[f"{v:.4g}" if isinstance(v, float) else str(v) for v in row] for row in frame.itertuples(index=False)]
    header = [str(c) for c in frame.columns]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(row) + " |" for row in cells])


def write_markdown(table, per_task, mean_ranks, t, path):
    lines = [f"# Vergleich Testlauf mit dem Paper (Budget t = {t})", ""]
    ok = (table["urteil"] == "ok").mean() if len(table) else float("nan")
    lines += [f"- Vergleiche: {len(table)}, davon im Rahmen (|z| <= 2 bzw. < 2 % Abweichung): {ok:.0%}",
              f"- Aufgaben: {table['dataset'].nunique()}, Methoden: {', '.join(sorted(table['method'].unique()))}",
              f"- Gleicher Gewinner wie im Paper: {(per_task['bester_wir'] == per_task['bester_paper']).mean():.0%} "
              f"der Aufgaben/Kennzahlen; mittlere Spearman-Korrelation der Rangfolgen: "
              f"{per_task['spearman'].mean():.2f}", ""]
    lines += ["## Auffaellige Abweichungen", ""]
    odd = table[table["urteil"] == "auffaellig"].sort_values("z", key=lambda s: -s.abs())
    lines += [_markdown(odd[["dataset", "metric", "method", "wir", "paper", "differenz", "z"]]) if len(odd) else "keine",
              ""]
    lines += ["## Mittlere Raenge (1 = beste Methode)", "", _markdown(mean_ranks, index=True), ""]
    lines += ["## Rangfolge je Aufgabe", "", _markdown(per_task), ""]
    lines += ["## Alle Werte", "", _markdown(table.drop(columns=["suite_id", "task_id"]))]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(results, t, out, reference="paper_reference_values.csv"):
    paper = pd.read_csv(reference)
    tasks = pd.read_csv("old_tasks_info.csv").set_index(["suite_id", "task_id"], drop=False)
    raw = load_results(results)
    if raw.empty:
        raise SystemExit(f"keine Ergebnisse unter {results} gefunden")
    table = compare(our_values(raw, t), paper, tasks, t)
    per_task, mean_ranks = rank_summary(table)
    table.to_csv(f"{out}.csv", index=False)
    write_markdown(table, per_task, mean_ranks, t, f"{out}.md")
    print(open(f"{out}.md").read().split("## Rangfolge je Aufgabe")[0])
    print(f"Geschrieben: {out}.csv, {out}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Ordner mit Results/, Results_SMAC/, Results_Default_Run/")
    parser.add_argument("--t", type=int, default=135)
    parser.add_argument("--out", default="repro_vergleich")
    args = parser.parse_args()
    main(args.results, args.t, args.out)
