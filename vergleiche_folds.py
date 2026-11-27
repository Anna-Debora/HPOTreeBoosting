"""vergleiche_folds.py -- Fold-Jobs (run_ablation.py --fold, Lauf xgb_alle) gegen das alte Format (ein Job, 5 Folds).

Fuer jede Zeile von jobs_paper/xgb_alle_vergleich/jobs.tsv (bash jobs_paper/submit_xgb_alle.sh vergleich, nach
Results_xgb_alle_vergleich) wird dieselbe Datei in Results_xgb_alle (Probe: bash jobs_paper/submit_xgb_alle.sh probe,
Fold-Jobs) gesucht und Zeile fuer Zeile verglichen: alle Spalten ausser runtime als Text (exakt, gleiche
Reihenfolge der Zeilen und Folds). runtime ist Wanduhrzeit und daher nie gleich.

Aufruf (Euler, ~/HPOTreeBoosting, .venv aktiv, wenn beide Arrays fertig sind):
    python vergleiche_folds.py
    python vergleiche_folds.py --alt A --neu B --jobs jobs.tsv     # andere Ordner (lokaler Test)
Exit-Code 0: alle gleich; 1: Abweichung oder Datei fehlt.
"""
import argparse
import csv
import sys
from pathlib import Path

import watch_paper as wp

IGNORED = ("runtime",)


def rows_without(path, ignored=IGNORED):
    """(Kopfzeile, Zeilen) als Listen von Textfeldern, ohne die Spalten in ignored."""
    rows = list(csv.reader(Path(path).read_text().splitlines()))
    keep = [i for i, name in enumerate(rows[0]) if name not in ignored]
    return [rows[0][i] for i in keep], [[row[i] for i in keep] for row in rows[1:]]


def compare(old, new):
    """'gleich' oder eine kurze Beschreibung der ersten Abweichung."""
    if not Path(old).exists():
        return "fehlt im alten Format"
    if not Path(new).exists():
        return "fehlt im Fold-Format"
    head_old, rows_old = rows_without(old)
    head_new, rows_new = rows_without(new)
    if head_old != head_new:
        return "Kopfzeilen verschieden"
    if len(rows_old) != len(rows_new):
        return f"{len(rows_old)} statt {len(rows_new)} Zeilen"
    for k, (a, b) in enumerate(zip(rows_old, rows_new), 2):
        if a != b:
            col = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            return f"Zeile {k}, Spalte {head_old[col]}: {a[col]} (alt) vs {b[col]} (Fold)"
    return "gleich"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", default="jobs_paper/xgb_alle_vergleich/jobs.tsv")
    parser.add_argument("--alt", default="Results_xgb_alle_vergleich", help="Ordner im alten Format")
    parser.add_argument("--neu", default="Results_xgb_alle", help="Ordner der Fold-Jobs")
    args = parser.parse_args(argv)
    import pandas as pd
    jobs = pd.read_csv(args.jobs, sep="\t", header=None, names=wp.COLUMNS, keep_default_na=False)
    bad = 0
    for job in jobs.itertuples():
        relative = Path(job.method, f"seed_{job.seed}", wp.library_of(job.flags), f"{job.suite}_{job.task}__num_leaves.csv")
        result = compare(Path(args.alt) / relative, Path(args.neu) / relative)
        bad += result != "gleich"
        n = len(rows_without(Path(args.alt) / relative)[1]) if result == "gleich" else 0
        print(f"{'OK ' if result == 'gleich' else 'XX '} {job.method:13s} {job.suite}_{job.task}: {result}"
              + (f" ({n} Zeilen, ohne runtime)" if n else ""))
    print(f"{len(jobs) - bad} von {len(jobs)} gleich" + ("" if not bad else f", {bad} abweichend oder fehlend"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
