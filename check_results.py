"""Ergebnisdateien der grossen Laeufe (Results_paper, Results_l1_270) auf Vollstaendigkeit pruefen.

Hintergrund (4.11.2026): Home-Quota voll ("OSError 122 Disk quota exceeded") -> leere oder abgeschnittene CSV.
Seit job_io.write_csv_atomic schreiben die Jobs atomar; alte Schaeden findet dieses Skript.

Je Job aus jobs_paper/<lauf>/jobs.tsv wird die erwartete Datei geprueft:
  defekt      leer, nicht lesbar, nur Kopfzeile, letzte Zeile ohne Zeilenende (abgeschnitten), Zeile mit anderer
              Spaltenzahl als die Kopfzeile, Pflichtspalten fehlen, Fold ausserhalb 0-4, Fold mit falscher Zeilenzahl
              (Zufallsgitter/TPE/GP-BO: --trials, deterministisches Gitter: 135 bzw. 270, Hyperband/SMAC: wie der
              groesste Fold), Default nicht 5 Folds x 270 Zeilen
  angefangen  weniger als 5 Folds, alle vollstaendig: kein Schaden, refill.py setzt mit --resume fort
  ok          vollstaendig
Ausserdem gemeldet: CSV-Dateien, die zu keinem Job gehoeren (nur gemeldet), uebrig gebliebene .tmp-Dateien
(Abbruch waehrend des Schreibens) und je Ergebnisordner smac3_output (Groesse, Dateizahl; wird nie geloescht).
Dateien von Jobs, die gerade in squeue stehen (wartend oder laufend), werden uebersprungen.

Aufruf (Euler, ~/HPOTreeBoosting, .venv aktiv):
    python check_results.py                    # nur auflisten (beide Laeufe)
    python check_results.py --run paper
    python check_results.py --loeschen         # defekte Dateien und .tmp-Reste loeschen; refill.py rechnet sie neu
Exit-Code 0: nichts defekt (oder alles geloescht); 1: defekte Dateien gefunden und nicht geloescht;
2: --loeschen ohne squeue (Abbruch: laufende Jobs nicht erkennbar).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

import watch_paper as wp
from run_ablation import expected_rows
from run_experiments import RESULT_COLUMNS

RUNS = ("paper", "l1_270")
FOLDS = 5
DEFAULT_ROWS = 270                  # methods.run_default: jede Zeile 270-mal
DEFAULT_COLUMNS = ("iter", "fold", "method", "task_id")


def result_path(job):
    """(Pfad, Art 'default'/'hpo', Zeilen je Fold oder None) eines Jobs (Zeile von jobs.tsv)."""
    if job.method == "default":
        return Path(job.folder) / "default" / "gpboost" / f"{job.suite}_{job.task}.csv", "default", DEFAULT_ROWS
    path = Path(job.folder) / job.method / f"seed_{job.seed}" / "gpboost" / f"{job.suite}_{job.task}__num_leaves.csv"
    return path, "hpo", expected_rows(job.method, int(job.trials), "--paper_grid" in str(job.flags))


def check_file(path, kind="hpo", n_rows=None):
    """('ok' | 'angefangen' | 'defekt', Grund)."""
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return "defekt", "leer"
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        return "defekt", f"unlesbar ({type(error).__name__})"
    if not text.endswith("\n"):
        return "defekt", "abgeschnitten (letzte Zeile ohne Zeilenende)"
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 2:
        return "defekt", "nur Kopfzeile"
    width = len(rows[0])
    bad = next((i for i, row in enumerate(rows[1:], 2) if len(row) != width), None)
    if bad is not None:
        return "defekt", f"Zeile {bad}: {len(rows[bad - 1])} statt {width} Spalten"
    try:
        frame = pd.read_csv(path)
    except Exception as error:                     # noqa: BLE001 - jede Lesefehlerart ist ein Defekt
        return "defekt", f"unlesbar ({type(error).__name__})"
    required = DEFAULT_COLUMNS if kind == "default" else RESULT_COLUMNS
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return "defekt", f"Spalten fehlen: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
    try:
        sizes = frame.groupby("fold").size()
        folds = {int(f) for f in sizes.index}
    except (TypeError, ValueError):
        return "defekt", "Spalte fold nicht ganzzahlig"
    if not folds <= set(range(FOLDS)):
        return "defekt", f"Folds ausserhalb 0-{FOLDS - 1}: {sorted(folds - set(range(FOLDS)))}"
    expected = n_rows or int(sizes.max())
    wrong = {int(f): int(n) for f, n in sizes.items() if n != expected}
    if wrong:
        return "defekt", "Zeilen je Fold: " + ", ".join(f"Fold {f} {n} statt {expected}" for f, n in wrong.items())
    if len(folds) < FOLDS:
        if kind == "default":
            return "defekt", f"nur {len(folds)} von {FOLDS} Folds"
        return "angefangen", f"{len(folds)}/{FOLDS} Folds"
    return "ok", ""


def busy_lines(run, queued):
    """Zeilen von jobs.tsv, deren letzte Abgabe in squeue steht."""
    last = {}
    for sub in wp.submissions(run):
        for i, line in enumerate(sub["lines"][:int(sub["n"] or len(sub["lines"]))]):
            last[line] = (sub["id"], i)
    return {line for line, key in last.items() if key in queued}


def folder_size(folder):
    total, count = 0, 0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                pass
    return total, count


def scan(runs, queued):
    """DataFrame mit lauf, zeile, pfad, zustand, grund (zustand: ok/angefangen/defekt/fehlt/laeuft/fremd/tmp)."""
    records, roots, known = [], set(), set()
    for run in runs:
        jobs = pd.read_csv(wp.JOBS / run / "jobs.tsv", sep="\t", header=None, names=wp.COLUMNS, keep_default_na=False)
        busy = busy_lines(run, queued)
        for line, job in enumerate(jobs.itertuples(), 1):
            path, kind, n_rows = result_path(job)
            roots.add(Path(job.folder))
            known.add(path)
            if line in busy:
                state, reason = "laeuft", "in squeue, nicht geprueft"
            elif not path.exists():
                state, reason = "fehlt", ""
            else:
                state, reason = check_file(path, kind, n_rows)
            records.append(dict(lauf=run, zeile=line, pfad=str(path), zustand=state, grund=reason))
            tmp = path.with_name(path.name + ".tmp")
            if tmp.exists():
                records.append(dict(lauf=run, zeile=line, pfad=str(tmp),
                                    zustand="laeuft" if line in busy else "tmp", grund="Rest eines Abbruchs"))
    for root in sorted(roots):
        for path in sorted(root.rglob("*.csv")):
            if path not in known and "smac3_output" not in path.parts:
                state, reason = check_file(path)
                records.append(dict(lauf="", zeile=0, pfad=str(path), zustand="fremd",
                                    grund=f"nicht in jobs.tsv ({state}{': ' + reason if reason else ''})"))
    return pd.DataFrame(records, columns=["lauf", "zeile", "pfad", "zustand", "grund"]), sorted(roots)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=(*RUNS, "beide"), default="beide")
    parser.add_argument("--loeschen", action="store_true",
                        help="defekte Dateien und .tmp-Reste loeschen (smac3_output nie)")
    parser.add_argument("--alle", action="store_true", help="alle defekten Dateien auflisten (sonst die ersten 50)")
    args = parser.parse_args(argv)
    runs = list(RUNS) if args.run == "beide" else [args.run]
    queued = wp.queued_tasks(wp.queue())
    if queued is None:
        if args.loeschen:
            print("squeue nicht verfuegbar: laufende Jobs nicht erkennbar -> nichts geloescht (nur auf Euler)")
            return 2
        print("(ohne squeue: auch Dateien laufender Jobs geprueft; ein gerade schreibender Job kann defekt wirken)")
        queued = {}
    table, roots = scan(runs, queued)
    counts = table["zustand"].value_counts().to_dict()
    print("Ergebnisdateien: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in
                                         ("ok", "angefangen", "defekt", "fehlt", "laeuft", "tmp", "fremd")))
    for root in roots:
        smac = root / "smac3_output"
        if smac.exists():
            size, count = folder_size(smac)
            print(f"{smac}: {count} Dateien, {size / 1e9:.2f} GB (nur gemeldet, wird nicht geloescht)")
    broken = table[table["zustand"].isin(["defekt", "tmp"])]
    shown = broken if args.alle else broken.head(50)
    for r in shown.itertuples():
        print(f"  {r.zustand:7s} {r.lauf} Zeile {r.zeile}: {r.pfad}  {r.grund}")
    if len(shown) < len(broken):
        print(f"  ... und {len(broken) - len(shown)} weitere (--alle)")
    for r in table[table["zustand"] == "fremd"].head(10).itertuples():
        print(f"  fremd   {r.pfad}  {r.grund}")
    if broken.empty:
        return 0
    if not args.loeschen:
        print(f"{len(broken)} defekte Dateien/Reste. Loeschen (danach rechnet refill.py sie neu): "
              "python check_results.py --loeschen")
        return 1
    removed = 0
    for path in broken["pfad"]:
        try:
            Path(path).unlink()
            removed += 1
        except OSError as error:
            print(f"  nicht geloescht: {path} ({error})")
    print(f"geloescht: {removed} von {len(broken)}; refill.py reicht die Jobs neu ein")
    return 0 if removed == len(broken) else 1


if __name__ == "__main__":
    sys.exit(main())
