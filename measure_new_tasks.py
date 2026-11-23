"""Laufzeit-Stichprobe fuer die 286 neuen Aufgaben (Suiten 1001/1002), Grundlage der Kostenschaetzung der grossen
Laeufe auf allen Datensaetzen (make_paper_jobs.py --run gpb_alle).

Wie jobs_paper/zeitmessung_mac.tsv fuer die 59 alten Aufgaben: Mac, 1 Thread, Fold 0, je 15 Versuche Zufallsgitter
und TPE (Paper-Einstellung: ohne lambda_l1, lineare Skalen) inkl. Test-Refits, Seed 27225; Sekunden je Versuch =
Maximum der beiden / 15. Gemessen wird nur run_fold (ohne Python-Start und Daten laden), je (Aufgabe, Methode) ein
eigener Prozess.

Stichprobe: die GROESSTE groessten neuen Aufgaben nach min(Zeilen, 100000) x Spalten (damit Zeit- und Speicherlimits
sichtbar werden) und STICHPROBE weitere, gleichmaessig ueber die Rangfolge dieser Groesse verteilt (alle
Groessenklassen von den kleinsten an), dazu ANKER alte Aufgaben aus zeitmessung_mac.tsv zum Abgleich (die alte
Messung hat womoeglich anders gemessen, z. B. mit Prozessstart; Faktor alt / neu wird bei der Hochrechnung auf die
neue Messung angewandt).

Hochrechnung (hochrechnen(), von make_paper_jobs.py benutzt): log(Sekunden je Versuch) linear in log(Zeilen),
log(Spalten), Anteil kategorialer Spalten und Klassifikation, angepasst auf die 59 alten (zeitmessung_mac.tsv) und die
gemessenen neuen Aufgaben (x Abgleichfaktor); gemessene Aufgaben behalten ihren Messwert. Reserve: Vorhersage x
exp(RESERVE_SD x Residuen-Standardabweichung).

Aufruf:  .venv/bin/python measure_new_tasks.py [--jobs 4]            # Stichprobe messen
         .venv/bin/python measure_new_tasks.py --hochrechnung        # nur Hochrechnung aus vorhandener Messung
Ausgabe: jobs_paper/zeitmessung_neu_mac.tsv (gemessene Aufgaben, Spalten wie zeitmessung_mac.tsv + anker),
         jobs_paper/zeitmessung_neu_hochrechnung.tsv (alle 345 Aufgaben: Sekunden je Versuch, Quelle)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("jobs_paper/zeitmessung_mac.tsv")
OUT = Path("jobs_paper/zeitmessung_neu_mac.tsv")
PRED = Path("jobs_paper/zeitmessung_neu_hochrechnung.tsv")
SUITES = (334, 335, 336, 337, 1001, 1002)
NEW_SUITES = (1001, 1002)
TRIALS = 15
SEED = 27225
STICHPROBE = 19
GROESSTE = 5
ANKER = ("335_361093", "336_361076", "337_361278", "334_361111", "335_361102", "337_361070")
MAX_ROWS = 100000          # ParameterOptimization._subsample_data
RESERVE_SD = 1.0


def sizes():
    """Alle 345 Aufgaben mit Zeilen (nach Unterabtastung), Spalten, Anteil kategorial, Klassifikation."""
    from task_types import is_classification
    rows = []
    for suite in SUITES:
        ids = np.load(f"task_indices/{suite}_task_indices.npy")
        with open(f"task_indices/{suite}_task_names.json") as handle:
            names = json.load(handle)
        for task, name in zip(ids, names):
            path = f"data/{suite}_{task}/{suite}_{task}"
            with open(f"{path}_X.csv") as handle:
                handle.readline()
                n = sum(1 for _ in handle)
            cat = np.load(f"{path}_categorical_indicator.npy")
            rows.append(dict(suite=suite, task=int(task), name=name, zeilen=min(n, MAX_ROWS), spalten=len(cat),
                             anteil_kat=float(cat.mean()) if len(cat) else 0.0,
                             klassifikation=int(is_classification(suite))))
    return pd.DataFrame(rows)


def auswahl(frame):
    new = frame[frame["suite"].isin(NEW_SUITES)].copy()
    new["groesse"] = new["zeilen"] * new["spalten"]
    new = new.sort_values(["groesse", "task"]).reset_index(drop=True)
    rest = new.iloc[:-GROESSTE]
    picks = np.unique(np.round(np.linspace(0, len(rest) - 1, STICHPROBE)).astype(int))
    chosen = [f"{r.suite}_{r.task}" for r in pd.concat([new.iloc[-GROESSTE:][::-1], rest.iloc[picks]]).itertuples()]
    return chosen + list(ANKER)


def one(task, method):
    import job_io
    import run_ablation
    from SMAC_method import ParameterOptimizationSMAC
    suite, task_id = task.split("_")
    path = f"data/{suite}_{task_id}/{suite}_{task_id}"
    job_io.quiet_logging()
    run_ablation.quiet_methods()
    run_ablation.apply_settings(method, TRIALS, True, False,
                                Path(os.environ.get("TMPDIR", "/tmp")) / "smac3_messung")
    X, y = pd.read_csv(f"{path}_X.csv"), pd.read_csv(f"{path}_y.csv")
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=np.load(f"{path}_categorical_indicator.npy"),
                                    suite_id=int(suite), seed=SEED, library="gpboost", **run_ablation.STRATEGY)
    train_index, test_index = next(iter(obj.splits))
    start = time.perf_counter()
    run_ablation.run_fold(obj, method, TRIALS, train_index, test_index, 2150, lambda_l1=False)
    return dict(task=task, method=method, sekunden=round(time.perf_counter() - start, 1))


def child(task, method):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONHASHSEED": "0"}
    result = subprocess.run([sys.executable, __file__, "--one", task, method], capture_output=True, text=True, env=env)
    lines = [l for l in result.stdout.splitlines() if l.startswith("{")]
    if result.returncode != 0 or not lines:
        print(f"FEHLER {task} {method}: {result.stderr[-1500:]}", flush=True)
        return dict(task=task, method=method, sekunden=float("nan"), rc=result.returncode)
    row = json.loads(lines[-1])
    print(f"  {task} {method}: {row['sekunden']} s", flush=True)
    return {**row, "rc": 0}


def measure(jobs):
    frame = sizes()
    tasks = auswahl(frame)
    print(f"Messung: {len(tasks)} Aufgaben ({len(tasks) - len(ANKER)} neu, {len(ANKER)} Anker) x random_search, tpe, "
          f"{TRIALS} Versuche, Fold 0, {jobs} parallel", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(lambda tm: child(*tm), [(t, m) for t in tasks for m in ("random_search", "tpe")]))
    wide = pd.DataFrame(rows).pivot(index="task", columns="method", values="sekunden")
    rc = pd.DataFrame(rows).pivot(index="task", columns="method", values="rc")
    out = []
    for task in tasks:
        suite, task_id = map(int, task.split("_"))
        name = frame.loc[(frame["suite"] == suite) & (frame["task"] == task_id), "name"].iloc[0]
        out.append(dict(suite=suite, task=task_id, name=name, rc_rs=int(rc.loc[task, "random_search"]),
                        rc_tpe=int(rc.loc[task, "tpe"]), random_search_s=wide.loc[task, "random_search"],
                        tpe_s=wide.loc[task, "tpe"], trials=TRIALS, anker=task in ANKER))
    pd.DataFrame(out).to_csv(OUT, sep="\t", index=False)
    print(f"geschrieben: {OUT}")


def hochrechnen(frame=None, old=OLD, new=OUT, verbose=True):
    """{(suite, task): Mac-Sekunden je Versuch} fuer alle Aufgaben (gemessen oder hochgerechnet mit Reserve) und
    die Tabelle dazu (auch nach PRED geschrieben)."""
    frame = sizes() if frame is None else frame
    old_t = pd.read_csv(old, sep="\t")
    old_t["sec"] = old_t[["random_search_s", "tpe_s"]].max(axis=1) / old_t["trials"]
    new_t = pd.read_csv(new, sep="\t").dropna(subset=["random_search_s", "tpe_s"])
    new_t["sec"] = new_t[["random_search_s", "tpe_s"]].max(axis=1) / new_t["trials"]
    anchor = new_t[new_t["anker"]].merge(old_t[["suite", "task", "sec"]], on=["suite", "task"], suffixes=("", "_alt"))
    calib = float(np.exp(np.mean(np.log(anchor["sec_alt"] / anchor["sec"])))) if len(anchor) else 1.0
    measured = {(int(r.suite), int(r.task)): float(r.sec) for r in old_t.itertuples()}
    measured.update({(int(r.suite), int(r.task)): float(r.sec) * calib
                     for r in new_t[~new_t["anker"]].itertuples()})
    frame = frame.copy()
    frame["gemessen"] = [measured.get((s, t)) for s, t in zip(frame["suite"], frame["task"])]
    design = np.column_stack([np.ones(len(frame)), np.log(frame["zeilen"]), np.log(frame["spalten"]),
                              frame["anteil_kat"], frame["klassifikation"]])
    fit = frame["gemessen"].notna().to_numpy()
    target = np.log(frame.loc[fit, "gemessen"].astype(float))
    coef, *_ = np.linalg.lstsq(design[fit], target, rcond=None)
    resid = target - design[fit] @ coef
    sd = float(np.std(resid, ddof=design.shape[1]))
    pred = np.exp(design @ coef + RESERVE_SD * sd)
    frame["hochgerechnet"] = pred
    frame["sek_versuch"] = frame["gemessen"].fillna(frame["hochgerechnet"])
    frame["quelle"] = np.where(frame["gemessen"].notna(),
                               np.where(frame["suite"].isin(NEW_SUITES), "neu gemessen", "alt gemessen"),
                               "hochgerechnet")
    frame.to_csv(PRED, sep="\t", index=False)
    if verbose:
        print(f"Abgleich alt/neu ({len(anchor)} Anker): Faktor {calib:.2f}; Fit auf {fit.sum()} Aufgaben, "
              f"Koeffizienten (1, log Zeilen, log Spalten, Anteil kat., Klassifikation) {np.round(coef, 3).tolist()}, "
              f"Residuen-SD {sd:.2f} (Reserve x {np.exp(RESERVE_SD * sd):.2f})")
    return {(int(r.suite), int(r.task)): float(r.sek_versuch) for r in frame.itertuples()}, frame


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], sys.argv[3])), flush=True)
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--hochrechnung", action="store_true", help="nur aus der vorhandenen Messung hochrechnen")
    args = parser.parse_args()
    if not args.hochrechnung:
        measure(args.jobs)
    hochrechnen()
