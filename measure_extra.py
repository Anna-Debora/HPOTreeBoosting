"""Kurz-Benchmark fuer den Lauf extra (BOHB, DEHB, HEBO) auf dem Mac: Wanduhrzeit, Anteil Training / Test-Refits /
Modell und Spitzenspeicher je Methode, jeweils Fold 0 mit den Einstellungen von paper (135 Versuche, ohne lambda_l1,
Hyperband-Rahmen R = 2150, eta = 2.8), Hyperband als Vergleich. Grundlage fuer METHOD_FACTOR, SURROGATE_SEC
(make_paper_jobs.py) und EXTRA_OVERHEAD_GB (speicher.py).

Je (Aufgabe, Methode) ein eigener Prozess (Spitzenspeicher = ru_maxrss dieses Prozesses), Aufgaben parallel
(--jobs), Methoden einer Aufgabe nacheinander.
  train_s  Zeit in _train_model_for_validation (Validierungs-Trainings)
  test_s   Zeit in _compute_test_scores (Test-Refits der Ergebniszeilen)
  modell_s Rest = Wanduhr - train_s - test_s (Surrogat/KDE/Evolution, Vorschlaege, Buchhaltung)
Auswertung je Methode (Mittel ueber die Aufgaben; ref_s = Mac-Sekunden je Versuch aus zeitmessung_mac.tsv):
  bohb, dehb: faktor = 0.6 x (train_s + test_s) / (train_s + test_s von Hyperband)   (0.6 = METHOD_FACTOR Hyperband)
  hebo:       faktor = (train_s + test_s) / (135 x ref_s)                             (TPE-artig, 1.0 erwartet)
  modell_s_je_versuch = modell_s / 135 (Mac)  -> SURROGATE_SEC = das x EULER_FACTOR (1.5)
  mehr_gb = Spitze - Spitze von Hyperband derselben Aufgabe (-> EXTRA_OVERHEAD_GB)

Aufruf (.venv_extra):
    .venv_extra/bin/python measure_extra.py                       # 4 Aufgaben, 4 Methoden, 4 parallel
    .venv_extra/bin/python measure_extra.py --tasks 336_361076 --methods bohb --jobs 1
Ausgabe: jobs_paper/extra/zeitmessung_extra_mac.tsv (je Aufgabe und Methode) und eine Zusammenfassung.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

TASKS = ("335_361093", "334_361111", "336_361076", "334_361110")   # analcatdata_supreme, eye_movements, wine, electricity
METHODS = ("hyperband", "bohb", "dehb", "hebo")
TRIALS = 135
HB_FACTOR = 0.6          # make_paper_jobs.METHOD_FACTOR["hyperband"]
EULER_FACTOR = 1.5       # make_paper_jobs.EULER_FACTOR
OUT = Path("jobs_paper/extra/zeitmessung_extra_mac.tsv")


def one(task, method, seed=27225):
    """Ein Lauf (Fold 0) in diesem Prozess; gibt dict mit Zeiten und Spitzenspeicher zurueck."""
    import numpy as np
    import job_io
    import run_ablation
    from SMAC_method import ParameterOptimizationSMAC

    suite, task_id = task.split("_")
    path = f"data/{suite}_{task_id}/{suite}_{task_id}"
    job_io.quiet_logging()
    run_ablation.quiet_methods()
    run_ablation.apply_settings(method, TRIALS, True, False, Path(os.environ.get("TMPDIR", "/tmp")) / "smac3_bench")
    X, y = pd.read_csv(f"{path}_X.csv"), pd.read_csv(f"{path}_y.csv")
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=np.load(f"{path}_categorical_indicator.npy"),
                                    suite_id=int(suite), seed=seed, library="gpboost", **run_ablation.STRATEGY)
    clock = {"train": 0.0, "test": 0.0}

    def timed(name, function):
        def wrapper(*a, **kw):
            start = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                clock[name] += time.perf_counter() - start
        return wrapper
    obj._train_model_for_validation = timed("train", obj._train_model_for_validation)
    obj._compute_test_scores = timed("test", obj._compute_test_scores)
    train_index, test_index = next(iter(obj.splits))
    start = time.perf_counter()
    rows = run_ablation.run_fold(obj, method, TRIALS, train_index, test_index, 2150, lambda_l1=False)
    total = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6      # macOS: Bytes, Linux: KB
    return dict(task=task, method=method, rows=len(rows), wanduhr_s=round(total, 1), train_s=round(clock["train"], 1),
                test_s=round(clock["test"], 1), modell_s=round(total - clock["train"] - clock["test"], 1),
                spitze_gb=round(rss_gb, 3), best_test=float(rows["current_best_test_score"].iloc[-1]))


def child(task, method):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONHASHSEED": "0"}
    result = subprocess.run([sys.executable, __file__, "--one", task, method], capture_output=True, text=True,
                            env=env)
    lines = [l for l in result.stdout.splitlines() if l.startswith("{")]
    if result.returncode != 0 or not lines:
        print(f"FEHLER {task} {method}: {result.stderr[-1500:]}", flush=True)
        return None
    row = json.loads(lines[-1])
    print(f"  {task} {method}: {row['wanduhr_s']} s (Training {row['train_s']}, Test {row['test_s']}, Modell "
          f"{row['modell_s']}), {row['rows']} Zeilen, {row['spitze_gb']} GB", flush=True)
    return row


def summarize(frame, timing="jobs_paper/zeitmessung_mac.tsv"):
    ref = pd.read_csv(timing, sep="\t")
    ref["task"] = ref["suite"].astype(str) + "_" + ref["task"].astype(str)
    ref["ref_s"] = ref[["random_search_s", "tpe_s"]].max(axis=1) / ref["trials"]
    frame = frame.merge(ref[["task", "name", "ref_s"]], on="task", how="left")
    hb = frame[frame["method"] == "hyperband"].set_index("task")
    out = []
    for r in frame[frame["method"] != "hyperband"].itertuples():
        work = r.train_s + r.test_s
        if r.method in ("bohb", "dehb") and r.task in hb.index:
            factor = HB_FACTOR * work / (hb.loc[r.task, "train_s"] + hb.loc[r.task, "test_s"])
        else:
            factor = work / (TRIALS * r.ref_s)
        out.append(dict(task=r.task, name=r.name, method=r.method, faktor=factor,
                        modell_s_je_versuch=r.modell_s / TRIALS,
                        mehr_gb=r.spitze_gb - hb.loc[r.task, "spitze_gb"] if r.task in hb.index else float("nan")))
    detail = pd.DataFrame(out)
    print("\nJe Aufgabe:")
    print(detail.round(3).to_string(index=False))
    agg = detail.groupby("method").agg(faktor=("faktor", "mean"), faktor_max=("faktor", "max"),
                                       modell_s_je_versuch=("modell_s_je_versuch", "mean"),
                                       modell_max=("modell_s_je_versuch", "max"), mehr_gb=("mehr_gb", "max"))
    agg["surrogate_sec_euler"] = agg["modell_max"] * EULER_FACTOR
    print("\nJe Methode (faktor -> METHOD_FACTOR, surrogate_sec_euler = max Modell-s je Versuch x 1.5 -> "
          "SURROGATE_SEC, mehr_gb = max -> EXTRA_OVERHEAD_GB):")
    print(agg.round(3).to_string())
    return detail, agg


def main(args):
    tasks, methods = args.tasks.split(","), args.methods.split(",")
    print(f"Benchmark: {len(tasks)} Aufgaben x {methods}, Fold 0, {args.jobs} parallel", flush=True)

    def per_task(task):
        return [child(task, m) for m in methods]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for rs in pool.map(per_task, tasks) for r in rs if r]
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out, sep="\t")
        frame = pd.concat([old[~old.set_index(["task", "method"]).index.isin(
            frame.set_index(["task", "method"]).index)], frame], ignore_index=True)
    frame.to_csv(out, sep="\t", index=False)
    print(f"geschrieben: {out}")
    summarize(frame)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], sys.argv[3])), flush=True)
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--summary", action="store_true", help="nur die vorhandene Datei auswerten")
    args = parser.parse_args()
    if args.summary:
        summarize(pd.read_csv(args.out, sep="\t"))
    else:
        main(args)
