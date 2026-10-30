"""Stand der grossen Laeufe (make_paper_jobs.py): pro Lauf fertig / angefangen / fehlt nach den Ergebnisdateien
und laufend / wartend / fehlgeschlagen / Timeout / Speicher nach Slurm (sacct, nur auf Euler).

Aufruf auf Euler (aus ~/HPOTreeBoosting, mit aktivierter .venv):
    python watch_paper.py                 # beide Laeufe
    python watch_paper.py --run paper     # nur einer
    python watch_paper.py --retry         # zusaetzlich arrays/retry_*.idx fuer abgeschickte Jobs, die ohne
                                          # fertige Datei geendet haben: nach Timeout naechstlaengere Partition, nach
                                          # Speicherfehler doppelter Speicher; abschicken mit
                                          # bash jobs_paper/submit.sh <lauf> retry
Ohne sacct (z. B. auf dem Mac nach dem Kopieren der Ergebnisse) nur der Dateistand.
"""
import argparse
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from run_ablation import expected_rows

JOBS = Path("jobs_paper")
COLUMNS = ["method", "seed", "suite", "task", "trials", "folder", "flags"]
PARTITIONS = (("normal.4h", 4), ("normal.24h", 24), ("normal.120h", 120))
FOLDS = 5


def file_state(job):
    """'fertig', 'angefangen (k/5)' oder 'fehlt'."""
    if job.method == "default":
        path = Path(job.folder) / "default" / "gpboost" / f"{job.suite}_{job.task}.csv"
        return "fertig" if path.exists() and path.stat().st_size > 0 else "fehlt"
    path = Path(job.folder) / job.method / f"seed_{job.seed}" / "gpboost" / f"{job.suite}_{job.task}__num_leaves.csv"
    if not path.exists():
        return "fehlt"
    try:
        sizes = pd.read_csv(path, usecols=["fold"])["fold"].value_counts()
    except Exception:
        return "angefangen (0/5)"
    n_rows = expected_rows(job.method, int(job.trials), "--paper_grid" in str(job.flags))
    done = sum(1 for n in sizes if n == (n_rows or sizes.max()))
    return "fertig" if done >= FOLDS else f"angefangen ({done}/5)"


def slurm_states(run):
    """{Zeile in jobs.tsv: (Zustand, Laufzeit, Job-ID, idx-Datei)} der letzten Abgabe je Zeile; leer ohne sacct."""
    submitted = JOBS / run / "submitted.tsv"
    if not submitted.exists():
        return {}
    subs = pd.read_csv(submitted, sep="\t", header=None, names=["id", "idx", "n", "zeit"], dtype={"id": str})
    try:
        text = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(subs["id"].str.split(";").str[0]),
                               "--format=JobID,State,ElapsedRaw"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    lines_of = {}
    for sub in subs.itertuples():
        idx = Path(sub.idx)
        lines_of[sub.id.split(";")[0]] = [int(x) for x in idx.read_text().split()] if idx.exists() else []
    states = {}
    for row in text.splitlines():
        job_id, state, elapsed = (row.split("|") + ["", ""])[:3]
        if "_" not in job_id or "[" in job_id:          # wartende Bereiche wie 123_[5-999] einzeln aufloesen
            base, _, rest = job_id.partition("_")
            if rest.startswith("["):
                for part in rest.strip("[]").split("%")[0].split(","):
                    start, _, end = part.partition("-")
                    for i in range(int(start), int(end or start) + 1):
                        if i < len(lines_of.get(base, [])):
                            states[lines_of[base][i]] = (state.split()[0], 0, base, i)
            continue
        base, index = job_id.split("_")
        if base in lines_of and int(index) < len(lines_of[base]):
            states[lines_of[base][int(index)]] = (state.split()[0], int(elapsed or 0), base, int(index))
    return states


def retry_name(job, state):
    """Partition, Speicher und Zeitlimit fuer die Wiederholung."""
    hours, mem, part = int(job.stunden), int(job.mem_gb), job.partition
    if state == "TIMEOUT":
        bigger = [p for p in PARTITIONS if p[1] > hours]
        part, hours = bigger[0] if bigger else PARTITIONS[-1]
    if state == "OUT_OF_MEMORY":
        mem *= 2
    return f"retry_{part}_{mem}G_{hours}h"


def watch(run, retry):
    jobs = pd.read_csv(JOBS / run / "jobs.tsv", sep="\t", header=None, names=COLUMNS, keep_default_na=False)
    costs = pd.read_csv(JOBS / run / "kosten.tsv", sep="\t")
    arrays = pd.read_csv(JOBS / run / "arrays.tsv", sep="\t").set_index("array")
    jobs[["partition", "mem_gb", "kernstunden"]] = costs[["partition", "mem_gb", "kernstunden"]]
    jobs["stunden"] = costs["array"].map(arrays["stunden"])      # Zeitlimit, mit dem das Array abgeschickt wurde
    jobs["line"] = jobs.index + 1
    jobs["datei"] = [file_state(j) for j in jobs.itertuples()]
    states = slurm_states(run)
    jobs["slurm"] = [states.get(line, ("nicht abgeschickt",))[0] for line in jobs["line"]]
    jobs["sekunden"] = [states.get(line, (None, 0))[1] for line in jobs["line"]]

    finished = jobs["datei"] == "fertig"
    print(f"===== Lauf {run}: {finished.sum()} von {len(jobs)} Jobs fertig ({finished.mean():.1%}) =====")
    print("Dateien:", dict(Counter(s.split(" (")[0] for s in jobs["datei"])))
    if states:
        print("Slurm (letzte Abgabe je Job):", dict(Counter(jobs["slurm"])))
        used = jobs["sekunden"].sum() / 3600
        print(f"Verbrauchte Kernstunden laut sacct: {used:,.0f} (Schaetzung fuer den ganzen Lauf: "
              f"{jobs['kernstunden'].sum():,.0f})")
    per_method = jobs.assign(fertig=finished).groupby("method")["fertig"].agg(["sum", "size"])
    print("Fertig je Methode:", ", ".join(f"{m} {r['sum']}/{r['size']}" for m, r in per_method.iterrows()))
    bad = jobs[jobs["slurm"].isin(["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL"]) & ~finished]
    for state, group in bad.groupby("slurm"):
        examples = ", ".join(f"Zeile {r.line} {r.method} {r.suite}_{r.task} seed {r.seed}"
                             for r in group.head(3).itertuples())
        print(f"  {state}: {len(group)} (z. B. {examples})")
    if len(bad):
        first = bad.iloc[0]
        base, index = states[first["line"]][2:]
        print(f"  Log des ersten: jobs_paper/logs/hpo-{run}_{base}_{index}.err")

    if retry:
        # nur abgeschickte und beendete Jobs ohne fertige Datei (nicht: laufend, wartend, nie abgeschickt)
        ended = ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "COMPLETED", "PREEMPTED"]
        todo = jobs[~finished & jobs["slurm"].isin(ended)]
        for old in (JOBS / run / "arrays").glob("retry_*.idx"):
            old.unlink()
        for name, group in todo.groupby([retry_name(j, s) for j, s in zip(todo.itertuples(), todo["slurm"])]):
            (JOBS / run / "arrays" / f"{name}.idx").write_text("\n".join(map(str, group["line"])) + "\n")
            print(f"  {name}.idx: {len(group)} Jobs")
        if len(todo):
            print(f"  abschicken: bash jobs_paper/submit.sh {run} retry")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("paper", "l1_270"), action="append")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    for run in args.run or ["paper", "l1_270"]:
        watch(run, args.retry)
