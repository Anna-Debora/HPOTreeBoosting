"""Job-Listen fuer die zwei grossen GPBoost-Laeufe auf den 59 Aufgaben der Suiten 334-337 (Grinsztajn et al.):

  paper   Paper-Einstellung (= Lauf A in run_ablation.py): 135 Versuche, ohne lambda_l1, deterministisches
          Gitter mit 135 Punkten wie im Paper-Code (--paper_grid)            -> Results_paper/
  l1_270  wie paper, aber lambda_l1 im Suchraum (linear (0, 100) wie im aktuellen Code) und 270 Versuche;
          deterministisches Gitter = heutiges Gitter mit 270 Punkten (bagging_fraction {0.5, 1}, ohne lambda_l1)
                                                                             -> Results_l1_270/
Beide: 20 Seeds des Papers, 5 Folds, Strategie num_leaves (Table 5, Abb. 1, 5-10), GPBoost, Kategorien nativ,
7 Methoden: Default, deterministisches Gitter, GP-BO, Hyperband (R = 2150, eta = 2.8), Zufallsgitter, SMAC, TPE.

Ein Job = (Aufgabe, Seed, Methode), 1 Kern; Default einmal pro Aufgabe (haengt nicht vom Seed ab).
  59 x 20 x 6 + 59 = 7139 Jobs pro Lauf.
Die Jobs schreiben nach jedem Fold und sind wiederholbar (run_ablation.py --resume: fertige Dateien
ueberspringen, angefangene ab dem ersten fehlenden Fold fortsetzen; Default: vorhandene Datei ueberspringen).

Zeitlimit pro Job aus dem Kostenmodell (siehe kosten()), Speicher aus der Speichermessung (memory_gb()).

Wellen (damit es frueh Zwischenergebnisse gibt): Welle 1 = Seeds 1-5 (Reihenfolge wie utils_modelling.seeds) plus
Default, Welle 2 = Seeds 6-10, Welle 3 = Seeds 11-15, Welle 4 = Seeds 16-20, jeweils alle 59 Aufgaben und alle
Methoden. Abgeschickt wird mit refill.py in Portionen (Euler: hoechstens ~1000 Jobs in der Schlange), in der
Reihenfolge der Spalte 'reihenfolge' in kosten.tsv: Welle, dann schnelle Partition zuerst (normal.4h vor 24h vor
120h), dann Aufgabe nach geschaetzter Laufzeit (kleine zuerst), Seed, Methode. Zeitlimit einer Portion = Spalte
'zeitlimit' (groesstes 'stunden' der Gruppe gleicher Welle, Partition und Speicher, wie bei den frueheren Arrays).

Eingefroren (31.10.2026): die Wellen-Arrays arrays/welle*.idx und arrays.tsv aus dem ersten Abschicken mit
submit.sh werden nicht mehr geschrieben oder geloescht, weil submitted.tsv auf Euler ueber sie die Array-Indizes
der schon abgeschickten Jobs auf Zeilen von jobs.tsv abbildet. jobs.tsv behaelt seine Reihenfolge.

Aufruf:  python make_paper_jobs.py [--out jobs_paper]
Ausgabe: <out>/<lauf>/jobs.tsv (eine Zeile je Job, ohne Kopfzeile, fuer das sbatch-Skript)
         <out>/<lauf>/kosten.tsv (mit Kopfzeile: Schaetzung, Partition, Zeitlimit, Speicher, Welle, Reihenfolge,
         Name des eingefrorenen Arrays, falls die Zeile in einem liegt)
         <out>/kosten_wellen.tsv (Kernstunden und Dauer je Lauf und Welle)
         <out>/<lauf>/arrays/probe.idx (wine_quality, Seed 27225, alle 7 Methoden)
         <out>/run_paper_job.sbatch, <out>/submit.sh, <out>/zusammenfassung.txt
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from utils_modelling import seeds as SEEDS
from task_types import is_classification

SUITES = (334, 335, 336, 337)
METHODS = ("default", "grid_search", "random_search", "tpe", "gp_bo", "hyperband", "SMAC")
RUNS = {"paper": dict(folder="Results_paper", trials=135, flags="--no_lambda_l1 --paper_grid"),
        "l1_270": dict(folder="Results_l1_270", trials=270, flags="")}
PROBE = (336, 361076, 27225)   # wine_quality: klein, im Testlauf und in den Ablationslaeufen schon gerechnet
FOLDS = 5
# Partitionen auf Euler: (Name, laengstes Zeitlimit in Stunden)
PARTITIONS = (("normal.4h", 4), ("normal.24h", 24), ("normal.120h", 120))
SAFETY = 2.0         # Zeitlimit = Schaetzung x SAFETY, aufgerundet auf volle Stunden, mindestens 1 h

# Kostenmodell (Kernstunden je Job, siehe README in REPRO_PLAN.md, Abschnitt "Kosten der grossen Laeufe"):
#   Sekunden je Versuch c_k pro Aufgabe = Wanduhrzeit lokal (Mac, 1 Thread) fuer 15 Versuche Zufallsgitter bzw.
#   TPE auf Fold 0 inkl. Test-Refits, geteilt durch 15 (jobs_paper/zeitmessung_mac.tsv), mal EULER_FACTOR.
#   Job = 5 Folds x Versuche x c_k x METHOD_FACTOR + Surrogat (GP-BO, SMAC) je Versuch.
EULER_FACTOR = 1.5            # Euler-Kern vs. Mac. Abgleich sulfur: Mac 15.7 s/Versuch, Euler-Testlauf 13-20 s/Versuch
                              # (Hauptjob 4 x 270 Versuche: Zeitlimit 12 h nach 2 fertigen Folds) -> 0.85-1.3, mit Reserve
# Verhaeltnis zum Zufallsgitter (Ablationslauf A, wine_quality, 135 Versuche, Euler): Paper-Gitter 0.5, Hyperband 1.2
# (nur Validierung); Hyperband macht 11 statt 135 Test-Refits -> etwa 0.6 eines Zufallsgitter-Jobs mit Refits
METHOD_FACTOR = {"grid_search": 0.6, "random_search": 1.0, "tpe": 1.0, "gp_bo": 1.0, "SMAC": 1.0, "hyperband": 0.6}
SURROGATE_SEC = {("gp_bo", 135): 1.6, ("gp_bo", 270): 3.6, ("SMAC", 135): 0.9, ("SMAC", 270): 1.4}  # Euler, wine_quality
DEFAULT_HOURS = 0.1
WAVE_SIZE = 5        # Seeds je Welle
CORES = (64, 110)    # Euler-Account: 64 Kerne je Partition, 110 in normal/bulk zusammen
MAX_SUBMIT = 17000   # hoechstens so viele abgeschickte (wartende + laufende) Jobs (QOS: tatsaechlich ~1000, refill.py)
MEM_STEPS = (4, 8, 12, 16, 24, 32)   # Speicherstufen je Kern in GB
# Speicher = naechste Stufe ueber MEM_FACTOR x Mac-Spitze + MEM_BASE (jobs_paper/speicher_mac.tsv, measure_memory.py).
# Abgleich: MiniBooNE Mac-Spitze 6.1 GB, auf Euler mit 4 GB OUT_OF_MEMORY (random_search, Job 19145396).
MEM_FACTOR, MEM_BASE = 1.5, 1.5


def tasks():
    rows = []
    for suite in SUITES:
        ids = np.load(f"task_indices/{suite}_task_indices.npy")
        with open(f"task_indices/{suite}_task_names.json") as handle:
            names = json.load(handle)
        rows += [dict(suite=suite, task=int(t), name=n) for t, n in zip(ids, names)]
    return pd.DataFrame(rows)


def seconds_per_trial(timing="jobs_paper/zeitmessung_mac.tsv"):
    """Mac-Sekunden je Versuch (inkl. Test-Refit) je Aufgabe: Maximum aus Zufallsgitter und TPE."""
    frame = pd.read_csv(timing, sep="\t")
    frame["sec"] = frame[["random_search_s", "tpe_s"]].max(axis=1) / frame["trials"]
    return {(int(r.suite), int(r.task)): float(r.sec) for r in frame.itertuples()}


def kosten(method, trials, sec):
    """Geschaetzte Kernstunden eines Jobs (5 Folds) auf Euler."""
    if method == "default":
        return DEFAULT_HOURS
    n = 135 if method == "hyperband" else trials   # Hyperband: fester Rahmen, etwa wie 135 Versuche
    per_trial = sec * EULER_FACTOR * METHOD_FACTOR[method] + SURROGATE_SEC.get((method, trials), 0.0)
    return FOLDS * n * per_trial / 3600


def mem_step(gb):
    """Kleinste Speicherstufe >= gb."""
    return next((m for m in MEM_STEPS if m >= gb), MEM_STEPS[-1])


def mac_peaks(path="jobs_paper/speicher_mac.tsv"):
    """{(suite, task): Spitze in GB} aus measure_memory.py; leer, wenn die Datei fehlt."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, sep="\t").dropna(subset=["peak_gb"])
    return {(int(r.suite), int(r.task)): float(r.peak_gb) for r in frame.itertuples()}


def memory_gb(suite, task, method="", peaks=None):
    """Speicher je Kern. Untergrenze nach Groesse der Rohdaten (die Daten werden ganz geladen, dann auf 100'000
    Zeilen unterabgetastet; Mac: delays_zurich_transport 306 MB -> 2.0 GB): 4 GB bis 100 MB, sonst 8 GB.
    HPO-Methoden: dazu die Histogramme von GPBoost beim teuersten Versuch (num_leaves 1024, max_bin bis 10'000),
    gemessen mit measure_memory.py: Stufe ueber MEM_FACTOR x Spitze + MEM_BASE. Default (31 Blaetter, 255 Bins)
    braucht nur die Untergrenze."""
    folder = Path(f"data/{suite}_{task}")
    mb = sum(f.stat().st_size for f in folder.glob("*")) / 1e6 if folder.exists() else 0
    floor = 4 if mb <= 100 else 8
    peak = (peaks or {}).get((suite, task))
    if method == "default" or peak is None:
        return floor
    return max(floor, mem_step(MEM_FACTOR * peak + MEM_BASE))


def partition(hours):
    limit = max(1, math.ceil(hours * SAFETY))
    for name, longest in PARTITIONS:
        if limit <= longest:
            return name, limit
    return PARTITIONS[-1][0], PARTITIONS[-1][1]


def wave(seed):
    """Welle 1-4 eines Seeds (Default, Seed 0: Welle 1)."""
    return 1 if seed not in SEEDS else SEEDS.index(seed) // WAVE_SIZE + 1


def build(run, sec, task_list, peaks=None):
    settings = RUNS[run]
    rows = []
    for t in task_list.itertuples():
        for method in METHODS:
            for seed in ([0] if method == "default" else SEEDS):
                hours = kosten(method, settings["trials"], sec[(t.suite, t.task)])
                name, limit = partition(hours)
                rows.append(dict(method=method, seed=seed, suite=t.suite, task=t.task, trials=settings["trials"],
                                 flags=settings["flags"] if method != "default" else "", folder=settings["folder"],
                                 name=t.name, kernstunden=round(hours, 3), partition=name, stunden=limit,
                                 mem_gb=memory_gb(t.suite, t.task, method, peaks), welle=wave(seed),
                                 sek_versuch=sec[(t.suite, t.task)]))
    return pd.DataFrame(rows)


def ordered(group):
    """Reihenfolge in einem Array: Aufgabe nach geschaetzter Laufzeit (kleine zuerst), dann Seed, dann Methode."""
    key = group.assign(_seed=[SEEDS.index(s) if s in SEEDS else -1 for s in group["seed"]],
                       _method=[METHODS.index(m) for m in group["method"]])
    return key.sort_values(["sek_versuch", "suite", "task", "_seed", "_method"], kind="stable")


def plan(jobs):
    """Spalten 'zeitlimit' (groesstes 'stunden' je Welle, Partition, Speicher) und 'reihenfolge' (1 = zuerst
    abschicken: Welle, schnelle Partition zuerst, kleine Aufgaben zuerst, Seed, Methode)."""
    jobs = jobs.copy()
    jobs["zeitlimit"] = jobs.groupby(["welle", "partition", "mem_gb"])["stunden"].transform("max").astype(int)
    hours = dict(PARTITIONS)
    keyed = ordered(jobs.assign(_part=jobs["partition"].map(hours)))
    order = keyed.sort_values(["welle", "_part"], kind="stable").index
    jobs.loc[order, "reihenfolge"] = np.arange(1, len(jobs) + 1)
    jobs["reihenfolge"] = jobs["reihenfolge"].astype(int)
    return jobs


def frozen_arrays(folder):
    """{Zeilennummer: Array-Name} aus den eingefrorenen arrays/*.idx (ohne retry_ und Portionen)."""
    names = {}
    for idx in sorted((folder / "arrays").glob("*.idx")):
        if idx.name.startswith("retry_"):
            continue
        for line in idx.read_text().split():
            names.setdefault(int(line), idx.stem)
    return names


def wave_costs(run, jobs, skip=()):
    """Kernstunden und Dauer je Welle (ohne die schon gerechneten Probe-Jobs)."""
    rows = []
    todo = jobs[~(jobs.index + 1).isin(list(skip))]
    for welle, group in todo.groupby("welle"):
        hours = group["kernstunden"].sum()
        row = dict(lauf=run, welle=int(welle), jobs=len(group), kernstunden=round(hours),
                   laengster_job_h=round(group["kernstunden"].max(), 1))
        for cores in CORES:
            # Untergrenze: gleichmaessig verteilt auf alle Kerne, aber nie kuerzer als der laengste Job
            row[f"tage_{cores}_kerne"] = round(max(hours / cores, group["kernstunden"].max()) / 24, 1)
        rows.append(row)
    return pd.DataFrame(rows)


SBATCH = """#!/bin/bash
#SBATCH --job-name=hpo-paper
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs_paper/logs/%x_%A_%a.out
#SBATCH --error=jobs_paper/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Ein Teiljob der grossen Laeufe (make_paper_jobs.py). Nicht direkt abschicken, sondern ueber
#   python refill.py (oder fuer die Probe: bash jobs_paper/submit.sh <lauf> probe)
# (setzt --array, --time, --mem-per-cpu, --partition und die Argumente <lauf> <idx-Datei>).
# Wiederholbar: fertige Ergebnisse werden uebersprungen, angefangene ab dem ersten fehlenden Fold fortgesetzt.

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

RUN=$1
IDX_FILE=$2
LINE_NO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IDX_FILE")
JOB_LINE=$(sed -n "${LINE_NO}p" "jobs_paper/$RUN/jobs.tsv")
IFS=$'\\t' read -r METHOD SEED SUITE_ID TASK_ID TRIALS FOLDER FLAGS <<< "$JOB_LINE"
echo "Lauf $RUN Zeile $LINE_NO: $METHOD seed=$SEED ${SUITE_ID}_${TASK_ID} trials=$TRIALS $FLAGS -> $FOLDER"
echo "Start $(date '+%F %T') auf $(hostname), Slurm-Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Zuerst: bash euler_prepare_repro_data.sh jobs_paper/aufgaben.tsv" >&2
    exit 2
fi

if [ "$METHOD" = "default" ]; then
    OUT="$FOLDER/default/gpboost/${SUITE_ID}_${TASK_ID}.csv"
    if [ -s "$OUT" ]; then
        echo "SCHON FERTIG: $OUT"
    else
        python run_default.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --result_folder "$FOLDER/default"
    fi
else
    python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \\
        --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS
fi
echo "Ende $(date '+%F %T')"
"""

SUBMIT = """#!/bin/bash
# Nur noch fuer die Probe. Die Wellen schickt refill.py in Portionen ab (QOS: hoechstens ~1000 Jobs in der Schlange):
#   python refill.py --run beide              # siehe python refill.py --help
#   bash jobs_paper/submit.sh paper probe     # 7 Jobs: wine_quality, Seed 27225, alle Methoden
# Optional: DRY=1 (nur anzeigen). Merkt sich die Job-ID in jobs_paper/<lauf>/submitted.tsv (fuer watch_paper.py).
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
RUN=$1; WHAT=$2
DIR=jobs_paper/$RUN
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --array=0-6 --partition=normal.4h --time=4:00:00 --mem-per-cpu=4G
                --job-name="hpo-$RUN" jobs_paper/run_paper_job.sbatch "$RUN" "$DIR/arrays/probe.idx")
           if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; exit 0; fi
           id=$("${cmd[@]}")
           printf '%s\\t%s\\t%s\\t%s\\tnormal.4h\\t4\\t4\\n' "$id" "$DIR/arrays/probe.idx" 7 "$(date '+%F %T')" >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe, 7 Teiljobs)" ;;
    *)     echo "$WHAT: Wellen und Wiederholungen jetzt mit refill.py (python refill.py --help)"; exit 1 ;;
esac
"""


def main(out, timing, memory="jobs_paper/speicher_mac.tsv"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sec = seconds_per_trial(timing)
    peaks = mac_peaks(memory)
    task_list = tasks()
    task_list.assign(classification=[is_classification(s) for s in task_list["suite"]]).to_csv(
        out / "aufgaben.tsv", sep="\t", index=False, columns=["suite", "task", "name", "classification"])
    summary, waves = [], []
    for run in RUNS:
        folder = out / run
        (folder / "arrays").mkdir(parents=True, exist_ok=True)
        jobs = plan(build(run, sec, task_list, peaks))
        jobs[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
            folder / "jobs.tsv", sep="\t", index=False, header=False)
        probe = jobs[(jobs["suite"] == PROBE[0]) & (jobs["task"] == PROBE[1])
                     & ((jobs["seed"] == PROBE[2]) | (jobs["method"] == "default"))]
        probe_lines = [i + 1 for i in probe.index]
        if not (folder / "arrays" / "probe.idx").exists():
            (folder / "arrays" / "probe.idx").write_text("\n".join(map(str, probe_lines)) + "\n")
        names = frozen_arrays(folder)
        jobs["array"] = [names.get(i + 1, "") for i in jobs.index]
        jobs.drop(columns="sek_versuch").to_csv(folder / "kosten.tsv", sep="\t", index=False)
        costs = wave_costs(run, jobs, skip=probe_lines)
        waves.append(costs)
        by_method = jobs.groupby("method")["kernstunden"].sum().round(0).astype(int).to_dict()
        hpo = jobs[jobs["method"] != "default"].drop_duplicates(["suite", "task"])
        big = hpo[hpo["mem_gb"] > 4].sort_values("mem_gb", ascending=False)
        summary += [f"Lauf {run} -> {RUNS[run]['folder']}/: {len(jobs)} Jobs, geschaetzt "
                    f"{jobs['kernstunden'].sum():,.0f} Kernstunden (Zeitlimits zusammen {jobs['stunden'].sum():,} h)",
                    f"  je Methode: {by_method}",
                    f"  teuerste Jobs: " + ", ".join(
                        f"{r.name} {r.method} {r.kernstunden:.1f} h" for r in
                        jobs.sort_values("kernstunden", ascending=False).drop_duplicates(["task", "method"])
                        .head(5).itertuples()),
                    "  Speicher je Job: " + ", ".join(f"{m} GB {n}" for m, n in
                                                     jobs["mem_gb"].value_counts().sort_index().items())
                    + "; Aufgaben ueber 4 GB: " + ", ".join(f"{r.name} {r.mem_gb}" for r in big.itertuples()),
                    f"  Probe (schon gerechnet): {len(probe)} Jobs (Zeilen {', '.join(map(str, probe_lines))})"]
        for c in costs.itertuples():
            summary.append(f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}"
                           f"{' + Default' if c.welle == 1 else ''}): {c.jobs} Jobs, {c.kernstunden:,} Kernstunden, "
                           f"bei 64 Kernen ~{c.tage_64_kerne} Tage, bei 110 Kernen ~{c.tage_110_kerne} Tage "
                           f"(laengster Job {c.laengster_job_h} h)")
        summary.append("")
    waves = pd.concat(waves, ignore_index=True)
    both = waves.groupby("welle").agg(jobs=("jobs", "sum"), kernstunden=("kernstunden", "sum"),
                                      laengster_job_h=("laengster_job_h", "max")).reset_index()
    for cores in CORES:
        both[f"tage_{cores}_kerne"] = (np.maximum(both["kernstunden"] / cores, both["laengster_job_h"]) / 24).round(1)
    waves = pd.concat([waves, both.assign(lauf="beide")], ignore_index=True)
    waves.to_csv(out / "kosten_wellen.tsv", sep="\t", index=False)
    summary += ["Beide Laeufe zusammen je Welle (teilen sich dieselben Kerne):"]
    summary += [f"  Welle {r.welle}: {r.jobs} Jobs, {r.kernstunden:,} Kernstunden, bei 64 Kernen ~{r.tage_64_kerne} "
                f"Tage, bei 110 Kernen ~{r.tage_110_kerne} Tage" for r in both.itertuples()]
    summary += [f"Alle Wellen beider Laeufe: {both['jobs'].sum():,} Jobs, {both['kernstunden'].sum():,} Kernstunden. "
                "Dauer = max(Kernstunden / Kerne, laengster Job), Untergrenze ohne Wartezeit in der Warteschlange. "
                "Abschicken: refill.py (Schlange <= 940).", ""]
    (out / "run_paper_job.sbatch").write_text(SBATCH)
    (out / "submit.sh").write_text(SUBMIT)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / ".gitkeep").touch()
    (out / "zusammenfassung.txt").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jobs_paper")
    parser.add_argument("--timing", default="jobs_paper/zeitmessung_mac.tsv")
    parser.add_argument("--memory", default="jobs_paper/speicher_mac.tsv", help="aus measure_memory.py")
    args = parser.parse_args()
    main(args.out, args.timing, args.memory)
