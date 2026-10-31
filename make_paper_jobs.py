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

Zeitlimit und Speicher pro Job aus dem Kostenmodell (siehe kosten()) und der Datengroesse.

Wellen (damit es frueh Zwischenergebnisse gibt): Welle 1 = Seeds 1-5 (Reihenfolge wie utils_modelling.seeds) plus
Default, Welle 2 = Seeds 6-10, Welle 3 = Seeds 11-15, Welle 4 = Seeds 16-20, jeweils alle 59 Aufgaben und alle
Methoden. Jede Welle wird nach (Partition, Speicher) in Slurm-Arrays mit hoechstens --max_array Teiljobs aufgeteilt
(Zeitlimit = groesstes der Gruppe, wie bisher). Innerhalb eines Arrays liegen die Indizes nach geschaetzter Laufzeit
der Aufgabe (kleine zuerst), dann Seed, dann Methode; Slurm startet Array-Teiljobs in Indexreihenfolge, also werden
kleine Aufgaben zuerst fuer alle Methoden und Seeds der Welle fertig. Die Probe-Jobs (schon gerechnet) fehlen in
den Wellen-Arrays. jobs.tsv selbst behaelt seine Reihenfolge (die Zeilennummern der Probe bleiben gueltig).

Aufruf:  python make_paper_jobs.py [--max_array 1000] [--out jobs_paper]
Ausgabe: <out>/<lauf>/jobs.tsv (eine Zeile je Job, ohne Kopfzeile, fuer das sbatch-Skript)
         <out>/<lauf>/kosten.tsv (mit Kopfzeile: Schaetzung, Zeitlimit, Speicher, Array)
         <out>/<lauf>/arrays/welle<w>.<partition>_<mem>G_<k>.idx (Zeilennummern in jobs.tsv je Array) und
         arrays.tsv (Uebersicht mit Spalte welle)
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
MAX_SUBMIT = 17000   # hoechstens so viele abgeschickte (wartende + laufende) Jobs


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


def memory_gb(suite, task):
    """Speicher je Kern nach Groesse der Rohdaten (die Daten werden ganz geladen, dann auf 100'000 Zeilen
    unterabgetastet). Gemessen (Mac, Laden + Kodierung): delays_zurich_transport 306 MB -> 2.0 GB Spitze,
    covertype 58 MB -> 1.2 GB, Higgs 419 MB -> 0.8 GB."""
    folder = Path(f"data/{suite}_{task}")
    mb = sum(f.stat().st_size for f in folder.glob("*")) / 1e6 if folder.exists() else 0
    return 4 if mb <= 100 else 8


def partition(hours):
    limit = max(1, math.ceil(hours * SAFETY))
    for name, longest in PARTITIONS:
        if limit <= longest:
            return name, limit
    return PARTITIONS[-1][0], PARTITIONS[-1][1]


def wave(seed):
    """Welle 1-4 eines Seeds (Default, Seed 0: Welle 1)."""
    return 1 if seed not in SEEDS else SEEDS.index(seed) // WAVE_SIZE + 1


def build(run, sec, task_list):
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
                                 mem_gb=memory_gb(t.suite, t.task), welle=wave(seed),
                                 sek_versuch=sec[(t.suite, t.task)]))
    return pd.DataFrame(rows)


def ordered(group):
    """Reihenfolge in einem Array: Aufgabe nach geschaetzter Laufzeit (kleine zuerst), dann Seed, dann Methode."""
    key = group.assign(_seed=[SEEDS.index(s) if s in SEEDS else -1 for s in group["seed"]],
                       _method=[METHODS.index(m) for m in group["method"]])
    return key.sort_values(["sek_versuch", "suite", "task", "_seed", "_method"], kind="stable")


def arrays(jobs, max_array, skip=()):
    """Je Welle Gruppen gleicher (Partition, Speicher); Zeitlimit = groesstes der Gruppe (auf die Partition
    gedeckelt). skip: Zeilennummern, die nicht mehr abgeschickt werden (Probe)."""
    out = []
    todo = jobs[~(jobs.index + 1).isin(list(skip))]
    for (welle, name, mem), group in todo.groupby(["welle", "partition", "mem_gb"], sort=True):
        lines = (ordered(group).index + 1).tolist()   # Zeilennummern in jobs.tsv (1-basiert, fuer sed -n)
        hours = int(group["stunden"].max())
        for chunk, start in enumerate(range(0, len(lines), max_array)):
            out.append(dict(array=f"welle{welle}.{name}_{mem}G_{chunk}", welle=int(welle), partition=name,
                            stunden=hours, mem_gb=mem, lines=lines[start:start + max_array]))
    return out


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
#   bash jobs_paper/submit.sh <lauf> <array|probe|alle>
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
# Schickt die Arrays eines Laufs ab und merkt sich die Job-IDs in jobs_paper/<lauf>/submitted.tsv
# (fuer watch_paper.py). Aufruf aus ~/HPOTreeBoosting auf Euler:
#   bash jobs_paper/submit.sh paper probe        # 7 Jobs: wine_quality, Seed 27225, alle Methoden
#   bash jobs_paper/submit.sh paper welle1       # alle Arrays der Welle 1 (Seeds 1-5 + Default), ebenso welle2..4
#   bash jobs_paper/submit.sh paper alle         # alle Wellen (fertige Jobs enden sofort)
#   bash jobs_paper/submit.sh paper welle1.normal.4h_4G_0   # ein einzelnes Array (Name aus arrays.tsv)
#   bash jobs_paper/submit.sh paper retry        # Wiederholung: arrays/retry_*.idx aus watch_paper.py --retry
# Optional: DRY=1 (nur anzeigen), THROTTLE=200 (hoechstens so viele Teiljobs eines Arrays gleichzeitig).
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
RUN=$1; WHAT=$2
DIR=jobs_paper/$RUN
mkdir -p jobs_paper/logs
submit() {   # $1 idx-Datei, $2 Partition, $3 Stunden, $4 Speicher GB
    local n; n=$(grep -c . "$1")
    [ "$n" -gt 0 ] || return 0
    local array="0-$((n - 1))${THROTTLE:+%$THROTTLE}"
    local cmd=(sbatch --parsable --array="$array" --partition="$2" --time="$3:00:00" --mem-per-cpu="$4G"
               --job-name="hpo-$RUN" jobs_paper/run_paper_job.sbatch "$RUN" "$1")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; return 0; fi
    local id; id=$("${cmd[@]}")
    printf '%s\\t%s\\t%s\\t%s\\n' "$id" "$1" "$n" "$(date '+%F %T')" >> "$DIR/submitted.tsv"
    echo "abgeschickt: Job $id, $n Teiljobs, $1 ($2, $3 h, $4 GB)"
}
case "$WHAT" in
    probe) submit "$DIR/arrays/probe.idx" normal.4h 4 4 ;;
    alle|welle[0-9]|welle[0-9][0-9])
           W=${WHAT#welle}; [ "$WHAT" = alle ] && W=
           FOUND=0
           while IFS=$'\\t' read -r NAME WELLE PART HOURS MEM N; do
               [ -z "$W" ] || [ "$WELLE" = "$W" ] || continue
               FOUND=1; submit "$DIR/arrays/$NAME.idx" "$PART" "$HOURS" "$MEM"
           done < <(tail -n +2 "$DIR/arrays.tsv")
           [ "$FOUND" = 1 ] || { echo "keine Arrays fuer $WHAT (siehe $DIR/arrays.tsv)"; exit 1; } ;;
    retry) for f in "$DIR"/arrays/retry_*.idx; do
               [ -e "$f" ] || { echo "keine retry_*.idx (zuerst: python watch_paper.py --retry)"; exit 1; }
               IFS=_ read -r _ PART MEMG HOURS <<< "$(basename "$f" .idx)"
               submit "$f" "$PART" "${HOURS%h}" "${MEMG%G}"; done ;;
    *)     LINE=$(awk -F'\\t' -v n="$WHAT" '$1 == n' "$DIR/arrays.tsv")
           [ -n "$LINE" ] || { echo "unbekannt: $WHAT (siehe $DIR/arrays.tsv)"; exit 1; }
           IFS=$'\\t' read -r NAME WELLE PART HOURS MEM N <<< "$LINE"
           submit "$DIR/arrays/$NAME.idx" "$PART" "$HOURS" "$MEM" ;;
esac
"""


def main(out, max_array, timing):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sec = seconds_per_trial(timing)
    task_list = tasks()
    task_list.assign(classification=[is_classification(s) for s in task_list["suite"]]).to_csv(
        out / "aufgaben.tsv", sep="\t", index=False, columns=["suite", "task", "name", "classification"])
    summary, waves = [], []
    for run in RUNS:
        folder = out / run
        (folder / "arrays").mkdir(parents=True, exist_ok=True)
        for old in (folder / "arrays").glob("*.idx"):
            if not old.name.startswith("retry_"):
                old.unlink()
        jobs = build(run, sec, task_list)
        jobs[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
            folder / "jobs.tsv", sep="\t", index=False, header=False)
        probe = jobs[(jobs["suite"] == PROBE[0]) & (jobs["task"] == PROBE[1])
                     & ((jobs["seed"] == PROBE[2]) | (jobs["method"] == "default"))]
        probe_lines = [i + 1 for i in probe.index]
        (folder / "arrays" / "probe.idx").write_text("\n".join(map(str, probe_lines)) + "\n")
        groups = arrays(jobs, max_array, skip=probe_lines)
        jobs["array"] = ""
        jobs.loc[probe.index, "array"] = "probe"
        for g in groups:
            (folder / "arrays" / f"{g['array']}.idx").write_text("\n".join(map(str, g["lines"])) + "\n")
            jobs.loc[[line - 1 for line in g["lines"]], "array"] = g["array"]
        pd.DataFrame([{k: g[k] for k in ("array", "welle", "partition", "stunden", "mem_gb")} | {"n": len(g["lines"])}
                      for g in groups]).to_csv(folder / "arrays.tsv", sep="\t", index=False)
        jobs.drop(columns="sek_versuch").to_csv(folder / "kosten.tsv", sep="\t", index=False)
        costs = wave_costs(run, jobs, skip=probe_lines)
        waves.append(costs)
        by_method = jobs.groupby("method")["kernstunden"].sum().round(0).astype(int).to_dict()
        summary += [f"Lauf {run} -> {RUNS[run]['folder']}/: {len(jobs)} Jobs, geschaetzt "
                    f"{jobs['kernstunden'].sum():,.0f} Kernstunden (Zeitlimits zusammen {jobs['stunden'].sum():,} h)",
                    f"  je Methode: {by_method}",
                    f"  teuerste Jobs: " + ", ".join(
                        f"{r.name} {r.method} {r.kernstunden:.1f} h" for r in
                        jobs.sort_values("kernstunden", ascending=False).drop_duplicates(["task", "method"])
                        .head(5).itertuples()),
                    f"  Probe (schon gerechnet, nicht in den Wellen): {len(probe)} Jobs "
                    f"(Zeilen {', '.join(map(str, probe_lines))})"]
        for c in costs.itertuples():
            part = [g for g in groups if g["welle"] == c.welle]
            summary += [f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}"
                        f"{' + Default' if c.welle == 1 else ''}): {c.jobs} Jobs, {c.kernstunden:,} Kernstunden, "
                        f"bei 64 Kernen ~{c.tage_64_kerne} Tage, bei 110 Kernen ~{c.tage_110_kerne} Tage "
                        f"(laengster Job {c.laengster_job_h} h)",
                        "    Arrays: " + ", ".join(f"{g['array']} ({len(g['lines'])} x {g['stunden']} h, "
                                                  f"{g['mem_gb']} GB)" for g in part)]
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
    summary += [f"Alle Wellen beider Laeufe: {both['jobs'].sum():,} Jobs (Grenze {MAX_SUBMIT:,} abgeschickte Jobs), "
                f"{both['kernstunden'].sum():,} Kernstunden. Dauer = max(Kernstunden / Kerne, laengster Job), "
                "Untergrenze ohne Wartezeit in der Warteschlange.", ""]
    (out / "run_paper_job.sbatch").write_text(SBATCH)
    (out / "submit.sh").write_text(SUBMIT)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / ".gitkeep").touch()
    (out / "zusammenfassung.txt").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jobs_paper")
    parser.add_argument("--max_array", type=int, default=1000,
                        help="hoechstens so viele Teiljobs je Array (Slurm MaxArraySize auf Euler pruefen)")
    parser.add_argument("--timing", default="jobs_paper/zeitmessung_mac.tsv")
    args = parser.parse_args()
    main(args.out, args.max_array, args.timing)
