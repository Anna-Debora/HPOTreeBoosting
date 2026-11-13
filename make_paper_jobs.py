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

Zeitlimit pro Job aus dem Kostenmodell (siehe kosten()), aufgerundet auf eine Stufe in HOUR_STEPS; Speicher je
Aufgabe und Methode nach speicher.py (Spitze x 1.3, mindestens 1 GB, Stufen 1, 1.5, 2, 3, 4, 6, 8, ... GB; Spitze
von Euler aus jobs_paper/speicher_euler.tsv, sonst vom Mac aus jobs_paper/speicher_mac.tsv).

Wellen (damit es frueh Zwischenergebnisse gibt): Welle 1 = Seeds 1-5 (Reihenfolge wie utils_modelling.seeds) plus
Default, Welle 2 = Seeds 6-10, Welle 3 = Seeds 11-15, Welle 4 = Seeds 16-20, jeweils alle 59 Aufgaben und alle
Methoden. Abgeschickt wird mit refill.py in Portionen (Euler: hoechstens ~1000 Jobs in der Schlange), in der
Reihenfolge der Spalte 'reihenfolge' in kosten.tsv: Welle, dann schnelle Partition zuerst (normal.4h vor 24h vor
120h), dann Aufgabe nach geschaetzter Laufzeit (kleine zuerst), Seed, Methode. refill.py buendelt Jobs gleicher
Welle, Partition, Spalte 'zeitlimit' (= 'stunden' aufgerundet auf HOUR_STEPS, je Job statt wie frueher das
Maximum der ganzen Gruppe) und Speicher 'mem_gb' zu einem Array.

Eingefroren (31.10.2026): die Wellen-Arrays arrays/welle*.idx und arrays.tsv aus dem ersten Abschicken mit
submit.sh werden nicht mehr geschrieben oder geloescht, weil submitted.tsv auf Euler ueber sie die Array-Indizes
der schon abgeschickten Jobs auf Zeilen von jobs.tsv abbildet. jobs.tsv behaelt seine Reihenfolge.

Lauf "extra" (eigener Aufruf, aendert nichts an paper/l1_270 und an den gemeinsamen Dateien in jobs_paper/):
  extra   BOHB, DEHB, HEBO mit den Einstellungen von paper (GPBoost, 135 Versuche, ohne lambda_l1, 20 Seeds, 5 Folds,
          num_leaves, Kategorien nativ), kein Default (der von paper gilt)            -> Results_extra/
          BOHB/DEHB im Hyperband-Rahmen von paper (R = 2150, eta = 2.8), HEBO 135 volle Auswertungen.
  59 x 20 x 3 = 3540 Jobs, gleiche Wellen (Welle 1 = Seeds 1-5 usw.), Zeitlimit- und Speicherregeln wie oben
  (speicher.py: Extra-Methoden mit Zuschlag EXTRA_OVERHEAD_GB), Account ls_math, eigenes sbatch-Skript
  jobs_paper/run_extra_job.sbatch (.venv_extra, Job-Name hpo-extra, Logs in jobs_paper/logs -> $SCRATCH).
  Probe: jobs_paper/extra/arrays/probe.idx = 3 Methoden x 2 kleine Aufgaben (EXTRA_PROBE) x Seed 27225,
  abschicken mit bash jobs_paper/submit_extra.sh probe.
  Aufruf: python make_paper_jobs.py --run extra
  Ausgabe: jobs_paper/extra/{jobs.tsv, kosten.tsv, kosten_wellen.tsv, zusammenfassung.txt, arrays/probe.idx},
           jobs_paper/run_extra_job.sbatch, jobs_paper/submit_extra.sh

Lauf "gpb" (eigener Aufruf, aendert nichts an paper/l1_270/extra und an den gemeinsamen Dateien in jobs_paper/):
  gpb     korrekte Skalen (run_ablation.py --scales gpboost: log fuer learning_rate, lambda_l2, num_leaves,
          min_data_in_leaf, max_bin) ohne lambda_l1 (--l1 off, fest in jobs.tsv, damit der Standard --l1 on nie
          greift), sonst Einstellungen von paper (GPBoost, 135 Versuche, 20 Seeds, 5 Folds, num_leaves, Kategorien
          nativ). Nur TPE, GP-BO, SMAC, Hyperband; Default, deterministisches Gitter und Zufallsgitter haengen nicht
          von der Skala ab und kommen aus Results_paper (make_table5.py --gpb)          -> Results_gpb/
  59 x 20 x 4 = 4720 Jobs, gleiche Wellen, Zeitlimit- und Speicherregeln wie oben, Kosten mit GPB_COST_FACTOR
  (kleine Lernraten auf der Log-Skala brauchen mehr Boosting-Runden), sbatch-Skript run_paper_job.sbatch (.venv),
  Job-Name hpo-gpb. Probe: jobs_paper/gpb/arrays/probe.idx = 4 Methoden x 2 kleine Aufgaben (GPB_PROBE) x Seed
  27225, abschicken mit bash jobs_paper/submit_gpb.sh probe.
  Aufruf: python make_paper_jobs.py --run gpb
  Ausgabe: jobs_paper/gpb/{jobs.tsv, kosten.tsv, kosten_wellen.tsv, zusammenfassung.txt, arrays/probe.idx},
           jobs_paper/submit_gpb.sh

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

import speicher
from utils_modelling import seeds as SEEDS
from task_types import is_classification

SUITES = (334, 335, 336, 337)
METHODS = ("default", "grid_search", "random_search", "tpe", "gp_bo", "hyperband", "SMAC")
RUNS = {"paper": dict(folder="Results_paper", trials=135, flags="--no_lambda_l1 --paper_grid"),
        "l1_270": dict(folder="Results_l1_270", trials=270, flags="")}
PROBE = (336, 361076, 27225)   # wine_quality: klein, im Testlauf und in den Ablationslaeufen schon gerechnet
EXTRA_METHODS = ("bohb", "dehb", "hebo")
EXTRA_RUN = dict(folder="Results_extra", trials=135, flags="--no_lambda_l1", methods=EXTRA_METHODS)
GPB_METHODS = ("tpe", "gp_bo", "SMAC", "hyperband")
# Mac, 4 Aufgaben (analcatdata_supreme, heloc, wine_quality, eye_movements), Fold 0, 40 Versuche, --l1 off, Wanduhr
# inkl. Test-Refits, gpboost / paper (13.11.2026): TPE 7.2 (411 / 57 s), GP-BO 2.7, SMAC 3.3, Hyperband 1.7 (R = 200);
# einzelne Aufgaben bis 9 (TPE eye_movements). Mit Reserve aufgerundet; nur fuer den Lauf gpb.
GPB_COST_FACTOR = {"tpe": 7.5, "gp_bo": 3.0, "SMAC": 3.5, "hyperband": 2.0}
GPB_RUN = dict(folder="Results_gpb", trials=135, flags="--l1 off --scales gpboost", methods=GPB_METHODS,
               cost_factor=GPB_COST_FACTOR)
ALL_RUNS = {**RUNS, "extra": EXTRA_RUN, "gpb": GPB_RUN}
ALL_METHODS = METHODS + EXTRA_METHODS
# Probe des Laufs extra: die zwei schnellsten Aufgaben laut zeitmessung_mac.tsv (Regression, Klassifikation)
EXTRA_PROBE = ((335, 361093), (337, 361278))   # analcatdata_supreme (Regression), heloc (Klassifikation); Seed 27225
GPB_PROBE = EXTRA_PROBE        # Probe des Laufs gpb: dieselben zwei Aufgaben, Seed 27225, alle 4 Methoden
FOLDS = 5
# Partitionen auf Euler: (Name, laengstes Zeitlimit in Stunden)
PARTITIONS = (("normal.4h", 4), ("normal.24h", 24), ("normal.120h", 120))
SAFETY = 2.0         # Zeitlimit = Schaetzung x SAFETY, aufgerundet auf volle Stunden, mindestens 1 h
# Zeitlimit-Stufen der Arrays: wenige Stufen, damit die Arrays gross bleiben, aber kein 1-h-Job mit 21 h Limit
# (kurze Limits kommen beim Backfill frueher dran; Grenzen 4, 24, 120 = Partitionen)
HOUR_STEPS = (1, 2, 4, 8, 12, 24, 48, 72, 120)

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
# Lauf extra (Mac-Kurzbenchmark 10.11.2026, measure_extra.py, jobs_paper/extra/zeitmessung_extra_mac.tsv; Fold 0 von
# analcatdata_supreme, eye_movements, wine_quality, electricity): Faktor = Trainings + Test-Refits relativ zu
# Hyperband (0.6) bzw. zu 135 x Sekunden je Versuch; Modell = KDE (BOHB), Evolution (DEHB), GP-Fit + NSGA-II (HEBO)
# je Versuch(-Aequivalent), groesster Mac-Wert x EULER_FACTOR. Gemessen (Mittel / Maximum):
#   BOHB Faktor 1.26 / 2.31 (Modell waehlt teurere Konfigurationen: spaeteres early stopping), Modell 0.53 / 0.63 s
#   DEHB Faktor 0.86 / 1.19, Modell 0.03 / 0.04 s
#   HEBO Faktor 0.55 / 1.00, Modell 2.86 / 3.55 s (unabhaengig von der Aufgabe: ~400-480 s je Fold)
METHOD_FACTOR.update({"bohb": 1.3, "dehb": 0.9, "hebo": 1.0})
SURROGATE_SEC.update({("bohb", 135): 0.9, ("dehb", 135): 0.06, ("hebo", 135): 5.3})
DEFAULT_HOURS = 0.1
WAVE_SIZE = 5        # Seeds je Welle
# Kerne fuer die Dauer-Schaetzung: 400 ~ gleichzeitig laufend unter ls_math (3.11.2026: ~358; vorher es_mge mit
# 64 je Partition), 8026 = ls_math-Limit in normal.120h (kleinste Kerngrenze, watch_paper.QOS_LIMITS)
CORES = (400, 8026)
MAX_SUBMIT = 17000   # QOS MaxSubmit ls_math: hoechstens so viele abgeschickte (wartende + laufende) Jobs
MEM_STEPS = speicher.MEM_STEPS     # Speicherstufen je Kern in GB (Regel in speicher.py)


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
    n = 135 if method in ("hyperband", "bohb", "dehb") else trials   # Hyperband-Rahmen: etwa wie 135 Versuche
    per_trial = sec * EULER_FACTOR * METHOD_FACTOR[method] + SURROGATE_SEC.get((method, trials), 0.0)
    return FOLDS * n * per_trial / 3600


def hour_step(hours):
    """Kleinste Zeitlimit-Stufe >= hours."""
    return next((h for h in HOUR_STEPS if h >= hours), HOUR_STEPS[-1])


def partition(hours):
    limit = max(1, math.ceil(hours * SAFETY))
    for name, longest in PARTITIONS:
        if limit <= longest:
            return name, limit
    return PARTITIONS[-1][0], PARTITIONS[-1][1]


def wave(seed):
    """Welle 1-4 eines Seeds (Default, Seed 0: Welle 1)."""
    return 1 if seed not in SEEDS else SEEDS.index(seed) // WAVE_SIZE + 1


def build(run, sec, task_list, memory=None):
    settings = ALL_RUNS[run]
    memory = memory or speicher.Plan(mac={}, euler=pd.DataFrame(columns=speicher.EULER_COLUMNS))
    rows = []
    for t in task_list.itertuples():
        for method in settings.get("methods", METHODS):
            for seed in ([0] if method == "default" else SEEDS):
                hours = kosten(method, settings["trials"], sec[(t.suite, t.task)])
                if "cost_factor" in settings:         # nur gpb; paper, l1_270, extra unveraendert
                    hours *= settings["cost_factor"][method]
                name, limit = partition(hours)
                rows.append(dict(method=method, seed=seed, suite=t.suite, task=t.task, trials=settings["trials"],
                                 flags=settings["flags"] if method != "default" else "", folder=settings["folder"],
                                 name=t.name, kernstunden=round(hours, 3), partition=name, stunden=limit,
                                 mem_gb=memory(t.suite, t.task, method), welle=wave(seed),
                                 sek_versuch=sec[(t.suite, t.task)]))
    return pd.DataFrame(rows)


def ordered(group):
    """Reihenfolge in einem Array: Aufgabe nach geschaetzter Laufzeit (kleine zuerst), dann Seed, dann Methode."""
    key = group.assign(_seed=[SEEDS.index(s) if s in SEEDS else -1 for s in group["seed"]],
                       _method=[ALL_METHODS.index(m) for m in group["method"]])
    return key.sort_values(["sek_versuch", "suite", "task", "_seed", "_method"], kind="stable")


def plan(jobs):
    """Spalten 'zeitlimit' ('stunden' aufgerundet auf HOUR_STEPS) und 'reihenfolge' (1 = zuerst abschicken: Welle,
    schnelle Partition zuerst, kleine Aufgaben zuerst, Seed, Methode)."""
    jobs = jobs.copy()
    jobs["zeitlimit"] = [hour_step(h) for h in jobs["stunden"]]
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
# Nur noch fuer die Probe. Die Wellen schickt refill.py in Portionen ab (Schlange bis --limit, Standard 4000):
#   python refill.py --run beide              # siehe python refill.py --help
#   bash jobs_paper/submit.sh paper probe     # 7 Jobs: wine_quality, Seed 27225, alle Methoden
# Optional: DRY=1 (nur anzeigen). Merkt sich die Job-ID in jobs_paper/<lauf>/submitted.tsv (fuer watch_paper.py).
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
RUN=$1; WHAT=$2
DIR=jobs_paper/$RUN
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-6 --partition=normal.4h --time=4:00:00 --mem-per-cpu=4G
                --job-name="hpo-$RUN" jobs_paper/run_paper_job.sbatch "$RUN" "$DIR/arrays/probe.idx")
           if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; exit 0; fi
           id=$("${cmd[@]}")
           printf '%s\\t%s\\t%s\\t%s\\tnormal.4h\\t4\\t4\\n' "$id" "$DIR/arrays/probe.idx" 7 "$(date '+%F %T')" >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe, 7 Teiljobs)" ;;
    *)     echo "$WHAT: Wellen und Wiederholungen jetzt mit refill.py (python refill.py --help)"; exit 1 ;;
esac
"""


SBATCH_EXTRA = """#!/bin/bash
#SBATCH --job-name=hpo-extra
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs_paper/logs/%x_%A_%a.out
#SBATCH --error=jobs_paper/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Ein Teiljob des Laufs extra (BOHB, DEHB, HEBO; python make_paper_jobs.py --run extra). Nicht direkt abschicken,
# sondern ueber  python refill.py --run extra  (oder --run alle; Probe: bash jobs_paper/submit_extra.sh probe).
# Wie run_paper_job.sbatch, aber mit .venv_extra (hpbandster, dehb, HEBO; euler_install_extra.sh), 1 Thread fuer
# OpenMP/BLAS/torch und Zwischendateien (DEHB-Log, matplotlib-Cache) in $TMPDIR bzw. $SCRATCH statt im Home.
# Wiederholbar: fertige Ergebnisse werden uebersprungen, angefangene ab dem ersten fehlenden Fold fortgesetzt.

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
if [ ! -x .venv_extra/bin/python ]; then
    echo "ABBRUCH: .venv_extra fehlt. Zuerst: bash euler_install_extra.sh" >&2
    exit 2
fi
source .venv_extra/bin/activate
set -u
export TMPDIR="${TMPDIR:-${SCRATCH:-/cluster/scratch/$USER}}"
export MPLCONFIGDIR="$TMPDIR/matplotlib_${SLURM_JOB_ID:-0}"

RUN=$1
IDX_FILE=$2
LINE_NO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IDX_FILE")
JOB_LINE=$(sed -n "${LINE_NO}p" "jobs_paper/$RUN/jobs.tsv")
IFS=$'\\t' read -r METHOD SEED SUITE_ID TASK_ID TRIALS FOLDER FLAGS <<< "$JOB_LINE"
echo "Lauf $RUN Zeile $LINE_NO: $METHOD seed=$SEED ${SUITE_ID}_${TASK_ID} trials=$TRIALS $FLAGS -> $FOLDER"
echo "Start $(date '+%F %T') auf $(hostname), Slurm-Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, TMPDIR $TMPDIR"

DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Zuerst: bash euler_prepare_repro_data.sh jobs_paper/aufgaben.tsv" >&2
    exit 2
fi
python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \\
    --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS
echo "Ende $(date '+%F %T')"
"""

SUBMIT_EXTRA = """#!/bin/bash
# Probe des Laufs extra: {n} Jobs (bohb, dehb, hebo x {names}, Seed {seed}), {part}, {hours} h, {mem} je Kern.
#   bash jobs_paper/submit_extra.sh probe          (DRY=1: nur anzeigen)
# Merkt sich die Job-ID in jobs_paper/extra/submitted.tsv (watch_paper.py zeigt sie, refill.py schickt sie nie doppelt).
# Die Wellen schickt refill.py:  python refill.py --run extra   (oder mit paper/l1_270 zusammen: --run alle)
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
WHAT=${{1:-}}
DIR=jobs_paper/extra
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-{last} --partition={part} --time={hours}:00:00
                --mem-per-cpu={mem} --job-name=hpo-extra jobs_paper/run_extra_job.sbatch extra "$DIR/arrays/probe.idx")
           if [ -n "${{DRY:-}}" ]; then echo "${{cmd[*]}}"; exit 0; fi
           id=$("${{cmd[@]}}")
           printf '%s\\t%s\\t%s\\t%s\\t{part}\\t{hours}\\t{mem_gb}\\n' "$id" "$DIR/arrays/probe.idx" {n} "$(date '+%F %T')" \\
               >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe extra, {n} Teiljobs)" ;;
    *)     echo "Aufruf: bash jobs_paper/submit_extra.sh probe   (Wellen: python refill.py --run extra)"; exit 1 ;;
esac
"""


def main_extra(out, timing, memory="jobs_paper/speicher_mac.tsv", euler="jobs_paper/speicher_euler.tsv"):
    """Lauf extra (BOHB, DEHB, HEBO): schreibt nur jobs_paper/extra/, run_extra_job.sbatch und submit_extra.sh."""
    out = Path(out)
    folder = out / "extra"
    (folder / "arrays").mkdir(parents=True, exist_ok=True)
    sec = seconds_per_trial(timing)
    peaks = speicher.Plan(mac=speicher.mac_peaks(memory), euler=speicher.euler_measurements(euler))
    jobs = plan(build("extra", sec, tasks(), peaks))
    jobs[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
        folder / "jobs.tsv", sep="\t", index=False, header=False)
    probe = jobs[np.array([(s, t) in EXTRA_PROBE for s, t in zip(jobs["suite"], jobs["task"])]) & (jobs["seed"] == PROBE[2])]
    probe_lines = [i + 1 for i in probe.index]
    if not (folder / "arrays" / "probe.idx").exists():
        (folder / "arrays" / "probe.idx").write_text("\n".join(map(str, probe_lines)) + "\n")
    jobs["array"] = ""
    jobs.drop(columns="sek_versuch").to_csv(folder / "kosten.tsv", sep="\t", index=False)
    costs = wave_costs("extra", jobs)
    costs.to_csv(folder / "kosten_wellen.tsv", sep="\t", index=False)
    by_method = jobs.groupby("method")["kernstunden"].sum().round(0).astype(int).to_dict()
    summary = [f"Lauf extra -> {EXTRA_RUN['folder']}/: {len(jobs)} Jobs ({', '.join(EXTRA_METHODS)} x "
               f"{jobs['task'].nunique()} Aufgaben x {jobs['seed'].nunique()} Seeds), geschaetzt "
               f"{jobs['kernstunden'].sum():,.0f} Kernstunden (Zeitlimits zusammen {jobs['stunden'].sum():,} h)",
               f"  je Methode: {by_method}",
               "  teuerste Jobs: " + ", ".join(
                   f"{r.name} {r.method} {r.kernstunden:.1f} h" for r in
                   jobs.sort_values("kernstunden", ascending=False).drop_duplicates(["task", "method"])
                   .head(5).itertuples()),
               "  Partitionen: " + ", ".join(f"{p} {n}" for p, n in jobs["partition"].value_counts().items()),
               "  Speicher je Job: " + ", ".join(f"{speicher.fmt(m)} GB {n}" for m, n in
                                                jobs["mem_gb"].value_counts().sort_index().items()),
               f"  Probe: {len(probe)} Jobs (Zeilen {', '.join(map(str, probe_lines))}): "
               + ", ".join(f"{r.name} {r.method} {r.kernstunden:.2f} h" for r in probe.itertuples())]
    for c in costs.itertuples():
        summary.append(f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}): {c.jobs} Jobs, "
                       f"{c.kernstunden:,} Kernstunden, "
                       + ", ".join(f"bei {k} Kernen ~{getattr(c, f'tage_{k}_kerne')} Tage" for k in CORES)
                       + f" (laengster Job {c.laengster_job_h} h)")
    summary += [f"Alle Wellen: {len(jobs):,} Jobs, {costs['kernstunden'].sum():,} Kernstunden. Abschicken: "
                "python refill.py --run extra (oder --run alle). Probe zuerst: bash jobs_paper/submit_extra.sh probe", ""]
    hours = int(probe["zeitlimit"].max())
    part = next(name for name, longest in PARTITIONS if hours <= longest)
    mem_gb = max(float(probe["mem_gb"].max()), 2.0)
    (out / "run_extra_job.sbatch").write_text(SBATCH_EXTRA)
    (out / "submit_extra.sh").write_text(SUBMIT_EXTRA.format(
        n=len(probe), last=len(probe) - 1, names=", ".join(probe["name"].unique()), seed=PROBE[2], part=part,
        hours=hours, mem=speicher.sbatch_mem(mem_gb), mem_gb=speicher.fmt(mem_gb)))
    (folder / "zusammenfassung.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    return jobs


SUBMIT_GPB = """#!/bin/bash
# Probe des Laufs gpb: {n} Jobs (tpe, gp_bo, SMAC, hyperband x {names}, Seed {seed}), {part}, {hours} h, {mem} je Kern.
#   bash jobs_paper/submit_gpb.sh probe          (DRY=1: nur anzeigen)
# Merkt sich die Job-ID in jobs_paper/gpb/submitted.tsv (watch_paper.py zeigt sie, refill.py schickt sie nie doppelt).
# Die Wellen schickt refill.py:  python refill.py --run gpb   (nicht in --run alle)
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
WHAT=${{1:-}}
DIR=jobs_paper/gpb
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-{last} --partition={part} --time={hours}:00:00
                --mem-per-cpu={mem} --job-name=hpo-gpb jobs_paper/run_paper_job.sbatch gpb "$DIR/arrays/probe.idx")
           if [ -n "${{DRY:-}}" ]; then echo "${{cmd[*]}}"; exit 0; fi
           id=$("${{cmd[@]}}")
           printf '%s\\t%s\\t%s\\t%s\\t{part}\\t{hours}\\t{mem_gb}\\n' "$id" "$DIR/arrays/probe.idx" {n} "$(date '+%F %T')" \\
               >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe gpb, {n} Teiljobs)" ;;
    *)     echo "Aufruf: bash jobs_paper/submit_gpb.sh probe   (Wellen: python refill.py --run gpb)"; exit 1 ;;
esac
"""


def main_gpb(out, timing, memory="jobs_paper/speicher_mac.tsv", euler="jobs_paper/speicher_euler.tsv"):
    """Lauf gpb (korrekte Skalen ohne lambda_l1; TPE, GP-BO, SMAC, Hyperband): schreibt nur jobs_paper/gpb/ und
    submit_gpb.sh (sbatch-Skript: run_paper_job.sbatch aus main())."""
    out = Path(out)
    folder = out / "gpb"
    (folder / "arrays").mkdir(parents=True, exist_ok=True)
    sec = seconds_per_trial(timing)
    peaks = speicher.Plan(mac=speicher.mac_peaks(memory), euler=speicher.euler_measurements(euler))
    jobs = plan(build("gpb", sec, tasks(), peaks))
    jobs[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
        folder / "jobs.tsv", sep="\t", index=False, header=False)
    probe = jobs[np.array([(s, t) in GPB_PROBE for s, t in zip(jobs["suite"], jobs["task"])]) & (jobs["seed"] == PROBE[2])]
    probe_lines = [i + 1 for i in probe.index]
    if not (folder / "arrays" / "probe.idx").exists():
        (folder / "arrays" / "probe.idx").write_text("\n".join(map(str, probe_lines)) + "\n")
    jobs["array"] = ""
    jobs.drop(columns="sek_versuch").to_csv(folder / "kosten.tsv", sep="\t", index=False)
    costs = wave_costs("gpb", jobs)
    costs.to_csv(folder / "kosten_wellen.tsv", sep="\t", index=False)
    by_method = jobs.groupby("method")["kernstunden"].sum().round(0).astype(int).to_dict()
    summary = [f"Lauf gpb -> {GPB_RUN['folder']}/: {len(jobs)} Jobs ({', '.join(GPB_METHODS)} x "
               f"{jobs['task'].nunique()} Aufgaben x {jobs['seed'].nunique()} Seeds), Flags '{GPB_RUN['flags']}', "
               f"geschaetzt {jobs['kernstunden'].sum():,.0f} Kernstunden (Zeitlimits zusammen "
               f"{jobs['stunden'].sum():,} h; Faktoren gegenueber paper {GPB_COST_FACTOR})",
               f"  je Methode: {by_method}",
               "  teuerste Jobs: " + ", ".join(
                   f"{r.name} {r.method} {r.kernstunden:.1f} h" for r in
                   jobs.sort_values("kernstunden", ascending=False).drop_duplicates(["task", "method"])
                   .head(5).itertuples()),
               "  Partitionen: " + ", ".join(f"{p} {n}" for p, n in jobs["partition"].value_counts().items()),
               "  Speicher je Job: " + ", ".join(f"{speicher.fmt(m)} GB {n}" for m, n in
                                                jobs["mem_gb"].value_counts().sort_index().items()),
               f"  Probe: {len(probe)} Jobs (Zeilen {', '.join(map(str, probe_lines))}): "
               + ", ".join(f"{r.name} {r.method} {r.kernstunden:.2f} h" for r in probe.itertuples())]
    for c in costs.itertuples():
        summary.append(f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}): {c.jobs} Jobs, "
                       f"{c.kernstunden:,} Kernstunden, "
                       + ", ".join(f"bei {k} Kernen ~{getattr(c, f'tage_{k}_kerne')} Tage" for k in CORES)
                       + f" (laengster Job {c.laengster_job_h} h)")
    summary += [f"Alle Wellen: {len(jobs):,} Jobs, {costs['kernstunden'].sum():,} Kernstunden. Abschicken: "
                "python refill.py --run gpb. Probe zuerst: bash jobs_paper/submit_gpb.sh probe", ""]
    hours = int(probe["zeitlimit"].max())
    part = next(name for name, longest in PARTITIONS if hours <= longest)
    mem_gb = max(float(probe["mem_gb"].max()), 2.0)
    (out / "submit_gpb.sh").write_text(SUBMIT_GPB.format(
        n=len(probe), last=len(probe) - 1, names=", ".join(probe["name"].unique()), seed=PROBE[2], part=part,
        hours=hours, mem=speicher.sbatch_mem(mem_gb), mem_gb=speicher.fmt(mem_gb)))
    (folder / "zusammenfassung.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    return jobs


def main(out, timing, memory="jobs_paper/speicher_mac.tsv", euler="jobs_paper/speicher_euler.tsv"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sec = seconds_per_trial(timing)
    peaks = speicher.Plan(mac=speicher.mac_peaks(memory), euler=speicher.euler_measurements(euler))
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
        big = jobs[jobs["method"] != "default"].groupby("name")["mem_gb"].max()
        big = big[big > 4].sort_values(ascending=False)
        summary += [f"Lauf {run} -> {RUNS[run]['folder']}/: {len(jobs)} Jobs, geschaetzt "
                    f"{jobs['kernstunden'].sum():,.0f} Kernstunden (Zeitlimits zusammen {jobs['stunden'].sum():,} h)",
                    f"  je Methode: {by_method}",
                    f"  teuerste Jobs: " + ", ".join(
                        f"{r.name} {r.method} {r.kernstunden:.1f} h" for r in
                        jobs.sort_values("kernstunden", ascending=False).drop_duplicates(["task", "method"])
                        .head(5).itertuples()),
                    "  Speicher je Job: " + ", ".join(f"{speicher.fmt(m)} GB {n}" for m, n in
                                                     jobs["mem_gb"].value_counts().sort_index().items())
                    + "; Aufgaben ueber 4 GB: " + ", ".join(f"{n} {speicher.fmt(m)}" for n, m in big.items()),
                    "  Mittlerer Speicher je Job und Partition (Kernstunden-gewichtet = was gleichzeitig laeuft): "
                    + ", ".join(f"{p} {np.average(g['mem_gb'], weights=g['kernstunden']):.2f} GB ({len(g)} Jobs, "
                                f"{g['kernstunden'].sum():,.0f} Kernstunden)"
                                for p, g in jobs.groupby("partition", sort=False)),
                    f"  Probe (schon gerechnet): {len(probe)} Jobs (Zeilen {', '.join(map(str, probe_lines))})"]
        for c in costs.itertuples():
            summary.append(f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}"
                           f"{' + Default' if c.welle == 1 else ''}): {c.jobs} Jobs, {c.kernstunden:,} Kernstunden, "
                           + ", ".join(f"bei {k} Kernen ~{getattr(c, f'tage_{k}_kerne')} Tage" for k in CORES)
                           + f" (laengster Job {c.laengster_job_h} h)")
        summary.append("")
    waves = pd.concat(waves, ignore_index=True)
    both = waves.groupby("welle").agg(jobs=("jobs", "sum"), kernstunden=("kernstunden", "sum"),
                                      laengster_job_h=("laengster_job_h", "max")).reset_index()
    for cores in CORES:
        both[f"tage_{cores}_kerne"] = (np.maximum(both["kernstunden"] / cores, both["laengster_job_h"]) / 24).round(1)
    waves = pd.concat([waves, both.assign(lauf="beide")], ignore_index=True)
    waves.to_csv(out / "kosten_wellen.tsv", sep="\t", index=False)
    summary += ["Beide Laeufe zusammen je Welle (teilen sich dieselben Kerne):"]
    summary += [f"  Welle {r.welle}: {r.jobs} Jobs, {r.kernstunden:,} Kernstunden, "
                + ", ".join(f"bei {k} Kernen ~{getattr(r, f'tage_{k}_kerne')} Tage" for k in CORES)
                for r in both.itertuples()]
    summary += [f"Alle Wellen beider Laeufe: {both['jobs'].sum():,} Jobs, {both['kernstunden'].sum():,} Kernstunden. "
                "Dauer = max(Kernstunden / Kerne, laengster Job), Untergrenze ohne Wartezeit in der Warteschlange. "
                "Abschicken: refill.py (Schlange <= --limit, Standard 4000).", ""]
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
    parser.add_argument("--euler", default="jobs_paper/speicher_euler.tsv", help="aus watch_paper.py --speicher")
    parser.add_argument("--run", choices=("paper+l1_270", "extra", "gpb"), default="paper+l1_270",
                        help="extra: nur den Lauf extra (BOHB, DEHB, HEBO) schreiben; gpb: nur den Lauf gpb (korrekte "
                             "Skalen ohne lambda_l1); Standard: paper und l1_270")
    args = parser.parse_args()
    if args.run == "extra":
        main_extra(args.out, args.timing, args.memory, args.euler)
    elif args.run == "gpb":
        main_gpb(args.out, args.timing, args.memory, args.euler)
    else:
        main(args.out, args.timing, args.memory, args.euler)
