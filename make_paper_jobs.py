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

Lauf "gpb_l1" (eigener Aufruf, aendert nichts an den anderen Laeufen und an den gemeinsamen Dateien):
  gpb_l1  wie gpb (Skalen nach GPBoost/engine.py, 20 Seeds, 59 Aufgaben, num_leaves), aber MIT lambda_l1 auf der
          Log-Skala (--l1 on --scales gpboost: untere Grenze 0 -> 1e-8, Bereich bis 100) und 270 Versuchen wie
          l1_270 (ein Hyperparameter mehr). Ausgewertet wird spaeter der Stand bei t = 135 (vergleich_l1.py
          --variante gpb). Methoden Zufallsgitter, SMAC, TPE, GP-BO (ohne Hyperband: haengt nicht von der
          Versuchszahl ab, von Fabio nicht verlangt)                                    -> Results_gpb_l1/
          Zufallsgitter: zieht lambda_l1 aus {0, 1, 10, 100} wie in l1_270 (Gitter haengt nicht von --scales ab).
  59 x 20 x 4 = 4720 Jobs, Wellen und Speicher wie oben, Kosten mit GPB_L1_COST_FACTOR. Reihenfolge: laengste
  Jobs zuerst (Spalte reihenfolge nach geschaetzten Kernstunden absteigend, ueber alle Wellen; refill.py haelt
  sich fuer Laeufe in LONGEST_FIRST daran statt Welle/Partition), damit die langen Jobs das Ende nicht verzoegern.
  Probe: 4 Methoden x GPB_PROBE x Seed 27225, abschicken mit bash jobs_paper/submit_gpb_l1.sh probe.
  Aufruf: python make_paper_jobs.py --run gpb_l1
  Ausgabe: jobs_paper/gpb_l1/{jobs.tsv, kosten.tsv, kosten_wellen.tsv, zusammenfassung.txt, arrays/probe.idx},
           jobs_paper/submit_gpb_l1.sh

Lauf "gpb_alle" (eigener Aufruf, aendert nichts an den anderen Laeufen und an den gemeinsamen Dateien):
  gpb_alle  GPBoost auf allen 345 Datensaetzen (59 alte Suiten 334-337 + 286 neue 1001/1002) mit allen Methoden
          (Default, deterministisches Gitter, Zufallsgitter, TPE, GP-BO, SMAC, Hyperband, BOHB, DEHB, HEBO), Skalen
          wie GPBoost (--scales gpboost), OHNE lambda_l1 (--l1 off), T = 135, 20 Seeds, 5 Folds, num_leaves,
          Kategorien nativ                                                               -> Results_gpb_alle/
          Gerechnet wird nur, was es noch nicht gibt (Wiederverwendung, pruefe_wiederverwendung()):
            59 alte Aufgaben: nur BOHB, DEHB, HEBO (Skalen gpboost; der Lauf extra hatte Skalen paper).
                TPE, GP-BO, SMAC, Hyperband kommen aus Results_gpb, Default und beide Gitter aus Results_paper
                (gleiche Seeds, T = 135, ohne lambda_l1; Gitter haengen nicht von der Skala ab).
            286 neue Aufgaben: alle 10 Methoden. Default, Gitter, Zufallsgitter mit den Flags von paper
                ("--no_lambda_l1 --paper_grid": 135er Gitter wie Results_paper), die anderen "--l1 off --scales gpboost".
  59 x 20 x 3 + 286 x (1 + 20 x 9) = 3540 + 51766 = 55306 Jobs. Sekunden je Versuch der neuen Aufgaben aus der
  Stichprobe + Hochrechnung (measure_new_tasks.py), Speicher aus jobs_paper/speicher_neu_mac.tsv (measure_memory.py
  --tasks jobs_paper/aufgaben_neu.tsv). Kosten mit GPB_ALLE_COST_FACTOR (gpb, BOHB/DEHB/HEBO gemessen, s. dort).
  Reihenfolge laengste Jobs zuerst (wie gpb_l1). sbatch-Skript jobs_paper/run_alle_job.sbatch: .venv fuer die
  sieben alten Methoden, .venv_extra fuer BOHB, DEHB, HEBO. Probe: GPB_ALLE_PROBE (Seed 27225), abschicken mit
  bash jobs_paper/submit_gpb_alle.sh probe. Daten der neuen Aufgaben auf Euler: bash euler_setup.sh.
  Aufruf: python make_paper_jobs.py --run gpb_alle
  Ausgabe: jobs_paper/gpb_alle/{jobs.tsv, kosten.tsv, kosten_wellen.tsv, zusammenfassung.txt, arrays/probe.idx},
           jobs_paper/aufgaben_alle.tsv, jobs_paper/run_alle_job.sbatch, jobs_paper/submit_gpb_alle.sh

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
import sys
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
GPB_L1_METHODS = ("random_search", "tpe", "gp_bo", "SMAC")
# Zufallsgitter haengt nicht von der Skala ab (Faktor 1 wie paper/l1_270); TPE, GP-BO, SMAC wie gpb
GPB_L1_COST_FACTOR = {**GPB_COST_FACTOR, "random_search": 1.0}
GPB_L1_RUN = dict(folder="Results_gpb_l1", trials=270, flags="--l1 on --scales gpboost", methods=GPB_L1_METHODS,
                  cost_factor=GPB_L1_COST_FACTOR, longest_first=True)
NEW_SUITES = (1001, 1002)
GPB_ALLE_METHODS = METHODS + EXTRA_METHODS
GPB_ALLE_OLD_METHODS = EXTRA_METHODS            # alte Aufgaben: der Rest kommt aus Results_gpb / Results_paper
PAPER_FLAGS = RUNS["paper"]["flags"]            # Default (ohne Flags), Gitter und Zufallsgitter: wie Results_paper
GPB_ALLE_FLAGS = {m: (PAPER_FLAGS if m in ("grid_search", "random_search") else GPB_RUN["flags"])
                  for m in GPB_ALLE_METHODS}
# Mac, 4 Aufgaben (analcatdata_supreme, heloc, wine_quality, eye_movements), Fold 0, --l1 off, Wanduhr, gpboost /
# paper (23.11.2026, .venv_extra, BOHB/DEHB R = 200, HEBO 40 Versuche, Wanduhr inkl. Prozessstart), Mittel / Maximum:
# BOHB 1.50 / 2.40, DEHB 1.92 / 3.77 (eye_movements), HEBO 1.01 / 1.21 (Modellzeit unabhaengig von der Skala).
# Mit Reserve aufgerundet wie bei gpb (Mittel -> Faktor).
GPB_ALLE_EXTRA_FACTOR = {"bohb": 2.0, "dehb": 2.5, "hebo": 1.3}
GPB_ALLE_COST_FACTOR = {"default": 1.0, "grid_search": 1.0, "random_search": 1.0, **GPB_COST_FACTOR,
                        **GPB_ALLE_EXTRA_FACTOR}
GPB_ALLE_RUN = dict(folder="Results_gpb_alle", trials=135, flags=GPB_RUN["flags"], methods=GPB_ALLE_METHODS,
                    flags_by_method=GPB_ALLE_FLAGS, cost_factor=GPB_ALLE_COST_FACTOR, longest_first=True)
# Probe: kleine neue Aufgaben (Regression, Klassifikation) mit allen 10 Methoden, dazu BOHB/DEHB/HEBO auf
# analcatdata_supreme (alte Aufgabe); Seed 27225 (Default: Seed 0)
GPB_ALLE_PROBE = ((1001, 920074), (1002, 930017), (335, 361093))


def library_run(library, folder):
    """Lauf xgb_alle / cat_alle: alle 345 Aufgaben, alle 10 Methoden, komplett neu (keine Wiederverwendung),
    --library <library> --l1 off --scales gpboost (Suchraeume SUCHRAEUME_XGB_CAT.md), Gitter zusaetzlich
    --paper_grid (135 Punkte), Default --library <library> (Bibliotheks-Default). Kosten = GPBoost-Kosten des Jobs
    (gleiche Faktoren wie gpb_alle) x Laufzeitverhaeltnis Bibliothek / GPBoost je Aufgabe (measure_new_tasks.py
    --library), Speicher aus jobs_paper/speicher_<library>_mac.tsv (measure_memory.py --library)."""
    base = f"--library {library} --l1 off --scales gpboost"
    flags = {m: base + (" --paper_grid" if m == "grid_search" else "") for m in GPB_ALLE_METHODS}
    return dict(folder=folder, trials=135, flags=base, methods=GPB_ALLE_METHODS, flags_by_method=flags,
                default_flags=f"--library {library}", cost_factor=GPB_ALLE_COST_FACTOR, longest_first=True,
                library=library)


XGB_ALLE_RUN = {**library_run("xgboost", "Results_xgb_alle"), "fold_split_minutes": 240, "bundle_minutes": (60, 120)}
# xgb_alle (Euler-Support 26.11.2026: keine Jobs unter 5-10 min). Alle Minuten sind "echte" Minuten = Schaetzung
# (kernstunden) x Faktor echt/geschaetzt je Methode aus gpb_alle Welle 1 (jobs_paper/faktor_echt_gpb_alle.tsv,
# python make_paper_jobs.py --faktor laufzeiten_gpb_alle.tsv):
#   Jobs mit mehr als 240 min je Fold -> 5 Fold-Jobs (--fold 0..4 in der Spalte flags, run_ablation.py --fold);
#   Jobs unter 60 min -> Buendel von 60-120 min (erst je Welle und Aufgabe, Reste aufgabenuebergreifend), alle
#   anderen je ein Buendel; jobs_paper/xgb_alle/buendel.tsv, run_buendel_job.sbatch, refill.py schickt Buendel ab.
#   Zeitlimit je Buendel: Summe Schaetzung x Faktor q90 x SAFETY; Speicher: Maximum der Zeilen.
#   Probe: Seed 27225 der Probe-Aufgaben als Fold-Jobs, je Methode ein Buendel mit den 5 Folds (Default dazu).
# Vergleich mit dem alten Format (ein Job, 5 Folds): XGB_VERGLEICH, bash jobs_paper/submit_xgb_alle.sh vergleich
FAKTOR_ECHT = "jobs_paper/faktor_echt_gpb_alle.tsv"
XGB_VERGLEICH = ((1001, 920074), (1002, 930017))     # housing_price_prediction (Regression), Customer_Personality (Kl.)
WELLE1_CORES = (60, 500, 600, 800, 1000, 2000)
CAT_ALLE_RUN = library_run("catboost", "Results_cat_alle")
ALLE_RUNS = {"gpb_alle": GPB_ALLE_RUN, "xgb_alle": XGB_ALLE_RUN, "cat_alle": CAT_ALLE_RUN}
LIBRARY_RUNS = ("xgb_alle", "cat_alle")
ALL_RUNS = {**RUNS, "extra": EXTRA_RUN, "gpb": GPB_RUN, "gpb_l1": GPB_L1_RUN, **ALLE_RUNS}
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


def tasks(suites=SUITES):
    rows = []
    for suite in suites:
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


def build(run, sec, task_list, memory=None, methods=None):
    """methods: Methoden fuer diese Aufgabenliste (Standard: die des Laufs; gpb_alle: alte Aufgaben nur BOHB/DEHB/HEBO).
    flags_by_method (gpb_alle): Flags je Methode statt einer Zeile fuer alle."""
    settings = ALL_RUNS[run]
    memory = memory or speicher.Plan(mac={}, euler=pd.DataFrame(columns=speicher.EULER_COLUMNS))
    flags = settings.get("flags_by_method", {})
    rows = []
    for t in task_list.itertuples():
        for method in methods or settings.get("methods", METHODS):
            for seed in ([0] if method == "default" else SEEDS):
                hours = kosten(method, settings["trials"], sec[(t.suite, t.task)])
                if "cost_factor" in settings:         # nur gpb, gpb_l1, gpb_alle; paper, l1_270, extra unveraendert
                    hours *= settings["cost_factor"][method]
                name, limit = partition(hours)
                rows.append(dict(method=method, seed=seed, suite=t.suite, task=t.task, trials=settings["trials"],
                                 flags=(flags.get(method, settings["flags"]) if method != "default"
                                        else settings.get("default_flags", "")),
                                 folder=settings["folder"],
                                 name=t.name, kernstunden=round(hours, 3), partition=name, stunden=limit,
                                 mem_gb=memory(t.suite, t.task, method), welle=wave(seed),
                                 sek_versuch=sec[(t.suite, t.task)]))
    return pd.DataFrame(rows)


def ordered(group):
    """Reihenfolge in einem Array: Aufgabe nach geschaetzter Laufzeit (kleine zuerst), dann Seed, dann Methode."""
    key = group.assign(_seed=[SEEDS.index(s) if s in SEEDS else -1 for s in group["seed"]],
                       _method=[ALL_METHODS.index(m) for m in group["method"]])
    return key.sort_values(["sek_versuch", "suite", "task", "_seed", "_method"], kind="stable")


def plan(jobs, longest_first=False):
    """Spalten 'zeitlimit' ('stunden' aufgerundet auf HOUR_STEPS) und 'reihenfolge' (1 = zuerst abschicken: Welle,
    schnelle Partition zuerst, kleine Aufgaben zuerst, Seed, Methode). longest_first (gpb_l1): geschaetzte
    Kernstunden absteigend ueber alle Wellen, bei Gleichstand Welle, Aufgabe, Seed, Methode."""
    jobs = jobs.copy()
    jobs["zeitlimit"] = [hour_step(h) for h in jobs["stunden"]]
    hours = dict(PARTITIONS)
    keyed = ordered(jobs.assign(_part=jobs["partition"].map(hours)))
    if longest_first:
        order = keyed.assign(_neg=-keyed["kernstunden"]).sort_values(["_neg", "welle"], kind="stable").index
    else:
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
# Probe des Laufs {run}: {n} Jobs ({methods} x {names}, Seed {seed}), {part}, {hours} h, {mem} je Kern.
#   bash jobs_paper/submit_{run}.sh probe          (DRY=1: nur anzeigen)
# Merkt sich die Job-ID in jobs_paper/{run}/submitted.tsv (watch_paper.py zeigt sie, refill.py schickt sie nie doppelt).
# Die Wellen schickt refill.py:  python refill.py --run {run}   (nicht in --run alle)
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
WHAT=${{1:-}}
DIR=jobs_paper/{run}
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-{last} --partition={part} --time={hours}:00:00
                --mem-per-cpu={mem} --job-name=hpo-{run} jobs_paper/run_paper_job.sbatch {run} "$DIR/arrays/probe.idx")
           if [ -n "${{DRY:-}}" ]; then echo "${{cmd[*]}}"; exit 0; fi
           id=$("${{cmd[@]}}")
           printf '%s\\t%s\\t%s\\t%s\\t{part}\\t{hours}\\t{mem_gb}\\n' "$id" "$DIR/arrays/probe.idx" {n} "$(date '+%F %T')" \\
               >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe {run}, {n} Teiljobs)" ;;
    *)     echo "Aufruf: bash jobs_paper/submit_{run}.sh probe   (Wellen: python refill.py --run {run})"; exit 1 ;;
esac
"""


def main_gpb(out, timing, memory="jobs_paper/speicher_mac.tsv", euler="jobs_paper/speicher_euler.tsv", run="gpb"):
    """Lauf gpb (korrekte Skalen ohne lambda_l1; TPE, GP-BO, SMAC, Hyperband) oder gpb_l1 (mit lambda_l1 log,
    270 Versuche; Zufallsgitter, TPE, GP-BO, SMAC): schreibt nur jobs_paper/<run>/ und submit_<run>.sh
    (sbatch-Skript: run_paper_job.sbatch aus main())."""
    settings = ALL_RUNS[run]
    methods, factors = settings["methods"], settings["cost_factor"]
    out = Path(out)
    folder = out / run
    (folder / "arrays").mkdir(parents=True, exist_ok=True)
    sec = seconds_per_trial(timing)
    peaks = speicher.Plan(mac=speicher.mac_peaks(memory), euler=speicher.euler_measurements(euler))
    jobs = plan(build(run, sec, tasks(), peaks), longest_first=settings.get("longest_first", False))
    jobs[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
        folder / "jobs.tsv", sep="\t", index=False, header=False)
    probe = jobs[np.array([(s, t) in GPB_PROBE for s, t in zip(jobs["suite"], jobs["task"])]) & (jobs["seed"] == PROBE[2])]
    probe_lines = [i + 1 for i in probe.index]
    if not (folder / "arrays" / "probe.idx").exists():
        (folder / "arrays" / "probe.idx").write_text("\n".join(map(str, probe_lines)) + "\n")
    jobs["array"] = ""
    jobs.drop(columns="sek_versuch").to_csv(folder / "kosten.tsv", sep="\t", index=False)
    costs = wave_costs(run, jobs)
    costs.to_csv(folder / "kosten_wellen.tsv", sep="\t", index=False)
    by_method = jobs.groupby("method")["kernstunden"].sum().round(0).astype(int).to_dict()
    summary = [f"Lauf {run} -> {settings['folder']}/: {len(jobs)} Jobs ({', '.join(methods)} x "
               f"{jobs['task'].nunique()} Aufgaben x {jobs['seed'].nunique()} Seeds), Flags '{settings['flags']}', "
               + (f"{settings['trials']} Versuche, " if run != "gpb" else "")
               + f"geschaetzt {jobs['kernstunden'].sum():,.0f} Kernstunden (Zeitlimits zusammen "
               f"{jobs['stunden'].sum():,} h; Faktoren gegenueber paper {factors})",
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
    if settings.get("longest_first"):
        summary.append("  Reihenfolge: laengste Jobs zuerst (geschaetzte Kernstunden absteigend, ueber alle Wellen)")
    for c in costs.itertuples():
        summary.append(f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}): {c.jobs} Jobs, "
                       f"{c.kernstunden:,} Kernstunden, "
                       + ", ".join(f"bei {k} Kernen ~{getattr(c, f'tage_{k}_kerne')} Tage" for k in CORES)
                       + f" (laengster Job {c.laengster_job_h} h)")
    summary += [f"Alle Wellen: {len(jobs):,} Jobs, {costs['kernstunden'].sum():,} Kernstunden. Abschicken: "
                f"python refill.py --run {run}. Probe zuerst: bash jobs_paper/submit_{run}.sh probe", ""]
    hours = int(probe["zeitlimit"].max())
    part = next(name for name, longest in PARTITIONS if hours <= longest)
    mem_gb = max(float(probe["mem_gb"].max()), 2.0)
    (out / f"submit_{run}.sh").write_text(SUBMIT_GPB.format(
        run=run, methods=", ".join(methods), n=len(probe), last=len(probe) - 1,
        names=", ".join(probe["name"].unique()), seed=PROBE[2], part=part,
        hours=hours, mem=speicher.sbatch_mem(mem_gb), mem_gb=speicher.fmt(mem_gb)))
    (folder / "zusammenfassung.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    return jobs


SBATCH_ALLE = """#!/bin/bash
#SBATCH --job-name=hpo-alle
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs_paper/logs/%x_%A_%a.out
#SBATCH --error=jobs_paper/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Ein Teiljob der Laeufe gpb_alle, xgb_alle, cat_alle (python make_paper_jobs.py --run <lauf>). Nicht direkt
# abschicken, sondern ueber  python refill.py --run <lauf>   (Probe: bash jobs_paper/submit_<lauf>.sh probe).
# Bibliothek aus der Spalte flags (--library, sonst gpboost), auch fuer den Default (run_default.py --library).
# venv je Methode: bohb, dehb, hebo mit .venv_extra (wie run_extra_job.sbatch, Zwischendateien in $TMPDIR/$SCRATCH),
# alle anderen mit .venv (wie run_paper_job.sbatch). Wiederholbar wie dort (--resume, Default: fertige Datei bleibt).

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"

RUN=$1
IDX_FILE=$2
LINE_NO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IDX_FILE")
JOB_LINE=$(sed -n "${LINE_NO}p" "jobs_paper/$RUN/jobs.tsv")
IFS=$'\\t' read -r METHOD SEED SUITE_ID TASK_ID TRIALS FOLDER FLAGS <<< "$JOB_LINE"
LIB=$(echo "$FLAGS" | sed -n 's/.*--library \\([a-z]*\\).*/\\1/p')
LIB=${LIB:-gpboost}

case "$METHOD" in
    bohb|dehb|hebo) VENV=.venv_extra
                    export TMPDIR="${TMPDIR:-${SCRATCH:-/cluster/scratch/$USER}}"
                    export MPLCONFIGDIR="$TMPDIR/matplotlib_${SLURM_JOB_ID:-0}" ;;
    *)              VENV=.venv ;;
esac
if [ ! -x "$VENV/bin/python" ]; then
    echo "ABBRUCH: $VENV fehlt (.venv_extra: bash euler_install_extra.sh)" >&2
    exit 2
fi
source "$VENV/bin/activate"
set -u
echo "Lauf $RUN Zeile $LINE_NO: $METHOD seed=$SEED ${SUITE_ID}_${TASK_ID} trials=$TRIALS $FLAGS -> $FOLDER ($VENV)"
echo "Start $(date '+%F %T') auf $(hostname), Slurm-Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# CPU-Modell fuer die Laufzeitauswertung (runtime-Spalte haengt vom Knotentyp ab)
echo "CPU: $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"

DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Neue Aufgaben: bash euler_setup.sh; alte: bash euler_prepare_repro_data.sh jobs_paper/aufgaben.tsv" >&2
    exit 2
fi

if [ "$METHOD" = "default" ]; then
    OUT="$FOLDER/default/$LIB/${SUITE_ID}_${TASK_ID}.csv"
    if [ -s "$OUT" ]; then
        echo "SCHON FERTIG: $OUT"
    else
        python run_default.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --result_folder "$FOLDER/default" \\
            --library "$LIB"
    fi
else
    python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \\
        --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS
fi
echo "Ende $(date '+%F %T')"
"""


def seconds_all(pred="jobs_paper/zeitmessung_neu_hochrechnung.tsv"):
    """Mac-Sekunden je Versuch fuer alle 345 Aufgaben (measure_new_tasks.py: alt gemessen, neu gemessen oder
    hochgerechnet mit Reserve) und die Quelle je Aufgabe."""
    frame = pd.read_csv(pred, sep="\t")
    sec = {(int(r.suite), int(r.task)): float(r.sek_versuch) for r in frame.itertuples()}
    return sec, {(int(r.suite), int(r.task)): r.quelle for r in frame.itertuples()}


REUSE = (("gpb", ("tpe", "gp_bo", "SMAC", "hyperband")), ("paper", ("default", "grid_search", "random_search")))


def pruefe_wiederverwendung(out, results=True):
    """Prueft, dass die wiederverwendeten Ergebnisse der 59 alten Aufgaben dieselben Einstellungen haben wie
    gpb_alle: Seeds, T, Flags (lambda_l1 aus, Skala) aus jobs_paper/<lauf>/jobs.tsv, und (results=True, falls die
    Ordner da sind) die Dateien: vollstaendig (check_results.check_file), lambda_l1 ueberall 0, Skala an der
    learning_rate erkennbar: Anteil < 0.01 bei log-uniform auf [0.001, 1] ~ 1/3, linear ~ 0.9 %. TPE, GP-BO, SMAC:
    in den zufaelligen Startversuchen (iter < 10) muss er ueber 20 % liegen; Hyperband (nur 11 ausgewaehlte Zeilen
    je Fold): ueber 3 % und mehr als das Dreifache derselben Methode mit linearen Skalen in Results_paper.
    Gibt (ok, Zeilen fuer zusammenfassung.txt) zurueck."""
    import check_results
    out = Path(out)
    lines, ok = ["Pruefung der wiederverwendeten Ergebnisse (59 alte Aufgaben):"], True
    old = {(t.suite, t.task) for t in tasks().itertuples()}
    for run, methods in REUSE:
        jobs = pd.read_csv(out / run / "jobs.tsv", sep="\t", header=None, keep_default_na=False,
                           names=["method", "seed", "suite", "task", "trials", "folder", "flags"])
        for method in methods:
            sub = jobs[jobs["method"] == method]
            want = "" if method == "default" else GPB_ALLE_FLAGS[method]
            seeds = sorted(set(sub["seed"])) == sorted([0] if method == "default" else SEEDS)
            budget = set(sub["trials"]) == {135}
            flags = set(sub["flags"]) == {want}
            cover = {(int(s), int(t)) for s, t in zip(sub["suite"], sub["task"])} == old
            good = seeds and budget and flags and cover
            text = (f"  {method:13s} aus {sub['folder'].iloc[0]}: {len(sub)} Jobs, Seeds {'gleich' if seeds else 'ANDERS'}, "
                    f"T = {sorted(set(sub['trials']))}, Flags '{sub['flags'].iloc[0]}' "
                    f"({'gleich' if flags else 'ANDERS als ' + repr(want)}), Aufgaben {'alle 59' if cover else 'NICHT alle 59'}")
            if results and Path(sub["folder"].iloc[0]).exists():
                states, l1, low, low_paper = [], [], [], []
                for job in sub.itertuples():
                    path, kind, n_rows = check_results.result_path(job)
                    state = check_results.check_file(path, kind, n_rows)[0] if path.exists() else "fehlt"
                    states.append(state)
                    if state == "ok" and kind == "hpo":
                        frame = pd.read_csv(path, usecols=lambda c: c in ("lambda_l1", "learning_rate", "iter"))
                        if "lambda_l1" in frame:
                            l1.append(bool((frame["lambda_l1"].fillna(0) == 0).all()))
                        if method in ("tpe", "gp_bo", "SMAC", "hyperband"):
                            rows = frame if method == "hyperband" else frame[frame["iter"] < 10]
                            low.append(float((rows["learning_rate"] < 0.01).mean()))
                            if method == "hyperband":
                                ref = Path("Results_paper") / path.relative_to(job.folder)
                                if ref.exists():
                                    low_paper.append(float((pd.read_csv(ref, usecols=["learning_rate"])
                                                            ["learning_rate"] < 0.01).mean()))
                n_ok = states.count("ok")
                good = good and n_ok == len(sub) and all(l1)
                text += f"; Dateien {n_ok}/{len(sub)} vollstaendig, lambda_l1 = 0 in {sum(l1)}/{len(l1)}"
                if low:
                    share = float(np.mean(low))
                    if method == "hyperband":
                        paper = float(np.mean(low_paper)) if low_paper else float("nan")
                        log_scale = share > 0.03 and not share <= 3 * paper
                        text += (f", learning_rate < 0.01 in {share:.1%} der Zeilen (Results_paper, linear: {paper:.1%}; "
                                 f"{'Log-Skala' if log_scale else 'NICHT log'})")
                    else:
                        log_scale = share > 0.2
                        text += (f", learning_rate < 0.01 in {share:.1%} der Startversuche (log ~33 %, linear ~1 %; "
                                 f"{'Log-Skala' if log_scale else 'NICHT log'})")
                    good = good and log_scale
            else:
                text += "; Dateien nicht geprueft (Ordner nicht da)"
            ok = ok and good
            lines.append(text + (" -> ok" if good else " -> PROBLEM"))
    lines.append("  Ergebnis: " + ("alle Einstellungen gleich, Wiederverwendung ok" if ok else
                                   "UNTERSCHIEDE, vor dem Zusammenfuehren klaeren"))
    return ok, lines


SUBMIT_ALLE = SUBMIT_GPB.replace("jobs_paper/run_paper_job.sbatch", "jobs_paper/run_alle_job.sbatch")
NEW_VS_GPB = {
    "gpb_alle": ["Neu gegenueber gpb: 286 neue Datensaetze (alle Methoden inkl. Default und Gitter), BOHB/DEHB/HEBO mit "
                 "Skalen gpboost (auch auf den 59 alten Aufgaben). Gleich: Skalen, ohne lambda_l1, T = 135, Seeds."],
    "xgb_alle": ["Neu gegenueber gpb: Bibliothek XGBoost (Suchraum SUCHRAEUME_XGB_CAT.md: max_leaves, min_child_weight = "
                 "min_data_in_leaf x0.25 bei Klassifikation, reg_lambda, subsample, colsample_bytree, max_bin; lossguide, "
                 "max_depth 0), alle 345 Datensaetze, alle 10 Methoden inkl. Default und Gitter, komplett neu gerechnet "
                 "(Gitter: XGBoost-Pfad, ohne lambda_l1, 135 Punkte). Gleich: Skalen wie GPBoost, ohne lambda_l1, T = 135, "
                 "Seeds, Folds, num_leaves."],
    "cat_alle": ["Neu gegenueber gpb: Bibliothek CatBoost (Suchraum SUCHRAEUME_XGB_CAT.md: depth [1, 10] linear statt "
                 "num_leaves, ohne min_data_in_leaf, l2_leaf_reg, border_count bis 1024, subsample mit bootstrap_type "
                 "Bernoulli, rsm), alle 345 Datensaetze, alle 10 Methoden inkl. Default und Gitter, komplett neu "
                 "gerechnet. Deterministisches Gitter: learning_rate {0.01, 0.1, 1} x lambda_l2 {0, 1, 10} x depth "
                 "{1, 2, 3, 5, 10} x subsample {0.5, 0.75, 1} = 135 (subsample ersetzt das wegfallende "
                 "min_data_in_leaf). Gleich: Skalen wie GPBoost, ohne lambda_l1, T = 135, Seeds, Folds."],
}
LIBRARY_NAMES = {"gpb_alle": "GPBoost", "xgb_alle": "XGBoost", "cat_alle": "CatBoost"}


def echt_faktoren(laufzeiten, out=FAKTOR_ECHT):
    """Faktor echte / geschaetzte Laufzeit je Methode aus gpb_alle (laufzeiten_gpb_alle.tsv von Euler: je Versuch
    line, state, sekunden, method, kernstunden). Gezaehlt werden Zeilen mit genau einem Versuch, COMPLETED (ohne
    Wiederholungen, OOM, TIMEOUT). Schreibt method, jobs, faktor (Median), faktor_q90, sekunden_median."""
    t = pd.read_csv(laufzeiten, sep="\t")
    one = t.groupby("line").filter(lambda g: len(g) == 1)
    one = one[(one["state"] == "COMPLETED") & (one["sekunden"] > 0) & (one["kernstunden"] > 0)]
    one = one.assign(r=one["sekunden"] / (one["kernstunden"] * 3600))
    table = one.groupby("method").agg(jobs=("r", "size"), faktor=("r", "median"),
                                      faktor_q90=("r", lambda r: r.quantile(0.9)),
                                      sekunden_median=("sekunden", "median")).reset_index()
    table.round(4).to_csv(out, sep="\t", index=False)
    return table


def faktoren(path=FAKTOR_ECHT):
    """{Methode: (Median, q90)} echt/geschaetzt; ohne Datei 1.0 (Schaetzung unveraendert)."""
    if not Path(path).exists():
        return {}
    return {r.method: (float(r.faktor), float(r.faktor_q90)) for r in pd.read_csv(path, sep="\t").itertuples()}


def split_folds(jobs, threshold, force):
    """Jobs (ausser Default) mit mehr als `threshold` echten Minuten je Fold (Spalte minuten_echt), und die Jobs mit
    force True, als 5 Fold-Jobs: flags + ' --fold k', Kernstunden und Minuten / 5, Partition und Zeitlimit je Fold."""
    rows = []
    for row, forced in zip(jobs.to_dict("records"), force):
        if row["method"] == "default" or not (row["minuten_echt"] / FOLDS > threshold or forced):
            rows.append(row)
            continue
        hours = row["kernstunden"] / FOLDS
        name, limit = partition(hours)
        rows += [{**row, "flags": f"{row['flags']} --fold {k}", "kernstunden": round(hours, 3),
                  "minuten_echt": round(row["minuten_echt"] / FOLDS, 2), "partition": name, "stunden": limit}
                 for k in range(FOLDS)]
    return pd.DataFrame(rows, columns=jobs.columns)


def make_bundles(jobs, low, high, factor, probe):
    """buendel.tsv (siehe XGB_ALLE_RUN): jobs mit Spalte minuten_echt, Zeile = Index + 1; probe: Maske der
    Probe-Zeilen (Seed 27225 als Fold-Jobs und Default der Probe-Aufgaben). Gibt (Tabelle, Probe-Buendel) zurueck."""
    groups, probe_groups = [], []
    lines = jobs.assign(line=jobs.index + 1, _probe=list(probe))
    for (suite, task), g in lines[lines["_probe"]].groupby(["suite", "task"], sort=False):
        default = list(g.loc[g["method"] == "default", "line"])
        for k, (method, m) in enumerate(g[g["method"] != "default"].groupby("method", sort=False)):
            probe_groups.append((default if k == 0 else []) + list(m["line"]))
    rest = lines[~lines["_probe"]]
    for welle, w in rest.groupby("welle"):
        short = w[w["minuten_echt"] < low]
        groups += [[l] for l in w.loc[w["minuten_echt"] >= low, "line"]]
        leftovers = []
        for _, g in short.groupby(["suite", "task"], sort=False):
            current, total = [], 0.0
            for r in g.sort_values("minuten_echt", ascending=False, kind="stable").itertuples():
                if current and total + r.minuten_echt > high:
                    (groups if total >= low else leftovers).append(current)
                    current, total = [], 0.0
                current.append(r.line)
                total += r.minuten_echt
                if total >= low:
                    groups.append(current)
                    current, total = [], 0.0
            if current:
                leftovers.append(current)
        minutes = lines.set_index("line")["minuten_echt"]
        current, total = [], 0.0
        for part in leftovers:
            size = float(minutes[part].sum())
            if current and total + size > high:
                groups.append(current)
                current, total = [], 0.0
            current += part
            total += size
            if total >= low:
                groups.append(current)
                current, total = [], 0.0
        if current:
            groups.append(current)
    by_line = lines.set_index("line")
    rows = []
    for zeilen in probe_groups + groups:
        sub = by_line.loc[zeilen]
        q90 = sum(h * factor.get(m, (1.0, 1.0))[1] for h, m in zip(sub["kernstunden"], sub["method"]))
        name, limit = partition(q90)
        rows.append(dict(zeilen=",".join(map(str, zeilen)), welle=int(sub["welle"].max()),
                         kernstunden=round(float(sub["kernstunden"].sum()), 3),
                         minuten_echt=round(float(sub["minuten_echt"].sum()), 1), partition=name, stunden=limit,
                         zeitlimit=hour_step(limit), mem_gb=float(sub["mem_gb"].max())))
    table = pd.DataFrame(rows)
    table.insert(0, "buendel", np.arange(1, len(table) + 1))
    order = table.assign(_neg=-table["minuten_echt"]).sort_values(["_neg", "welle"], kind="stable").index
    table.loc[order, "reihenfolge"] = np.arange(1, len(table) + 1)
    table["reihenfolge"] = table["reihenfolge"].astype(int)
    return table, list(range(1, len(probe_groups) + 1))


def welle1_tage(hours, cores=WELLE1_CORES):
    """Tage fuer Welle 1 bei so vielen gleichzeitigen Kernen (Jobs): laengste Jobs zuerst auf den naechsten
    freien Kern (wie refill.py, LONGEST_FIRST), Ende des letzten Jobs."""
    import heapq
    out = {}
    for c in cores:
        free = [0.0] * c
        for h in sorted(hours, reverse=True):
            heapq.heappush(free, heapq.heappop(free) + h)
        out[c] = round(max(free) / 24, 1)
    return out


SUBMIT_VERGLEICH = """    vergleich) # Vergleich mit dem alten Format: {n} Jobs (ein Job, 5 Folds) nach Results_xgb_alle_vergleich
           cmd=(sbatch --parsable --account=ls_math --array=0-{last} --partition={part} --time={hours}:00:00
                --mem-per-cpu={mem} --job-name=hpo-xgb_alle_vergleich jobs_paper/run_alle_job.sbatch xgb_alle_vergleich
                jobs_paper/xgb_alle_vergleich/alle.idx)
           if [ -n "${{DRY:-}}" ]; then echo "${{cmd[*]}}"; exit 0; fi
           id=$("${{cmd[@]}}")
           printf '%s\\t%s\\t%s\\t%s\\t{part}\\t{hours}\\t{mem_gb}\\n' "$id" jobs_paper/xgb_alle_vergleich/alle.idx {n} \\
               "$(date '+%F %T')" >> jobs_paper/xgb_alle_vergleich/submitted.tsv
           echo "abgeschickt: Job $id (Vergleich altes Format, {n} Teiljobs); danach: python vergleiche_folds.py" ;;
"""


def vergleich_jobs(out, whole):
    """Lauf xgb_alle_vergleich: XGB_VERGLEICH x 9 Methoden, Seed 27225, wie xgb_alle, aber ein Job mit 5 Folds
    (altes Format) nach Results_xgb_alle_vergleich; schreibt jobs.tsv und alle.idx, gibt den submit-Fall zurueck."""
    rows = whole[np.array([(s, t) in XGB_VERGLEICH for s, t in zip(whole["suite"], whole["task"])])
                 & (whole["seed"] == PROBE[2])].assign(folder="Results_xgb_alle_vergleich")
    folder = Path(out) / "xgb_alle_vergleich"
    folder.mkdir(parents=True, exist_ok=True)
    rows[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
        folder / "jobs.tsv", sep="\t", index=False, header=False)
    (folder / "alle.idx").write_text("\n".join(map(str, range(1, len(rows) + 1))) + "\n")
    hours = hour_step(max(int(rows["stunden"].max()), 1))
    part = next(name for name, longest in PARTITIONS if hours <= longest)
    mem_gb = max(float(rows["mem_gb"].max()), 2.0)
    return SUBMIT_VERGLEICH.format(n=len(rows), last=len(rows) - 1, part=part, hours=hours,
                                   mem=speicher.sbatch_mem(mem_gb), mem_gb=speicher.fmt(mem_gb))


def main_gpb_alle(out, check=True, **kw):
    return main_alle(out, "gpb_alle", check=check, **kw)


def main_alle(out, run="gpb_alle", pred="jobs_paper/zeitmessung_neu_hochrechnung.tsv",
              memory="jobs_paper/speicher_mac.tsv", memory_new="jobs_paper/speicher_neu_mac.tsv",
              euler="jobs_paper/speicher_euler.tsv", check=True):
    """Laeufe gpb_alle, xgb_alle, cat_alle: schreibt jobs_paper/<run>/, aufgaben_alle.tsv, run_alle_job.sbatch und
    submit_<run>.sh. gpb_alle: alte Aufgaben nur BOHB/DEHB/HEBO (Rest wiederverwendet, pruefe_wiederverwendung);
    xgb_alle, cat_alle: alles neu, Sekunden je Versuch x Verhaeltnis Bibliothek / GPBoost, Speicher der Bibliothek."""
    settings = ALL_RUNS[run]
    library = settings.get("library", "gpboost")
    out = Path(out)
    folder = out / run
    (folder / "arrays").mkdir(parents=True, exist_ok=True)
    sec, source = seconds_all(pred)
    if library == "gpboost":
        peaks = speicher.Plan(mac={**speicher.mac_peaks(memory_new), **speicher.mac_peaks(memory)},
                              euler=speicher.euler_measurements(euler))
    else:
        import measure_new_tasks
        ratio, _ = measure_new_tasks.verhaeltnisse(library, verbose=False)
        sec = {k: v * ratio[k] for k, v in sec.items()}
        # nur die Mac-Messung der Bibliothek (Euler-Messungen in speicher_euler.tsv stammen von GPBoost)
        peaks = speicher.Plan(mac=speicher.mac_peaks(out / f"speicher_{library}_mac.tsv"),
                              euler=pd.DataFrame(columns=speicher.EULER_COLUMNS))
    old, new = tasks(), tasks(NEW_SUITES)
    every = pd.concat([old, new], ignore_index=True)
    every.assign(classification=[is_classification(s) for s in every["suite"]]).to_csv(
        out / "aufgaben_alle.tsv", sep="\t", index=False, columns=["suite", "task", "name", "classification"])
    old_methods = GPB_ALLE_OLD_METHODS if run == "gpb_alle" else None
    jobs = pd.concat([build(run, sec, old, peaks, methods=old_methods), build(run, sec, new, peaks)],
                     ignore_index=True)
    whole = jobs
    bundled = "bundle_minutes" in settings
    if bundled:
        factor = faktoren()
        jobs = jobs.assign(minuten_echt=[round(h * 60 * factor.get(m, (1.0, 1.0))[0], 2)
                                         for h, m in zip(jobs["kernstunden"], jobs["method"])])
        whole = jobs
        in_probe = np.array([(s, t) in GPB_ALLE_PROBE for s, t in zip(jobs["suite"], jobs["task"])])
        jobs = split_folds(jobs, settings["fold_split_minutes"], in_probe & (jobs["seed"] == PROBE[2]).values)
    jobs = plan(jobs, longest_first=True)
    jobs[["method", "seed", "suite", "task", "trials", "folder", "flags"]].to_csv(
        folder / "jobs.tsv", sep="\t", index=False, header=False)
    probe = jobs[np.array([(s, t) in GPB_ALLE_PROBE for s, t in zip(jobs["suite"], jobs["task"])])
                 & jobs["seed"].isin([PROBE[2], 0])]
    probe_lines = [i + 1 for i in probe.index]
    if bundled:
        table, probe_lines = make_bundles(jobs, *settings["bundle_minutes"], factor,
                                          jobs.index.isin(probe.index))
        table.to_csv(folder / "buendel.tsv", sep="\t", index=False)
        probe_bundles = table.iloc[[b - 1 for b in probe_lines]]
    if not (folder / "arrays" / "probe.idx").exists():
        (folder / "arrays" / "probe.idx").write_text("\n".join(map(str, probe_lines)) + "\n")
    jobs["quelle_zeit"] = [source[(s, t)] for s, t in zip(jobs["suite"], jobs["task"])]
    jobs["array"] = ""
    jobs.drop(columns="sek_versuch").to_csv(folder / "kosten.tsv", sep="\t", index=False)
    costs = wave_costs(run, jobs)
    costs.to_csv(folder / "kosten_wellen.tsv", sep="\t", index=False)
    is_new = jobs["suite"].isin(NEW_SUITES)
    by_method = jobs.groupby("method")["kernstunden"].sum().round(0).astype(int).to_dict()
    by_source = jobs.groupby("quelle_zeit")["kernstunden"].sum().round(0).astype(int).to_dict()
    summary = [f"Lauf {run} -> {settings['folder']}/: {len(jobs):,} Jobs, {LIBRARY_NAMES[run]}, Skalen gpboost, ohne lambda_l1, "
               f"T = 135, 20 Seeds, 5 Folds, num_leaves; geschaetzt {jobs['kernstunden'].sum():,.0f} Kernstunden "
               f"(Zeitlimits zusammen {jobs['stunden'].sum():,} h)",
               f"  59 alte Aufgaben: {(~is_new).sum():,} Jobs ("
               f"{'bohb, dehb, hebo x 20 Seeds' if run == 'gpb_alle' else 'Default + 9 Methoden x 20 Seeds'}), "
               f"{jobs.loc[~is_new, 'kernstunden'].sum():,.0f} Kernstunden",
               f"  {new.shape[0]} neue Aufgaben: {is_new.sum():,} Jobs (Default + 9 Methoden x 20 Seeds), "
               f"{jobs.loc[is_new, 'kernstunden'].sum():,.0f} Kernstunden",
               f"  Flags: Default '{settings.get('default_flags', '')}'; " + "; ".join(
                   f"{', '.join(m for m in GPB_ALLE_METHODS if m != 'default' and settings['flags_by_method'][m] == f)}"
                   f" '{f}'" for f in dict.fromkeys(settings['flags_by_method'][m] for m in GPB_ALLE_METHODS
                                                    if m != 'default')),
               f"  Faktoren gegenueber paper: {settings['cost_factor']}",
               f"  je Methode: {by_method}",
               f"  Kernstunden nach Quelle der Sekunden je Versuch: {by_source} (hochgerechnet = mit Reserve)"
               + ("" if library == "gpboost" else f"; Sekunden = GPBoost x Verhaeltnis {library}/GPBoost je Aufgabe "
                  "(measure_new_tasks.py --library, Stichprobe 30 Aufgaben, sonst hochgerechnet mit Reserve)"),
               "  teuerste Jobs: " + ", ".join(
                   f"{r.name} {r.method} {r.kernstunden:.1f} h" for r in
                   jobs.sort_values("kernstunden", ascending=False).drop_duplicates(["task", "method"])
                   .head(5).itertuples()),
               "  Partitionen: " + ", ".join(f"{p} {n}" for p, n in jobs["partition"].value_counts().items()),
               "  Speicher je Job: " + ", ".join(f"{speicher.fmt(m)} GB {n}" for m, n in
                                                jobs["mem_gb"].value_counts().sort_index().items()),
               f"  Probe: {len(probe)} Jobs (Zeilen {', '.join(map(str, probe_lines))}): "
               + ", ".join(f"{r.name} {r.method} {r.kernstunden:.2f} h" for r in probe.itertuples()),
               "  Reihenfolge: laengste Jobs zuerst (geschaetzte Kernstunden absteigend, ueber alle Wellen)"]
    if bundled:
        split = jobs["flags"].str.contains("--fold")
        w1, b1 = jobs[jobs["welle"] == 1], table[table["welle"] == 1]
        hours = b1["minuten_echt"] / 60
        longest = b1.loc[b1["minuten_echt"].idxmax()]
        summary += [f"  Faktor echt/geschaetzt je Methode (gpb_alle Welle 1, {FAKTOR_ECHT}): "
                    + (", ".join(f"{m} {f[0]:.2f} (q90 {f[1]:.2f})" for m, f in factor.items()) or "keine Datei, 1.0"),
                    f"  Fold-Jobs: {int(split.sum()) // FOLDS:,} Jobs mit mehr als {settings['fold_split_minutes']} "
                    f"echten min je Fold (und Seed {PROBE[2]} der Probe-Aufgaben) in je 5 Fold-Jobs geteilt "
                    f"(--fold 0-4, gleiche Ergebnisdatei): {len(jobs):,} Zeilen in jobs.tsv",
                    f"  Buendel (buendel.tsv, Slurm-Jobs): {len(table):,}; Zeilen unter {settings['bundle_minutes'][0]} "
                    f"echten min zu {settings['bundle_minutes'][0]}-{settings['bundle_minutes'][1]} min gebuendelt",
                    f"  Welle 1 allein (gerechnet wird nur Welle 1: refill.py --wellen 1): {len(b1):,} Slurm-Jobs "
                    f"({len(w1):,} Zeilen), echte Dauer je Job Median {hours.median() * 60:.0f} min, Minimum "
                    f"{hours.min() * 60:.1f} min, unter 10 min {int((hours < 1 / 6).sum())} (davon Probe "
                    f"{int(((hours < 1 / 6) & b1['buendel'].isin(probe_lines)).sum())}), unter 30 min "
                    f"{int((hours < 0.5).sum())}; laengster Job {longest['minuten_echt'] / 60:.1f} h echt "
                    f"(Zeitlimit {int(longest['zeitlimit'])} h, {longest['partition']}), groesstes Zeitlimit "
                    f"{int(b1['zeitlimit'].max())} h; {hours.sum():,.0f} echte Kernstunden",
                    "  Welle 1 Dauer (laengste zuerst, gleichzeitige Jobs -> Tage): " + ", ".join(
                        f"{c} -> {d}" for c, d in welle1_tage(hours).items())]
    for c in costs.itertuples():
        summary.append(f"  Welle {c.welle} (Seeds {(c.welle - 1) * WAVE_SIZE + 1}-{c.welle * WAVE_SIZE}"
                       f"{' + Default' if c.welle == 1 else ''}): {c.jobs:,} Jobs, {c.kernstunden:,} Kernstunden, "
                       + ", ".join(f"bei {k} Kernen ~{getattr(c, f'tage_{k}_kerne')} Tage" for k in CORES)
                       + f" (laengster Job {c.laengster_job_h} h)"
                       + (" [Zeilen von jobs.tsv, Schaetzung ohne Faktor echt/geschaetzt; echte Werte oben]"
                          if bundled else ""))
    summary += [f"Alle Wellen: {len(jobs):,} Jobs, {costs['kernstunden'].sum():,} Kernstunden. Abschicken: "
                f"python refill.py --run {run}. Probe zuerst: bash jobs_paper/submit_{run}.sh probe",
                *NEW_VS_GPB[run],
                "Vor dem Start auf Euler: neue Daten (bash euler_setup.sh), .venv_extra (bash euler_install_extra.sh).", ""]
    if check and run == "gpb_alle":
        summary += [*pruefe_wiederverwendung(out)[1], ""]
    hours = int(probe["zeitlimit"].max())
    mem_gb = max(float(probe["mem_gb"].max()), 2.0)
    if bundled:                             # Probe-Buendel; Zeitlimit mindestens 4 h wie bisher
        hours = max(int(probe_bundles["zeitlimit"].max()), 4)
        mem_gb = max(float(probe_bundles["mem_gb"].max()), 2.0)
    part = next(name for name, longest in PARTITIONS if hours <= longest)
    (out / "run_alle_job.sbatch").write_text(SBATCH_ALLE)
    submit = SUBMIT_ALLE.format(
        run=run, methods=", ".join(GPB_ALLE_METHODS), n=len(probe), last=len(probe) - 1,
        names=", ".join(probe["name"].unique()), seed=PROBE[2], part=part,
        hours=hours, mem=speicher.sbatch_mem(mem_gb), mem_gb=speicher.fmt(mem_gb))
    if bundled:
        n = len(probe_lines)
        submit = submit.replace(f"--array=0-{len(probe) - 1} ", f"--array=0-{n - 1} ", 1) \
            .replace(f"jobs_paper/run_alle_job.sbatch {run}", f"jobs_paper/run_buendel_job.sbatch {run}", 1) \
            .replace(f"probe.idx\" {len(probe)} ", f"probe.idx\" {n} ", 1) \
            .replace(f"{len(probe)} Teiljobs", f"{n} Buendel", 1) \
            .replace(f"Probe des Laufs {run}: {len(probe)} Jobs", f"Probe des Laufs {run}: {n} Buendel mit {len(probe)} Zeilen", 1)
        submit = submit.replace("    *)     echo", vergleich_jobs(out, whole) + "    *)     echo", 1)
    (out / f"submit_{run}.sh").write_text(submit)
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
    parser.add_argument("--faktor", help="laufzeiten_gpb_alle.tsv (Euler) -> " + FAKTOR_ECHT + ", dann Ende")
    parser.add_argument("--run", choices=("paper+l1_270", "extra", "gpb", "gpb_l1", "gpb_alle", "xgb_alle", "cat_alle"), default="paper+l1_270",
                        help="extra: nur den Lauf extra (BOHB, DEHB, HEBO) schreiben; gpb: nur den Lauf gpb (korrekte "
                             "Skalen ohne lambda_l1); gpb_l1: wie gpb mit lambda_l1 (log) und 270 Versuchen; "
                             "gpb_alle: GPBoost, alle Methoden und Datensaetze, Skalen gpboost, ohne lambda_l1; "
                             "Standard: paper und l1_270")
    args = parser.parse_args()
    if args.faktor:
        print(echt_faktoren(args.faktor).round(3).to_string(index=False))
        sys.exit(0)
    if args.run == "extra":
        main_extra(args.out, args.timing, args.memory, args.euler)
    elif args.run in ALLE_RUNS:
        main_alle(args.out, args.run, memory=args.memory, euler=args.euler)
    elif args.run in ("gpb", "gpb_l1"):
        main_gpb(args.out, args.timing, args.memory, args.euler, run=args.run)
    else:
        main(args.out, args.timing, args.memory, args.euler)
