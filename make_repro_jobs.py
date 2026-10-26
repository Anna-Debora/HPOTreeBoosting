"""Erzeugt (startet aber NICHT) den kleinen Reproduktions-Testlauf auf den alten Suiten 334-337.

Ziel: stichprobenartig pruefen, ob der heutige Code die Ergebnisse des Papers (Koster & Sigrist) im
Wesentlichen wiedergibt. Vergleich danach mit compare_with_paper.py.

Auswahl (Begruendung in REPRO_PLAN.md):
  - Bibliothek: nur GPBoost (das Paper nutzt nur GPBoost/LightGBM), kein XGBoost, kein CatBoost
  - Suchraum: nur num_leaves (Hauptergebnis des Papers, Abbildungen 1 und 5-10)
  - Methoden: Default, deterministisches Gitter, Zufallsgitter, TPE, GP-BO (ein Hauptjob) und SMAC
    (eigener Job, nur num_leaves); Hyperband nicht (im Paper zweitschlechteste Methode, teuer)
  - Seeds: die ersten zwei der 20 Paper-Seeds (27225, 32244)
  - Aufgaben: REPRO_TASKS (beide Aufgabentypen, alle vier Suiten, eindeutige Namen, deutliche
    Unterschiede zwischen den Methoden im Paper, kleine bis mittlere Datensaetze)

Ausgabe in --out (Standard jobs_repro/):
  jobs.tsv               eine Zeile je Job: Art, Suite, Task, Seed
  run_repro_array.sbatch Slurm-Array; Aufruf auf Euler: cd ~/HPOTreeBoosting && sbatch jobs_repro/run_repro_array.sbatch
  kosten.tsv             geschaetzte Kernstunden je Job
Ergebnisse landen in Results_repro/ (Results/, Results_SMAC/, Results_Default_Run/).

Aufruf:  python make_repro_jobs.py [--out jobs_repro] [--seeds 27225 32244]
"""
import argparse
import shlex
from pathlib import Path

PAPER_SEEDS = (27225, 32244)
STRATEGY = "num_leaves"
LIBRARY = "gpboost"
RESULTS = "Results_repro"
# (suite, task, Name, gemessene Sekunden pro Versuch inkl. Test-Neuanpassung)
# Messung 26.10.2026 lokal (1 Thread, 27 Zufallsversuche auf Fold 0, Strategie num_leaves, Seed 27225)
REPRO_TASKS = (
    (336, 361076, "wine_quality", 0.38),
    (336, 361085, "sulfur", 0.54),
    (335, 361097, "Mercedes_Benz_Greener_Manufacturing", 0.85),
    (334, 361286, "compas-two-years", 0.04),
    (337, 361065, "MagicTelescope", 0.66),
    (337, 361278, "heloc", 0.30),
    (337, 361055, "credit", 0.23),
)
TRIALS = 270                 # pro Methode und Fold
FOLDS = 5
MAIN_METHODS = 4             # deterministisches Gitter (270 Konfigurationen), Zufallsgitter, TPE, GP-BO
GP_OVERHEAD_SEC = 650        # Surrogat-Zeit von GP-BO fuer 270 Aufrufe und einen Fold (gemessen 607 s, scikit-optimize)
SMAC_OVERHEAD_SEC = 0.15     # Surrogat-Zeit von SMAC pro Versuch (Schaetzung)
SAFETY = 2.0                 # Euler-Kern langsamer als lokal, Streuung zwischen Konfigurationen
TIME_LIMIT_HOURS = 12        # GP-BO-Surrogat allein ~1.5 h pro Hauptjob, daher nicht die 4-h-Partition
PARTITION = "normal.24h"


def estimate_hours(kind, sec_per_trial):
    if kind == "default":
        return 0.05
    if kind == "main":
        seconds = FOLDS * (MAIN_METHODS * TRIALS * sec_per_trial + GP_OVERHEAD_SEC)
    else:
        seconds = FOLDS * TRIALS * (sec_per_trial + SMAC_OVERHEAD_SEC)
    return SAFETY * seconds / 3600


def build_jobs(seeds):
    jobs = []
    for suite, task, name, sec in REPRO_TASKS:
        jobs.append(("default", suite, task, 0, name, estimate_hours("default", sec)))
        for seed in seeds:
            jobs.append(("main", suite, task, seed, name, estimate_hours("main", sec)))
            jobs.append(("smac", suite, task, seed, name, estimate_hours("smac", sec)))
    return jobs


def sbatch_script(job_count, out):
    logs = f"{out}/logs"
    return f"""#!/bin/bash
#SBATCH --job-name=hpo-repro
#SBATCH --partition={PARTITION}
#SBATCH --time={TIME_LIMIT_HOURS:02d}:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-{job_count - 1}
#SBATCH --output={logs}/%x_%A_%a.out
#SBATCH --error={logs}/%x_%A_%a.err
#SBATCH --export=ALL
# Reproduktions-Testlauf (siehe REPRO_PLAN.md). Abschicken aus ~/HPOTreeBoosting:
#   cd ~/HPOTreeBoosting && sbatch {out}/run_repro_array.sbatch
# (die Pfade oben sind relativ zu diesem Ordner)

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {shlex.quote(out)}/jobs.tsv)
IFS=$'\\t' read -r KIND SUITE_ID TASK_ID SEED NAME <<< "$JOB_LINE"
echo "Job $SLURM_ARRAY_TASK_ID: $KIND $SUITE_ID $TASK_ID seed=$SEED ($NAME)"

case "$KIND" in
  main)
    python run_experiments.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --seed "$SEED" \\
        --library {LIBRARY} --strategy {STRATEGY} --result_folder {RESULTS}/Results ;;
  smac)
    python run_experiment_H_SMAC.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --seed "$SEED" \\
        --library {LIBRARY} --only smac --strategy {STRATEGY} --result_folder_SMAC {RESULTS}/Results_SMAC ;;
  default)
    python run_default.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" \\
        --library {LIBRARY} --result_folder {RESULTS}/Results_Default_Run ;;
  *)
    echo "unbekannte Job-Art: $KIND" >&2; exit 1 ;;
esac
"""


def main(out="jobs_repro", seeds=PAPER_SEEDS):
    jobs = build_jobs(seeds)
    folder = Path(out)
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    (folder / "logs" / ".gitkeep").touch()
    (folder / "jobs.tsv").write_text("".join(f"{k}\t{s}\t{t}\t{seed}\t{n}\n" for k, s, t, seed, n, _ in jobs))
    (folder / "kosten.tsv").write_text("art\tsuite\ttask\tseed\tname\tkernstunden_geschaetzt\n" + "".join(
        f"{k}\t{s}\t{t}\t{seed}\t{n}\t{h:.2f}\n" for k, s, t, seed, n, h in jobs))
    script = folder / "run_repro_array.sbatch"
    script.write_text(sbatch_script(len(jobs), out))
    script.chmod(0o755)
    total = sum(h for *_, h in jobs)
    longest = max(jobs, key=lambda j: j[-1])
    print(f"{len(jobs)} Jobs ({sum(j[0] == 'main' for j in jobs)} Haupt-, {sum(j[0] == 'smac' for j in jobs)} SMAC-, "
          f"{sum(j[0] == 'default' for j in jobs)} Default-Jobs)")
    print(f"geschaetzte Kernstunden gesamt: {total:.1f} (laengster Job: {longest[0]} {longest[4]} "
          f"seed {longest[3]}: {longest[-1]:.1f} h, Zeitlimit {TIME_LIMIT_HOURS} h)")
    print(f"Geschrieben: {folder}/jobs.tsv, {folder}/kosten.tsv, {script}")
    print(f"Starten (auf Euler, NICHT automatisch): cd ~/HPOTreeBoosting && sbatch {script}")
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jobs_repro")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    args = parser.parse_args()
    main(args.out, args.seeds)
