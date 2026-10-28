"""Erzeugt (startet aber NICHT) die Ursachen-Laeufe A/B auf wine_quality, siehe run_ablation.py.

  Lauf A (Paper-Einstellung): 135 Versuche, ohne lambda_l1  -> Results_ablation/A
  Lauf B (nur Versuche):      135 Versuche, mit lambda_l1   -> Results_ablation/B
Je Teiljob eine Methode, ein Lauf und ein Seed (27225, 32244), 5 Folds.

  --group 1 (jobs_ablation):  Zufallsgitter, TPE, SMAC je A und B: 12 Teiljobs
  --group 2 (jobs_ablation2): die uebrigen Methoden, 10 Teiljobs
      Deterministisches Gitter  nur A (--paper_grid, 135er-Gitter des Papers); B = Testlauf, denn das heutige Gitter
                                hat kein lambda_l1 und keine Versuchszahl (270 Punkte, Results_repro)
      GP-BO                     A und B (135 Aufrufe); Testlauf mit 270 Aufrufen liegt in Results_repro
      Hyperband                 A und B; keine Versuchszahl (R = 2150, factor = 2.8), daher ist B zugleich die
                                Testlauf-Einstellung (mit lambda_l1), ein eigener Testlauf waere dieselbe Rechnung
      Default                   kein Teiljob: params = {}, haengt weder von lambda_l1 noch von der Versuchszahl
                                noch vom Seed ab; liegt in Results_repro/Results_Default_Run

Kosten: Sekunden pro Versuch aus der Spalte runtime des Testlaufs auf Euler (wine_quality, Versuche mit
lambda_l1 <= 0.5, groesseres der beiden Seed-Mittel; SMAC inklusive Surrogat), dazu der Test-Refit jedes Versuchs
(etwa ein Training wie bei der Validierung), Sicherheitsfaktor 2. Gruppe 2 mit Sekunden pro Fold:
  Gitter     runtime der 135 Punkte mit bagging_fraction = 1 im Testlauf (40 s) + 135 Test-Refits (0.3 s)
  GP-BO      runtime der ersten 135 Aufrufe im Testlauf inkl. Surrogat (268 s, waechst mit der Aufrufzahl;
             alle 270 Aufrufe: 1161 s) + 135 Test-Refits
  Hyperband  lokal gemessen 31 s je Fold (3573 Trainings mit wenigen Runden, 11 Test-Refits), x 1.5 fuer Euler

Aufruf:  python make_ablation_jobs.py [--group 1|2] [--out jobs_ablation | jobs_ablation2]
Ausgabe: jobs.tsv, kosten.tsv, run_ablation_array.sbatch, logs/.gitkeep
"""
import argparse
from pathlib import Path

SUITE, TASK, NAME = 336, 361076, "wine_quality"
SEEDS = (27225, 32244)
TRIALS = 135
FOLDS = 5
RUNS = {"A": "--no_lambda_l1", "B": ""}
# (Validierung s/Versuch, Test-Refit s/Versuch), gemessen im Testlauf 18544431
SEC_PER_TRIAL = {"random_search": (0.80, 0.80), "tpe": (0.73, 0.73), "SMAC": (5.18, 0.80)}
SAFETY = 2.0
TIME_LIMIT_HOURS = 12
PARTITION = "normal.24h"
RESULTS = "Results_ablation"
# Gruppe 2: (Methode, Lauf, Zusatzflags), Sekunden pro Fold (siehe oben)
GROUP2 = (("grid_search", "A", "--paper_grid"),
          ("gp_bo", "A", "--no_lambda_l1"), ("gp_bo", "B", ""),
          ("hyperband", "A", "--no_lambda_l1"), ("hyperband", "B", ""))
SEC_PER_FOLD2 = {"grid_search": 40 + 135 * 0.30, "gp_bo": 268 + 135 * 0.30, "hyperband": 31 * 1.5}
TIME_LIMIT_HOURS2 = 4
PARTITION2 = "normal.4h"


def build_jobs(group=1):
    """[(Lauf, Methode, Seed, Flags, Kernstunden)]"""
    if group == 2:
        return [(run, method, seed, flags, SAFETY * FOLDS * SEC_PER_FOLD2[method] / 3600)
                for method, run, flags in GROUP2 for seed in SEEDS]
    return [(run, method, seed, RUNS[run], SAFETY * FOLDS * TRIALS * sum(SEC_PER_TRIAL[method]) / 3600)
            for run in RUNS for method in SEC_PER_TRIAL for seed in SEEDS]


def sbatch_script(job_count, out, partition=PARTITION, hours=TIME_LIMIT_HOURS, name="hpo-ablation"):
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --time={hours:02d}:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-{job_count - 1}
#SBATCH --output={out}/logs/%x_%A_%a.out
#SBATCH --error={out}/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Ursachen-Laeufe A/B auf wine_quality (siehe run_ablation.py, make_ablation_jobs.py).
# Abschicken aus ~/HPOTreeBoosting:
#   Probe nur Teiljob 0:  sbatch --array=0 {out}/run_ablation_array.sbatch
#   danach der Rest:      sbatch --array=1-{job_count - 1} {out}/run_ablation_array.sbatch
# Bereits vorhandene Ergebnisdateien werden nicht ueberschrieben (Teiljob endet mit Exit-Code 3).

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {out}/jobs.tsv)
IFS=$'\\t' read -r RUN METHOD SEED SUITE_ID TASK_ID TRIALS FLAGS <<< "$JOB_LINE"
echo "Job $SLURM_ARRAY_TASK_ID: Lauf $RUN $METHOD seed=$SEED $SUITE_ID $TASK_ID trials=$TRIALS $FLAGS"

# data/ ist nicht im Git: ohne die Daten sofort mit klarer Meldung abbrechen
DATA_X="data/${{SUITE_ID}}_${{TASK_ID}}/${{SUITE_ID}}_${{TASK_ID}}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Zuerst auf Euler ausfuehren: bash euler_prepare_repro_data.sh" >&2
    exit 2
fi

python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \\
    --trials "$TRIALS" --result_folder "{RESULTS}/$RUN" $FLAGS
"""


def main(out=None, group=1):
    out = out or ("jobs_ablation2" if group == 2 else "jobs_ablation")
    jobs = build_jobs(group)
    folder = Path(out)
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    (folder / "logs" / ".gitkeep").touch()
    (folder / "jobs.tsv").write_text("".join(
        f"{run}\t{method}\t{seed}\t{SUITE}\t{TASK}\t{TRIALS}\t{flags}\n" for run, method, seed, flags, _ in jobs))
    (folder / "kosten.tsv").write_text("lauf\tmethode\tseed\tkernstunden_geschaetzt\n" + "".join(
        f"{run}\t{method}\t{seed}\t{hours:.2f}\n" for run, method, seed, _, hours in jobs))
    hours = TIME_LIMIT_HOURS2 if group == 2 else TIME_LIMIT_HOURS
    script = folder / "run_ablation_array.sbatch"
    script.write_text(sbatch_script(len(jobs), out, PARTITION2 if group == 2 else PARTITION, hours,
                                    "hpo-ablation2" if group == 2 else "hpo-ablation"))
    script.chmod(0o755)
    longest = max(jobs, key=lambda j: j[-1])
    print(f"{len(jobs)} Teiljobs, geschaetzte Kernstunden gesamt: {sum(j[-1] for j in jobs):.1f} "
          f"(laengster: {longest[1]} {longest[-1]:.2f} h, Zeitlimit {hours} h)")
    for method in dict.fromkeys(j[1] for j in jobs):
        print(f"  {method}: {sum(j[-1] for j in jobs if j[1] == method):.2f} h")
    print(f"Geschrieben: {folder}/jobs.tsv, {folder}/kosten.tsv, {script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out", default=None, help="Standard: jobs_ablation bzw. jobs_ablation2")
    args = parser.parse_args()
    main(args.out, args.group)
