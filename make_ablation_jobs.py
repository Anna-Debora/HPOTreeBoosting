"""Erzeugt (startet aber NICHT) die Ursachen-Laeufe A/B auf wine_quality, siehe run_ablation.py.

  Lauf A (Paper-Einstellung): 135 Versuche, ohne lambda_l1  -> Results_ablation/A
  Lauf B (nur Versuche):      135 Versuche, mit lambda_l1   -> Results_ablation/B
Je Lauf, Methode (Zufallsgitter, TPE, SMAC) und Seed (27225, 32244) ein Teiljob: 12 Teiljobs, 5 Folds je Teiljob.

Kosten: Sekunden pro Versuch aus der Spalte runtime des Testlaufs auf Euler (wine_quality, Versuche mit
lambda_l1 <= 0.5, groesseres der beiden Seed-Mittel; SMAC inklusive Surrogat), dazu der Test-Refit jedes Versuchs
(etwa ein Training wie bei der Validierung), Sicherheitsfaktor 2.

Aufruf:  python make_ablation_jobs.py [--out jobs_ablation]
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


def build_jobs():
    return [(run, method, seed, SAFETY * FOLDS * TRIALS * sum(SEC_PER_TRIAL[method]) / 3600)
            for run in RUNS for method in SEC_PER_TRIAL for seed in SEEDS]


def sbatch_script(job_count, out):
    return f"""#!/bin/bash
#SBATCH --job-name=hpo-ablation
#SBATCH --partition={PARTITION}
#SBATCH --time={TIME_LIMIT_HOURS:02d}:00:00
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


def main(out="jobs_ablation"):
    jobs = build_jobs()
    folder = Path(out)
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    (folder / "logs" / ".gitkeep").touch()
    (folder / "jobs.tsv").write_text("".join(
        f"{run}\t{method}\t{seed}\t{SUITE}\t{TASK}\t{TRIALS}\t{RUNS[run]}\n" for run, method, seed, _ in jobs))
    (folder / "kosten.tsv").write_text("lauf\tmethode\tseed\tkernstunden_geschaetzt\n" + "".join(
        f"{run}\t{method}\t{seed}\t{hours:.2f}\n" for run, method, seed, hours in jobs))
    script = folder / "run_ablation_array.sbatch"
    script.write_text(sbatch_script(len(jobs), out))
    script.chmod(0o755)
    longest = max(jobs, key=lambda j: j[-1])
    print(f"{len(jobs)} Teiljobs, geschaetzte Kernstunden gesamt: {sum(j[-1] for j in jobs):.1f} "
          f"(laengster: {longest[1]} {longest[-1]:.2f} h, Zeitlimit {TIME_LIMIT_HOURS} h)")
    print(f"Geschrieben: {folder}/jobs.tsv, {folder}/kosten.tsv, {script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="jobs_ablation")
    main(parser.parse_args().out)
