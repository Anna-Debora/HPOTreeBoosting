#!/bin/bash
#SBATCH --job-name=hpo-repro
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-34
#SBATCH --output=jobs_repro/logs/%x_%A_%a.out
#SBATCH --error=jobs_repro/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Reproduktions-Testlauf (siehe REPRO_PLAN.md). Vorher einmal: bash euler_prepare_repro_data.sh
# Abschicken aus ~/HPOTreeBoosting:
#   cd ~/HPOTreeBoosting && sbatch jobs_repro/run_repro_array.sbatch   (Probe nur Teiljob 0: sbatch --array=0 ...)
# (die Pfade oben sind relativ zu diesem Ordner)

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs_repro/jobs.tsv)
IFS=$'\t' read -r KIND SUITE_ID TASK_ID SEED NAME <<< "$JOB_LINE"
echo "Job $SLURM_ARRAY_TASK_ID: $KIND $SUITE_ID $TASK_ID seed=$SEED ($NAME)"

# data/ ist nicht im Git: ohne die Daten sofort mit klarer Meldung abbrechen
DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Zuerst auf Euler ausfuehren: bash euler_prepare_repro_data.sh" >&2
    exit 2
fi

case "$KIND" in
  main)
    python run_experiments.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --seed "$SEED" \
        --library gpboost --strategy num_leaves --result_folder Results_repro/Results ;;
  smac)
    python run_experiment_H_SMAC.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --seed "$SEED" \
        --library gpboost --only smac --strategy num_leaves --result_folder_SMAC Results_repro/Results_SMAC ;;
  default)
    python run_default.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" \
        --library gpboost --result_folder Results_repro/Results_Default_Run ;;
  *)
    echo "unbekannte Job-Art: $KIND" >&2; exit 1 ;;
esac
