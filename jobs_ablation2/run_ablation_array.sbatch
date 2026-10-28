#!/bin/bash
#SBATCH --job-name=hpo-ablation2
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-9
#SBATCH --output=jobs_ablation2/logs/%x_%A_%a.out
#SBATCH --error=jobs_ablation2/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Ursachen-Laeufe A/B auf wine_quality (siehe run_ablation.py, make_ablation_jobs.py).
# Abschicken aus ~/HPOTreeBoosting:
#   Probe nur Teiljob 0:  sbatch --array=0 jobs_ablation2/run_ablation_array.sbatch
#   danach der Rest:      sbatch --array=1-9 jobs_ablation2/run_ablation_array.sbatch
# Bereits vorhandene Ergebnisdateien werden nicht ueberschrieben (Teiljob endet mit Exit-Code 3).

set -eo pipefail     # -u erst nach "module load" und "activate"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs_ablation2/jobs.tsv)
IFS=$'\t' read -r RUN METHOD SEED SUITE_ID TASK_ID TRIALS FLAGS <<< "$JOB_LINE"
echo "Job $SLURM_ARRAY_TASK_ID: Lauf $RUN $METHOD seed=$SEED $SUITE_ID $TASK_ID trials=$TRIALS $FLAGS"

# data/ ist nicht im Git: ohne die Daten sofort mit klarer Meldung abbrechen
DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Zuerst auf Euler ausfuehren: bash euler_prepare_repro_data.sh" >&2
    exit 2
fi

python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \
    --trials "$TRIALS" --result_folder "Results_ablation/$RUN" $FLAGS
