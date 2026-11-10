#!/bin/bash
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
IFS=$'\t' read -r METHOD SEED SUITE_ID TASK_ID TRIALS FOLDER FLAGS <<< "$JOB_LINE"
echo "Lauf $RUN Zeile $LINE_NO: $METHOD seed=$SEED ${SUITE_ID}_${TASK_ID} trials=$TRIALS $FLAGS -> $FOLDER"
echo "Start $(date '+%F %T') auf $(hostname), Slurm-Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, TMPDIR $TMPDIR"

DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
if [ ! -s "$DATA_X" ]; then
    echo "ABBRUCH: $DATA_X fehlt oder ist leer. Zuerst: bash euler_prepare_repro_data.sh jobs_paper/aufgaben.tsv" >&2
    exit 2
fi
python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \
    --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS
echo "Ende $(date '+%F %T')"
