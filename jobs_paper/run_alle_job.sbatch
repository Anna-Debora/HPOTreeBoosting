#!/bin/bash
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
IFS=$'\t' read -r METHOD SEED SUITE_ID TASK_ID TRIALS FOLDER FLAGS <<< "$JOB_LINE"
LIB=$(echo "$FLAGS" | sed -n 's/.*--library \([a-z]*\).*/\1/p')
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
        python run_default.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" --result_folder "$FOLDER/default" \
            --library "$LIB"
    fi
else
    python run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" --task_id "$TASK_ID" \
        --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS
fi
echo "Ende $(date '+%F %T')"
