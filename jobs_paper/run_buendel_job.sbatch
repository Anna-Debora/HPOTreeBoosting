#!/bin/bash
#SBATCH --job-name=hpo-buendel
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs_paper/logs/%x_%A_%a.out
#SBATCH --error=jobs_paper/logs/%x_%A_%a.err
#SBATCH --export=ALL
# Ein Buendel des Laufs xgb_alle (python make_paper_jobs.py --run xgb_alle): rechnet die Zeilen von jobs.tsv aus
# jobs_paper/<lauf>/buendel.tsv nacheinander, jede wie run_alle_job.sbatch (gleiche Befehle, eigene Ergebnisdatei,
# --resume, fertige Zeilen/Folds: SCHON FERTIG). Nicht direkt abschicken, sondern ueber  python refill.py --run <lauf>
# (Probe: bash jobs_paper/submit_<lauf>.sh probe). Die idx-Datei nennt Buendelnummern (Spalte buendel).
# Bricht eine Zeile mit einem Fehler ab, laufen die uebrigen weiter; Exit-Code 1 am Ende (refill wiederholt das
# Buendel, fertige Zeilen werden uebersprungen). OOM/TIMEOUT beenden das ganze Buendel (refill eskaliert es).
# gpb_alle benutzt weiter run_alle_job.sbatch.

set -eo pipefail     # -u erst nach "module load"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
set -u

RUN=$1
IDX_FILE=$2
BUNDLE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IDX_FILE")
LINES=$(awk -F'\t' -v b="$BUNDLE" 'NR > 1 && $1 == b {print $2}' "jobs_paper/$RUN/buendel.tsv" | tr ',' ' ')
if [ -z "$LINES" ]; then
    echo "ABBRUCH: Buendel $BUNDLE nicht in jobs_paper/$RUN/buendel.tsv" >&2
    exit 2
fi
echo "Lauf $RUN Buendel $BUNDLE: Zeilen $LINES"
echo "Start $(date '+%F %T') auf $(hostname), Slurm-Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CPU: $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
SCRATCH_TMP="${TMPDIR:-${SCRATCH:-/cluster/scratch/$USER}}"

RC=0
for LINE_NO in $LINES; do
    JOB_LINE=$(sed -n "${LINE_NO}p" "jobs_paper/$RUN/jobs.tsv")
    IFS=$'\t' read -r METHOD SEED SUITE_ID TASK_ID TRIALS FOLDER FLAGS <<< "$JOB_LINE"
    LIB=$(echo "$FLAGS" | sed -n 's/.*--library \([a-z]*\).*/\1/p')
    LIB=${LIB:-gpboost}
    case "$METHOD" in
        bohb|dehb|hebo) VENV=.venv_extra ;;
        *)              VENV=.venv ;;
    esac
    if [ ! -x "$VENV/bin/python" ]; then
        echo "ABBRUCH: $VENV fehlt (.venv_extra: bash euler_install_extra.sh)" >&2
        exit 2
    fi
    echo "-- Zeile $LINE_NO: $METHOD seed=$SEED ${SUITE_ID}_${TASK_ID} trials=$TRIALS $FLAGS -> $FOLDER ($VENV), $(date '+%T')"
    DATA_X="data/${SUITE_ID}_${TASK_ID}/${SUITE_ID}_${TASK_ID}_X.csv"
    if [ ! -s "$DATA_X" ]; then
        echo "FEHLER: $DATA_X fehlt oder ist leer (bash euler_setup.sh)" >&2
        RC=1
        continue
    fi
    if [ "$METHOD" = "default" ]; then
        OUT="$FOLDER/default/$LIB/${SUITE_ID}_${TASK_ID}.csv"
        if [ -s "$OUT" ]; then
            echo "SCHON FERTIG: $OUT"
        else
            "$VENV/bin/python" run_default.py --suite_id "$SUITE_ID" --task_id "$TASK_ID" \
                --result_folder "$FOLDER/default" --library "$LIB" || { echo "FEHLER: Zeile $LINE_NO" >&2; RC=1; }
        fi
    elif [ "$VENV" = .venv_extra ]; then
        TMPDIR="$SCRATCH_TMP" MPLCONFIGDIR="$SCRATCH_TMP/matplotlib_${SLURM_JOB_ID:-0}" \
            "$VENV/bin/python" run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" \
            --task_id "$TASK_ID" --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS \
            || { echo "FEHLER: Zeile $LINE_NO" >&2; RC=1; }
    else
        "$VENV/bin/python" run_ablation.py --method "$METHOD" --seed "$SEED" --suite_id "$SUITE_ID" \
            --task_id "$TASK_ID" --trials "$TRIALS" --result_folder "$FOLDER" --resume $FLAGS \
            || { echo "FEHLER: Zeile $LINE_NO" >&2; RC=1; }
    fi
done
echo "Ende $(date '+%F %T'), Exit-Code $RC"
exit $RC
