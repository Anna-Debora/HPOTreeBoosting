#!/bin/bash
# Probe des Laufs gpb_alle: 23 Jobs (default, grid_search, random_search, tpe, gp_bo, hyperband, SMAC, bohb, dehb, hebo x analcatdata_supreme, housing_price_prediction, Customer_Personality_Analysis, Seed 27225), normal.4h, 4 h, 2G je Kern.
#   bash jobs_paper/submit_gpb_alle.sh probe          (DRY=1: nur anzeigen)
# Merkt sich die Job-ID in jobs_paper/gpb_alle/submitted.tsv (watch_paper.py zeigt sie, refill.py schickt sie nie doppelt).
# Die Wellen schickt refill.py:  python refill.py --run gpb_alle   (nicht in --run alle)
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
WHAT=${1:-}
DIR=jobs_paper/gpb_alle
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-22 --partition=normal.4h --time=4:00:00
                --mem-per-cpu=2G --job-name=hpo-gpb_alle jobs_paper/run_alle_job.sbatch gpb_alle "$DIR/arrays/probe.idx")
           if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; exit 0; fi
           id=$("${cmd[@]}")
           printf '%s\t%s\t%s\t%s\tnormal.4h\t4\t2\n' "$id" "$DIR/arrays/probe.idx" 23 "$(date '+%F %T')" \
               >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe gpb_alle, 23 Teiljobs)" ;;
    *)     echo "Aufruf: bash jobs_paper/submit_gpb_alle.sh probe   (Wellen: python refill.py --run gpb_alle)"; exit 1 ;;
esac
