#!/bin/bash
# Probe des Laufs gpb: 8 Jobs (tpe, gp_bo, SMAC, hyperband x analcatdata_supreme, heloc, Seed 27225), normal.4h, 4 h, 2G je Kern.
#   bash jobs_paper/submit_gpb.sh probe          (DRY=1: nur anzeigen)
# Merkt sich die Job-ID in jobs_paper/gpb/submitted.tsv (watch_paper.py zeigt sie, refill.py schickt sie nie doppelt).
# Die Wellen schickt refill.py:  python refill.py --run gpb   (nicht in --run alle)
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
WHAT=${1:-}
DIR=jobs_paper/gpb
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-7 --partition=normal.4h --time=4:00:00
                --mem-per-cpu=2G --job-name=hpo-gpb jobs_paper/run_paper_job.sbatch gpb "$DIR/arrays/probe.idx")
           if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; exit 0; fi
           id=$("${cmd[@]}")
           printf '%s\t%s\t%s\t%s\tnormal.4h\t4\t2\n' "$id" "$DIR/arrays/probe.idx" 8 "$(date '+%F %T')" \
               >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe gpb, 8 Teiljobs)" ;;
    *)     echo "Aufruf: bash jobs_paper/submit_gpb.sh probe   (Wellen: python refill.py --run gpb)"; exit 1 ;;
esac
