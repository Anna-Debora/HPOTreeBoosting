#!/bin/bash
# Probe des Laufs gpb_l1: 8 Jobs (random_search, tpe, gp_bo, SMAC x analcatdata_supreme, heloc, Seed 27225), normal.24h, 12 h, 2G je Kern.
#   bash jobs_paper/submit_gpb_l1.sh probe          (DRY=1: nur anzeigen)
# Merkt sich die Job-ID in jobs_paper/gpb_l1/submitted.tsv (watch_paper.py zeigt sie, refill.py schickt sie nie doppelt).
# Die Wellen schickt refill.py:  python refill.py --run gpb_l1   (nicht in --run alle)
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
WHAT=${1:-}
DIR=jobs_paper/gpb_l1
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --account=ls_math --array=0-7 --partition=normal.24h --time=12:00:00
                --mem-per-cpu=2G --job-name=hpo-gpb_l1 jobs_paper/run_paper_job.sbatch gpb_l1 "$DIR/arrays/probe.idx")
           if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; exit 0; fi
           id=$("${cmd[@]}")
           printf '%s\t%s\t%s\t%s\tnormal.24h\t12\t2\n' "$id" "$DIR/arrays/probe.idx" 8 "$(date '+%F %T')" \
               >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe gpb_l1, 8 Teiljobs)" ;;
    *)     echo "Aufruf: bash jobs_paper/submit_gpb_l1.sh probe   (Wellen: python refill.py --run gpb_l1)"; exit 1 ;;
esac
