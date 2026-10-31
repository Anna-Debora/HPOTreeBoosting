#!/bin/bash
# Nur noch fuer die Probe. Die Wellen schickt refill.py in Portionen ab (QOS: hoechstens ~1000 Jobs in der Schlange):
#   python refill.py --run beide              # siehe python refill.py --help
#   bash jobs_paper/submit.sh paper probe     # 7 Jobs: wine_quality, Seed 27225, alle Methoden
# Optional: DRY=1 (nur anzeigen). Merkt sich die Job-ID in jobs_paper/<lauf>/submitted.tsv (fuer watch_paper.py).
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
RUN=$1; WHAT=$2
DIR=jobs_paper/$RUN
mkdir -p jobs_paper/logs
case "$WHAT" in
    probe) cmd=(sbatch --parsable --array=0-6 --partition=normal.4h --time=4:00:00 --mem-per-cpu=4G
                --job-name="hpo-$RUN" jobs_paper/run_paper_job.sbatch "$RUN" "$DIR/arrays/probe.idx")
           if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; exit 0; fi
           id=$("${cmd[@]}")
           printf '%s\t%s\t%s\t%s\tnormal.4h\t4\t4\n' "$id" "$DIR/arrays/probe.idx" 7 "$(date '+%F %T')" >> "$DIR/submitted.tsv"
           echo "abgeschickt: Job $id (Probe, 7 Teiljobs)" ;;
    *)     echo "$WHAT: Wellen und Wiederholungen jetzt mit refill.py (python refill.py --help)"; exit 1 ;;
esac
