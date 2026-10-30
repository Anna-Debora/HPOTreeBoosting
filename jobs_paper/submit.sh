#!/bin/bash
# Schickt die Arrays eines Laufs ab und merkt sich die Job-IDs in jobs_paper/<lauf>/submitted.tsv
# (fuer watch_paper.py). Aufruf aus ~/HPOTreeBoosting auf Euler:
#   bash jobs_paper/submit.sh paper probe        # 7 Jobs: wine_quality, Seed 27225, alle Methoden
#   bash jobs_paper/submit.sh paper alle         # alle Arrays des Laufs (fertige Jobs enden sofort)
#   bash jobs_paper/submit.sh paper normal.4h_4G_0   # ein einzelnes Array (Name aus arrays.tsv)
#   bash jobs_paper/submit.sh paper retry        # Wiederholung: arrays/retry_*.idx aus watch_paper.py --retry
# Optional: DRY=1 (nur anzeigen), THROTTLE=200 (hoechstens so viele Teiljobs eines Arrays gleichzeitig).
set -euo pipefail
cd "$HOME/HPOTreeBoosting" 2>/dev/null || true
RUN=$1; WHAT=$2
DIR=jobs_paper/$RUN
mkdir -p jobs_paper/logs
submit() {   # $1 idx-Datei, $2 Partition, $3 Stunden, $4 Speicher GB
    local n; n=$(grep -c . "$1")
    [ "$n" -gt 0 ] || return 0
    local array="0-$((n - 1))${THROTTLE:+%$THROTTLE}"
    local cmd=(sbatch --parsable --array="$array" --partition="$2" --time="$3:00:00" --mem-per-cpu="$4G"
               --job-name="hpo-$RUN" jobs_paper/run_paper_job.sbatch "$RUN" "$1")
    if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; return 0; fi
    local id; id=$("${cmd[@]}")
    printf '%s\t%s\t%s\t%s\n' "$id" "$1" "$n" "$(date '+%F %T')" >> "$DIR/submitted.tsv"
    echo "abgeschickt: Job $id, $n Teiljobs, $1 ($2, $3 h, $4 GB)"
}
case "$WHAT" in
    probe) submit "$DIR/arrays/probe.idx" normal.4h 4 4 ;;
    alle)  tail -n +2 "$DIR/arrays.tsv" | while IFS=$'\t' read -r NAME PART HOURS MEM N; do
               submit "$DIR/arrays/$NAME.idx" "$PART" "$HOURS" "$MEM"; done ;;
    retry) for f in "$DIR"/arrays/retry_*.idx; do
               [ -e "$f" ] || { echo "keine retry_*.idx (zuerst: python watch_paper.py --retry)"; exit 1; }
               IFS=_ read -r _ PART MEMG HOURS <<< "$(basename "$f" .idx)"
               submit "$f" "$PART" "${HOURS%h}" "${MEMG%G}"; done ;;
    *)     LINE=$(awk -F'\t' -v n="$WHAT" '$1 == n' "$DIR/arrays.tsv")
           [ -n "$LINE" ] || { echo "unbekannt: $WHAT (siehe $DIR/arrays.tsv)"; exit 1; }
           IFS=$'\t' read -r NAME PART HOURS MEM N <<< "$LINE"
           submit "$DIR/arrays/$NAME.idx" "$PART" "$HOURS" "$MEM" ;;
esac
