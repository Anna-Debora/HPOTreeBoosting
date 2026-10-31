#!/bin/bash
#SBATCH --job-name=hpo-refill
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=jobs_paper/logs/refill_%j.out
#SBATCH --error=jobs_paper/logs/refill_%j.out
# Selbstfortsetzender Nachfueller. Nicht direkt abschicken, sondern: python refill.py --run beide --daemon
# Beenden: python refill.py --stop  (= scancel -n hpo-refill, trifft auch den wartenden Nachfolger)
# Ablauf: gleich zu Beginn einen Nachfolger mit --dependency=afterany abschicken (startet erst, wenn dieser Job
# endet, auch nach Timeout oder Knotenausfall), dann alle REFILL_INTERVAL Sekunden (Standard 600) refill.py
# laufen lassen, 20 min vor dem Zeitlimit enden. Exit-Code 3 von refill.py (nichts mehr zu tun) oder die Datei
# jobs_paper/refill.stop beenden die Kette (der Nachfolger wird gestrichen).
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

INTERVAL=${REFILL_INTERVAL:-600}
LIMIT_S=$(( 24 * 3600 - 20 * 60 ))
START=$(date +%s)
NEXT=$(squeue --me -h -n hpo-refill -t PENDING -o %i | head -1)
if [ -z "$NEXT" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" jobs_paper/refill_daemon.sbatch "$@")
    echo "Nachfolger: $NEXT"
fi
stop_chain() {
    echo "$(date '+%F %T') $1 -> Kette beendet"
    scancel "${NEXT%%;*}" 2>/dev/null || true
    rm -f jobs_paper/refill.stop
    exit 0
}
while [ $(( $(date +%s) - START )) -lt "$LIMIT_S" ]; do
    [ -e jobs_paper/refill.stop ] && stop_chain "jobs_paper/refill.stop gefunden"
    set +e
    python refill.py "$@"
    RC=$?
    set -e
    [ "$RC" -eq 3 ] && stop_chain "nichts mehr zu tun"
    [ "$RC" -ne 0 ] && echo "$(date '+%F %T') refill.py Exit-Code $RC (naechster Versuch in $INTERVAL s)"
    sleep "$INTERVAL"
done
echo "$(date '+%F %T') Zeitlimit naht, Nachfolger $NEXT uebernimmt"
