#!/bin/bash
#SBATCH --job-name=hpo-refill
#SBATCH --account=ls_math
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=normal.4h
#SBATCH --time=4:00:00
#SBATCH --output=jobs_paper/logs/refill_%j.out
#SBATCH --error=jobs_paper/logs/refill_%j.out
# Selbstfortsetzender Nachfueller. Nicht direkt abschicken, sondern: python refill.py --run beide --daemon
#   (Standard normal.4h, 4:00:00; anders: --daemon-partition normal.24h --daemon-zeit 24:00:00)
# Beenden: python refill.py --stop  (= scancel -n hpo-refill, trifft laufenden Daemon und wartenden Nachfolger,
# egal in welcher Partition)
# Ablauf: gleich zu Beginn einen Nachfolger mit --dependency=afterany abschicken (gleiche Partition und gleiches
# Zeitlimit; startet erst, wenn dieser Job endet, auch nach Timeout oder Knotenausfall), dann alle REFILL_INTERVAL
# Sekunden (Standard 600) refill.py laufen lassen, REFILL_MARGIN Sekunden (Standard 20 min) vor dem Zeitlimit enden.
# Exit-Code 3 von refill.py (nichts mehr zu tun), Exit-Code 4 (Home-Quota ueber 80 %, quota.py) oder die Datei
# jobs_paper/refill.stop beenden die Kette (der Nachfolger wird gestrichen). Laeuft schon ein anderer
# hpo-refill-Job, endet dieser sofort (nie zwei Daemons).
# Partition, Zeitlimit und Account kommen aus DAEMON_PARTITION, DAEMON_TIME, DAEMON_SECONDS, DAEMON_ACCOUNT (von
# refill.py --daemon gesetzt, mit --export=ALL an den Nachfolger vererbt); ohne sie gelten die #SBATCH-Werte oben.
# Speicher 1 GB: refill.py --run beide braucht ~0.25 GB (Mac, 1.11.2026).
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

PART=${DAEMON_PARTITION:-normal.4h}
TIME=${DAEMON_TIME:-4:00:00}
ACCOUNT=${DAEMON_ACCOUNT:-ls_math}
SECONDS_TOTAL=${DAEMON_SECONDS:-14400}
INTERVAL=${REFILL_INTERVAL:-600}
MARGIN=${REFILL_MARGIN:-1200}
LIMIT_S=$(( SECONDS_TOTAL - MARGIN ))
START=$(date +%s)
echo "$(date '+%F %T') Daemon $SLURM_JOB_ID in $PART ($ACCOUNT), Zeitlimit $TIME, laeuft bis $LIMIT_S s"

OTHERS=$(squeue --me -h -n hpo-refill -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID" || true)
if [ -n "$OTHERS" ]; then
    echo "$(date '+%F %T') anderer Daemon laeuft schon: $OTHERS -> dieser endet ohne Nachfolger"
    exit 0
fi
NEXT=$(squeue --me -h -n hpo-refill -t PENDING -o %i | head -1)
if [ -z "$NEXT" ]; then
    NEXT=$(sbatch --parsable --account="$ACCOUNT" --partition="$PART" --time="$TIME" --export=ALL \
           --dependency=afterany:"$SLURM_JOB_ID" jobs_paper/refill_daemon.sbatch "$@")
    echo "Nachfolger: $NEXT ($ACCOUNT, $PART, $TIME)"
else
    echo "Nachfolger wartet schon: $NEXT"
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
    [ "$RC" -eq 4 ] && stop_chain "Home-Quota fast voll (siehe oben)"
    [ "$RC" -ne 0 ] && echo "$(date '+%F %T') refill.py Exit-Code $RC (naechster Versuch in $INTERVAL s)"
    [ $(( $(date +%s) - START + INTERVAL )) -lt "$LIMIT_S" ] || break
    sleep "$INTERVAL"
done
echo "$(date '+%F %T') Zeitlimit naht, Nachfolger $NEXT uebernimmt"
