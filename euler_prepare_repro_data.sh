#!/bin/bash
# euler_prepare_repro_data.sh -- erzeugt auf Euler NUR die Daten der 7 Aufgaben des Reproduktions-Testlaufs
# (REPRO_PLAN.md, Liste aus jobs_repro/jobs.tsv) und prueft die Pakete. Startet keine Jobs, committet nichts.
#
# Aufruf auf Euler:
#     cd ~/HPOTreeBoosting && bash euler_prepare_repro_data.sh
#
# data/ ist nicht im Git. Die alten Aufgaben (Suiten 334-337) holt prepare_one_dataset.py von OpenML
# (bereits nach Grinsztajn aufbereitet) und legt sie als data/<suite>_<task>/<suite>_<task>_{X,y}.csv und
# _categorical_indicator.npy ab. Schon vollstaendige Ordner werden nicht neu geladen; unvollstaendige
# (z. B. abgebrochener Download) werden neu erzeugt. Dauer: wenige Minuten.

set -eo pipefail     # -u erst nach "module load" und "activate", die vertragen es oft nicht
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u

LOGS=setup_logs
mkdir -p "$LOGS"

echo "===== Pakete ====="
python - <<'EOF'
import importlib, sys
from importlib.metadata import version as installed_version
# Importname: (Paketname bei pip, verlangte Version oder None)
need = {"gpboost": ("gpboost", "1.7.4"), "smac": ("smac", "2.4.1"), "optuna": ("optuna", None),
        "skopt": ("scikit-optimize", None), "ConfigSpace": ("ConfigSpace", None), "openml": ("openml", None),
        "sklearn": ("scikit-learn", None), "pandas": ("pandas", None), "numpy": ("numpy", None)}
bad = []
for name, (dist, version) in need.items():
    try:
        importlib.import_module(name)
        found = installed_version(dist)
    except Exception as error:
        bad.append(f"{name} fehlt oder laesst sich nicht laden ({error})")
        continue
    print(f"  {name:12s} {found}")
    if version and found != version:
        bad.append(f"{name} ist {found}, erwartet {version}")
if bad:
    print("FEHLER:", *bad, sep="\n  ")
    print("Behebung: pip install -r requirements.txt  (in der aktivierten .venv)")
    sys.exit(1)
print("OK: alle Pakete da")
EOF

echo
echo "===== Daten der Testaufgaben ====="
complete() {   # alle drei Dateien vorhanden und nicht leer
    local dir="data/$1_$2" stem="data/$1_$2/$1_$2"
    [ -s "${stem}_X.csv" ] && [ -s "${stem}_y.csv" ] && [ -s "${stem}_categorical_indicator.npy" ]
}
TASKS=$(cut -f2,3 jobs_repro/jobs.tsv | sort -u)
echo "$TASKS" | while read -r SUITE TASK; do
    if complete "$SUITE" "$TASK"; then
        echo "  vorhanden:  data/${SUITE}_${TASK}"
    else
        echo "  lade:       data/${SUITE}_${TASK} ..."
        rm -rf "data/${SUITE}_${TASK}"
        python prepare_one_dataset.py --suite_id "$SUITE" --task_id "$TASK" > "$LOGS/prepare_${SUITE}_${TASK}.log" 2>&1 \
            || { echo "  FEHLER beim Download, siehe $LOGS/prepare_${SUITE}_${TASK}.log"; tail -5 "$LOGS/prepare_${SUITE}_${TASK}.log"; }
    fi
done

echo
echo "===== Kontrolle ====="
N_OK=0; N_ALL=0
while read -r SUITE TASK; do
    N_ALL=$((N_ALL + 1))
    X="data/${SUITE}_${TASK}/${SUITE}_${TASK}_X.csv"
    if complete "$SUITE" "$TASK"; then
        N_OK=$((N_OK + 1))
        echo "  OK      $X  ($(($(wc -l < "$X") - 1)) Zeilen)"
    else
        echo "  FEHLT   $X (oder y/categorical_indicator fehlt bzw. ist leer)"
    fi
done <<< "$TASKS"
echo
if [ "$N_OK" -eq "$N_ALL" ] && [ "$N_ALL" -eq 7 ]; then
    echo "FERTIG: alle 7 Testaufgaben haben ihre Daten. Naechster Schritt (Probe):"
    echo "    sbatch --array=0 jobs_repro/run_repro_array.sbatch"
else
    echo "NICHT FERTIG: $N_OK von $N_ALL Aufgaben vollstaendig (erwartet 7). Bitte die Meldungen oben pruefen."
    exit 1
fi
