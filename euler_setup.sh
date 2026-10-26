#!/bin/bash
# euler_setup.sh -- einmal auf Euler ausfuehren, bereitet alles fuer die Laeufe vor. Startet KEINE Jobs,
# committet und pusht nichts.
#
# Aufruf auf Euler (nach dem Einloggen mit ssh andubach@euler.ethz.ch), eine Zeile:
#     cd ~/HPOTreeBoosting && git fetch origin && git show origin/xgboost-catboost:euler_setup.sh > ~/euler_setup.sh && bash ~/euler_setup.sh 2>&1 | tee ~/euler_setup.log
# (Die Kopie nach ~/euler_setup.sh ist Absicht: das Skript macht selbst git pull und wuerde sich sonst
#  waehrend des Laufens ueberschreiben.)
#
# Was das Skript macht:
#   1. Module laden, virtuelle Umgebung aktivieren, Branch xgboost-catboost holen (git pull)
#   2. fehlende Pakete nachinstallieren (huggingface_hub, xgboost, catboost in den lokalen Versionen)
#   3. Daten der neuen Suiten 1001/1002 neu erzeugen (data/ ist nicht im Git):
#        prepare_new_datasets.py (OpenML), prepare_bcco.py (Hugging Face), prepare_talent.py (TALENT, Google Drive)
#      Sie lesen new_tasks.csv, excluded_duplicates.csv usw. aus dem Git.
#   4. Daten der Aufgaben fuer den Reproduktions-Testlauf (alte Suiten) holen
#   5. pruefen: Task-Listen identisch mit dem Git-Stand, zu jeder Aufgabe ein Datenordner
#   6. Job-Dateien fuer den Reproduktions-Testlauf erzeugen (jobs_repro/), aber nicht abschicken
# Dauer: grob 30-90 Minuten (vor allem Downloads). Wenn die Verbindung abbrechen koennte, vorher `tmux` starten.

set -eo pipefail     # -u erst nach "module load" und "activate", die vertragen es oft nicht

BRANCH=xgboost-catboost
TALENT_ZIP_URL="https://drive.usercontent.google.com/download?id=1Rr2uZsZzabshoa0KX0uCne-qMWFpbyAo&export=download&confirm=t"
TALENT_ZIP_SHA256=4c8481107153593eb98ef5bb677b00dedfa04cd4c31441babf696a8d100f207c
TALENT_ZIP="$HOME/talent_download/dataset-250115.zip"
# Diese Dateien schreiben die Skripte neu; massgeblich ist der Stand im Git (wird am Ende wiederhergestellt)
GENERATED=(preprocessing_report.csv preprocessing_report_bcco.csv preprocessing_report_talent.csv
           task_indices/1001_task_indices.npy task_indices/1001_task_names.json
           task_indices/1002_task_indices.npy task_indices/1002_task_names.json)
SETUP_LOGS=setup_logs

step() { echo; echo "===== $* ====="; }

step "1. Umgebung und Code"
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$HOME/HPOTreeBoosting"
source .venv/bin/activate
set -u
mkdir -p "$SETUP_LOGS"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
fi
# erzeugte Dateien eines frueheren Laufs zuruecksetzen, damit git pull nicht blockiert
restore_generated() { for f in "${GENERATED[@]}"; do git checkout -- "$f" 2>/dev/null || true; done; }
restore_generated
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "FEHLER: Es gibt geaenderte Dateien auf Euler (git status zeigt sie). Bitte zuerst klaeren, nichts wurde veraendert."
    git status --short --untracked-files=no
    exit 1
fi
git pull --ff-only origin "$BRANCH"
git log --oneline -1

step "2. Pakete"
need_install() { python -c "import $1" 2>/dev/null && return 1 || return 0; }
if need_install huggingface_hub; then pip install "huggingface_hub==1.8.0"; fi
if need_install xgboost; then pip install "xgboost==2.1.4"; fi
if need_install catboost; then pip install "catboost==1.2.10"; fi
pip show gpboost | grep -E "^(Name|Version)"
python -c "import gpboost, sys; v = gpboost.__version__; print('gpboost', v); sys.exit(0 if v == '1.7.4' else 'gpboost muss 1.7.4 sein')"

step "3a. OpenML-Aufgaben (new_tasks.csv)"
python prepare_new_datasets.py new_tasks.csv > "$SETUP_LOGS/prepare_new_datasets.log" 2>&1
tail -1 "$SETUP_LOGS/prepare_new_datasets.log"

step "3b. BCCO (Hugging Face)"
python prepare_bcco.py > "$SETUP_LOGS/prepare_bcco.log" 2>&1
tail -1 "$SETUP_LOGS/prepare_bcco.log"

step "3c. TALENT (Google Drive)"
if [ ! -f "$TALENT_ZIP" ]; then
    mkdir -p "$(dirname "$TALENT_ZIP")"
    curl -L --fail --retry 3 -o "$TALENT_ZIP" "$TALENT_ZIP_URL"
fi
echo "$TALENT_ZIP_SHA256  $TALENT_ZIP" | sha256sum -c -
python prepare_talent.py --zip "$TALENT_ZIP" > "$SETUP_LOGS/prepare_talent.log" 2>&1
tail -1 "$SETUP_LOGS/prepare_talent.log"

step "4. Daten fuer den Reproduktions-Testlauf (alte Suiten)"
python make_repro_jobs.py --out jobs_repro > /dev/null
cut -f2,3 jobs_repro/jobs.tsv | sort -u | while read -r SUITE TASK; do
    if [ ! -d "data/${SUITE}_${TASK}" ]; then
        python prepare_one_dataset.py --suite_id "$SUITE" --task_id "$TASK" > "$SETUP_LOGS/prepare_${SUITE}_${TASK}.log" 2>&1
    fi
    echo "data/${SUITE}_${TASK}: $(ls "data/${SUITE}_${TASK}" | wc -l) Dateien"
done

step "5. Kontrolle"
for f in preprocessing_report.csv preprocessing_report_bcco.csv preprocessing_report_talent.csv; do
    cp "$f" "$SETUP_LOGS/euler_$f"
done
if git diff --quiet -- task_indices; then
    echo "OK: Task-Listen identisch mit dem Git-Stand"
else
    echo "WARNUNG: Die auf Euler erzeugten Task-Listen weichen vom Git-Stand ab (Unterschiede siehe unten)."
    echo "         Massgeblich bleibt der Git-Stand; bitte die Abweichung melden."
    python - <<'EOF'
import json, subprocess
for suite in (1001, 1002):
    path = f"task_indices/{suite}_task_names.json"
    euler = set(json.load(open(path)))
    git = set(json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"])))
    print(suite, "nur auf Euler:", sorted(euler - git), "| fehlt auf Euler:", sorted(git - euler))
EOF
fi
restore_generated
python - <<'EOF'
import os, numpy as np
missing, counts = [], {}
for suite in (334, 335, 336, 337, 1001, 1002):
    ids = np.load(f"task_indices/{suite}_task_indices.npy").tolist()
    counts[suite] = len(ids)
    if suite >= 1000:
        missing += [f"{suite}_{t}" for t in ids if not os.path.isdir(f"data/{suite}_{t}")]
print("Aufgaben pro Suite:", counts)
print("OK: zu jeder neuen Aufgabe gibt es einen Datenordner" if not missing else f"FEHLT: {missing}")
EOF
git status --short --untracked-files=no

step "6. Reproduktions-Testlauf vorbereitet (NICHT gestartet)"
mkdir -p jobs_repro/logs
python make_repro_jobs.py --out jobs_repro
echo
echo "Fertig. Zum Starten des Testlaufs (erst wenn du willst):"
echo "    cd ~/HPOTreeBoosting && sbatch jobs_repro/run_repro_array.sbatch"
