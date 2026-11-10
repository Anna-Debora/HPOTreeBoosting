#!/bin/bash
# euler_install_extra.sh -- legt das GETRENNTE Environment .venv_extra fuer den Lauf "extra" (BOHB, DEHB, HEBO) an.
# Das bestehende .venv wird nicht veraendert (nur gelesen: pip freeze). Startet keine Jobs.
#
# Aufruf auf Euler (Login-Knoten, ~/HPOTreeBoosting auf Branch extra-methods):
#     cd ~/HPOTreeBoosting && bash euler_install_extra.sh 2>&1 | tee $SCRATCH/install_extra.log
# Nochmal von vorn (altes .venv_extra wird erst nach Rueckfrage geloescht):  FORCE=1 bash euler_install_extra.sh
#
# Was das Skript macht:
#   1. Module laden (python/3.9.18 wie .venv), Platz pruefen (quota.py: Home unter 70 % der Soft Quota; das
#      Environment braucht ~1.2 GB und ~36 000 Dateien), pip-Cache nach $SCRATCH (nicht ins Home)
#   2. Basis = pip freeze von .venv (gleiche Versionen von gpboost, optuna, smac, ConfigSpace, pandas, sklearn ...),
#      ausser numpy 1.24.4 und scipy 1.12.0 (von HEBO/GPy erzwungen, siehe requirements_extra.txt)
#   3. python -m venv .venv_extra; torch 2.8.0 als CPU-Version (download.pytorch.org/whl/cpu, ~200 MB statt
#      ~3 GB mit CUDA); dann Basis + requirements_extra.txt mit festen Versionen; pip check
#   4. Kontrolle: Versionen ausgeben, .venv unveraendert (pip freeze vorher = nachher), Kurztest der drei
#      Methoden (run_ablation.py, 1 Fold, kleines Budget, Ergebnis nach $SCRATCH/extra_install_test, danach geloescht)
set -eo pipefail

ROOT="$HOME/HPOTreeBoosting"
VENV=.venv_extra
TORCH="torch==2.8.0+cpu"
TORCH_INDEX=https://download.pytorch.org/whl/cpu

step() { echo; echo "===== $* ====="; }

step "1. Umgebung und Platz"
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd "$ROOT"
SCRATCH_DIR="${SCRATCH:-/cluster/scratch/$USER}"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip_cache_extra"
export TMPDIR="${TMPDIR:-$SCRATCH_DIR/tmp_install_extra}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python --version
[ -x .venv/bin/python ] || { echo "FEHLER: .venv fehlt (das bestehende Environment wird als Basis gelesen)"; exit 1; }
.venv/bin/python - <<'PY'
import sys
import quota
ok, lines = quota.check(home_limit=0.7)
info = quota.usage()
print(quota.describe("home", info["home"]))
print(quota.describe("scratch", info["scratch"]))
print("\n".join(lines))
sys.exit(0 if ok else 4)
PY

if [ -e "$VENV" ]; then
    if [ -z "${FORCE:-}" ]; then
        echo "$VENV gibt es schon -> nichts getan. Neu anlegen: FORCE=1 bash euler_install_extra.sh"
        exit 1
    fi
    read -r -p "$VENV loeschen und neu anlegen? [j/N] " answer
    [ "$answer" = "j" ] || { echo "abgebrochen"; exit 1; }
    rm -rf "$VENV"
fi

step "2. Basis aus .venv (nur lesen)"
BASE="$TMPDIR/base_constraints.txt"
.venv/bin/python -m pip freeze --all | sort > "$TMPDIR/venv_freeze_vorher.txt"
grep -v -E '^(numpy|scipy|pip|setuptools|wheel)==|^-e | @ ' "$TMPDIR/venv_freeze_vorher.txt" > "$BASE"
echo "$(wc -l < "$BASE") Pakete aus .venv als feste Versionen; numpy/scipy aus requirements_extra.txt:"
grep -E '^(numpy|scipy|gpboost|optuna|smac|ConfigSpace|pandas|scikit-learn)==' "$TMPDIR/venv_freeze_vorher.txt" || true

step "3. $VENV anlegen und installieren"
python -m venv "$VENV"
source "$VENV/bin/activate"
set -u
python -m pip install --upgrade pip
python -m pip install --no-deps --index-url "$TORCH_INDEX" "$TORCH"
python -m pip install -c "$BASE" -r "$BASE" -r requirements_extra.txt
python -m pip check

step "4. Kontrolle"
python - <<'PY'
from importlib.metadata import version
for p in ("numpy", "scipy", "pandas", "scikit-learn", "gpboost", "optuna", "smac", "ConfigSpace", "torch",
          "hpbandster", "DEHB", "HEBO", "GPy", "pymoo", "statsmodels", "gpytorch"):
    print(f"{p:14s} {version(p)}")
import torch, hpbandster, dehb, hebo   # noqa: F401  (Import muss gehen)
print("torch CUDA:", torch.version.cuda, "(None = CPU-Version, gewollt)")
PY
.venv/bin/python -m pip freeze --all | sort > "$TMPDIR/venv_freeze_nachher.txt"
if cmp -s "$TMPDIR/venv_freeze_vorher.txt" "$TMPDIR/venv_freeze_nachher.txt"; then
    echo ".venv unveraendert (pip freeze vorher = nachher)"
else
    echo "FEHLER: .venv hat sich veraendert:"; diff "$TMPDIR/venv_freeze_vorher.txt" "$TMPDIR/venv_freeze_nachher.txt"; exit 1
fi
TEST_DIR="$SCRATCH_DIR/extra_install_test"
rm -rf "$TEST_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for METHOD in bohb dehb hebo; do
    python run_ablation.py --method "$METHOD" --seed 27225 --suite_id 336 --task_id 361076 --folds 1 \
        --trials 6 --hyperband_R 30 --no_lambda_l1 --result_folder "$TEST_DIR" | tail -2
done
python - "$TEST_DIR" <<'PY'
import sys, glob
import pandas as pd
for path in sorted(glob.glob(f"{sys.argv[1]}/*/seed_27225/gpboost/*.csv")):
    frame = pd.read_csv(path)
    print(f"{frame['method'].iloc[0]:5s} {len(frame)} Zeilen, bester Test-R2 {frame['current_best_test_score'].iloc[-1]:.3f}")
PY
rm -rf "$TEST_DIR"
du -sh "$VENV"
echo
echo "Fertig: $VENV. Naechster Schritt: python make_paper_jobs.py --run extra (mit .venv), dann die Probe:"
echo "    bash jobs_paper/submit_extra.sh probe"
