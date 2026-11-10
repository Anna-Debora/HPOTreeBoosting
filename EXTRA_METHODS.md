# Lauf „extra“: BOHB, DEHB, HEBO

Drei zusätzliche HPO-Methoden mit denselben Einstellungen wie der Lauf **paper**: GPBoost, 135 Versuche,
ohne `lambda_l1`, alle 59 Aufgaben der Suiten 334–337, 20 Seeds, 5 Folds, Strategie `num_leaves`, Kategorien
nativ (kein One-Hot). Ergebnisse in `Results_extra/` (nicht in `Results_paper/`, damit die Normierung der
7-Methoden-Tabelle gleich bleibt). Die zusammengeführte Tabelle erzeugt `make_table5.py --extra` als
`table5_extra.*`.

## Bibliotheken (getrenntes Environment `.venv_extra`)

| Methode | Bibliothek | Version | Original? |
|---|---|---|---|
| BOHB | `hpbandster` (automl) | 0.7.4 | ja, Original-Code des BOHB-Papers. Benutzt wird nur der Konfigurationsgenerator `hpbandster.optimizers.config_generators.bohb.BOHB` (KDE), im Prozess, ohne Nameserver, ohne Pyro-Worker, ohne Dateien |
| DEHB | `dehb` (automl/DEHB) | 0.1.2 | ja, über `ask`/`tell`, `n_workers=1`, kein Dask-Client |
| HEBO | `HEBO` (huawei-noah) | 0.3.6 | ja, Standardwerte (Modell `gp`, MACE, NSGA-II), dazu GPy 1.13.2, pymoo 0.6.0, torch 2.8.0 (CPU) |

hpbandster läuft noch: 0.7.4 (2019) geht mit ConfigSpace 1.2.2 (nur Deprecation-Warnungen), statsmodels
0.14.6 und Python 3.9. Getestet wurden modellbasierte Vorschläge, ganze und kategorische Parameter.

**Abweichung im Environment (nicht im Verfahren):** HEBO 0.3.5/0.3.6 verlangt `numpy<1.25`, GPy 1.13.2
`numpy<2` und `scipy<=1.12` (paramz bricht mit numpy 2 ab). `.venv_extra` hat deshalb **numpy 1.24.4** statt
2.0.2 und **scipy 1.12.0** statt 1.13.1. Alle anderen Pakete haben genau die Versionen von `.venv` (aus
`pip freeze`). Geprüft: TPE, Hyperband und das Zufallsgitter liefern in beiden Environments **bitgleiche**
Ergebnisse (heloc, 1 Fold, 10 Versuche). Die Zielfunktion ist also dieselbe. `.venv` wird nicht verändert.
Die Alternative mit numpy 2 wäre HEBO 0.3.2 (die letzte Version ohne numpy-Grenze). Sie bricht aber beim
Import ab (pymoo 0.6 hat `pymoo.factory` entfernt, GPy/paramz geht nicht mit numpy 2). Diese Kombination
lässt sich also nicht installieren.

## Budget: BOHB und DEHB im Hyperband-Rahmen

`hyperband_method` (methods.py) hat die Fidelity **Boosting-Runden** (`num_boost_round`, mit early stopping 20
auf dem Validierungsteil), `R = 2150`, `eta = 2.8`, `s_max = 7`, also 8 Brackets s = 7 … 0 mit
`n = ceil(8·eta^s/(s+1))` Konfigurationen und `r_k = floor(R·eta^(k−s))` Runden. Jede Rung trainiert neu
(kein Warmstart). Nach jeder Rung bleiben die besten `floor(n_k/eta)`. Ergebniszeilen sind die Überlebenden
jedes Brackets (zusammen **11 je Fold**, 3573 Auswertungen). Die Test-Refits aller Zeilen nutzen `best_iter`
des besten Überlebenden des letzten Brackets. Die Laufzeit einer Zeile ist die Wanduhr des Brackets geteilt
durch die Zahl seiner Zeilen. In `make_table5.py` zählt der Endwert (die letzte Zeile).

| Bracket s | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|---|---|---|
| Konfigurationen × Runden (erste Rung) | 1350 × 1 | 551 × 4 | 230 × 12 | 99 × 34 | 44 × 97 | 21 × 274 | 12 × 767 | 8 × 2150 |
| Ergebniszeilen | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 2 |

**BOHB** und **DEHB** nutzen genau diesen Rahmen: dieselben Brackets in derselben Reihenfolge, dieselbe Zahl
von Konfigurationen je Rung, dieselben Budgets, dieselben Ergebniszeilen, dieselbe Laufzeit-Verteilung und
dasselbe `best_iter`. Ein Test prüft, dass die Budget-Folge aller Trainings in Aufrufreihenfolge mit der von
`hyperband_method` übereinstimmt (`test_extra_methods.FrameTests`, `ScheduleTests`). Anders ist nur die Wahl der
Konfigurationen:

* BOHB: das KDE-Modell von hpbandster mit seinen Standardwerten (top 15 %, 64 Kandidaten, 1/3 zufällig,
  Bandbreitenfaktor 3), Ergebnisse jeder Rung mit ihrem Budget. Wie hpbandster mit einem Worker werden die
  Konfigurationen der ersten Rung einzeln gezogen, jede nach dem Ergebnis der vorigen.
* DEHB: Populationen je Budget, Beförderung, Mutation, Crossover und Selektion von DEHB.

**Abweichung von den Originalen (gewollt, für die Vergleichbarkeit mit Hyperband):** hpbandster und DEHB
berechnen ihre Brackets mit `n0 = floor(s_max+1 / (s+1))·eta^s` und Budgets `max·eta^−i` ohne Abrunden. Hier
gelten stattdessen die Formeln von `hyperband_method` (ceil statt floor, ganze Runden, gleiches Abbruchkriterium).
Bei DEHB werden dafür `_get_next_iteration` und `_pre_compute_fidelity_spacing` überschrieben
(`methods.scheduled_dehb_class`). Bei BOHB läuft die Successive-Halving-Schleife von `hyperband_method`, und der
hpbandster-Generator liefert die Konfigurationen. Ein Durchgang besteht aus 8 Brackets wie bei Hyperband
(hpbandster/DEHB würden sonst eine Zahl von Iterationen bzw. ein Zeitbudget nehmen).

**HEBO** ist single-fidelity: 135 volle Auswertungen (1000 Runden mit early stopping), wie TPE/SMAC/GP-BO,
mit 135 Ergebniszeilen je Fold.

## Suchraum, Zielfunktion, Zufall

* Suchraum wie TPE/SMAC im Lauf paper: `learning_rate` [0.001, 1], `min_data_in_leaf` [1, 1000] ganz,
  `lambda_l2` [0, 100], `max_bin` [255, min(n, 10000)] ganz, `bagging_fraction` und `feature_fraction` [0.5, 1],
  `num_leaves` [2, 1024] ganz, `max_depth = −1` fest, `lambda_l1 = 0` fest. Alle Bereiche linear wie bei den
  bestehenden Methoden (keine log-Skala). Ganze Zahlen sind in ConfigSpace `Integer` und in HEBO `int`.
  `max_depth` ist (für die Strategie max_depth) kategorisch wie bei TPE/SMAC. Log-Skalen werden unterstützt
  (`log=True` → ConfigSpace `log`, HEBO `pow`/`pow_int`), aber nicht benutzt.
* Zielfunktion `_train_model_for_validation` (gleicher Train/Validierungs-Split), Test-Refits
  `_compute_test_scores`, gleiches CSV-Format (`RESULT_COLUMNS`), `--resume` pro Fold, atomares Schreiben
  (`job_io.write_csv_atomic`).
* Seed: zu Beginn jedes Folds `self.seed` für den globalen numpy-Zufall (hpbandster), `ConfigurationSpace(seed)`,
  DEHB `seed`, torch, `random` und das Sobol-Startdesign von HEBO. Gleicher Seed ergibt gleiche Zahlen (getestet).
* 1 Kern: `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`, DEHB ohne Dask, BOHB ohne Nameserver.
* Temporäre Dateien: nur DEHB schreibt (`dehb.log`), in einen Ordner unter `$TMPDIR` bzw. `$SCRATCH`
  (`job_io.temp_dir`, nie unter `$HOME`), der danach gelöscht wird. `MPLCONFIGDIR` liegt im sbatch-Skript
  unter `$TMPDIR`.

## Dateien

| Datei | Zweck |
|---|---|
| `methods.py` | `bohb_method`, `dehb_method`, `hebo_method` (+ `run_bohb/run_dehb/run_hebo`), `hyperband_schedule` |
| `run_ablation.py` | `--method bohb|dehb|hebo` |
| `make_paper_jobs.py --run extra` | `jobs_paper/extra/` (3540 Jobs), `jobs_paper/run_extra_job.sbatch`, `jobs_paper/submit_extra.sh` |
| `refill.py --run extra` / `--run alle` | Nachfüllen (`--run beide` bleibt paper + l1_270) |
| `watch_paper.py` | zeigt extra mit an, sobald `jobs_paper/extra/jobs.tsv` existiert |
| `check_results.py --run extra` | Ergebnisdateien prüfen |
| `speicher.py` | Extra-Methoden: Plan der HPO-Methoden der Aufgabe + `EXTRA_OVERHEAD_GB`, eigene Art „extra“ |
| `make_table5.py --extra` | `table5_extra.*` (9 HPO-Methoden + Default) |
| `measure_extra.py` | Kurz-Benchmark (Zeit, Speicher) → `jobs_paper/extra/zeitmessung_extra_mac.tsv` |
| `euler_install_extra.sh`, `requirements_extra.txt` | `.venv_extra` |
| `test_extra_methods.py` | Tests (in `.venv_extra`, ~1 Minute) |
