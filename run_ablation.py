"""Ein Teiljob der Ursachen-Laeufe A/B: Zufallsgitter, TPE, SMAC, GP-BO, Hyperband oder deterministisches Gitter
mit waehlbarer Versuchszahl und mit oder ohne lambda_l1, ohne Aenderung an methods.py oder SMAC_method.py.

Hintergrund (Testlauf vs. Paper, siehe repro_table.md): TPE, SMAC und Zufallsgitter liegen bei t = 135 unter
dem Paper. Seit dem Paper sind zwei Dinge anders: lambda_l1 ist im Suchraum, und es laufen 270 statt 135
Versuche (das beeinflusst t = 135 ueber best_iter des besten Versuchs aller 270 und ueber das SMAC-Startdesign
min(10 x Parameter, 0.25 x n_trials)).
  Lauf A (Paper-Einstellung): --trials 135 --no_lambda_l1
  Lauf B (nur Versuche):      --trials 135
  Deterministisches Gitter:   --paper_grid (nur Lauf A, siehe unten)

Wie die Einstellungen ohne Codeaenderung greifen (nur in diesem Prozess):
  Versuche   Zufallsgitter: grid_search_method(num_try_random=N) direkt aufgerufen
             TPE:   optuna Study.optimize mit n_trials=N statt 270
             SMAC:  Scenario mit n_trials=N statt 270
             GP-BO: gp_minimize mit n_calls=N statt 270
             Hyperband und deterministisches Gitter haben keine Versuchszahl (--trials wird ignoriert):
                    Hyperband hat R = 2150, factor = 2.8 fest, also 8 Brackets und 11 Ergebniszeilen je Fold
                    (das Paper legt sie in plot_maker gleichmaessig auf t = 1 ... 135, Wert bei t = 135 = letzte Zeile)
  lambda_l1  Zufallsgitter: lambda_l1 aus dem Gitter entfernt -> genau das Gitter des Paper-Codes (gleiche
                    Groesse, gleiche Reihenfolge der Ziehung mit demselben Seed); GPBoost nimmt dann den Default 0
             TPE:   suggest_float('lambda_l1', 0, 0) -> fester Wert 0, wird nicht gesampelt
             SMAC:  lambda_l1 als Konstante 0 (SMAC laesst Konstanten im Sobol-Startdesign weg)
             GP-BO: lambda_l1 aus dem Raum von gp_minimize entfernt, in jeden Aufruf als 0 eingesetzt und in
                    x_iters wieder eingefuegt -> genau der Raum des Paper-Codes (gleiche Reihenfolge)
             Hyperband: suggest_float('lambda_l1_<bracket>_<i>', 0, 0) -> fester Wert 0; optuna zieht dafuer
                    nichts (distribution.single()), die uebrigen Ziehungen bleiben wie im Paper-Code
             Deterministisches Gitter: hat heute kein lambda_l1 (methods.grid_search_method, num_try_random=None),
                    aber bagging_fraction = [0.5, 1] (270 Punkte). --paper_grid nimmt bagging_fraction heraus ->
                    das 135er-Gitter des Paper-Codes (bagging_fraction = 1). Das aendert auch best_iter, das fuer
                    die Test-Refits aller Gitterpunkte gilt; deshalb reicht die Teilmenge des Testlaufs nicht.
Zusaetzliche Methoden (Lauf "extra", nur in .venv_extra: hpbandster, dehb, HEBO; methods.py):
  bohb, dehb  im Rahmen von Hyperband (R = --hyperband_R, eta = 2.8): gleiche Brackets, Budgets (Boosting-Runden),
              Zahl der Konfigurationen je Rung und 11 Ergebniszeilen je Fold; --trials wird ignoriert
  hebo        --trials volle Auswertungen (single-fidelity) wie TPE/SMAC/GP-BO
  lambda_l1   --no_lambda_l1: lambda_l1 nicht im Suchraum, fest 0 (Argument lambda_l1=False der Methoden)
  Alle sequentiell, 1 Kern (torch 1 Thread, kein Dask, BOHB ohne Nameserver), Seed = --seed je Fold neu gesetzt.
  DEHB schreibt sein Log in einen temporaeren Ordner unter $TMPDIR bzw. $SCRATCH (job_io.temp_dir), nie ins Home.
Skalen (--scales, methods.SCALES; Bereiche gleich, nur die Skala nicht):
  paper    Standard, alter Zustand (alle Zahl-Parameter linear), bitgleich zu den Laeufen vor dieser Option.
  gpboost  wie GPBoost (engine.py, _parse_tuning_search_space): log fuer learning_rate, lambda_l1, lambda_l2
           (untere Grenze 0 -> 1e-8), num_leaves, min_data_in_leaf, max_bin; linear fuer bagging_fraction,
           feature_fraction, max_depth; n_iter unveraendert. TPE/Hyperband: suggest_*(log=True); GP-BO:
           prior='log-uniform'; SMAC, BOHB, DEHB: ConfigSpace log=True; HEBO: pow / pow_int. Zufallsgitter und
           deterministisches Gitter (feste Wertelisten) bleiben gleich. --l1 off: TPE, GP-BO, SMAC, Hyperband lassen
           lambda_l1 selbst weg (fest 0; SMAC als Konstante wie oben), das Zufallsgitter wie oben ohne lambda_l1.
           Eigenen --result_folder verwenden (die Ergebnisdateien haben keine Spalte fuer die Skala).
SMAC schreibt seine Zwischendateien unter <result_folder>/smac3_output statt ins Arbeitsverzeichnis.
Logs: nur Start, Fehler und Ende (job_io.quiet_logging, quiet_methods: optuna, SMAC, gpboost, Zufallsgitter leise).
Ergebnisdateien werden atomar geschrieben (job_io.write_csv_atomic: .tmp, dann umbenennen).
Bestehende Ergebnisdateien werden nie ueberschrieben (Abbruch mit Exit-Code 3), ausser mit --resume:
  vollstaendige Datei (alle Folds)  -> nichts tun, Exit-Code 0
  angefangene Datei (Zeitlimit)     -> fertige Folds behalten, ab dem ersten fehlenden Fold weiterrechnen.
  Das ergibt dieselben Zahlen wie ein Lauf am Stueck: jede Methode setzt ihren Sampler pro Fold neu mit dem
  Seed auf (TPESampler, gp_minimize, SMAC-Scenario, RandomState) und min_score/best_iter pro Aufruf zurueck.

Aufruf:  python run_ablation.py --method {random_search,tpe,SMAC,gp_bo,hyperband,grid_search,bohb,dehb,hebo} \
             --seed 27225 \
             --result_folder Results_ablation/A [--trials 135] [--l1 on|off] [--no_lambda_l1] [--paper_grid] \
             [--scales paper|gpboost] [--library gpboost|xgboost|catboost] [--folds 5] \
             [--suite_id 336 --task_id 361076] [--hyperband_R 2150] [--resume]
Bibliothek (--library, Standard gpboost): xgboost und catboost nur mit --scales gpboost, Suchraeume nach
SUCHRAEUME_XGB_CAT.md (methods.py). --paper_grid gilt fuer deren Gitter in methods._library_grid.
Ausgabe: <result_folder>/<method>/seed_<seed>/<library>/<suite>_<task>__num_leaves.csv (Spalten wie run_experiments.py),
         nach jedem Fold angehaengt, damit ein Abbruch durch das Zeitlimit die fertigen Folds behaelt.
--fold K (Lauf xgb_alle, lange Jobs je Fold aufgeteilt): nur Fold K rechnen, in dieselbe Datei. Unter einer Sperre
  (fcntl.lockf auf <Ordner>/.fold.lock, wie refill.py auf $HOME) wird der Block von Fold K eingefuegt und die Bloecke
  als Textzeilen nach Fold sortiert (nicht mit pandas neu formatiert): mit allen 5 Folds ist die Datei bis auf die
  Spalte runtime dieselbe wie bei einem Job mit 5 Folds. Fold K schon vollstaendig -> nichts tun, Exit-Code 0.
  Ohne --fold aendert sich nichts.
"""
import argparse
import csv
import fcntl
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import job_io
import methods
import SMAC_method
from run_experiments import RESULT_COLUMNS
from SMAC_method import ParameterOptimizationSMAC
from task_types import is_classification

LIBRARY = "gpboost"                             # Standard; --library xgboost|catboost (Laeufe xgb_alle, cat_alle)
LIBRARIES = ("gpboost", "xgboost", "catboost")
EXTRA_METHODS = ("bohb", "dehb", "hebo")       # Lauf "extra", nur in .venv_extra
STRATEGY = dict(try_max_depth=False, try_num_leaves=True, joint_tuning_depth_leaves=False, try_num_iter=False)


def gp_minimize_without(name, gp_minimize):
    """gp_minimize ohne die Dimension name; sie wird in jedem Aufruf und in x_iters als 0 eingesetzt."""
    def wrapped(func, dimensions, **kw):
        index = [d.name for d in dimensions].index(name)
        insert = lambda x: [*x[:index], 0.0, *x[index:]]
        result = gp_minimize(lambda x: func(insert(list(x))), dimensions[:index] + dimensions[index + 1:], **kw)
        result.x_iters = [insert(list(x)) for x in result.x_iters]
        result.x = insert(list(result.x))
        return result
    return wrapped


def quiet_methods():
    """SMAC auf WARNING (sonst setzt jeder HPOFacade-Aufruf das Logging auf INFO zurueck) und das Zufallsgitter
    ohne Ausgabe je Kombination und je cv-Runde (verbose_eval=3 in methods.py; Fehler druckt es bei 0 weiter)."""
    facade = SMAC_method.HPOFacade
    SMAC_method.HPOFacade = lambda *a, **kw: facade(*a, **{"logging_level": logging.WARNING, **kw})
    grid = methods.modified_grid_search_tune_parameters
    methods.modified_grid_search_tune_parameters = lambda *a, **kw: grid(*a, **{**kw, "verbose_eval": 0})
    # BOHB: hpbandster meldet jeden Rueckfall auf eine Zufallskonfiguration als Warnung; gezaehlt wird er in
    # obj.model_based_picks und je Fold einmal ausgegeben
    logging.getLogger("hpbandster").setLevel(logging.ERROR)


def apply_settings(method, trials, no_lambda_l1, paper_grid, smac_output, scales="paper"):
    """Setzt Versuchszahl und lambda_l1 fuer diesen Prozess, ohne die Module auf der Platte zu aendern.
    scales="gpboost": TPE, GP-BO, SMAC und Hyperband lassen lambda_l1 selbst weg (ParameterOptimization(
    lambda_l1=False)), hier wird dann nur noch das Zufallsgitter angepasst."""
    if method != "hyperband":   # Hyperband ruft optimize mit der Zahl der Brackets auf
        optimize = optuna.study.Study.optimize
        optuna.study.Study.optimize = lambda self, func, n_trials=None, **kw: optimize(self, func, n_trials=trials,
                                                                                        **kw)
    gp_minimize = methods.gp_minimize
    methods.gp_minimize = lambda func, dimensions, n_calls=None, **kw: gp_minimize(func, dimensions, n_calls=trials,
                                                                                   **kw)

    scenario = SMAC_method.Scenario
    SMAC_method.Scenario = lambda *a, **kw: scenario(*a, **{**kw, "n_trials": trials, "output_directory": smac_output})

    if paper_grid:
        grid = methods.modified_grid_search_tune_parameters
        methods.modified_grid_search_tune_parameters = lambda param_grid, *a, **kw: grid(
            {k: v for k, v in param_grid.items() if k != "bagging_fraction"} if kw.get("num_try_random") is None
            else param_grid, *a, **kw)

    if not no_lambda_l1:
        return
    if scales == "paper":
        suggest_float = optuna.trial.Trial.suggest_float
        optuna.trial.Trial.suggest_float = lambda self, name, low, high, **kw: (
            suggest_float(self, name, 0.0, 0.0, **kw) if name == "lambda_l1" or name.startswith("lambda_l1_")
            else suggest_float(self, name, low, high, **kw))

        methods.gp_minimize = gp_minimize_without("lambda_l1", methods.gp_minimize)

    grid_search = methods.modified_grid_search_tune_parameters
    methods.modified_grid_search_tune_parameters = lambda param_grid, *a, **kw: grid_search(
        {k: v for k, v in param_grid.items() if k != "lambda_l1"}, *a, **kw)

    if scales == "paper":
        configuration_space = SMAC_method.ConfigurationSpace
        SMAC_method.ConfigurationSpace = lambda space=None, **kw: configuration_space(
            space={**space, "lambda_l1": 0.0} if space and "lambda_l1" in space else space, **kw)


def run_fold(obj, method, trials, train_index, test_index, hyperband_R, lambda_l1=True):
    data = dict(X_train_full=obj.X.iloc[train_index], y_train_full=obj.y.iloc[train_index],
                X_test=obj.X.iloc[test_index], y_test=obj.y.iloc[test_index])
    if method == "random_search":
        return obj.grid_search_method(**data, num_try_random=trials)
    if method == "grid_search":
        return obj.grid_search_method(**data)
    if method == "tpe":
        return obj.tpe_method(**data)
    if method == "gp_bo":
        return obj.gp_bo_method(**data)
    if method == "hyperband":
        return obj.hyperband_method(**data, R=hyperband_R).drop(columns="param_ind")
    if method == "bohb":
        return obj.bohb_method(**data, R=hyperband_R, lambda_l1=lambda_l1)
    if method == "dehb":
        return obj.dehb_method(**data, R=hyperband_R, lambda_l1=lambda_l1)
    if method == "hebo":
        return obj.hebo_method(**data, n_trials=trials, lambda_l1=lambda_l1)
    return obj.run_smac(**data)


def expected_rows(method, trials, paper_grid, hyperband_R=2150):
    """Zeilen pro fertigem Fold; None, wo die Zahl nicht vorab feststeht (Hyperband, SMAC). BOHB und DEHB: die
    Ergebniszeilen des Hyperband-Rahmens (11 bei R = 2150)."""
    if method in ("random_search", "tpe", "gp_bo", "hebo"):
        return trials
    if method in ("bohb", "dehb"):
        return methods.hyperband_rows(hyperband_R)
    if method == "grid_search":
        return 135 if paper_grid else 270
    return None


def finished_folds(output, n_rows):
    """Folds, die in output vollstaendig stehen. Eine unvollstaendige oder unlesbare Datei wird auf die
    vollstaendigen Folds gekuerzt bzw. geloescht."""
    try:
        old = pd.read_csv(output)
        sizes = old.groupby("fold").size()
    except Exception as error:
        print(f"  {output} unlesbar ({error}), wird neu gerechnet", flush=True)
        output.unlink()
        return set()
    if n_rows is None:          # Hyperband, SMAC: ein halb geschriebener Fold ist kuerzer als die anderen
        n_rows = sizes.max()
    done = {int(f) for f, n in sizes.items() if n == n_rows}
    if len(done) < len(sizes):
        job_io.write_csv_atomic(old[old["fold"].isin(done)], output)
        print(f"  unvollstaendige Folds {sorted(set(sizes.index) - done)} verworfen", flush=True)
    if not done:
        output.unlink()
    return done


def fold_done(output, fold, n_rows):
    """--fold: steht Fold `fold` vollstaendig in output? Folds werden ganz und atomar geschrieben (write_fold),
    deshalb reicht bei Hyperband/SMAC (n_rows None) das Vorhandensein."""
    if not output.exists():
        return False
    sizes = pd.read_csv(output, usecols=["fold"])["fold"].value_counts()
    return int(sizes.get(fold, 0)) == (n_rows or int(sizes.get(fold, 0)) or -1)


def merge_fold_text(old_text, block_text, fold):
    """Datei-Text mit dem Block von Fold `fold` (block_text: Kopfzeile + Zeilen), Bloecke nach Fold sortiert.
    Zeilen werden nur umgestellt, nicht neu formatiert; ein alter Block desselben Folds wird ersetzt."""
    header, *new_rows = block_text.splitlines(keepends=True)
    if not old_text:
        return block_text
    old_header, *old_rows = old_text.splitlines(keepends=True)
    if old_header != header:
        raise ValueError("Kopfzeile der Ergebnisdatei passt nicht zum neuen Fold")
    position = next(csv.reader([header])).index("fold")
    keyed = [(int(float(next(csv.reader([row]))[position])), row) for row in old_rows]
    keyed = [(f, row) for f, row in keyed if f != fold] + [(fold, row) for row in new_rows]
    keyed.sort(key=lambda item: item[0])          # stabil: Reihenfolge innerhalb eines Folds bleibt
    return header + "".join(row for _, row in keyed)


def write_fold(results, output, fold):
    """--fold: Block von Fold `fold` unter Sperre in output einfuegen (gleichzeitige Folds derselben Datei)."""
    block = io.StringIO()
    results.to_csv(block, index=False)
    with open(output.parent / ".fold.lock", "a") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        try:
            old = output.read_text() if output.exists() else ""
            tmp = job_io.tmp_path(output)
            try:
                with open(tmp, "w", newline="") as handle:
                    handle.write(merge_fold_text(old, block.getvalue(), fold))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, output)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        finally:
            fcntl.lockf(lock, fcntl.LOCK_UN)


def main(args):
    path = f"data/{args.suite_id}_{args.task_id}"
    data_x = os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv")
    if not os.path.isfile(data_x) or os.path.getsize(data_x) == 0:
        sys.exit(f"ABBRUCH: {data_x} fehlt oder ist leer")
    folder = Path(args.result_folder) / args.method / f"seed_{args.seed}" / args.library
    output = folder / f"{args.suite_id}_{args.task_id}__num_leaves.csv"
    done = set()
    n_rows = expected_rows(args.method, args.trials, args.paper_grid, args.hyperband_R)
    if args.fold is not None:
        if fold_done(output, args.fold, n_rows):
            print(f"SCHON FERTIG: {output} (Fold {args.fold})", flush=True)
            return
        done = set(range(args.folds)) - {args.fold}      # nur Fold K rechnen; fertige Folds bleiben (write_fold)
    elif output.exists():
        if not args.resume:
            print(f"ABBRUCH: {output} existiert schon und wird nicht ueberschrieben", file=sys.stderr)
            sys.exit(3)
        done = finished_folds(output, n_rows)
        if len(done) >= args.folds:
            print(f"SCHON FERTIG: {output} ({len(done)} Folds)", flush=True)
            return
        if done:
            print(f"WEITER: {output} hat die Folds {sorted(done)}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    job_io.quiet_logging()
    quiet_methods()
    apply_settings(args.method, args.trials, args.no_lambda_l1, args.paper_grid,
                   Path(args.result_folder) / "smac3_output", scales=args.scales)

    X = pd.read_csv(data_x)
    y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{args.task_id}_categorical_indicator.npy"))
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=int(args.suite_id),
                                    seed=int(args.seed), library=args.library, scales=args.scales,
                                    lambda_l1=args.scales == "paper" or not args.no_lambda_l1, **STRATEGY)
    obj.paper_grid = args.paper_grid        # XGBoost/CatBoost-Gitter (methods._library_grid); GPBoost: apply_settings
    print(f"{args.method} {args.library} seed={args.seed} trials={args.trials} lambda_l1={'aus' if args.no_lambda_l1 else 'an'} "
          f"Skalen={args.scales} "
          f"{'Paper-Gitter ' if args.paper_grid else ''}-> {output}", flush=True)
    for fold, (train_index, test_index) in enumerate(list(obj.splits)[:args.folds]):
        if fold in done:
            continue
        results = run_fold(obj, args.method, args.trials, train_index, test_index,
                           args.hyperband_R, lambda_l1=not args.no_lambda_l1).reset_index(drop=True)
        if args.method == "bohb":
            print(f"  Fold {fold}: BOHB {obj.model_based_picks[0]} von {obj.model_based_picks[1]} Konfigurationen "
                  "modellbasiert (Rest zufaellig: random_fraction 1/3, Modell noch nicht da)", flush=True)
        results.insert(0, "iter", results.index)
        if args.no_lambda_l1:
            results["lambda_l1"] = 0.0
        if args.paper_grid and args.library != "catboost":   # CatBoost: subsample ist Achse des Gitters
            results["bagging_fraction"] = 1.0   # wie df_trials['bagging_fraction'] = 1 im Paper-Code
        results["fold"] = fold
        results["method"] = args.method
        results["task_id"] = int(args.task_id)
        results["library"] = args.library
        results["classification"] = 1 if is_classification(int(args.suite_id)) else 0
        results = results.reindex(columns=[*RESULT_COLUMNS, *[c for c in results.columns if c not in RESULT_COLUMNS]])
        if args.fold is not None:
            write_fold(results, output, fold)
            print(f"Fold {fold} fertig -> {output}", flush=True)
            return
        job_io.write_csv_atomic(results, output, append=True)
        done.add(fold)
    print(f"Folds fertig: {sorted(done)} -> {output}", flush=True)


def parse_args(argv=None):
    """Argumente; --l1 off und --no_lambda_l1 setzen beide args.no_lambda_l1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("random_search", "tpe", "SMAC", "gp_bo", "hyperband", "grid_search",
                                             *EXTRA_METHODS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result_folder", required=True, help="z. B. Results_ablation/A")
    parser.add_argument("--trials", type=int, default=135)
    parser.add_argument("--no_lambda_l1", action="store_true", help="wie --l1 off (alter Name, in jobs.tsv)")
    parser.add_argument("--l1", choices=("on", "off"), default=None,
                        help="off: lambda_l1 nicht im Suchraum, fest 0 (Paper-Setting); Standard on, mit "
                             "--no_lambda_l1 off")
    parser.add_argument("--scales", choices=methods.SCALES, default="paper",
                        help="paper: alle Skalen linear wie im Paper (Standard, bitgleich); gpboost: Log-Skalen wie "
                             "GPBoost fuer TPE, GP-BO, SMAC, Hyperband, BOHB, DEHB, HEBO (Gitter unveraendert)")
    parser.add_argument("--paper_grid", action="store_true",
                        help="deterministisches Gitter ohne bagging_fraction (135 Punkte wie im Paper-Code)")
    parser.add_argument("--library", choices=LIBRARIES, default=LIBRARY,
                        help="gpboost (Standard), xgboost oder catboost; xgboost/catboost nur mit --scales gpboost "
                             "(Suchraeume nach SUCHRAEUME_XGB_CAT.md)")
    parser.add_argument("--folds", type=int, default=5, help="nur fuer lokale Proben kleiner als 5")
    parser.add_argument("--fold", type=int, choices=range(5), default=None,
                        help="nur diesen Fold (0-4) rechnen und in die gemeinsame Datei einfuegen (xgb_alle)")
    parser.add_argument("--hyperband_R", type=int, default=2150, help="nur fuer lokale Proben kleiner als 2150")
    parser.add_argument("--resume", action="store_true",
                        help="fertige Datei ueberspringen, angefangene ab dem ersten fehlenden Fold fortsetzen")
    parser.add_argument("--suite_id", default="336")
    parser.add_argument("--task_id", default="361076")
    args = parser.parse_args(argv)
    if args.l1 == "on" and args.no_lambda_l1:
        parser.error("--l1 on und --no_lambda_l1 widersprechen sich")
    args.no_lambda_l1 = args.no_lambda_l1 or args.l1 == "off"
    if args.library != "gpboost" and args.scales != "gpboost":
        parser.error("--library xgboost/catboost nur mit --scales gpboost (freigegebene Suchraeume)")
    return args


if __name__ == "__main__":
    main(parse_args())
