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
SMAC schreibt seine Zwischendateien unter <result_folder>/smac3_output statt ins Arbeitsverzeichnis.
Bestehende Ergebnisdateien werden nie ueberschrieben (Abbruch mit Exit-Code 3), ausser mit --resume:
  vollstaendige Datei (alle Folds)  -> nichts tun, Exit-Code 0
  angefangene Datei (Zeitlimit)     -> fertige Folds behalten, ab dem ersten fehlenden Fold weiterrechnen.
  Das ergibt dieselben Zahlen wie ein Lauf am Stueck: jede Methode setzt ihren Sampler pro Fold neu mit dem
  Seed auf (TPESampler, gp_minimize, SMAC-Scenario, RandomState) und min_score/best_iter pro Aufruf zurueck.

Aufruf:  python run_ablation.py --method {random_search,tpe,SMAC,gp_bo,hyperband,grid_search} --seed 27225 \
             --result_folder Results_ablation/A [--trials 135] [--no_lambda_l1] [--paper_grid] [--folds 5] \
             [--suite_id 336 --task_id 361076] [--hyperband_R 2150] [--resume]
Ausgabe: <result_folder>/<method>/seed_<seed>/gpboost/<suite>_<task>__num_leaves.csv (Spalten wie run_experiments.py),
         nach jedem Fold angehaengt, damit ein Abbruch durch das Zeitlimit die fertigen Folds behaelt.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import methods
import SMAC_method
from run_experiments import RESULT_COLUMNS
from SMAC_method import ParameterOptimizationSMAC
from task_types import is_classification

LIBRARY = "gpboost"
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


def apply_settings(method, trials, no_lambda_l1, paper_grid, smac_output):
    """Setzt Versuchszahl und lambda_l1 fuer diesen Prozess, ohne die Module auf der Platte zu aendern."""
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
    suggest_float = optuna.trial.Trial.suggest_float
    optuna.trial.Trial.suggest_float = lambda self, name, low, high, **kw: (
        suggest_float(self, name, 0.0, 0.0, **kw) if name == "lambda_l1" or name.startswith("lambda_l1_")
        else suggest_float(self, name, low, high, **kw))

    methods.gp_minimize = gp_minimize_without("lambda_l1", methods.gp_minimize)

    grid_search = methods.modified_grid_search_tune_parameters
    methods.modified_grid_search_tune_parameters = lambda param_grid, *a, **kw: grid_search(
        {k: v for k, v in param_grid.items() if k != "lambda_l1"}, *a, **kw)

    configuration_space = SMAC_method.ConfigurationSpace
    SMAC_method.ConfigurationSpace = lambda space=None, **kw: configuration_space(
        space={**space, "lambda_l1": 0.0} if space and "lambda_l1" in space else space, **kw)


def run_fold(obj, method, trials, train_index, test_index, hyperband_R):
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
    return obj.run_smac(**data)


def expected_rows(method, trials, paper_grid):
    """Zeilen pro fertigem Fold; None, wo die Zahl nicht vorab feststeht (Hyperband, SMAC)."""
    if method in ("random_search", "tpe", "gp_bo"):
        return trials
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
        old[old["fold"].isin(done)].to_csv(output, index=False)
        print(f"  unvollstaendige Folds {sorted(set(sizes.index) - done)} verworfen", flush=True)
    if not done:
        output.unlink()
    return done


def main(args):
    path = f"data/{args.suite_id}_{args.task_id}"
    data_x = os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv")
    if not os.path.isfile(data_x) or os.path.getsize(data_x) == 0:
        sys.exit(f"ABBRUCH: {data_x} fehlt oder ist leer")
    folder = Path(args.result_folder) / args.method / f"seed_{args.seed}" / LIBRARY
    output = folder / f"{args.suite_id}_{args.task_id}__num_leaves.csv"
    done = set()
    if output.exists():
        if not args.resume:
            print(f"ABBRUCH: {output} existiert schon und wird nicht ueberschrieben", file=sys.stderr)
            sys.exit(3)
        done = finished_folds(output, expected_rows(args.method, args.trials, args.paper_grid))
        if len(done) >= args.folds:
            print(f"SCHON FERTIG: {output} ({len(done)} Folds)", flush=True)
            return
        if done:
            print(f"WEITER: {output} hat die Folds {sorted(done)}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    apply_settings(args.method, args.trials, args.no_lambda_l1, args.paper_grid,
                   Path(args.result_folder) / "smac3_output")

    X = pd.read_csv(data_x)
    y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{args.task_id}_categorical_indicator.npy"))
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=int(args.suite_id),
                                    seed=int(args.seed), library=LIBRARY, **STRATEGY)
    print(f"{args.method} seed={args.seed} trials={args.trials} lambda_l1={'aus' if args.no_lambda_l1 else 'an'} "
          f"{'Paper-Gitter ' if args.paper_grid else ''}-> {output}", flush=True)
    for fold, (train_index, test_index) in enumerate(list(obj.splits)[:args.folds]):
        if fold in done:
            continue
        results = run_fold(obj, args.method, args.trials, train_index, test_index,
                           args.hyperband_R).reset_index(drop=True)
        results.insert(0, "iter", results.index)
        if args.no_lambda_l1:
            results["lambda_l1"] = 0.0
        if args.paper_grid:
            results["bagging_fraction"] = 1.0   # wie df_trials['bagging_fraction'] = 1 im Paper-Code
        results["fold"] = fold
        results["method"] = args.method
        results["task_id"] = int(args.task_id)
        results["library"] = LIBRARY
        results["classification"] = 1 if is_classification(int(args.suite_id)) else 0
        results = results.reindex(columns=[*RESULT_COLUMNS, *[c for c in results.columns if c not in RESULT_COLUMNS]])
        first = not output.exists()
        results.to_csv(output, index=False, mode="w" if first else "a", header=first)
        print(f"  Fold {fold}: {len(results)} Versuche, bester Test-Wert {results['test_score'].max():.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("random_search", "tpe", "SMAC", "gp_bo", "hyperband", "grid_search"),
                        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result_folder", required=True, help="z. B. Results_ablation/A")
    parser.add_argument("--trials", type=int, default=135)
    parser.add_argument("--no_lambda_l1", action="store_true", help="lambda_l1 = 0 fest (Suchraum des Papers)")
    parser.add_argument("--paper_grid", action="store_true",
                        help="deterministisches Gitter ohne bagging_fraction (135 Punkte wie im Paper-Code)")
    parser.add_argument("--folds", type=int, default=5, help="nur fuer lokale Proben kleiner als 5")
    parser.add_argument("--hyperband_R", type=int, default=2150, help="nur fuer lokale Proben kleiner als 2150")
    parser.add_argument("--resume", action="store_true",
                        help="fertige Datei ueberspringen, angefangene ab dem ersten fehlenden Fold fortsetzen")
    parser.add_argument("--suite_id", default="336")
    parser.add_argument("--task_id", default="361076")
    main(parser.parse_args())
