"""Ein Teiljob der Ursachen-Laeufe A/B: Zufallsgitter, TPE oder SMAC mit waehlbarer Versuchszahl und mit oder
ohne lambda_l1, ohne Aenderung an methods.py oder SMAC_method.py.

Hintergrund (Testlauf vs. Paper, siehe repro_table.md): TPE, SMAC und Zufallsgitter liegen bei t = 135 unter
dem Paper. Seit dem Paper sind zwei Dinge anders: lambda_l1 ist im Suchraum, und es laufen 270 statt 135
Versuche (das beeinflusst t = 135 ueber best_iter des besten Versuchs aller 270 und ueber das SMAC-Startdesign
min(10 x Parameter, 0.25 x n_trials)).
  Lauf A (Paper-Einstellung): --trials 135 --no_lambda_l1
  Lauf B (nur Versuche):      --trials 135

Wie die Einstellungen ohne Codeaenderung greifen (nur in diesem Prozess):
  Versuche   Zufallsgitter: grid_search_method(num_try_random=N) direkt aufgerufen
             TPE:   optuna Study.optimize mit n_trials=N statt 270
             SMAC:  Scenario mit n_trials=N statt 270
  lambda_l1  Zufallsgitter: lambda_l1 aus dem Gitter entfernt -> genau das Gitter des Paper-Codes (gleiche
                    Groesse, gleiche Reihenfolge der Ziehung mit demselben Seed); GPBoost nimmt dann den Default 0
             TPE:   suggest_float('lambda_l1', 0, 0) -> fester Wert 0, wird nicht gesampelt
             SMAC:  lambda_l1 als Konstante 0 (SMAC laesst Konstanten im Sobol-Startdesign weg)
SMAC schreibt seine Zwischendateien unter <result_folder>/smac3_output statt ins Arbeitsverzeichnis.
Bestehende Ergebnisdateien werden nie ueberschrieben (Abbruch mit Exit-Code 3).

Aufruf:  python run_ablation.py --method {random_search,tpe,SMAC} --seed 27225 --result_folder Results_ablation/A \
             [--trials 135] [--no_lambda_l1] [--folds 5] [--suite_id 336 --task_id 361076]
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


def apply_settings(trials, no_lambda_l1, smac_output):
    """Setzt Versuchszahl und lambda_l1 fuer diesen Prozess, ohne die Module auf der Platte zu aendern."""
    optimize = optuna.study.Study.optimize
    optuna.study.Study.optimize = lambda self, func, n_trials=None, **kw: optimize(self, func, n_trials=trials, **kw)

    scenario = SMAC_method.Scenario
    SMAC_method.Scenario = lambda *a, **kw: scenario(*a, **{**kw, "n_trials": trials, "output_directory": smac_output})

    if not no_lambda_l1:
        return
    suggest_float = optuna.trial.Trial.suggest_float
    optuna.trial.Trial.suggest_float = lambda self, name, low, high, **kw: (
        suggest_float(self, name, 0.0, 0.0, **kw) if name == "lambda_l1" else suggest_float(self, name, low, high, **kw))

    grid_search = methods.modified_grid_search_tune_parameters
    methods.modified_grid_search_tune_parameters = lambda param_grid, *a, **kw: grid_search(
        {k: v for k, v in param_grid.items() if k != "lambda_l1"}, *a, **kw)

    configuration_space = SMAC_method.ConfigurationSpace
    SMAC_method.ConfigurationSpace = lambda space=None, **kw: configuration_space(
        space={**space, "lambda_l1": 0.0} if space and "lambda_l1" in space else space, **kw)


def run_fold(obj, method, trials, train_index, test_index):
    data = dict(X_train_full=obj.X.iloc[train_index], y_train_full=obj.y.iloc[train_index],
                X_test=obj.X.iloc[test_index], y_test=obj.y.iloc[test_index])
    if method == "random_search":
        return obj.grid_search_method(**data, num_try_random=trials)
    if method == "tpe":
        return obj.tpe_method(**data)
    return obj.run_smac(**data)


def main(args):
    path = f"data/{args.suite_id}_{args.task_id}"
    data_x = os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv")
    if not os.path.isfile(data_x) or os.path.getsize(data_x) == 0:
        sys.exit(f"ABBRUCH: {data_x} fehlt oder ist leer")
    folder = Path(args.result_folder) / args.method / f"seed_{args.seed}" / LIBRARY
    output = folder / f"{args.suite_id}_{args.task_id}__num_leaves.csv"
    if output.exists():
        print(f"ABBRUCH: {output} existiert schon und wird nicht ueberschrieben", file=sys.stderr)
        sys.exit(3)
    folder.mkdir(parents=True, exist_ok=True)
    apply_settings(args.trials, args.no_lambda_l1, Path(args.result_folder) / "smac3_output")

    X = pd.read_csv(data_x)
    y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{args.task_id}_categorical_indicator.npy"))
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=int(args.suite_id),
                                    seed=int(args.seed), library=LIBRARY, **STRATEGY)
    print(f"{args.method} seed={args.seed} trials={args.trials} lambda_l1={'aus' if args.no_lambda_l1 else 'an'} "
          f"-> {output}", flush=True)
    for fold, (train_index, test_index) in enumerate(list(obj.splits)[:args.folds]):
        results = run_fold(obj, args.method, args.trials, train_index, test_index).reset_index(drop=True)
        results.insert(0, "iter", results.index)
        if args.no_lambda_l1:
            results["lambda_l1"] = 0.0
        results["fold"] = fold
        results["method"] = args.method
        results["task_id"] = int(args.task_id)
        results["library"] = LIBRARY
        results["classification"] = 1 if is_classification(int(args.suite_id)) else 0
        results = results.reindex(columns=[*RESULT_COLUMNS, *[c for c in results.columns if c not in RESULT_COLUMNS]])
        results.to_csv(output, index=False, mode="w" if fold == 0 else "a", header=fold == 0)
        print(f"  Fold {fold}: {len(results)} Versuche, bester Test-Wert {results['test_score'].max():.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("random_search", "tpe", "SMAC"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result_folder", required=True, help="z. B. Results_ablation/A")
    parser.add_argument("--trials", type=int, default=135)
    parser.add_argument("--no_lambda_l1", action="store_true", help="lambda_l1 = 0 fest (Suchraum des Papers)")
    parser.add_argument("--folds", type=int, default=5, help="nur fuer lokale Proben kleiner als 5")
    parser.add_argument("--suite_id", default="336")
    parser.add_argument("--task_id", default="361076")
    main(parser.parse_args())
