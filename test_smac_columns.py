"""Check sampled SMAC parameter ranges on abalone with 20 trials per strategy."""

import os

import numpy as np
import pandas as pd

import SMAC_method


ROOT = os.path.dirname(os.path.abspath(__file__))
SUITE_ID = 335
TASK_ID = 361288
SEED = 27225
N_TRIALS = 20


_original_scenario = SMAC_method.Scenario
SMAC_method.Scenario = lambda *args, **kwargs: _original_scenario(
    *args, **{**kwargs, "n_trials": N_TRIALS}
)
_original_split_data = SMAC_method.ParameterOptimizationSMAC._split_data
SMAC_method.ParameterOptimizationSMAC._split_data = lambda self: list(
    _original_split_data(self)
)[:1]


def report_range(frame, column, low, high, strategy, violations, mask=None):
    values = pd.to_numeric(frame[column], errors="coerce")
    checked = values if mask is None else values.loc[mask]
    invalid = checked.isna() | (checked < low) | (checked > high)
    if invalid.any():
        for row_index, value in checked.loc[invalid].items():
            violations.append(
                f"{strategy}: {column}={value!r} außerhalb [{low}, {high}] "
                f"(Zeile {row_index})"
            )
    if checked.empty:
        print(f"  {column}: keine zu prüfenden Werte")
    else:
        print(f"  {column}: min={checked.min():.6g}, max={checked.max():.6g}, "
              f"n={len(checked)}, Bereich=[{low}, {high}]")


def main():
    data_path = os.path.join(ROOT, "data", f"{SUITE_ID}_{TASK_ID}")
    X = pd.read_csv(os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_X.csv"))
    y = pd.read_csv(os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_y.csv"))
    categorical_indicator = np.load(
        os.path.join(data_path, f"{SUITE_ID}_{TASK_ID}_categorical_indicator.npy")
    )
    dataset_rows = len(X)
    strategies = {
        "max_depth": {"try_max_depth": True},
        "num_leaves": {"try_num_leaves": True, "try_max_depth": False},
        "joint": {"joint_tuning_depth_leaves": True, "try_max_depth": False},
    }
    violations = []

    for strategy, flags in strategies.items():
        optimizer = SMAC_method.ParameterOptimizationSMAC(
            X=X.copy(), y=y.copy(), categorical_indicator=categorical_indicator.copy(),
            suite_id=SUITE_ID, seed=SEED, **flags
        )
        result = optimizer.method_smac()
        print(f"\n{strategy}: {len(result)} Zeilen, erwartete {N_TRIALS}; "
              f"Spalten lambda_l1/lambda_l2 vorhanden: "
              f"{'lambda_l1' in result and 'lambda_l2' in result}")
        if len(result) != N_TRIALS:
            violations.append(f"{strategy}: {len(result)} Zeilen statt {N_TRIALS}")

        report_range(result, "lambda_l1", 0, 100, strategy, violations)
        report_range(result, "lambda_l2", 0, 100, strategy, violations)
        report_range(result, "min_data_in_leaf", 1, 1000, strategy, violations)
        report_range(result, "max_bin", 255, dataset_rows, strategy, violations)
        report_range(result, "learning_rate", 0.001, 1, strategy, violations)
        report_range(result, "bagging_fraction", 0.5, 1, strategy, violations)
        report_range(result, "feature_fraction", 0.5, 1, strategy, violations)
        report_range(result, "num_leaves", 2, 1024, strategy, violations)

        if strategy == "max_depth":
            report_range(result, "max_depth", 1, 10, strategy, violations)
        elif strategy == "joint":
            report_range(result, "max_depth", -1, 10, strategy, violations)
        else:
            fixed_depths = pd.to_numeric(result["max_depth"], errors="coerce")
            if not fixed_depths.eq(-1).all():
                violations.append(
                    "num_leaves: max_depth ist ein Fixparameter und sollte durchgehend -1 sein"
                )
            print("  max_depth: nicht getunt, fixer Wert -1 (nicht als Tiefe geprüft)")

        equal_pairs = result["lambda_l1"].eq(result["lambda_l2"])
        equal_count = int(equal_pairs.sum())
        print(f"  lambda_l1 == lambda_l2: {equal_count}/{len(result)} Zeilen")
        if equal_count:
            rows = result.index[equal_pairs].tolist()
            violations.append(
                f"{strategy}: lambda_l1 und lambda_l2 sind in {equal_count} Zeilen identisch: {rows}"
            )

    print("\nVerletzungen:")
    if violations:
        for violation in violations:
            print(f"- {violation}")
    else:
        print("- keine")


if __name__ == "__main__":
    main()
