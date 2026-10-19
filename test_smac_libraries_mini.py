"""Mini test: SMAC runs for gpboost, xgboost and catboost (regression and classification).

Uses small synthetic data (no download needed), 1 fold and 12 trials per strategy.
Run:  python test_smac_libraries_mini.py
"""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

import SMAC_method
import run_experiment_H_SMAC

TRIALS = 12
SEED = 27225
CASES = {"regression": (335, 999001), "classification": (334, 999002)}
LIBRARIES = ("gpboost", "xgboost", "catboost")


def write_dataset(root, suite, task, classification):
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.uniform(size=n),
        "cat": rng.choice(["u", "v", "w"], size=n),
    })
    signal = X["a"] + 0.5 * X["b"] + (X["cat"] == "u") * 1.0
    if classification:
        y = pd.DataFrame({"y": (signal + rng.normal(scale=0.5, size=n) > 0.5).astype(int)})
    else:
        y = pd.DataFrame({"y": signal + rng.normal(scale=0.3, size=n)})
    folder = os.path.join(root, "data", f"{suite}_{task}")
    os.makedirs(folder)
    X.to_csv(os.path.join(folder, f"{suite}_{task}_X.csv"), index=False)
    y.to_csv(os.path.join(folder, f"{suite}_{task}_y.csv"), index=False)
    np.save(os.path.join(folder, f"{suite}_{task}_categorical_indicator.npy"),
            np.array([False, False, False, True]))


def main():
    original_scenario = SMAC_method.Scenario
    SMAC_method.Scenario = lambda *a, **k: original_scenario(*a, **{**k, "n_trials": TRIALS})
    original_split = SMAC_method.ParameterOptimizationSMAC._split_data
    SMAC_method.ParameterOptimizationSMAC._split_data = lambda self: list(original_split(self))[:1]

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="smac-libs-") as temp:
        os.chdir(temp)
        try:
            for kind, (suite, task) in CASES.items():
                write_dataset(temp, suite, task, kind == "classification")
            for kind, (suite, task) in CASES.items():
                for library in LIBRARIES:
                    run_experiment_H_SMAC.main(argparse.Namespace(
                        suite_id=str(suite), task_id=str(task), seed=str(SEED), library=library,
                        only="smac", result_folder_SMAC=os.path.join(temp, "Results_SMAC"),
                        result_folder_H=os.path.join(temp, "Results_Hyperband")))
                    path = os.path.join(temp, "Results_SMAC", f"seed_{SEED}", library, f"{suite}_{task}.csv")
                    assert os.path.exists(path), f"missing result file {path}"
                    frame = pd.read_csv(path)
                    assert len(frame) == 3 * TRIALS, (kind, library, len(frame))
                    assert np.isfinite(frame["val_score"]).all(), (kind, library)
                    assert (frame["val_score"] != 1e99).all(), (kind, library)
                    assert (frame["runtime"] > 0).all(), (kind, library)
                    assert frame["test_score"].notna().all(), (kind, library)
                    assert set(frame["fold"]) == {0}
                    print(f"OK {kind:14s} {library:9s} rows={len(frame)} "
                          f"val_score {frame['val_score'].min():.3f}..{frame['val_score'].max():.3f} "
                          f"runtime_sum={frame['runtime'].sum():.1f}s")
            # the three libraries must not overwrite each other
            for kind, (suite, task) in CASES.items():
                dirs = sorted(os.listdir(os.path.join(temp, "Results_SMAC", f"seed_{SEED}")))
                assert dirs == sorted(LIBRARIES), dirs
            print("ALL OK")
        finally:
            os.chdir(old_cwd)
            SMAC_method.Scenario = original_scenario
            SMAC_method.ParameterOptimizationSMAC._split_data = original_split


if __name__ == "__main__":
    main()
