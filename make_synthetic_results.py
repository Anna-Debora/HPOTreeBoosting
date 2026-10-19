"""Generate deterministic result CSVs in the repository's experiment layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (1101, 2202)
LIBRARIES = ("gpboost", "xgboost", "catboost")
TASKS = ((335, 97001, "synthetic_regression_a"),
         (336, 97002, "synthetic_regression_b"),
         (334, 97003, "synthetic_classification"))
MAIN_METHODS = ("grid_search", "random_search", "tpe", "gp_bo")
STRATEGIES = {
    "max_depth": (False, False, False),
    "num_leaves": (True, False, False),
    "joint": (False, True, False),
    "num_iter": (False, False, True),
}
METHOD_OFFSET = {"grid_search": 0.00, "random_search": 0.035, "tpe": 0.065,
                 "gp_bo": 0.095, "hyperband": 0.055, "SMAC": 0.08, "default": -0.06}
LIBRARY_OFFSET = {"gpboost": 0.0, "xgboost": 0.018, "catboost": 0.032}


def _row_values(rng, iteration, task_index, seed_index, fold, method, library,
                classification, flags, count):
    progress = 1.0 - np.exp(-np.arange(count, dtype=float) / 48.0)
    trial_index = iteration
    base = 0.67 if classification else 0.28
    task_offset = (task_index - 1) * (0.025 if classification else 0.08)
    seed_offset = (seed_index - 0.5) * 0.012
    fold_offset = (fold - 2) * 0.004
    noise = rng.normal(0.0, 0.006, count)
    quality = (base + task_offset + seed_offset + fold_offset
               + METHOD_OFFSET[method] + LIBRARY_OFFSET[library]
               + 0.10 * progress + noise)
    if classification:
        test_score = np.clip(quality, 0.05, 0.99)
        test_rmse = np.full(count, np.nan)
        test_log_loss = np.maximum(0.05, 1.2 - test_score + rng.normal(0, 0.008, count))
        objective = 1.0 - test_score + rng.normal(0, 0.01, count)
        best_score = np.maximum.accumulate(test_score)
        best_rmse = np.full(count, np.nan)
        best_log_loss = np.minimum.accumulate(test_log_loss)
    else:
        test_score = np.clip(quality, -0.8, 0.95)
        test_rmse = np.maximum(0.4, 3.6 - 1.8 * test_score + rng.normal(0, 0.015, count))
        test_log_loss = np.full(count, np.nan)
        objective = test_rmse + rng.normal(0, 0.02, count)
        best_score = np.maximum.accumulate(test_score)
        best_rmse = np.minimum.accumulate(test_rmse)
        best_log_loss = np.full(count, np.nan)

    try_leaves, joint, num_iter = flags
    depths = rng.integers(1, 11, count) if not try_leaves else np.full(count, -1)
    leaves = rng.integers(2, 1025, count) if try_leaves or joint else np.full(count, 1024)
    if joint:
        depths = rng.choice(np.asarray([-1, *range(1, 11)]), count)
    if num_iter:
        depths = rng.integers(1, 11, count)
    runtime_scale = {"gpboost": 1.0, "xgboost": 1.3, "catboost": 1.7}[library]
    runtime_method = {"grid_search": 1.0, "random_search": 0.95, "tpe": 1.05,
                     "gp_bo": 1.15, "hyperband": 0.8, "SMAC": 1.1, "default": 1.0}[method]
    runtime = runtime_scale * runtime_method * (0.018 + 0.00002 * (iteration % 17))
    if method == "default":
        runtime = np.full(count, np.nan)
    return {
        "iter": trial_index,
        "learning_rate": rng.uniform(0.001, 1.0, count),
        "min_data_in_leaf": rng.integers(1, 1001, count),
        "max_depth": depths,
        "lambda_l1": rng.uniform(0.0, 100.0, count),
        "lambda_l2": rng.uniform(0.0, 100.0, count),
        "num_leaves": leaves,
        "max_bin": rng.integers(255, 4178, count),
        "bagging_fraction": rng.uniform(0.5, 1.0, count),
        "feature_fraction": rng.uniform(0.5, 1.0, count),
        "val_score": objective,
        "runtime": np.full(count, runtime),
        "test_score": test_score,
        "test_log_loss": test_log_loss,
        "test_f1_score": np.full(count, np.nan if not classification else 0.7),
        "test_rmse": test_rmse,
        "current_best_test_score": best_score,
        "current_best_test_log_loss": best_log_loss,
        "current_best_test_f1_score": np.full(count, np.nan if not classification else 0.7),
        "current_best_test_rmse": best_rmse,
        "try_num_leaves": try_leaves,
        "joint_tuning_depth_leaves": joint,
        "try_num_iter": num_iter,
        "fold": fold,
        "method": method,
        "task_id": None,
        "library": library,
        "classification": int(classification),
        "n_iter": rng.integers(1, 1001, count) if num_iter else np.full(count, np.nan),
    }


def _block(seed, library, task_index, fold, task, method, strategy, count):
    suite_id, task_id, _ = task
    classification = suite_id in (334, 337)
    stable_seed = (seed * 31 + LIBRARIES.index(library) * 100000 + task_id * 7 + fold * 13
                   + list(MAIN_METHODS + ("hyperband", "SMAC", "default")).index(method) * 101
                   + list(STRATEGIES).index(strategy) * 17)
    rng = np.random.default_rng(stable_seed)
    flags = STRATEGIES.get(strategy, (False, False, False))
    data = _row_values(rng, np.arange(count), task_index, SEEDS.index(seed), fold,
                       method, library, classification, flags, count)
    data["task_id"] = np.full(count, task_id)
    return pd.DataFrame(data)


def generate(results_dir, iterations=270, hyperband_iterations=11):
    root = Path(results_dir)
    for suite_id in (334, 335, 336, 337):
        suite_tasks = [task for task in TASKS if task[0] == suite_id]
        indices = root / "task_indices"
        indices.mkdir(parents=True, exist_ok=True)
        np.save(indices / f"{suite_id}_task_indices.npy",
                np.asarray([task[1] for task in suite_tasks], dtype=int))
        (indices / f"{suite_id}_task_names.json").write_text(
            json.dumps([task[2] for task in suite_tasks], indent=2)
        )

    for task_index, task in enumerate(TASKS):
        suite_id, task_id, _ = task
        classification = suite_id in (334, 337)
        for library in LIBRARIES:
            default_folds = []
            for fold in range(5):
                block = _block(SEEDS[0], library, task_index, fold, task, "default", "max_depth", iterations)
                block["method"] = "default"
                default_folds.append(block)
            default_path = root / "Results_Default_Run" / library / f"{suite_id}_{task_id}.csv"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(default_folds, ignore_index=True).to_csv(default_path, index=False)

            for seed in SEEDS:
                main_frames = []
                for fold in range(5):
                    for method in MAIN_METHODS:
                        strategies = ("max_depth", "num_leaves") if method == "grid_search" else tuple(STRATEGIES)
                        for strategy in strategies:
                            main_frames.append(_block(seed, library, task_index, fold, task,
                                                      method, strategy, iterations))
                main_path = root / "Results" / f"seed_{seed}" / library / f"{suite_id}_{task_id}.csv"
                main_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(main_frames, ignore_index=True).to_csv(main_path, index=False)

                hyperband_frames = []
                for fold in range(5):
                    for strategy in ("max_depth", "num_leaves", "joint"):
                        hyperband_frames.append(_block(seed, library, task_index, fold, task,
                                                       "hyperband", strategy, hyperband_iterations))
                hyperband_path = (root / "Results_Hyperband_2007" / f"seed_{seed}" / library
                                  / f"{suite_id}_{task_id}.csv")
                hyperband_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(hyperband_frames, ignore_index=True).to_csv(hyperband_path, index=False)

                smac_frames = []
                for fold in range(5):
                    for strategy in ("max_depth", "num_leaves", "joint"):
                        smac_frames.append(_block(seed, library, task_index, fold, task,
                                                  "SMAC", strategy, iterations))
                smac_path = (root / "Results_SMAC" / f"seed_{seed}" / library
                             / f"{suite_id}_{task_id}.csv")
                smac_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(smac_frames, ignore_index=True).to_csv(smac_path, index=False)

    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="synthetic_results")
    parser.add_argument("--iterations", type=int, default=270)
    parser.add_argument("--hyperband-iterations", type=int, default=11)
    args = parser.parse_args()
    path = generate(args.out, args.iterations, args.hyperband_iterations)
    print(f"Synthetic results written to {path}")
    print(f"Tasks={len(TASKS)}, seeds={len(SEEDS)}, libraries={len(LIBRARIES)}, iterations={args.iterations}")


if __name__ == "__main__":
    main()
