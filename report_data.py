"""Load HPO result folders into task/seed-aligned arrays for report generation."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import utils_plot_maker as upm


METHODS = ("grid_search", "random_search", "tpe", "gp_bo", "hyperband", "SMAC")
STRATEGIES = ("num_leaves", "max_depth", "joint", "num_iter")
METRICS = ("R2", "RMSE", "Accuracy", "LogLoss")
_FLAG_NAMES = ("try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter")


def _truthy(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _task_is_classification(suite_id):
    return int(suite_id) in (334, 337)


def _read_task_file(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text())
        return [_task_record(record, index) for index, record in enumerate(records)]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        return [_task_record(row.to_dict(), index) for index, row in frame.iterrows()]

    tasks = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        pieces = value.replace(",", " ").split()
        if len(pieces) < 2:
            raise ValueError(f"Task-Zeile braucht suite_id und task_id: {line!r}")
        record = {"suite_id": pieces[0], "task_id": pieces[1]}
        if len(pieces) > 2:
            record["name"] = " ".join(pieces[2:])
        tasks.append(_task_record(record, len(tasks)))
    return tasks


def _task_record(record, index):
    if isinstance(record, dict):
        suite_id = int(record.get("suite_id", record.get("suite", 335)))
        task_id = int(record.get("task_id", record.get("task", record.get("id"))))
        name = record.get("name", record.get("task_name", str(task_id)))
    else:
        suite_id, task_id, *label = record if isinstance(record, (list, tuple)) else (335, record)
        suite_id, task_id = int(suite_id), int(task_id)
        name = " ".join(map(str, label)) if label else str(task_id)
    return {
        "suite_id": suite_id,
        "task_id": task_id,
        "name": str(name),
        "classification": _task_is_classification(suite_id),
        "index": index,
    }


def discover_tasks(results_root, tasks_file=None, project_root=None):
    """Discover task names/IDs from task_indices or a supplied task list."""
    if tasks_file:
        return _read_task_file(tasks_file)

    result_indices = Path(results_root) / "task_indices"
    if result_indices.exists():
        tasks = _tasks_from_indices(result_indices)
        if tasks:
            return tasks

    root = Path(results_root)
    identifiers = set()
    for pattern in ("Results/seed_*/*/*.csv", "Results_Hyperband_2007/seed_*/*/*.csv",
                    "Results_SMAC/seed_*/*.csv", "Results_Default_Run/*/*.csv",
                    "seed_*/*/*.csv"):
        for path in root.glob(pattern):
            stem = path.stem.split("_")
            if len(stem) >= 2 and stem[0].isdigit() and stem[1].isdigit():
                identifiers.add((int(stem[0]), int(stem[1])))
    if identifiers:
        return [
            _task_record({"suite_id": suite, "task_id": task, "name": str(task)}, index)
            for index, (suite, task) in enumerate(sorted(identifiers))
        ]

    project_indices = Path(project_root or Path(__file__).parent) / "task_indices"
    if project_indices.exists():
        tasks = _tasks_from_indices(project_indices)
        if tasks:
            return tasks
    return []


def _tasks_from_indices(directory):
    tasks = []
    for suite_id in (334, 335, 336, 337):
        ids_file = directory / f"{suite_id}_task_indices.npy"
        names_file = directory / f"{suite_id}_task_names.json"
        if not ids_file.exists():
            continue
        task_ids = np.load(ids_file).tolist()
        task_names = json.loads(names_file.read_text()) if names_file.exists() else []
        for index, task_id in enumerate(task_ids):
            name = task_names[index] if index < len(task_names) else str(task_id)
            tasks.append(_task_record(
                {"suite_id": suite_id, "task_id": task_id, "name": name}, len(tasks)
            ))
    return tasks


def discover_seeds(results_root, libraries):
    root = Path(results_root)
    result = {}
    for library in libraries:
        found = set()
        for path in root.glob(f"Results/seed_*/{library}/*.csv"):
            seed_name = path.parents[1].name
            if seed_name.startswith("seed_"):
                found.add(int(seed_name.removeprefix("seed_")))
        for path in root.glob(f"Results_Hyperband_2007/seed_*/{library}/*.csv"):
            seed_name = path.parents[1].name
            if seed_name.startswith("seed_"):
                found.add(int(seed_name.removeprefix("seed_")))
        for path in root.glob(f"seed_*/{library}/*.csv"):
            seed_name = path.parent.parent.name
            if seed_name.startswith("seed_"):
                found.add(int(seed_name.removeprefix("seed_")))
        for path in root.glob(f"seed_*/{library}/*.csv"):
            seed_name = path.parent.parent.name
            if seed_name.startswith("seed_"):
                found.add(int(seed_name.removeprefix("seed_")))
        result[library] = sorted(found)
    return result


def _cell_mask(frame, strategy):
    leaves = _truthy(frame["try_num_leaves"])
    joint = _truthy(frame["joint_tuning_depth_leaves"])
    num_iter = _truthy(frame["try_num_iter"])
    if strategy == "num_leaves":
        return leaves & ~num_iter
    if strategy == "max_depth":
        return ~leaves & ~joint & ~num_iter
    if strategy == "joint":
        return joint & ~num_iter
    return num_iter


def _metric_column(metric, oracle=True):
    if metric in ("R2", "Accuracy"):
        return "current_best_test_score" if oracle else "test_score"
    if metric == "RMSE":
        return "current_best_test_rmse" if oracle else "test_rmse"
    return "current_best_test_log_loss" if oracle else "test_log_loss"


def _strategy_index(frame, method, strategy):
    method_mask = frame["method"].astype(str).eq(method)
    return frame.loc[method_mask & _cell_mask(frame, strategy)].copy()


def _records_for_cell(frame, method, strategy, task, max_iterations, file_path, warnings_list):
    if frame is None or frame.empty:
        return None
    required = {"method", *_FLAG_NAMES, "fold"}
    missing = required.difference(frame.columns)
    if missing:
        warnings_list.append(f"Inkompatibles CSV {file_path}: fehlende Spalten {sorted(missing)}")
        return None
    rows = _strategy_index(frame, method, strategy)
    if rows.empty:
        return None
    if "iter" in rows.columns:
        rows["_iter"] = pd.to_numeric(rows["iter"], errors="coerce")
    else:
        rows["_iter"] = rows.groupby("fold", sort=False).cumcount()
    rows["_iter"] = rows["_iter"].fillna(rows.groupby("fold").cumcount()).astype(int)
    rows = rows.loc[(rows["_iter"] >= 0) & (rows["_iter"] < max_iterations)]
    fold_ids = sorted(pd.to_numeric(rows["fold"], errors="coerce").dropna().unique())
    if not fold_ids:
        return None

    result = {
        "scores": {metric: np.full((max_iterations, len(fold_ids)), np.nan) for metric in METRICS},
        "val_score": np.full((max_iterations, len(fold_ids)), np.nan),
        "runtime": np.full((max_iterations, len(fold_ids)), np.nan),
        "val_selected": {metric: np.full((max_iterations, len(fold_ids)), np.nan) for metric in METRICS},
        "folds": fold_ids,
    }
    for fold_index, fold_id in enumerate(fold_ids):
        fold_rows = rows.loc[pd.to_numeric(rows["fold"], errors="coerce") == fold_id]
        fold_rows = fold_rows.sort_values("_iter", kind="stable")
        fold_rows = fold_rows.drop_duplicates("_iter", keep="last")
        source_indices = fold_rows["_iter"].to_numpy(dtype=int)
        if method == "hyperband" and len(source_indices) > 1:
            indices = np.rint(np.linspace(0, max_iterations - 1, len(source_indices))).astype(int)
        else:
            indices = source_indices
        val = pd.to_numeric(fold_rows.get("val_score", np.nan), errors="coerce").to_numpy(dtype=float)
        result["val_score"][indices, fold_index] = val
        if "runtime" in fold_rows.columns:
            per_trial = pd.to_numeric(fold_rows["runtime"], errors="coerce").to_numpy(dtype=float)
            result["runtime"][indices, fold_index] = np.cumsum(np.nan_to_num(per_trial, nan=0.0))

        for metric in METRICS:
            oracle_col = _metric_column(metric, oracle=True)
            if oracle_col in fold_rows:
                result["scores"][metric][indices, fold_index] = pd.to_numeric(
                    fold_rows[oracle_col], errors="coerce"
                ).to_numpy(dtype=float)
            test_col = _metric_column(metric, oracle=False)
            if test_col not in fold_rows or "val_score" not in fold_rows:
                continue
            test_values = pd.to_numeric(fold_rows[test_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(val) & np.isfinite(test_values)
            if not np.any(valid):
                continue
            best_order = np.argsort(np.where(valid, val, np.inf), kind="stable")
            best_seen = np.full(max_iterations, np.nan)
            best_validation = np.inf
            selected_test = np.nan
            for iteration, validation, test_value, is_valid in zip(indices, val, test_values, valid):
                if is_valid and validation < best_validation:
                    best_validation = validation
                    selected_test = test_value
                best_seen[iteration:] = selected_test
            result["val_selected"][metric][:, fold_index] = best_seen

        if method == "hyperband" and len(indices):
            end = indices[-1] + 1
            for metric in METRICS:
                for target in (result["scores"][metric], result["val_selected"][metric]):
                    block = target[:end, fold_index]
                    valid_values = np.flatnonzero(np.isfinite(block))
                    if valid_values.size:
                        last = valid_values[0]
                        for position in range(end):
                            if np.isfinite(block[position]):
                                last = position
                            else:
                                block[position] = block[last]
            block = result["val_score"][:end, fold_index]
            valid_values = np.flatnonzero(np.isfinite(block))
            if valid_values.size:
                last = valid_values[0]
                for position in range(end):
                    if np.isfinite(block[position]):
                        last = position
                    else:
                        block[position] = block[last]

    return result


def _fold_mean(values):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def _file_for(results_root, method, library, seed, task):
    root = Path(results_root)
    if method == "Default":
        return root / "Results_Default_Run" / library / f"{task['suite_id']}_{task['task_id']}.csv"
    if method == "hyperband":
        return root / "Results_Hyperband_2007" / f"seed_{seed}" / library / f"{task['suite_id']}_{task['task_id']}.csv"
    if method == "SMAC":
        return root / "Results_SMAC" / f"seed_{seed}" / f"{task['suite_id']}_{task['task_id']}.csv"
    standard = root / "Results" / f"seed_{seed}" / library / f"{task['suite_id']}_{task['task_id']}.csv"
    if standard.exists():
        return standard
    return root / f"seed_{seed}" / library / f"{task['suite_id']}_{task['task_id']}.csv"


def _read_frame(path, warnings_list, required=False):
    if not path.exists():
        warnings_list.append(f"Fehlt: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # Preserve the report if one result file is malformed.
        warnings_list.append(f"CSV nicht lesbar {path}: {type(exc).__name__}: {exc}")
        return None


def _create_scores_dict_if_compatible(results_root, library, tasks, seeds, iterations, warnings_list):
    """Reuse the project's score loader for complete fixed-layout result sets."""
    if not tasks or not seeds or iterations < 1:
        return {}
    class_groups = {
        True: [task for task in tasks if task["classification"]],
        False: [task for task in tasks if not task["classification"]],
    }
    saved = {name: getattr(upm, name) for name in (
        "seeds", "NUM_SEEDS", "FOLDS", "NUM_FOLDS", "NUM_ITERS", "NUM_TASKS",
        "NUM_REGR_TASKS", "NUM_CLASS_TASKS", "HPO_METHODS", "NUM_METHODS",
    ) if hasattr(upm, name)}
    loaded = {}
    old_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="report_task_indices-") as temp:
            os.makedirs(os.path.join(temp, "task_indices"), exist_ok=True)
            for suite_id in (334, 335, 336, 337):
                suite_tasks = [task for task in tasks if task["suite_id"] == suite_id]
                np.save(os.path.join(temp, "task_indices", f"{suite_id}_task_indices.npy"),
                        np.asarray([task["task_id"] for task in suite_tasks], dtype=int))
                with open(os.path.join(temp, "task_indices", f"{suite_id}_task_names.json"), "w") as file:
                    json.dump([task["name"] for task in suite_tasks], file)
            os.chdir(temp)
            for classification, selected in class_groups.items():
                if not selected:
                    continue
                upm.seeds = list(seeds)
                upm.NUM_SEEDS = len(seeds)
                upm.FOLDS = sorted({fold for fold in range(5)})
                upm.NUM_FOLDS = len(upm.FOLDS)
                upm.NUM_ITERS = iterations
                upm.NUM_CLASS_TASKS = len(class_groups[True])
                upm.NUM_REGR_TASKS = len(class_groups[False])
                upm.NUM_TASKS = len(selected)
                expected_methods = list(METHODS)
                if library != "gpboost":
                    expected_methods.remove("SMAC")
                upm.HPO_METHODS = expected_methods
                upm.NUM_METHODS = len(expected_methods)
                try:
                    for metric, kwargs in (
                        ("score", {}), ("rmse", {"rmse": True}),
                    ) if not classification else (
                        ("accuracy", {}), ("logloss", {"logloss": True}),
                    ):
                        scores, names = upm.create_scores_dict(
                            str(results_root), classification=classification,
                            library=library, **kwargs
                        )
                        loaded[(classification, metric)] = (scores, names, selected)
                except Exception as exc:
                    warnings_list.append(
                        f"create_scores_dict übersprungen für {library}/"
                        f"{'classification' if classification else 'regression'}: "
                        f"{type(exc).__name__}: {exc}"
                    )
    finally:
        os.chdir(old_cwd)
        for name, value in saved.items():
            setattr(upm, name, value)
    return loaded


def load_report_data(results_root, libraries, tasks_file=None, project_root=None):
    """Load scores, validation losses, cumulative runtimes and val-selected scores.

    Score arrays use ``(strategy, iteration, task, seed)``; default arrays use
    ``(iteration, task, seed)``. Fold-specific runtimes are cumulatively summed
    in trial order before averaging across folds. Missing runtimes remain NaN.
    """
    results_root = Path(results_root).resolve()
    libraries = list(libraries)
    tasks = discover_tasks(results_root, tasks_file, project_root)
    seeds_by_library = discover_seeds(results_root, libraries)
    warnings_list = []
    missing_files = []
    max_iterations = 0
    for library in libraries:
        for task in tasks:
            for seed in seeds_by_library.get(library, []):
                for method in ("random_search", "tpe", "gp_bo", "grid_search", "hyperband"):
                    path = _file_for(results_root, method, library, seed, task)
                    if path.exists():
                        try:
                            frame = pd.read_csv(path)
                            if "iter" in frame:
                                frame["iter"] = pd.to_numeric(frame["iter"], errors="coerce")
                                grouping = [column for column in ("method", *_FLAG_NAMES, "fold")
                                            if column in frame]
                                method_frame = frame
                                if method not in ("hyperband", "SMAC") and "method" in frame:
                                    method_frame = frame.loc[frame["method"].astype(str).eq(method)]
                                counts = method_frame.groupby(grouping, dropna=False)["iter"].count()
                                if not counts.empty:
                                    max_iterations = max(max_iterations, int(counts.max()))
                        except Exception:
                            continue
            default_path = _file_for(results_root, "Default", library, None, task)
            if default_path.exists():
                try:
                    default_frame = pd.read_csv(default_path, usecols=["fold"])
                    default_counts = default_frame.groupby("fold").size()
                    if not default_counts.empty:
                        max_iterations = max(max_iterations, int(default_counts.max()))
                except Exception:
                    pass
    max_iterations = max(max_iterations, 1)
    task_count = len(tasks)
    result = {
        "results_root": str(results_root), "libraries": libraries, "tasks": tasks,
        "seeds": seeds_by_library, "iterations": max_iterations,
        "methods": list(METHODS), "strategies": list(STRATEGIES),
        "scores": {}, "val_score": {}, "runtime": {}, "val_selected": {},
        "warnings": warnings_list, "missing_files": missing_files,
        "create_scores_dict": {},
    }

    for library in libraries:
        seeds = seeds_by_library.get(library, [])
        seed_count = max(len(seeds), 1)
        result["scores"][library] = {
            metric: {
                **{method: np.full((4, max_iterations, task_count, seed_count), np.nan)
                   for method in METHODS},
                "Default": np.full((max_iterations, task_count, seed_count), np.nan),
            }
            for metric in METRICS
        }
        result["val_selected"][library] = {
            metric: {
                **{method: np.full((4, max_iterations, task_count, seed_count), np.nan)
                   for method in METHODS},
                "Default": np.full((max_iterations, task_count, seed_count), np.nan),
            }
            for metric in METRICS
        }
        result["val_score"][library] = {
            **{method: np.full((4, max_iterations, task_count, seed_count), np.nan)
               for method in METHODS},
            "Default": np.full((4, max_iterations, task_count, seed_count), np.nan),
        }
        result["runtime"][library] = {
            **{method: np.full((4, max_iterations, task_count, seed_count), np.nan)
               for method in METHODS},
            "Default": np.full((4, max_iterations, task_count, seed_count), np.nan),
        }
        for task_index, task in enumerate(tasks):
            default_path = _file_for(results_root, "Default", library, None, task)
            default_frame = _read_frame(default_path, warnings_list)
            if default_frame is None:
                missing_files.append(str(default_path))
            else:
                for metric in METRICS:
                    column = _metric_column(metric, oracle=True)
                    if column not in default_frame:
                        continue
                    folds = []
                    for _, fold_rows in default_frame.groupby("fold", sort=True):
                        ordered = fold_rows.sort_values("iter", kind="stable") if "iter" in fold_rows else fold_rows
                        vals = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
                        folds.append(vals[:max_iterations])
                    if folds:
                        stack = np.full((max_iterations, len(folds)), np.nan)
                        for fold_index, values in enumerate(folds):
                            stack[:len(values), fold_index] = values
                        fold_count = np.isfinite(stack).sum(axis=1)
                        fold_mean = np.divide(
                            np.nansum(stack, axis=1), fold_count,
                            out=np.full(max_iterations, np.nan), where=fold_count > 0
                        )
                        result["scores"][library][metric]["Default"][:, task_index, :] = fold_mean[:, None]
                        result["val_selected"][library][metric]["Default"][:, task_index, :] = fold_mean[:, None]

            for seed_index, seed in enumerate(seeds):
                frame_by_method = {}
                for method in METHODS:
                    path = _file_for(results_root, method, library, seed, task)
                    if method == "SMAC" and library != "gpboost":
                        message = f"SMAC wird nur für gpboost gelesen ({library} übersprungen)."
                        if message not in warnings_list:
                            warnings_list.append(message)
                        continue
                    frame = _read_frame(path, warnings_list)
                    if frame is None:
                        missing_files.append(str(path))
                        continue
                    if method == "hyperband":
                        method_values = frame
                    elif method == "SMAC":
                        method_values = frame
                    else:
                        method_values = frame.loc[frame["method"].astype(str).eq(method)] if "method" in frame else frame.iloc[0:0]
                    frame_by_method[method] = method_values

                for method, frame in frame_by_method.items():
                    for strategy_index, strategy in enumerate(STRATEGIES):
                        cell = _records_for_cell(
                            frame, method, strategy, task, max_iterations,
                            _file_for(results_root, method, library, seed, task), warnings_list
                        )
                        if cell is None:
                            continue
                        result["val_score"][library][method][strategy_index, :, task_index, seed_index] = _fold_mean(cell["val_score"])
                        result["runtime"][library][method][strategy_index, :, task_index, seed_index] = _fold_mean(cell["runtime"])
                        for metric in METRICS:
                            result["scores"][library][metric][method][strategy_index, :, task_index, seed_index] = _fold_mean(cell["scores"][metric])
                            result["val_selected"][library][metric][method][strategy_index, :, task_index, seed_index] = _fold_mean(cell["val_selected"][metric])

        loaded = _create_scores_dict_if_compatible(
            results_root, library, tasks, seeds, max_iterations, warnings_list
        )
        result["create_scores_dict"][library] = loaded
    result["missing_files"] = sorted(set(missing_files))
    result["warnings"] = list(dict.fromkeys(warnings_list))
    return result
