"""CSV and booktabs table generation for the HPO comparison report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from aggregation_methods import elo, improvability
from report_data import METHODS, METRICS, STRATEGIES
from report_plots import _metric_payload, _normalized_payload, normalized_available


PRIMARY = {"regression": "R2", "classification": "Accuracy"}
ERROR_METRIC = {"regression": "RMSE", "classification": "LogLoss"}


def _mean(values, axis=None):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.nansum(values, axis=axis)
    mean = np.divide(total, count, out=np.full(np.shape(total), np.nan, dtype=float), where=count > 0)
    return float(mean) if np.ndim(mean) == 0 else mean


def _domain_task_indices(data, domain):
    return [index for index, task in enumerate(data["tasks"])
            if task["classification"] == (domain == "classification")]


def _available_methods(data, library, metric, task_indices):
    return [method for method in METHODS
            if np.isfinite(data["scores"][library][metric][method][:, :, task_indices, :]).any()]


def _aggregation_dict(data, library, metric, task_indices, methods):
    arrays = {method: data["scores"][library][metric][method][:, :, task_indices, :].copy()
              for method in methods}
    default_with_seeds = data["scores"][library][metric]["Default"][:, task_indices, :]
    default = _mean(default_with_seeds, axis=2)
    return {
        **arrays,
        "Default": default,
        "classification": metric in ("Accuracy", "LogLoss"),
        "rmse": metric == "RMSE",
        "logloss": metric == "LogLoss",
        "normalized": False,
    }


def _rank_scores(data, library, metric, strategy_index, budget_index, task_indices, methods):
    output = {method: [] for method in methods}
    output["Default"] = []
    greater_is_better = metric in ("R2", "Accuracy")
    for task_index in task_indices:
        values = []
        names = []
        for method in methods:
            method_array = data["scores"][library][metric][method]
            if budget_index >= method_array.shape[1]:
                value = np.nan
            else:
                value = _mean(method_array[strategy_index, budget_index, task_index, :])
            names.append(method)
            values.append(value)
        default = _mean(data["scores"][library][metric]["Default"][budget_index, task_index, :])
        names.append("Default")
        values.append(default)
        numeric = np.asarray(values, dtype=float)
        valid = np.isfinite(numeric)
        if np.any(valid):
            ranks = rankdata(-numeric[valid] if greater_is_better else numeric[valid], method="average")
            for name, rank in zip(np.asarray(names)[valid], ranks):
                output[name].append(float(rank))
    return {method: _mean(ranks) for method, ranks in output.items()}


def _metric_value(data, library, metric, method, strategy_index, budget_index, task_indices):
    values = data["scores"][library][metric][method][strategy_index, budget_index, task_indices, :]
    return _mean(values)


def _write_pair(frame, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{stem}.csv", index=False)
    try:
        latex = frame.to_latex(index=False, escape=True, longtable=True,
                               caption=stem.replace("_", " "), label=f"tab:{stem}")
        latex = latex.replace("\\begin{longtable}", "\\begin{longtable}")
        latex = latex.replace("\\toprule", "\\toprule")
        (directory / f"{stem}.tex").write_text(latex)
    except Exception as exc:
        (directory / f"{stem}.tex").write_text(
            "% LaTeX export failed: " + f"{type(exc).__name__}: {exc}\n"
        )


def _overall_tables(data, out_dir, budgets):
    rows = []
    for library in data["libraries"]:
        for domain in ("regression", "classification"):
            task_indices = _domain_task_indices(data, domain)
            if not task_indices:
                continue
            score_metric = PRIMARY[domain]
            error_metric = ERROR_METRIC[domain]
            methods = _available_methods(data, library, score_metric, task_indices)
            payload = _metric_payload(data, library, score_metric, task_indices, methods)
            has_normalization = normalized_available(data, library, domain)
            normalized = _normalized_payload(payload) if has_normalization else None
            if len(task_indices) < 5:
                data["warnings"].append(
                    f"Elo für {library}/{domain} basiert auf nur {len(task_indices)} Tasks (<5): "
                    "Bootstrap-Ratings sind nicht belastbar."
                )
            error_result = _aggregation_dict(data, library, error_metric, task_indices, methods)
            score_result = _aggregation_dict(data, library, score_metric, task_indices, methods)
            improv = improvability(error_result, budgets=budgets)
            try:
                ratings = elo(error_result, budgets=budgets)
            except ValueError:
                ratings = {
                    name: {strategy: np.full(len(budgets), np.nan)
                          for strategy in STRATEGIES}
                    for name in ["Default", *methods]
                }
                data["warnings"].append(
                    f"Elo nicht bestimmbar für {library}/{domain}: "
                    "keine vergleichbaren Entrants an mindestens einem Budget."
                )
            accuracy_improv = None
            accuracy_elo = None
            if domain == "classification":
                accuracy_result = _aggregation_dict(data, library, "Accuracy", task_indices, methods)
                accuracy_result["classification"] = True
                accuracy_improv = improvability(accuracy_result, budgets=budgets)
                try:
                    accuracy_elo = elo(accuracy_result, budgets=budgets)
                except ValueError:
                    accuracy_elo = {}

            for strategy_index, strategy in enumerate(STRATEGIES):
                for budget_position, budget in enumerate(budgets):
                    budget_index = budget - 1
                    for method in methods:
                        if has_normalization and budget_index < normalized[method].shape[1]:
                            norm_value = _mean(normalized[method][strategy_index, budget_index])
                        else:
                            norm_value = np.nan
                        error_value = (improv.get(method, {}).get(strategy, np.full(len(budgets), np.nan))
                                       [budget_position])
                        elo_value = (ratings.get(method, {}).get(strategy, np.full(len(budgets), np.nan))
                                     [budget_position])
                        ranks = _rank_scores(data, library, score_metric, strategy_index,
                                             budget_index, task_indices, methods)
                        rows.append({
                            "library": library, "domain": domain, "strategy": strategy,
                            "method": method, "budget": budget, "normalized_score": norm_value,
                            "improvability": error_value, "elo": elo_value,
                            "mean_task_rank": ranks.get(method, np.nan),
                        })
                    rows.append({
                        "library": library, "domain": domain, "strategy": strategy,
                        "method": "Default", "budget": budget,
                        "normalized_score": (_mean(normalized["Default"][budget_index, :])
                                             if has_normalization else np.nan),
                        "improvability": improv["Default"][strategy][budget_position],
                        "elo": ratings["Default"][strategy][budget_position],
                        "mean_task_rank": _rank_scores(
                            data, library, score_metric, strategy_index, budget_index,
                            task_indices, methods
                        ).get("Default", np.nan),
                    })
                    if domain == "classification":
                        for method in methods:
                            rows.append({
                                "library": library, "domain": domain, "strategy": strategy,
                                "method": method, "budget": budget,
                                "normalized_score": np.nan, "improvability_1_minus_accuracy":
                                accuracy_improv.get(method, {}).get(strategy, np.full(len(budgets), np.nan))[budget_position],
                                "elo_1_minus_accuracy": accuracy_elo.get(method, {}).get(
                                    strategy, np.full(len(budgets), np.nan))[budget_position],
                                "mean_task_rank": np.nan,
                            })
    return pd.DataFrame(rows)


def _per_task_table(data, out_dir, budgets):
    end_index = budgets[-1] - 1
    rows = []
    for library in data["libraries"]:
        for task_index, task in enumerate(data["tasks"]):
            domain = "classification" if task["classification"] else "regression"
            metric = PRIMARY[domain]
            methods = _available_methods(data, library, metric, [task_index])
            errors_metric = ERROR_METRIC[domain]
            errors = _aggregation_dict(data, library, errors_metric, [task_index], methods)
            improv = improvability(errors, budgets=[budgets[-1]])
            for strategy_index, strategy in enumerate(STRATEGIES):
                values = {}
                for method in methods:
                    values[method] = _metric_value(data, library, metric, method,
                                                   strategy_index, end_index, [task_index])
                default = _mean(data["scores"][library][metric]["Default"][end_index, task_index, :])
                rank_values = list(values.values()) + [default]
                rank_names = list(values) + ["Default"]
                greater = metric in ("R2", "Accuracy")
                numeric = np.asarray(rank_values, dtype=float)
                valid = np.isfinite(numeric)
                ranks = {}
                if np.any(valid):
                    current = rankdata(-numeric[valid] if greater else numeric[valid], method="average")
                    ranks = dict(zip(np.asarray(rank_names)[valid], current))
                for method in methods:
                    rows.append({
                        "library": library, "suite_id": task["suite_id"],
                        "task_id": task["task_id"], "task": task["name"],
                        "domain": domain, "strategy": strategy, "method": method,
                        "budget": budgets[-1], "raw_score": values[method],
                        "improvability": improv.get(method, {}).get(strategy, [np.nan])[0],
                        "rank": ranks.get(method, np.nan),
                    })
    return pd.DataFrame(rows)


def _library_comparison(data, budgets):
    budget_index = budgets[-1] - 1
    rows = []
    for domain in ("regression", "classification"):
        task_indices = _domain_task_indices(data, domain)
        if not task_indices:
            continue
        metric = PRIMARY[domain]
        for strategy_index, strategy in enumerate(STRATEGIES):
            for method in METHODS:
                per_library_errors = {}
                raw_scores = {}
                for library in data["libraries"]:
                    if not np.isfinite(data["scores"][library][metric][method]).any():
                        continue
                    array = data["scores"][library][metric][method]
                    vals = array[strategy_index, budget_index, task_indices, :]
                    raw_scores[library] = _mean(vals)
                    error = 1.0 - vals if metric in ("R2", "Accuracy") else vals
                    per_library_errors[library] = _mean(error)
                if not raw_scores:
                    continue
                best_error = min(per_library_errors.values())
                for library, score in raw_scores.items():
                    error = per_library_errors[library]
                    relative_improvement = 0.0 if error == 0 and best_error == 0 else (
                        1 - best_error / error if error else np.nan
                    )
                    rows.append({
                        "domain": domain, "strategy": strategy, "method": method,
                        "library": library, "budget": budgets[-1], "best_score": score,
                        "improvability_across_libraries": relative_improvement,
                    })
    return pd.DataFrame(rows)


def _runtime_table(data):
    rows = []
    for library in data["libraries"]:
        for task_index, task in enumerate(data["tasks"]):
            for strategy_index, strategy in enumerate(STRATEGIES):
                for method in METHODS:
                    array = data["runtime"][library].get(method)
                    if array is None:
                        continue
                    values = array[strategy_index, :, task_index, :]
                    totals = []
                    for seed_index in range(values.shape[1]):
                        finite = values[:, seed_index][np.isfinite(values[:, seed_index])]
                        if finite.size:
                            totals.append(float(finite[-1]))
                    if totals:
                        rows.append({
                            "library": library, "task": task["name"],
                            "suite_id": task["suite_id"], "task_id": task["task_id"],
                            "strategy": strategy, "method": method,
                            "runtime_median_seconds": float(np.median(totals)),
                            "runtime_sum_seconds": float(np.sum(totals)),
                            "seed_runs": len(totals),
                        })
    return pd.DataFrame(rows)


def make_all_tables(data, out_dir, budgets):
    directory = Path(out_dir) / "tables"
    _write_pair(_overall_tables(data, directory, budgets), directory, "overall_metrics")
    _write_pair(_per_task_table(data, directory, budgets), directory, "per_dataset_end_budget")
    _write_pair(_library_comparison(data, budgets), directory, "library_comparison")
    _write_pair(_runtime_table(data), directory, "runtime_summary")
    return directory
