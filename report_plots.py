"""Plot routines for score, library and runtime comparisons."""

from __future__ import annotations

import copy
import io
import re
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils_plot_maker as upm
from report_data import METHODS, METRICS, STRATEGIES


METHOD_LABELS = {
    "grid_search": "Grid search", "random_search": "Random search", "tpe": "TPE",
    "gp_bo": "GP-BO", "hyperband": "Hyperband", "SMAC": "SMAC", "Default": "Default",
}
COLORS = {
    "grid_search": "#2f6690", "random_search": "#f28e2b", "tpe": "#3a923a",
    "gp_bo": "#c23b52", "hyperband": "#7653a6", "SMAC": "#2a9d8f",
    "Default": "#343a40", "gpboost": "#2f6690", "xgboost": "#e07a2d",
    "catboost": "#39845a",
}
STRATEGY_LABELS = {"num_leaves": "Number of leaves", "max_depth": "Maximum depth",
                   "joint": "Joint", "num_iter": "Number of iterations"}
METRIC_LABELS = {"R2": "R²", "RMSE": "RMSE", "Accuracy": "Accuracy", "LogLoss": "Log loss"}


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("_")


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _mean(values, axis):
    values = np.asarray(values, dtype=float)
    valid_count = np.sum(np.isfinite(values), axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, valid_count, out=np.full_like(total, np.nan, dtype=float), where=valid_count > 0)


@contextmanager
def _plot_globals(task_count, methods):
    names = ("NUM_TASKS", "NUM_REGR_TASKS", "NUM_CLASS_TASKS", "HPO_METHODS", "NUM_METHODS")
    saved = {name: getattr(upm, name) for name in names}
    try:
        upm.NUM_TASKS = task_count
        upm.NUM_REGR_TASKS = task_count
        upm.NUM_CLASS_TASKS = task_count
        upm.HPO_METHODS = list(methods)
        upm.NUM_METHODS = len(methods)
        yield
    finally:
        for name, value in saved.items():
            setattr(upm, name, value)


def _available_methods(data, library, task_indices):
    available = []
    for method in METHODS:
        values = data["scores"][library]["Accuracy" if any(
            data["tasks"][index]["classification"] for index in task_indices
        ) else "R2"][method]
        if np.isfinite(np.take(values, task_indices, axis=2)).any():
            available.append(method)
    return available


def _metric_payload(data, library, metric, task_indices, methods=None, source="scores"):
    method_names = list(methods or METHODS)
    metric_arrays = data[source][library][metric]
    payload = {method: np.take(metric_arrays[method], task_indices, axis=2).copy()
               for method in method_names if method != "Default" and method in metric_arrays}
    default = np.take(metric_arrays["Default"], task_indices, axis=1)
    payload["Default"] = _mean(default, axis=2)
    payload.update({"classification": metric in ("Accuracy", "LogLoss"),
                    "rmse": metric == "RMSE", "logloss": metric == "LogLoss",
                    "normalized": False})
    return payload


def _normalized_payload(payload, percentiles=(10, 90)):
    methods = [method for method in METHODS if method in payload]
    task_count = next((payload[method].shape[2] for method in methods if method in payload),
                      payload["Default"].shape[1])
    with _plot_globals(task_count, methods):
        try:
            with redirect_stdout(io.StringIO()):
                return upm.normalize_scores(copy.deepcopy(payload), percentiles)
        except (TypeError, ValueError, FloatingPointError):
            raw = copy.deepcopy(payload)
            lower_better = raw["rmse"] or raw["logloss"]
            stack = [raw["Default"][None, :, :]] + [raw[name] for name in methods]
            all_values = np.concatenate(stack, axis=0)
            lows = np.nanpercentile(all_values, percentiles[0], axis=(0, 1, 3))
            highs = np.nanpercentile(all_values, percentiles[1], axis=(0, 1, 3))
            width = highs - lows
            with np.errstate(divide="ignore", invalid="ignore"):
                if lower_better:
                    for name in methods:
                        raw[name] = (highs[None, None, :, None] - raw[name]) / width[None, None, :, None]
                    raw["Default"] = (highs[None, :] - raw["Default"]) / width[None, :]
                else:
                    for name in methods:
                        raw[name] = (raw[name] - lows[None, None, :, None]) / width[None, None, :, None]
                    raw["Default"] = (raw["Default"] - lows[None, :]) / width[None, :]
            raw["normalized"] = True
            return raw


def _avg_std(payload, metric, method, strategy_index):
    methods = [name for name in METHODS if name in payload]
    with _plot_globals(payload[method].shape[2], methods):
        try:
            _, std = upm.compute_avg_std(
                {metric: payload}, metric, method, strategy_index, randomness="both"
            )
            return np.asarray(std, dtype=float)
        except (TypeError, ValueError, IndexError, KeyError):
            values = payload[method][strategy_index]
            return np.nanstd(values, axis=(1, 2))


def normalized_available(data, library, domain):
    """True if create_scores_dict succeeded for this library/domain (all files present)."""
    classification = domain == "classification"
    loaded = data.get("create_scores_dict", {}).get(library, {})
    return (classification, "accuracy" if classification else "score") in loaded


def _domain_groups(data):
    return {
        "regression": [i for i, task in enumerate(data["tasks"]) if not task["classification"]],
        "classification": [i for i, task in enumerate(data["tasks"]) if task["classification"]],
    }


def _x_positions(method, length, iterations):
    if method == "hyperband" and length > 1:
        return np.linspace(1, iterations, length)
    return np.arange(1, length + 1)


def _plot_overall(data, library, domain, metric, strategy, task_indices, methods,
                  out_dir, language, normalize=True):
    payload = _metric_payload(data, library, metric, task_indices, methods)
    normalized = _normalized_payload(payload) if normalize else payload
    strategy_index = STRATEGIES.index(strategy)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for method in methods:
        if method == "Default":
            continue
        values = normalized[method][strategy_index]
        mean = _mean(values, axis=(1, 2))
        valid = np.isfinite(mean)
        if not valid.any():
            continue
        x = _x_positions(method, len(mean), data["iterations"])
        color = COLORS[method]
        ax.plot(x[valid], mean[valid], label=METHOD_LABELS[method], color=color, linewidth=2)
        if not normalize:
            # Band: +-1 std across seeds of the task-averaged curve (only with >1 seed).
            per_seed = _mean(values, axis=1)
            if np.isfinite(per_seed).any(axis=0).sum() > 1:
                with np.errstate(invalid="ignore"):
                    spread = np.nan_to_num(np.nanstd(per_seed, axis=1))
                ax.fill_between(x[valid], (mean - spread)[valid], (mean + spread)[valid],
                                color=color, alpha=0.13)
        else:
            std = _avg_std(normalized, metric, method, strategy_index)
            count = np.sum(np.isfinite(values), axis=(1, 2))
            margin = np.divide(2.0 * std, np.sqrt(count), out=np.zeros_like(std), where=count > 0)
            ax.fill_between(x[valid], (mean - margin)[valid], (mean + margin)[valid], color=color, alpha=0.13)
    if "Default" in normalized:
        default = _mean(normalized["Default"][:, :, None], axis=(1, 2))
        if np.isfinite(default).any():
            ax.plot(np.arange(1, len(default) + 1), default, "--", color=COLORS["Default"],
                    label=METHOD_LABELS["Default"], linewidth=1.8)
    xlabel = "Number of trials" if language == "en" else "Anzahl der Durchläufe"
    if normalize:
        ylabel = "Normalized score" if language == "en" else "Normalisierter Score"
    else:
        ylabel = (f"{METRIC_LABELS[metric]} (not normalized)" if language == "en"
                  else f"{METRIC_LABELS[metric]} (nicht normalisiert)")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{library} | {domain} | {STRATEGY_LABELS[strategy]} | {METRIC_LABELS[metric]}")
    ax.set_xlim(1, data["iterations"])
    if normalize:
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, color="#d8dde2", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, Path(out_dir) / f"overall_{library}_{domain}_{strategy}_{metric}")


def _plot_per_task(data, library, task_index, metric, strategy, methods, out_dir, language):
    task = data["tasks"][task_index]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    strategy_index = STRATEGIES.index(strategy)
    for method in methods:
        if method == "Default":
            values = data["scores"][library][metric]["Default"][:, task_index, :]
            line = _mean(values, axis=1)
            x = np.arange(1, len(line) + 1)
        else:
            values = data["scores"][library][metric][method][strategy_index, :, task_index, :]
            line = _mean(values, axis=1)
            valid_length = np.flatnonzero(np.isfinite(line))
            if not len(valid_length):
                continue
            x = _x_positions(method, len(line), data["iterations"])
        valid = np.isfinite(line)
        if valid.any():
            ax.plot(x[valid], line[valid], color=COLORS[method], label=METHOD_LABELS[method], linewidth=1.8)
    ax.set(xlabel="Number of trials" if language == "en" else "Anzahl der Durchläufe",
           ylabel=METRIC_LABELS[metric], title=f"{task['name']} | {library} | {STRATEGY_LABELS[strategy]}")
    ax.grid(True, color="#d8dde2", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, Path(out_dir) / "per_dataset" / f"{_slug(task['name'])}_{library}_{strategy}_{metric}")


def _plot_library_methods(data, domain, metric, strategy, task_indices, out_dir, language):
    strategy_index = STRATEGIES.index(strategy)
    libraries = data["libraries"]
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        found = False
        for library in libraries:
            values = data["scores"][library][metric].get(method)
            if values is None:
                continue
            values = np.take(values[strategy_index], task_indices, axis=1)
            mean = _mean(values, axis=(1, 2))
            valid = np.isfinite(mean)
            if not valid.any():
                continue
            found = True
            ax.plot(_x_positions(method, len(mean), data["iterations"])[valid], mean[valid],
                    label=library, color=COLORS[library], linewidth=2)
        if not found:
            plt.close(fig)
            continue
        ax.set(xlabel="Number of trials" if language == "en" else "Anzahl der Durchläufe",
               ylabel=METRIC_LABELS[metric], title=f"{domain} | {STRATEGY_LABELS[strategy]} | {METHOD_LABELS[method]}")
        ax.grid(True, color="#d8dde2", linestyle="--", linewidth=0.6)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, Path(out_dir) / f"library_{domain}_{strategy}_{metric}_{method}")


def _plot_best_library(data, domain, metric, strategy, task_indices, out_dir, language):
    strategy_index = STRATEGIES.index(strategy)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for library in data["libraries"]:
        methods = [method for method in METHODS if method in data["scores"][library][metric]]
        stack = np.stack([
            np.take(data["scores"][library][metric][method][strategy_index], task_indices, axis=1)
            for method in methods
        ])
        best_by_task_seed = np.nanmax(stack, axis=0) if metric in ("R2", "Accuracy") else np.nanmin(stack, axis=0)
        mean = _mean(best_by_task_seed, axis=(1, 2))
        valid = np.isfinite(mean)
        if valid.any():
            ax.plot(np.arange(1, len(mean) + 1)[valid], mean[valid], label=library,
                    color=COLORS[library], linewidth=2)
    ax.set(xlabel="Number of trials" if language == "en" else "Anzahl der Durchläufe",
           ylabel=METRIC_LABELS[metric], title=f"Best method by library | {domain} | {STRATEGY_LABELS[strategy]}")
    ax.grid(True, color="#d8dde2", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, Path(out_dir) / f"best_method_by_library_{domain}_{strategy}_{metric}")


def _step_curve(times, scores, grid, best_so_far=True):
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(times) & np.isfinite(scores) & (times > 0)
    times, scores = times[valid], scores[valid]
    if not len(times):
        return np.full(len(grid), np.nan)
    order = np.argsort(times, kind="stable")
    times, scores = times[order], scores[order]
    best = np.maximum.accumulate(scores) if best_so_far else scores
    indices = np.searchsorted(times, grid, side="right") - 1
    curve = np.full(len(grid), np.nan)
    available = indices >= 0
    curve[available] = best[indices[available]]
    return curve


def runtime_step_curves(times, scores, reference_time, grid, selection="oracle"):
    """Public hand-testable helper for cumulative best-score runtime curves."""
    times = np.asarray(times, dtype=float)
    if reference_time <= 0:
        return np.full(len(grid), np.nan)
    if selection not in ("oracle", "val"):
        raise ValueError("selection must be 'oracle' or 'val'")
    return _step_curve(times / reference_time, scores, grid, best_so_far=selection == "oracle")


def _runtime_grid(data, library, domain_indices, strategy_index):
    random_runtime = data["runtime"][library].get("random_search")
    if random_runtime is None:
        return np.logspace(-3, 0, 100)
    reference = np.take(random_runtime[strategy_index], domain_indices, axis=1)
    reference_totals = np.nanmax(reference, axis=0)
    ratios = []
    for method in METHODS:
        values = data["runtime"][library].get(method)
        if values is None:
            continue
        selected = np.take(values[strategy_index], domain_indices, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = selected / reference_totals[None, :, :]
        positive = scaled[np.isfinite(scaled) & (scaled > 0)]
        if positive.size:
            ratios.extend(positive.tolist())
    if not ratios:
        return np.logspace(-3, 0, 100)
    return np.logspace(np.log10(max(min(ratios), 1e-5)), np.log10(max(max(ratios), 1.0)), 100)


def _runtime_scores(data, library, metric, selection):
    source = "scores" if selection == "oracle" else "val_selected"
    return data[source][library][metric]


def _plot_runtime(data, library, task_indices, strategy, out_dir, language, selection,
                  normalize=True):
    metric = "Accuracy"
    method_names = [method for method in METHODS if method in data["runtime"][library]]
    strategy_index = STRATEGIES.index(strategy)
    grid = _runtime_grid(data, library, task_indices, strategy_index)
    scores_by_method = _runtime_scores(data, library, metric, selection)
    score_payload = _metric_payload(
        data, library, metric, task_indices,
        [method for method in method_names if method != "Default"],
        source="scores" if selection == "oracle" else "val_selected",
    )
    normalized_scores = _normalized_payload(score_payload) if normalize else score_payload
    reference = scores_by_method.get("random_search")
    if reference is None:
        data["warnings"].append(f"Laufzeitplot übersprungen für {library}: Random Search fehlt.")
        return
    lines = {}
    for method in method_names:
        time_array = np.take(data["runtime"][library][method][strategy_index], task_indices, axis=1)
        score_array = normalized_scores[method][strategy_index]
        curves = []
        for task_pos in range(time_array.shape[1]):
            for seed_pos in range(time_array.shape[2]):
                reference_curve = reference[strategy_index, :, task_indices[task_pos], seed_pos]
                finite_reference = reference_curve[np.isfinite(reference_curve)]
                reference_time = finite_reference[-1] if finite_reference.size else np.nan
                curves.append(runtime_step_curves(
                    time_array[:, task_pos, seed_pos], score_array[:, task_pos, seed_pos],
                    reference_time, grid, selection=selection
                ))
        if curves:
            lines[method] = _mean(np.stack(curves), axis=0)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for method, values in lines.items():
        if np.isfinite(values).any():
            ax.plot(grid, values, label=METHOD_LABELS[method], color=COLORS[method], linewidth=2)
    ax.set_xscale("log")
    ax.set(xlabel="Cumulative runtime / Random Search runtime" if language == "en" else "Kumulierte Laufzeit / Random-Search-Laufzeit",
           ylabel=("Normalized accuracy" if language == "en" else "Normalisierte Accuracy") if normalize
           else ("Accuracy (not normalized)" if language == "en" else "Accuracy (nicht normalisiert)"),
           title=f"Accuracy vs runtime | {library} | {STRATEGY_LABELS[strategy]} | selection={selection}")
    ax.grid(True, which="both", color="#d8dde2", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, Path(out_dir) / "runtime" / f"accuracy_{library}_{strategy}_{selection}")

    for task_index in task_indices:
        task = data["tasks"][task_index]
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        for method in method_names:
            time_values = _mean(data["runtime"][library][method][strategy_index, :, task_index, :], axis=1)
            score_values = _mean(normalized_scores[method][strategy_index, :, task_index, :], axis=1)
            valid = np.isfinite(time_values) & np.isfinite(score_values) & (time_values > 0)
            if valid.any():
                ax.plot(time_values[valid], score_values[valid], label=METHOD_LABELS[method],
                        color=COLORS[method], linewidth=1.8)
        ax.set_xscale("log")
        ax.set(xlabel="Cumulative runtime (seconds)" if language == "en" else "Kumulierte Laufzeit (Sekunden)",
               ylabel="Accuracy", title=f"{task['name']} | {library} | selection={selection}")
        ax.grid(True, which="both", color="#d8dde2", linestyle="--", linewidth=0.6)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        _save(
            fig,
            Path(out_dir) / "runtime" / "per_dataset" / f"{_slug(task['name'])}_{library}_{strategy}_{selection}",
        )


def make_all_plots(data, out_dir, language="en", selection="val"):
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    domain_groups = _domain_groups(data)
    primary_metric = {"regression": "R2", "classification": "Accuracy"}
    for library in data["libraries"]:
        for domain, task_indices in domain_groups.items():
            if not task_indices:
                continue
            methods = _available_methods(data, library, task_indices)
            plot_methods = methods + (["Default"] if np.isfinite(np.take(
                data["scores"][library][primary_metric[domain]]["Default"], task_indices, axis=1
            )).any() else [])
            metric = primary_metric[domain]
            normalize = normalized_available(data, library, domain)
            if not normalize:
                data["warnings"].append(
                    f"Plots für {library}/{domain} sind NICHT normalisiert (Rohwerte {METRIC_LABELS[metric]}, "
                    "Mittel über Folds, Tasks; Band = ±1 Std über Seeds bei >1 Seed): "
                    "create_scores_dict nicht möglich (fehlende Dateien oder inkompatibles Layout, siehe Warnung 'create_scores_dict übersprungen')."
                )
            for strategy in STRATEGIES:
                _plot_overall(data, library, domain, metric, strategy, task_indices,
                              plot_methods, plots_dir, language, normalize=normalize)
                for task_index in task_indices:
                    _plot_per_task(data, library, task_index, metric, strategy,
                                   methods + (["Default"] if "Default" in plot_methods else []), plots_dir, language)
                if domain == "classification":
                    _plot_runtime(data, library, task_indices, strategy, plots_dir, language,
                                  selection, normalize=normalize)

    for domain, task_indices in domain_groups.items():
        if not task_indices:
            continue
        metric = primary_metric[domain]
        for strategy in STRATEGIES:
            _plot_library_methods(data, domain, metric, strategy, task_indices, plots_dir, language)
            _plot_best_library(data, domain, metric, strategy, task_indices, plots_dir, language)
    return plots_dir
