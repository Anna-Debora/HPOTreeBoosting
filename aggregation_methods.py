"""Aggregationsmetriken fuer HPO-Ergebnisse ueber mehrere Datensaetze."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


_TUNING_STRATEGIES = ("num_leaves", "max_depth", "joint", "num_iter")
_METHOD_TUNING_COUNTS = {"grid_search": 2, "hyperband": 3}
_ELO_SCALE = 400.0
_ELO_INIT_RATING = 1000.0
_ELO_BOOTSTRAP_ROUNDS = 200
_BT_RIDGE = 0.5 / (1e6 * np.log(10.0) ** 2)


def _prepare_results(result_dict):
    """Validiere das Ergebnisformat und liefere Scores samt Metadaten."""
    if "Default" not in result_dict:
        raise KeyError("result_dict muss den Eintrag 'Default' enthalten.")

    default = np.asarray(result_dict["Default"], dtype=float)
    if default.ndim != 2:
        raise ValueError("result_dict['Default'] muss die Form (Iterationen, Tasks) haben.")

    methods = {}
    task_count = default.shape[1]
    seed_count = None
    for method, values in result_dict.items():
        if method in {"Default", "classification", "rmse", "logloss", "normalized"}:
            continue
        array = np.asarray(values, dtype=float)
        if array.ndim != 4:
            continue
        if array.shape[2] != task_count:
            raise ValueError(f"{method}: Task-Achse passt nicht zu 'Default'.")
        if seed_count is None:
            seed_count = array.shape[3]
        elif array.shape[3] != seed_count:
            raise ValueError("Alle HPO-Methoden muessen gleich viele Seeds haben.")

        strategy_count = _METHOD_TUNING_COUNTS.get(method, array.shape[0])
        if strategy_count > array.shape[0]:
            raise ValueError(f"{method}: weniger Tuningstrategien als erwartet.")
        methods[method] = (array, strategy_count)

    if not methods:
        raise ValueError("Keine HPO-Methoden mit vierdimensionalen Ergebnisarrays gefunden.")
    if seed_count is None or seed_count < 1:
        raise ValueError("Die Ergebnisarrays muessen mindestens einen Seed enthalten.")

    return default[0], methods, task_count, seed_count


def _as_errors(values, result_dict):
    """Wandle Scores in Fehler um; kleinere Fehler sind stets besser."""
    if result_dict.get("rmse", False) or result_dict.get("logloss", False):
        return np.asarray(values, dtype=float)
    return 1.0 - np.asarray(values, dtype=float)


def _strategy_entries(default_scores, methods, budget, strategy_index, seed_count, result_dict):
    """Erzeuge Task-mal-Seed-Matrizen fuer Default und eine Strategie."""
    entries = {"Default": np.repeat(default_scores[:, None], seed_count, axis=1)}
    for method, (array, strategy_count) in methods.items():
        if strategy_index >= strategy_count or budget >= array.shape[1]:
            continue
        name = _TUNING_STRATEGIES[strategy_index]
        entries[f"{method}:{name}"] = array[strategy_index, budget, :, :]

    return {
        name: _as_errors(scores, result_dict)
        for name, scores in entries.items()
    }


def _available_strategies(methods):
    return tuple(
        strategy_index
        for strategy_index in range(max(count for _, count in methods.values()))
        if any(strategy_index < count for _, count in methods.values())
    )


def _mean_over_tasks_and_seeds(values):
    """Gleiches Gewicht fuer Tasks; Seeds werden zuerst innerhalb des Tasks gemittelt."""
    seed_counts = np.sum(np.isfinite(values), axis=1)
    task_means = np.divide(
        np.nansum(values, axis=1),
        seed_counts,
        out=np.full(values.shape[0], np.nan),
        where=seed_counts > 0,
    )
    valid = np.isfinite(task_means)
    return float(np.mean(task_means[valid])) if np.any(valid) else float("nan")


def improvability(result_dict):
    """Berechne Improvability je HPO-Methode, Tuningstrategie und Budget.

    Parameters
    ----------
    result_dict : dict
        Metrik-Ergebnisdictionary aus ``create_scores_dict``. HPO-Arrays haben
        die Form ``(Tuningstrategie, Iteration, Task, Seed)``; ``Default`` hat
        die Form ``(Iteration, Task)``. Die Fold-Achse ist dort bereits gemittelt.
        ``rmse`` und ``logloss`` kennzeichnen Fehlermetriken; sonst gelten R2
        und Accuracy als Scores, bei denen groesser besser ist.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Verschachteltes Dictionary ``Methode -> Tuningstrategie -> Werte``.
        Jeder Wertevektor enthaelt einen gleich-task-gewichteten Improvability-
        Wert pro Iterationsbudget. ``Default`` wird als Vergleichseintrag fuer
        jede Strategie mit ausgegeben.

    Formel
    ------
    Pro Task und Seed wird fuer jede Methode der Fehler ``err`` bestimmt. Bei
    RMSE/Log Loss ist das der Rohwert; bei R2/Accuracy wird ``err = 1 - Score``
    verwendet. Mit dem kleinsten beobachteten Fehler ``bestErr`` lautet der
    Wert ``1 - bestErr / err``. Er ist 0 fuer die beste Methode und wird ueber
    Seeds innerhalb jedes Tasks sowie anschliessend gleichgewichtet ueber Tasks
    gemittelt. Der undefinierte Fall ``err = bestErr = 0`` wird als 0 behandelt.

    Abweichung: TabArena beschreibt Improvability fuer Fehlermetriken. Die
    Score-zu-Fehler-Abbildung ``1 - Score`` erweitert die Definition hier auf
    R2 und Accuracy. Anders als bei TabArena werden Ergebnisse pro
    Tuningstrategie und Iterationsbudget statt nur fuer einen Leaderboard-
    Endpunkt berechnet.
    """
    default_scores, methods, _, seed_count = _prepare_results(result_dict)
    max_budget = max(array.shape[1] for array, _ in methods.values())
    output = {"Default": {}}
    for method, (_, strategy_count) in methods.items():
        output[method] = {
            _TUNING_STRATEGIES[index]: np.full(max_budget, np.nan)
            for index in range(strategy_count)
        }

    for strategy_index in _available_strategies(methods):
        strategy_name = _TUNING_STRATEGIES[strategy_index]
        budget_count = max(
            array.shape[1]
            for array, count in methods.values()
            if strategy_index < count
        )
        output["Default"][strategy_name] = np.full(budget_count, np.nan)
        for method, (_, strategy_count) in methods.items():
            if strategy_index < strategy_count:
                output[method][strategy_name] = np.full(budget_count, np.nan)

        for budget in range(budget_count):
            entries = _strategy_entries(
                default_scores, methods, budget, strategy_index, seed_count, result_dict
            )
            names = list(entries)
            errors = np.stack([entries[name] for name in names])
            available = np.isfinite(errors)
            best_error = np.min(np.where(available, errors, np.inf), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = 1.0 - best_error[None, :, :] / errors
            scores = np.where((errors == 0.0) & (best_error[None, :, :] == 0.0), 0.0, scores)
            scores = np.where(available, np.clip(scores, 0.0, 1.0), np.nan)

            for entrant_index, name in enumerate(names):
                method_name, _, tuning_name = name.partition(":")
                output[method_name][tuning_name or strategy_name][budget] = (
                    _mean_over_tasks_and_seeds(scores[entrant_index])
                )

    return output


def _fit_bradley_terry(wins, scale, init_rating, anchor_index):
    """Schaetze Bradley-Terry-Ratings aus symmetrischen Paar-Winzahlen."""
    entrant_count = wins.shape[0]
    if entrant_count < 2:
        raise ValueError("Elo benoetigt mindestens zwei vergleichbare Eintraege.")

    pairs = [(left, right) for left in range(entrant_count) for right in range(left + 1, entrant_count)]
    pair_wins = np.asarray([wins[left, right] for left, right in pairs])
    pair_losses = np.asarray([wins[right, left] for left, right in pairs])
    pair_counts = pair_wins + pair_losses
    if not np.any(pair_counts > 0):
        raise ValueError("Es gibt keine auswertbaren paarweisen Vergleiche.")

    def objective(log_strength):
        loss = _BT_RIDGE * float(log_strength @ log_strength)
        gradient = 2.0 * _BT_RIDGE * log_strength.copy()
        for (left, right), won, lost, count in zip(
            pairs, pair_wins, pair_losses, pair_counts
        ):
            if count == 0:
                continue
            difference = log_strength[left] - log_strength[right]
            log_normalizer = np.logaddexp(log_strength[left], log_strength[right])
            loss += count * log_normalizer - won * log_strength[left] - lost * log_strength[right]
            expected_left = count / (1.0 + np.exp(-difference))
            gradient[left] += expected_left - won
            gradient[right] += (count - expected_left) - lost
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(entrant_count),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12},
    )
    centered = fit.x - np.mean(fit.x)
    ratings = init_rating + scale * centered / np.log(10.0)
    return ratings + (init_rating - ratings[anchor_index])


def _task_win_totals(entries, seed_count):
    """Erzeuge pro Task eine Matrix gewichteter, halb geteilter Siege."""
    names = list(entries)
    entrant_count = len(names)
    first_entry = entries[names[0]]
    task_count = first_entry.shape[0]
    task_wins = np.zeros((task_count, entrant_count, entrant_count), dtype=float)

    for task_index in range(task_count):
        for seed_index in range(seed_count):
            errors = np.asarray([entries[name][task_index, seed_index] for name in names])
            valid_indices = np.flatnonzero(np.isfinite(errors))
            for left_pos, left in enumerate(valid_indices):
                for right in valid_indices[left_pos + 1 :]:
                    weight = 1.0 / seed_count
                    if errors[left] < errors[right]:
                        task_wins[task_index, left, right] += weight
                    elif errors[left] > errors[right]:
                        task_wins[task_index, right, left] += weight
                    else:
                        task_wins[task_index, left, right] += 0.5 * weight
                        task_wins[task_index, right, left] += 0.5 * weight
    return names, task_wins


def elo(result_dict, bootstrap_rounds=_ELO_BOOTSTRAP_ROUNDS, random_state=0, scale=_ELO_SCALE):
    """Berechne Elo je HPO-Methode, Tuningstrategie und Iterationsbudget.

    Parameters
    ----------
    result_dict : dict
        Ergebnisdictionary aus ``create_scores_dict``. HPO-Arrays haben die
        Form ``(Tuningstrategie, Iteration, Task, Seed)``; ``Default`` hat die
        Form ``(Iteration, Task)`` und wird fuer alle Seeds wiederverwendet.
        Die Fold-Achse ist bereits gemittelt.
    bootstrap_rounds : int, default=200
        Anzahl Task-Bootstrap-Stichproben. Jede Stichprobe zieht die Taskzahl
        mit Zuruecklegen und behaelt alle zugehoerigen Seed-Vergleiche.
    random_state : int, default=0
        Seed fuer reproduzierbares Task-Bootstrapping.
    scale : float, default=400
        Elo-Skalierung. Eine Differenz von ``scale`` Punkten entspricht einer
        erwarteten Gewinnchance von 10:1.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Verschachteltes Dictionary ``Methode -> Tuningstrategie -> Werte``.
        Jeder Vektor enthaelt den Median der Bootstrap-Elo-Werte je Budget.
        ``Default`` wird als Anker fuer jede Strategie mit ausgegeben und hat
        an jedem Budget exakt 1000 Elo.

    Formel
    ------
    Pro Task/Seed werden Methoden paarweise verglichen. Der bessere Fehler
    erhaelt einen Sieg, bei exakt gleichen Werten erhalten beide einen halben
    Sieg. Bradley-Terry schaetzt Log-Staerken ``t`` mit
    ``P(i gewinnt j) = 1 / (1 + 10 ** ((t_j - t_i) / scale))``. Ratings werden
    auf Mittelwert 1000 zentriert und danach so verschoben, dass ``Default``
    genau 1000 betraegt.

    Abweichungen zu TabArena: Der Default-Lauf ersetzt den Default-Random-Forest
    als Elo-Anker. Das Projekt speichert keinen Random-Forest; seine Werte
    werden stattdessen fuer jeden Seed wiederverwendet. Das Paper nennt 200
    Bootstrap-Stichproben fuer 95%-Intervalle, waehrend Anhang D.2 und der
    aktuelle TabArena-Code 100 nennen. Hier werden 200 (Paper-Haupttext/D.1)
    verwendet; da nur ein Wert verlangt ist, wird der Bootstrap-Median als
    Schaetzung ausgegeben, nicht ein Konfidenzintervall.
    """
    if not isinstance(bootstrap_rounds, int) or bootstrap_rounds < 1:
        raise ValueError("bootstrap_rounds muss eine positive ganze Zahl sein.")
    if scale <= 0:
        raise ValueError("scale muss groesser als 0 sein.")

    default_scores, methods, _, seed_count = _prepare_results(result_dict)
    output = {"Default": {}}
    for method, (_, strategy_count) in methods.items():
        output[method] = {
            _TUNING_STRATEGIES[index]: np.full(
                max(array.shape[1] for array, _ in methods.values()), np.nan
            )
            for index in range(strategy_count)
        }

    rng = np.random.default_rng(random_state)
    for strategy_index in _available_strategies(methods):
        strategy_name = _TUNING_STRATEGIES[strategy_index]
        budget_count = max(
            array.shape[1]
            for array, count in methods.values()
            if strategy_index < count
        )
        output["Default"][strategy_name] = np.full(budget_count, _ELO_INIT_RATING)
        for method, (_, strategy_count) in methods.items():
            if strategy_index < strategy_count:
                output[method][strategy_name] = np.full(budget_count, np.nan)

        for budget in range(budget_count):
            entries = _strategy_entries(
                default_scores, methods, budget, strategy_index, seed_count, result_dict
            )
            names, task_wins = _task_win_totals(entries, seed_count)
            if "Default" not in names:
                raise ValueError("'Default' muss als Elo-Anker vergleichbar sein.")
            anchor_index = names.index("Default")
            bootstrap_ratings = np.empty((bootstrap_rounds, len(names)))
            task_count = task_wins.shape[0]
            for round_index in range(bootstrap_rounds):
                sampled_tasks = rng.integers(0, task_count, size=task_count)
                sampled_wins = np.sum(task_wins[sampled_tasks], axis=0)
                bootstrap_ratings[round_index] = _fit_bradley_terry(
                    sampled_wins, scale, _ELO_INIT_RATING, anchor_index
                )

            median_ratings = np.median(bootstrap_ratings, axis=0)
            for entrant_index, name in enumerate(names):
                method_name, _, tuning_name = name.partition(":")
                output[method_name][tuning_name or strategy_name][budget] = median_ratings[
                    entrant_index
                ]

    return output