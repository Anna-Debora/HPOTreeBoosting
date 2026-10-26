"""Traegt vorbereitete Aufgaben in task_indices/{suite}_task_indices.npy und _task_names.json ein.
Bestehende Eintraege bleiben erhalten; gleiche task_id wird ersetzt (so koennen mehrere Skripte
nacheinander Aufgaben zu denselben Pseudo-Suiten 1001/1002 hinzufuegen)."""
import json
import os

import numpy as np


def register_tasks(suite, tasks, directory="task_indices"):
    """tasks: Liste von (task_id, name)."""
    os.makedirs(directory, exist_ok=True)
    ids_file = os.path.join(directory, f"{suite}_task_indices.npy")
    names_file = os.path.join(directory, f"{suite}_task_names.json")
    entries = {}
    if os.path.exists(ids_file) and os.path.exists(names_file):
        old_ids = np.load(ids_file).tolist()
        with open(names_file) as f:
            old_names = json.load(f)
        entries = dict(zip(old_ids, old_names))
    entries.update({int(t): n for t, n in tasks})
    ids = sorted(entries)
    np.save(ids_file, np.array(ids, dtype=int))
    with open(names_file, "w") as f:
        json.dump([entries[i] for i in ids], f)
    return len(ids)


def unregister_range(suite, first, stop, directory="task_indices"):
    """Entfernt alle Eintraege mit first <= task_id < stop (z. B. um eine Quelle neu aufzubauen)."""
    ids_file = os.path.join(directory, f"{suite}_task_indices.npy")
    names_file = os.path.join(directory, f"{suite}_task_names.json")
    if not (os.path.exists(ids_file) and os.path.exists(names_file)):
        return 0
    with open(names_file) as f:
        entries = dict(zip(np.load(ids_file).tolist(), json.load(f)))
    kept = {t: n for t, n in entries.items() if not first <= t < stop}
    ids = sorted(kept)
    np.save(ids_file, np.array(ids, dtype=int))
    with open(names_file, "w") as f:
        json.dump([kept[i] for i in ids], f)
    return len(entries) - len(kept)
