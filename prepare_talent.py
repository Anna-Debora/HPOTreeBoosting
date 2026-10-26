"""Laedt die TALENT-Datensaetze (Ye et al., Basis-Benchmark mit 300 Datensaetzen, Stand 2025-01, Google Drive:
dataset-250115.zip), verarbeitet sie wie Grinsztajn et al. (siehe preprocess_tabular.py) und legt sie wie die
OpenML-Aufgaben ab:
    data/{suite}_{task}/{suite}_{task}_X.csv, _y.csv, _categorical_indicator.npy
TALENT liefert pro Datensatz N_{train,val,test}.npy (numerisch), C_{...}.npy (kategorial), y_{...}.npy und
info.json. Train/Val/Test werden zusammengefuegt (die Aufteilung macht unser Code selbst in 5 Folds).
Kategorial = alle Spalten aus C, numerisch = alle Spalten aus N (wie der Kategorie-Indikator bei OpenML).
TALENT hat keine eigenen IDs: task_id = 920000 + laufende Nummer (Regression, Suite 1001) bzw.
930000 + laufende Nummer (bin- und multiclass, Suite 1002), alphabetisch nach Ordnername.

Duplikate (wie bei BCCO): uebersprungen wird, wenn der Name schon in den alten Suiten, in new_tasks.csv
oder unter den registrierten BCCO-Aufgaben vorkommt, wenn die OpenML-Datensatz-ID schon in den alten Suiten
oder in union.csv steckt, oder wenn die Aufgabe in excluded_duplicates.csv steht (dort landen auch die
Treffer des Inhaltsvergleichs, siehe check_content_duplicates.py).
Die TALENT-Eintraege in task_indices werden bei jedem Lauf neu aufgebaut (IDs 920000-939999), damit
spaeter ausgeschlossene Aufgaben auch wieder verschwinden.

Viele .npy-Dateien sind Objekt-Arrays (Pickle). Sie werden mit einem eingeschraenkten Unpickler geladen,
der nur Numpy-Arrays, Zahlen und Texte zulaesst (np.load(allow_pickle=True) koennte beliebigen Code ausfuehren).

Aufruf:  python prepare_talent.py --zip dataset-250115.zip      (entpackt nach talent_raw/)
         python prepare_talent.py --root talent_raw/data          (schon entpackt)
         python prepare_talent.py --root ... --limit 5            (nur die ersten 5 zum Testen)
"""
import argparse
import json
import os
import pickle
import re
import zipfile

import numpy as np
import pandas as pd

from preprocess_tabular import preprocess, selection_problem
from task_registry import register_tasks, unregister_range

REGRESSION_SUITE, CLASSIFICATION_SUITE = 1001, 1002
FIRST_ID = {REGRESSION_SUITE: 920000, CLASSIFICATION_SUITE: 930000}
ID_RANGE = (920000, 940000)
ARTIFICIAL = {"2dplanes"}
MISSING_STRINGS = {"", "nan", "NaN", "NAN", "None", "none", "?", "NA", "N/A", "null"}
GDRIVE_FILE_ID = "1Rr2uZsZzabshoa0KX0uCne-qMWFpbyAo"   # dataset-250115.zip im TALENT-Drive-Ordner


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("builtins", "str"), ("builtins", "int"), ("builtins", "float"), ("builtins", "bool"),
        ("builtins", "complex"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"nicht erlaubt in .npy: {module}.{name}")


def load_npy(path):
    """Wie np.load, Objekt-Arrays aber nur mit eingeschraenktem Unpickler."""
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            np.lib.format._check_version(version)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            array = _SafeUnpickler(f).load()
        return np.asarray(array).reshape(shape, order="F" if fortran else "C")


def read_dataset(folder):
    """Gibt (X, y, kategorial-Indikator, info) mit allen Zeilen aus train/val/test zurueck."""
    info = json.load(open(os.path.join(folder, "info.json")))
    parts = {}
    for kind in ("N", "C", "y"):
        arrays = [load_npy(os.path.join(folder, f"{kind}_{split}.npy")) for split in ("train", "val", "test")
                  if os.path.exists(os.path.join(folder, f"{kind}_{split}.npy"))]
        if arrays:
            parts[kind] = np.concatenate([a.reshape(len(a), -1) if kind != "y" else a.ravel() for a in arrays])
    frames, indicator = [], []
    if "N" in parts:
        names = list(info.get("num_feature_intro") or [])
        if len(names) != parts["N"].shape[1]:
            names = [f"N{i}" for i in range(parts["N"].shape[1])]
        frames.append(pd.DataFrame(parts["N"].astype(float), columns=names))
        indicator += [False] * len(names)
    if "C" in parts:
        names = list(info.get("cat_feature_intro") or [])
        if len(names) != parts["C"].shape[1]:
            names = [f"C{i}" for i in range(parts["C"].shape[1])]
        cat = pd.DataFrame(parts["C"].astype(object), columns=names)
        cat = cat.apply(lambda col: col.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                                            or str(v).strip() in MISSING_STRINGS else str(v)))
        frames.append(cat)
        indicator += [True] * len(names)
    X = pd.concat(frames, axis=1)
    X.columns = _unique(X.columns)
    y = pd.Series(parts["y"])
    if info.get("task_type") == "regression":
        y = y.astype(float)
    else:
        y = y.map(lambda v: np.nan if str(v).strip() in MISSING_STRINGS else str(v))
    return X, y, indicator, info


def _unique(columns):
    seen, out = {}, []
    for c in map(str, columns):
        seen[c] = seen.get(c, -1) + 1
        out.append(c if seen[c] == 0 else f"{c}_{seen[c]}")
    return out


def openml_id(info):
    if info.get("openml_id") is not None:
        return int(info["openml_id"])
    match = re.search(r"openml\.org.*[?&]id=(\d+)", str(info.get("source", "")))
    return int(match.group(1)) if match else None


def known_names():
    """Normierte Namen aus alten Suiten, new_tasks.csv und registrierten BCCO-Aufgaben."""
    names = set()
    for suite in (334, 335, 336, 337):
        path = f"task_indices/{suite}_task_names.json"
        if os.path.exists(path):
            names |= {norm(n) for n in json.load(open(path))}
    if os.path.exists("new_tasks.csv"):
        names |= {norm(n) for n in pd.read_csv("new_tasks.csv")["name"]}
    if os.path.exists("preprocessing_report_bcco.csv"):
        bcco = pd.read_csv("preprocessing_report_bcco.csv")
        names |= {norm(n) for n in bcco.loc[bcco["status"] == "ok", "name"]}
    return names


def known_openml_ids():
    ids = set()
    if os.path.exists("old_tasks_info.csv"):
        ids |= set(pd.read_csv("old_tasks_info.csv")["dataset_id"].astype(int))
    if os.path.exists("union.csv"):
        union = pd.read_csv("union.csv")
        ids |= {int(v) for v in union["dataset_id"] if str(v).isdigit()}
    return ids


def excluded():
    """(suite_id, task_id) und (suite_id, normierter Name) aus excluded_duplicates.csv."""
    if not os.path.exists("excluded_duplicates.csv"):
        return set(), set()
    table = pd.read_csv("excluded_duplicates.csv")
    return ({(int(s), int(t)) for s, t in zip(table["suite_id"], table["task_id"])},
            {(int(s), norm(n)) for s, n in zip(table["suite_id"], table["name"])})


def task_list(root):
    """Alle TALENT-Datensaetze mit vergebener (suite, task_id), alphabetisch nach Ordnername."""
    entries = []
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if os.path.isfile(os.path.join(path, "info.json")):
            info = json.load(open(os.path.join(path, "info.json")))
            suite = REGRESSION_SUITE if info.get("task_type") == "regression" else CLASSIFICATION_SUITE
            entries.append([suite, folder, info])
    counters = dict(FIRST_ID)
    for entry in entries:
        entry.insert(1, counters[entry[0]])
        counters[entry[0]] += 1
    return entries   # [suite, task_id, folder, info]


def extract(zip_path, target="talent_raw"):
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.startswith("/") or ".." in member.split("/"):
                raise ValueError(f"verdaechtiger Pfad im Archiv: {member}")
        archive.extractall(target)
    return os.path.join(target, "data")


def main(root, limit=None, balance=True):
    skip_names, skip_ids = known_names(), known_openml_ids()
    excluded_ids, excluded_names = excluded()
    report, registered = [], {REGRESSION_SUITE: [], CLASSIFICATION_SUITE: []}
    entries = task_list(root)
    if limit:
        entries = entries[:limit]
    for suite, task_id, folder, meta in entries:
        name = meta.get("name") or folder
        print(f"[{suite}_{task_id}] {folder} ...", flush=True)
        info = {"suite_id": suite, "task_id": task_id, "name": folder, "talent_task_type": meta.get("task_type"),
                "openml_id": openml_id(meta), "source": meta.get("source")}
        try:
            names = {norm(folder), norm(name)}
            if names & skip_names:
                info["status"] = "UEBERSPRUNGEN: Name schon vorhanden (Duplikat)"
            elif info["openml_id"] in skip_ids:
                info["status"] = "UEBERSPRUNGEN: OpenML-Datensatz schon vorhanden (Duplikat)"
            elif (suite, task_id) in excluded_ids or any((suite, n) in excluded_names for n in names):
                info["status"] = "UEBERSPRUNGEN: in excluded_duplicates.csv ausgeschlossen"
            elif names & {norm(a) for a in ARTIFICIAL}:
                info["status"] = "UEBERSPRUNGEN: kuenstlich erzeugte Daten"
            else:
                frame, y, categorical, _ = read_dataset(os.path.join(root, folder))
                X, y, indicator, details = preprocess(frame, y, categorical, classification=(suite == CLASSIFICATION_SUITE),
                                                      balance=balance)
                info.update(details)
                problem = "nach der Bereinigung bleibt nichts uebrig" if (X.shape[1] == 0 or len(X) == 0) \
                    else selection_problem(details)
                if problem:
                    info["status"] = f"AUSGESCHIEDEN: {problem}"
                else:
                    path = f"data/{suite}_{task_id}"
                    os.makedirs(path, exist_ok=True)
                    X.to_csv(f"{path}/{suite}_{task_id}_X.csv", index=False)
                    y.rename("target").to_csv(f"{path}/{suite}_{task_id}_y.csv", index=False)
                    np.save(f"{path}/{suite}_{task_id}_categorical_indicator.npy", indicator)
                    info["status"] = "ok"
                    registered[suite].append((task_id, folder))
        except Exception as error:
            info["status"] = f"FEHLER: {error}"
        print("   ", info.get("status"), {k: info[k] for k in ("rows_before", "rows_after", "cols_after") if k in info},
              flush=True)
        report.append(info)
    pd.DataFrame(report).to_csv("preprocessing_report_talent.csv", index=False)
    for suite, tasks in registered.items():
        unregister_range(suite, *ID_RANGE)
        if tasks:
            register_tasks(suite, tasks)
    ok = sum(len(v) for v in registered.values())
    print(f"\nfertig: {ok} TALENT-Aufgaben vorbereitet (Regression {len(registered[REGRESSION_SUITE])}, "
          f"Klassifikation {len(registered[CLASSIFICATION_SUITE])}); Bericht: preprocessing_report_talent.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", help="dataset-250115.zip (wird nach talent_raw/ entpackt)")
    source.add_argument("--root", help="Ordner mit den entpackten TALENT-Datensaetzen (je ein Unterordner)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()
    main(extract(args.zip) if args.zip else args.root, args.limit, balance=not args.no_balance)
