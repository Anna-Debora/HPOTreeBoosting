"""Inhaltsvergleich: findet Datensaetze mit gleichem Inhalt, auch wenn Name und Quelle verschieden sind.

Verglichen werden die TALENT-Kandidaten (Status ok in preprocessing_report_talent.csv, Rohdaten vor dem
Ausgleich der Klassen) mit
  - den alten Suiten 334-337 (von OpenML, wie im Paper),
  - den bisherigen neuen Aufgaben in task_indices/1001 und 1002 (OpenML und BCCO, Ordner data/),
  - untereinander.
Vorgehen pro Paar:
 1. Jede Spalte (auch das Ziel) mit mindestens MIN_UNIQUE verschiedenen Zahlenwerten wird auf 5 signifikante
    Stellen gerundet; ihre Wertemenge ist der "Fingerabdruck" (gehasht, bei grossen Mengen eine feste
    Hash-Stichprobe, damit Teilmengen erkennbar bleiben).
 2. Spalte A passt zu Spalte B, wenn mindestens MATCH_SHARE der Werte der einen Spalte in der anderen
    vorkommen (so werden auch Teilstichproben und ausgeglichene Klassen erkannt). Spalten werden eins zu
    eins zugeordnet.
 3. Zeilenvergleich: Anteil der Zeilen (auf bis zu 6 zugeordneten Spalten), die im anderen Datensatz
    genau so vorkommen.
Urteil:
  identisch  = gleicher Aufgabentyp (Regression/Klassifikation), mindestens 3 Spalten zugeordnet, Ziel gleich
               (falls bei beiden pruefbar), mindestens 90 % der Zeilen gleich (Regression: in beide Richtungen;
               Klassifikation: in eine Richtung, weil die Klassen ausgeglichen werden) und entweder mindestens 80 % der pruefbaren
               Spalten beider Datensaetze zugeordnet, oder der Kandidat enthaelt mindestens 90 % der Spalten
               des bestehenden Datensatzes (und ist selbst zu mindestens 60 % zugeordnet: alter Datensatz plus
               Zusatzspalten wie id, wie bei house_sales_reduced)
               -> wird mit --apply in excluded_duplicates.csv eingetragen
  Variante   = mindestens 2 Spalten zugeordnet und mindestens 50 % der Zeilen gleich, aber nicht identisch
               (z. B. Teilmenge der Merkmale oder Zeilen, anderes Ziel, Regression statt Klassifikation)
               -> bleibt drin, steht nur im Bericht
  Name aehnlich  = ein normierter Name ist Anfang des anderen (z. B. pole/pol) -> von Hand pruefen
  nicht pruefbar = weniger als 3 pruefbare Spalten (z. B. nur 0/1-Merkmale) -> von Hand pruefen
Bei zwei identischen TALENT-Kandidaten bleibt der alphabetisch erste.

Aufruf:  python check_content_duplicates.py --talent-root talent_raw/data [--apply]
Ausgabe: content_duplicates_talent.csv

Zweite Betriebsart (--new-since REV): Kandidaten sind alle Aufgaben in task_indices/1001 und 1002, die im
Git-Stand REV noch nicht registriert waren (aus allen Quellen, vorbereitete Daten aus data/). Verglichen wird
mit den alten Suiten, den schon in REV registrierten Aufgaben und untereinander. So werden die Datensaetze
geprueft, die erst durch gelockerte Kriterien dazukommen.
Aufruf:  python check_content_duplicates.py --new-since HEAD [--apply] [--out content_duplicates_neu.csv]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

MIN_UNIQUE = 50
MATCH_SHARE = 0.9
FULL_LIMIT = 20000      # bis zu so vielen verschiedenen Werten wird die ganze Menge gespeichert
SAMPLE_MOD = 16         # sonst nur Hashes mit h % SAMPLE_MOD == 0 (gleiche Auswahl in allen Datensaetzen)
MIN_SAMPLE = 30
MAX_ROWS = 300000
ROW_KEY_COLUMNS = 6


def rounded(values):
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.where(v == 0, 0, np.floor(np.log10(np.abs(v))))
        scale = 10.0 ** (4 - exponent)
        out = np.round(v * scale) / scale
    return out


def fingerprint(column):
    v = rounded(column)
    v = v[~np.isnan(v)]
    if len(v) < 0.9 * len(column):
        return None
    hashes = np.unique(pd.util.hash_array(v))
    if len(hashes) < MIN_UNIQUE:
        return None
    sample = set(hashes[hashes % SAMPLE_MOD == 0].tolist())
    return {"full": set(hashes.tolist()) if len(hashes) <= FULL_LIMIT else None, "sample": sample}


def containment(a, b):
    """Anteil der Werte von a, die in b vorkommen (None = nicht schaetzbar)."""
    if a["full"] is not None and b["full"] is not None:
        return len(a["full"] & b["full"]) / len(a["full"])
    if len(a["sample"]) < MIN_SAMPLE:
        return None
    return len(a["sample"] & b["sample"]) / len(a["sample"])


def profile(name, frame, classification, raw_name):
    if len(frame) > MAX_ROWS:
        frame = frame.sample(n=MAX_ROWS, random_state=0)
    columns = {}
    for col in frame.columns:
        fp = fingerprint(frame[col])
        if fp is not None:
            columns[col] = fp
    union = set().union(*[fp["sample"] for fp in columns.values()]) if columns else set()
    return {"name": name, "frame": frame, "columns": columns, "union": union, "classification": classification,
            "key": norm(raw_name)}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def similar_names(a, b, min_length=4):
    short, long = sorted((a, b), key=len)
    return len(short) >= min_length and long.startswith(short)


def compare(p, q):
    if not p["columns"] or not q["columns"]:
        return None
    small = min(len(p["union"]), len(q["union"]))
    if small >= MIN_SAMPLE and len(p["union"] & q["union"]) / small < 0.2:
        return None                                # schnelle Vorauswahl: fast keine gemeinsamen Werte
    scores = []
    for a, fa in p["columns"].items():
        for b, fb in q["columns"].items():
            values = [c for c in (containment(fa, fb), containment(fb, fa)) if c is not None]
            if values and max(values) >= MATCH_SHARE:
                scores.append((max(values), a, b))
    used_a, used_b, pairs = set(), set(), []
    for score, a, b in sorted(scores, reverse=True):
        if a not in used_a and b not in used_b:
            used_a.add(a), used_b.add(b), pairs.append((a, b))
    if not pairs:
        return None
    key_pairs = pairs[:ROW_KEY_COLUMNS]
    keys_p = _row_keys(p["frame"], [a for a, _ in key_pairs])
    keys_q = _row_keys(q["frame"], [b for _, b in key_pairs])
    rows_p_in_q = float(np.isin(keys_p, keys_q).mean())
    rows_q_in_p = float(np.isin(keys_q, keys_p).mean())
    target = None
    if "__ziel__" in p["columns"] and "__ziel__" in q["columns"]:
        target = ("__ziel__", "__ziel__") in pairs
    return {"ziel_gleich": target, "zugeordnete_spalten": len(pairs), "pruefbare_spalten_a": len(p["columns"]),
            "pruefbare_spalten_b": len(q["columns"]), "anteil_spalten_a": len(pairs) / len(p["columns"]),
            "anteil_spalten_b": len(pairs) / len(q["columns"]), "zeilen_a_in_b": rows_p_in_q,
            "zeilen_b_in_a": rows_q_in_p, "spalten": "; ".join(f"{a}={b}" for a, b in pairs[:8])}


def _row_keys(frame, columns):
    parts = [pd.util.hash_array(rounded(frame[c])) for c in columns]
    key = np.zeros(len(frame), dtype=np.uint64)
    for part in parts:
        key = key * np.uint64(1000003) ^ part
    return key


def verdict(result, same_type=True, classification=False):
    if result is None:
        return "verschieden"
    rows = max(result["zeilen_a_in_b"], result["zeilen_b_in_a"])
    rows_needed = rows if classification else min(result["zeilen_a_in_b"], result["zeilen_b_in_a"])
    a, b = result["anteil_spalten_a"], result["anteil_spalten_b"]   # a = Kandidat, b = bestehend
    if (same_type and result["zugeordnete_spalten"] >= 3 and result["ziel_gleich"] is not False
            and rows_needed >= 0.9 and (min(a, b) >= 0.8 or (b >= 0.9 and a >= 0.6))):
        return "identisch"
    if result["zugeordnete_spalten"] >= 2 and rows >= 0.5:
        return "Variante"
    return "verschieden"


def load_old():
    import openml
    old = pd.read_csv("old_tasks_info.csv")
    profiles = []
    for _, row in old.iterrows():
        dataset = openml.datasets.get_dataset(int(row["dataset_id"]), download_data=True,
                                              download_qualities=False, download_features_meta_data=False)
        X, y, _, _ = dataset.get_data(target=row["target"], dataset_format="dataframe")
        X["__ziel__"] = y
        profiles.append(profile(f"{row['name']} ({row['suite_id']}, {row['task_id']})", X,
                                row["suite_id"] in (334, 337), row["name"]))
        print("alt:", profiles[-1]["name"], len(profiles[-1]["columns"]), flush=True)
    return profiles


def load_registered(first=0, stop=920000):
    profiles = []
    for suite in (1001, 1002):
        ids = np.load(f"task_indices/{suite}_task_indices.npy").tolist()
        names = json.load(open(f"task_indices/{suite}_task_names.json"))
        for task_id, name in zip(ids, names):
            if not first <= task_id < stop:
                continue
            path = f"data/{suite}_{task_id}/{suite}_{task_id}"
            X = pd.read_csv(f"{path}_X.csv")
            X["__ziel__"] = pd.read_csv(f"{path}_y.csv").iloc[:, 0]
            profiles.append(profile(f"{name} ({suite}, {task_id})", X, suite == 1002, name))
    print("bisherige neue Aufgaben:", len(profiles), flush=True)
    return profiles


def load_talent(root):
    from prepare_talent import read_dataset
    report = pd.read_csv("preprocessing_report_talent.csv")
    profiles = []
    for _, row in report[report["status"] == "ok"].sort_values("name").iterrows():
        X, y, _, _ = read_dataset(os.path.join(root, row["name"]))
        X["__ziel__"] = y
        p = profile(f"{row['name']} ({row['suite_id']}, {row['task_id']})", X, row["suite_id"] == 1002, row["name"])
        p.update(suite_id=int(row["suite_id"]), task_id=int(row["task_id"]), folder=row["name"])
        profiles.append(p)
    print("TALENT-Kandidaten:", len(profiles), flush=True)
    return profiles


def registered_in(rev):
    """{(suite, task_id)} aus task_indices im Git-Stand rev."""
    import io
    import subprocess
    ids = set()
    for suite in (1001, 1002):
        raw = subprocess.check_output(["git", "show", f"{rev}:task_indices/{suite}_task_indices.npy"])
        ids |= {(suite, int(t)) for t in np.load(io.BytesIO(raw))}
    return ids


def load_split_by(previous):
    """Registrierte Aufgaben: (schon in previous, neu dazugekommen)."""
    old, new = [], []
    for suite in (1001, 1002):
        ids = np.load(f"task_indices/{suite}_task_indices.npy").tolist()
        names = json.load(open(f"task_indices/{suite}_task_names.json"))
        for task_id, name in zip(ids, names):
            path = f"data/{suite}_{task_id}/{suite}_{task_id}"
            X = pd.read_csv(f"{path}_X.csv")
            X["__ziel__"] = pd.read_csv(f"{path}_y.csv").iloc[:, 0]
            p = profile(f"{name} ({suite}, {task_id})", X, suite == 1002, name)
            p.update(suite_id=suite, task_id=int(task_id), folder=name)
            (old if (suite, task_id) in previous else new).append(p)
    print(f"registriert: {len(old)} schon vorher, {len(new)} neu", flush=True)
    return old, new


def source_of(task_id):
    return "TALENT" if task_id >= 920000 else "BCCO" if task_id >= 900000 else "OpenML"


def main(talent_root, apply=False, out="content_duplicates_talent.csv", new_since=None):
    if new_since:
        before, candidates = load_split_by(registered_in(new_since))
        existing = [("alt", p) for p in load_old()] + [("neu", p) for p in before]
    else:
        candidates = load_talent(talent_root)
        existing = [("alt", p) for p in load_old()] + [("neu", p) for p in load_registered()]
    rows = []
    for i, c in enumerate(candidates):
        base = {"suite_id": c["suite_id"], "task_id": c["task_id"], "talent": c["folder"]}
        if len(c["columns"]) < 3:
            rows.append({**base, "urteil": "nicht pruefbar", "pruefbare_spalten_a": len(c["columns"])})
            print(f"nicht pruefbar  {c['name']}", flush=True)
        others = existing + [("TALENT", d) for d in candidates[:i]]
        for kind, e in others:
            result = compare(c, e)
            v = verdict(result, c["classification"] == e["classification"], c["classification"])
            if v != "verschieden":
                rows.append({**base, "vergleich_mit": e["name"], "art": kind, "urteil": v, **result})
                print(f"{v:10s} {c['name']}  <->  {e['name']}", flush=True)
            if similar_names(c["key"], e["key"]):
                rows.append({**base, "vergleich_mit": e["name"], "art": kind, "urteil": "Name aehnlich"})
                print(f"Name aehnlich  {c['name']}  <->  {e['name']}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    print("Bericht:", out)
    if apply and len(table):
        identical = table[table["urteil"] == "identisch"].drop_duplicates(["suite_id", "task_id"])
        excluded = pd.read_csv("excluded_duplicates.csv")
        known = set(zip(excluded["suite_id"], excluded["task_id"]))
        new = [{"suite_id": r.suite_id, "task_id": r.task_id, "name": r.talent,
                "entsprechende_alte_aufgabe": r.vergleich_mit,
                "grund": f"{source_of(r.task_id)}, Inhaltsvergleich: {int(r.zugeordnete_spalten)} Spalten gleich, "
                         f"{max(r.zeilen_a_in_b, r.zeilen_b_in_a):.0%} der Zeilen gleich",
                "sicherheit": "sicher"} for r in identical.itertuples() if (r.suite_id, r.task_id) not in known]
        if new:
            pd.concat([excluded, pd.DataFrame(new)], ignore_index=True).to_csv("excluded_duplicates.csv", index=False)
        print(f"{len(new)} Aufgaben in excluded_duplicates.csv eingetragen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--talent-root")
    mode.add_argument("--new-since", help="Git-Stand, gegen den neu registrierte Aufgaben bestimmt werden")
    parser.add_argument("--apply", action="store_true", help="identische Treffer in excluded_duplicates.csv eintragen")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.talent_root, args.apply, args.out or ("content_duplicates_neu.csv" if args.new_since
                                                    else "content_duplicates_talent.csv"), args.new_since)
