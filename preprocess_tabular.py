"""Vorverarbeitung und Auswahlkriterien wie bei Grinsztajn et al. (2022), Abschnitte 3.1 und 3.2.

Vorverarbeitung (3.2), in dieser Reihenfolge:
 1. Fehlende Werte: erst Spalten mit vielen fehlenden Werten entfernen, dann alle Zeilen mit mindestens
    einem fehlenden Wert. ("many" ist im Paper nicht beziffert: wir nehmen > MAX_MISSING_COLUMN_SHARE.)
 2. Klassifikation: binaer machen mit den zwei haeufigsten Klassen; optional gleich viele Zeilen je Klasse.
 3. Kategoriale Merkmale mit mehr als 20 Stufen entfernen.
 4. Numerische Merkmale mit weniger als 10 verschiedenen Werten entfernen; solche mit genau 2 Werten
    werden stattdessen kategorial.
Auswahlkriterien (3.1), die sich pruefen lassen: mindestens 4 Merkmale, mindestens 3000 Zeilen,
Verhaeltnis Merkmale/Zeilen unter 1/10. (Nicht automatisch pruefbar: Zeitreihen, kuenstliche oder
deterministische Daten, "zu einfach".) One-Hot-Kodierung und die Unterstichprobe auf 100'000 Zeilen
passieren spaeter in methods.py.
"""
import numpy as np
import pandas as pd

MAX_LEVELS = 20
MIN_UNIQUE_NUMERIC = 10
MAX_MISSING_COLUMN_SHARE = 0.5   # eigene Annahme, im Paper steht nur "many"
MIN_FEATURES = 4
MIN_SAMPLES = 3000
MAX_FEATURE_SAMPLE_RATIO = 0.1


def preprocess(X, y, categorical_indicator, classification, balance=True, seed=0):
    X = X.reset_index(drop=True).copy()
    y = y.reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(np.asarray(y).ravel())
    categorical = dict(zip(X.columns, [bool(c) for c in categorical_indicator]))
    info = {"rows_before": len(X), "cols_before": X.shape[1]}

    # 1. fehlende Werte
    many_missing = [c for c in X.columns if X[c].isna().mean() > MAX_MISSING_COLUMN_SHARE]
    X = X.drop(columns=many_missing)
    info["dropped_missing_columns"] = len(many_missing)
    keep = X.notna().all(axis=1) & y.notna()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    info["rows_after_missing"] = len(X)

    # 2. Klassen
    if classification:
        y = y.astype(object)  # kategoriales Ziel: ungenutzte Kategorien sonst mit Anzahl 0 mitgezaehlt
        top_two = y.value_counts().index[:2]
        if len(top_two) < 2:
            raise ValueError("weniger als 2 Klassen nach der Bereinigung")
        keep = y.isin(top_two)
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        if balance:
            n_each = int(y.value_counts()[list(top_two)].min())
            picked = pd.concat([y[y == c].sample(n=n_each, random_state=seed).index.to_series()
                                for c in top_two]).sort_values()
            X, y = X.loc[picked].reset_index(drop=True), y.loc[picked].reset_index(drop=True)
        ordered = sorted(top_two, key=lambda v: str(v))
        y = y.map({ordered[0]: 0, ordered[1]: 1}).astype(int)
        info["classes_kept"] = f"{ordered[0]}|{ordered[1]}"

    # 3./4. Kardinalitaet
    dropped_cat, dropped_num, to_categorical = [], [], []
    for col in list(X.columns):
        n_unique = X[col].nunique()
        if categorical.get(col, False):
            if n_unique > MAX_LEVELS:
                dropped_cat.append(col)
        elif n_unique < MIN_UNIQUE_NUMERIC:
            if n_unique == 2:
                to_categorical.append(col)
            else:
                dropped_num.append(col)
    X = X.drop(columns=dropped_cat + dropped_num)
    for col in to_categorical:
        categorical[col] = True
    info["dropped_categorical"] = len(dropped_cat)
    info["dropped_numeric"] = len(dropped_num)
    info["numeric_to_categorical"] = len(to_categorical)

    indicator = np.array([categorical.get(c, False) for c in X.columns], dtype=bool)
    info["rows_after"] = len(X)
    info["cols_after"] = X.shape[1]
    return X, y, indicator, info


def selection_problem(info):
    """Gibt den Grund zurueck, falls die Aufgabe die Auswahlkriterien nicht erfuellt, sonst None."""
    n, d = info["rows_after"], info["cols_after"]
    if d < MIN_FEATURES:
        return f"zu wenige Merkmale ({d} < {MIN_FEATURES})"
    if n < MIN_SAMPLES:
        return f"zu wenige Zeilen ({n} < {MIN_SAMPLES})"
    if d / n >= MAX_FEATURE_SAMPLE_RATIO:
        return f"zu hochdimensional (Merkmale/Zeilen = {d / n:.3f} >= 0.1)"
    return None
