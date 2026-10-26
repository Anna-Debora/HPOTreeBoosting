"""Liest die Kurven pro Datensatz aus den Abbildungen 5 bis 10 des Papers (Koster & Sigrist, "Selecting
Hyperparameters for Tree-Boosting", arXiv:2602.05786v3) und speichert sie als Tabelle.

Das Paper hat pro Datensatz keine Zahlentabellen, nur Grafiken. Diese sind Vektorgrafiken (Matplotlib):
Achsen, Gitterlinien mit Beschriftung, eine farbige Linie pro Methode, Legende. Das Skript liest die
Linienzuege direkt aus dem PDF und rechnet die Koordinaten ueber die Gitterlinien in Werte um. Die
Genauigkeit ist damit etwa 1/1000 der Achsenspanne (keine Pixel, sondern die Originalkoordinaten).
Pro Methode gibt es drei Linien: Mittelwert und die Grenzen des 95-%-Konfidenzintervalls ueber die
20 Seeds ("uncertainty due to the randomness in the hyperparameter selection methods"); gespeichert werden
alle drei (value, ci_low, ci_high).
"Default" ist eine waagrechte Linie oder, wenn sie ausserhalb der Achse liegt, als Text "Default = ..."
angegeben (dann mit den gerundeten Stellen aus dem Text).

Abbildungen: 5 und 6 = R2, 7 und 8 = RMSE (Regression, 36 Datensaetze), 9 = Accuracy, 10 = Log Loss
(Klassifikation, 23 Datensaetze). Kurven = Mittel ueber 20 Seeds und 5 Folds, Suchraum "num_leaves",
Budget 135 Versuche (Hyperband: 9 Stufen, gleichmaessig auf der x-Achse verteilt).

Aufruf:  python extract_paper_figures.py PFAD/2602.05786v3.pdf [--out paper_reference_values.csv]
Ausgabe: eine Zeile je (Abbildung, Feld, Datensatz, Methode, t) mit dem Kurvenwert; t = 45, 60, ..., 135.
Datensaetze, die in zwei Suiten vorkommen (z. B. diamonds in 335 und 336), stehen zweimal in den Bildern
und lassen sich dort nicht unterscheiden: Spalte "doppelt" = True, fuer Vergleiche nicht verwenden.
Benoetigt pypdf (pip install pypdf), nur fuer dieses Skript.
"""
import argparse
import re

import numpy as np
import pandas as pd

FIGURES = {15: (5, "R2"), 16: (6, "R2"), 17: (7, "RMSE"), 18: (8, "RMSE"), 19: (9, "Accuracy"), 20: (10, "Log Loss")}
TRIALS = (45, 60, 75, 90, 105, 120, 135)
METHODS = ("Default", "Deterministic Grid", "GP-BO", "Hyperband", "Random Grid", "SMAC", "TPE")
TOKEN = re.compile(rb"\((?:\\.|[^\\)])*\)|\[|\]|/[^\s/\[\]()<>]+|-?\d*\.\d+|-?\d+|[A-Za-z*'\"]+")


def _number(token):
    try:
        return float(token)
    except ValueError:
        return None


def parse(stream):
    """Gibt Linienzuege (Punkte, Farbe, Clip) und Texte (x, y, Text) in Reihenfolge zurueck."""
    events, stack, path, text_parts = [], [], [], None
    state = {"ctm": np.eye(3), "stroke": (0.0, 0.0, 0.0), "clip": None}
    saved = []
    for raw in TOKEN.findall(stream):
        token = raw.decode("latin1")
        value = _number(token)
        if value is not None or token.startswith("/"):
            stack.append(value if value is not None else token)
            continue
        if token.startswith("("):
            if text_parts is not None:
                text_parts.append(token[1:-1].replace("\\(", "(").replace("\\)", ")"))
            continue
        if token == "[":
            text_parts = []
            continue
        if token == "]":
            continue
        op = token
        if op == "q":
            saved.append(dict(state))
        elif op == "Q":
            state = saved.pop() if saved else state
        elif op == "cm" and len(stack) >= 6:
            a, b, c, d, e, f = stack[-6:]
            state["ctm"] = np.array([[a, b, 0], [c, d, 0], [e, f, 1]]) @ state["ctm"]
        elif op == "RG" and len(stack) >= 3:
            state["stroke"] = tuple(stack[-3:])
        elif op == "G" and stack:
            state["stroke"] = (stack[-1],) * 3
        elif op == "re" and len(stack) >= 4:
            path = [("re", tuple(stack[-4:]))]
        elif op == "W":
            if path and path[0][0] == "re":
                state["clip"] = path[0][1]
        elif op == "m" and len(stack) >= 2:
            path = [tuple(stack[-2:])]
        elif op == "l" and len(stack) >= 2:
            path.append(tuple(stack[-2:]))
        elif op == "c" and len(stack) >= 6:
            path.append(tuple(stack[-2:]))
        elif op == "S":
            points = [p for p in path if p and p[0] != "re"]
            if points:
                events.append(("path", np.array(points, dtype=float), state["stroke"], state["clip"]))
            path = []
        elif op in ("f", "n", "B", "b"):
            path = []
        elif op == "Do" and stack and "minus" in str(stack[-1]) and events and events[-1][0] == "text":
            kind, x, y, text = events[-1]          # Minuszeichen als eigenes Glyphen-Objekt nach dem Text
            events[-1] = (kind, x, y, "-" + text)
        elif op == "TJ" and text_parts is not None:
            x, y = state["ctm"][2, 0], state["ctm"][2, 1]
            events.append(("text", x, y, "".join(text_parts)))
            text_parts = None
        stack = []
    return events


def _is_gray(color):
    return max(color) - min(color) < 1e-6


def digitize(events):
    axes = {}
    pending = None          # letzte Gitterlinie, die naechste Zahl ist ihre Beschriftung
    legend_line = None
    legend = {}
    texts = []
    for event in events:
        if event[0] == "path":
            _, points, color, clip = event
            if clip is not None and clip[2] > 100 and clip[3] > 100:
                ax = axes.setdefault(clip, {"x": [], "y": [], "curves": []})
                x0, y0, w, h = clip
                if _is_gray(color) and len(points) == 2:
                    if abs(points[0, 0] - points[1, 0]) < 1e-6 and abs(points[1, 1] - points[0, 1]) > 0.9 * h:
                        pending = (clip, "x", points[0, 0])
                    elif abs(points[0, 1] - points[1, 1]) < 1e-6 and abs(points[1, 0] - points[0, 0]) > 0.9 * w:
                        pending = (clip, "y", points[0, 1])
                elif not _is_gray(color):
                    ax["curves"].append((color, points))
            elif clip is None and not _is_gray(color) and len(points) == 3 \
                    and np.ptp(points[:, 1]) < 1e-6 and 20 < np.ptp(points[:, 0]) < 40:
                legend_line = color
        else:
            _, x, y, text = event
            value = _number(text.replace("−", "-"))
            if pending is not None and value is not None:
                clip, axis, coord = pending
                axes[clip][axis].append((coord, value))
                pending = None
            elif legend_line is not None and text in METHODS:
                legend[tuple(round(c, 4) for c in legend_line)] = text
                legend_line = None
            else:
                texts.append((x, y, text))
    return axes, legend, texts


def _linear(pairs):
    coords, values = np.array(pairs, dtype=float).T
    slope, intercept = np.polyfit(coords, values, 1)
    return lambda c: slope * np.asarray(c) + intercept


def _inverse(pairs):
    coords, values = np.array(pairs, dtype=float).T
    slope, intercept = np.polyfit(values, coords, 1)
    return lambda v: slope * np.asarray(v) + intercept


def extract(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    rows, panel = [], 0
    for page_number, (figure, metric) in FIGURES.items():
        page = reader.pages[page_number - 1]
        xobjects = page["/Resources"]["/XObject"]
        form = [xobjects[k].get_object() for k in xobjects if xobjects[k].get_object().get("/Subtype") == "/Form"][0]
        axes, legend, texts = digitize(parse(form.get_data()))
        for clip, ax in sorted(axes.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
            x0, y0, w, h = clip
            if len(ax["x"]) < 2 or len(ax["y"]) < 2:
                continue
            title = [t for t in texts if y0 + h < t[1] < y0 + h + 40 and x0 - 20 < t[0] < x0 + w
                     and not t[2].startswith("Default")]
            name = _title(texts, clip)
            offset = sum(float(t[2]) for t in texts if t[2][:1] in "+-" and _number(t[2]) is not None
                         and abs(t[0] - x0) < 5 and y0 + h < t[1] < y0 + h + 20)   # Matplotlib-Offset, z. B. "+1"
            y_value, x_coord = _linear(ax["y"]), _inverse(ax["x"])
            panel += 1
            groups = {}
            for color, points in ax["curves"]:
                method = legend.get(tuple(round(c, 4) for c in color))
                if method is not None:
                    groups.setdefault(method, []).append(points)
            for method, paths in groups.items():
                curves = []
                for points in paths:
                    order = np.argsort(points[:, 0])
                    xs, ys = points[order, 0], y_value(points[order, 1]) + offset
                    curves.append([float(np.interp(x_coord(t), xs, ys)) for t in TRIALS])
                curves = np.array(curves)
                mean = curves[0] if len(curves) != 3 else np.median(curves, axis=0)
                low, high = curves.min(axis=0), curves.max(axis=0)
                for i, t in enumerate(TRIALS):
                    rows.append({"figure": figure, "panel": panel, "metric": metric, "dataset": name, "method": method,
                                 "t": t, "value": mean[i], "ci_low": low[i], "ci_high": high[i], "quelle": "Kurve"})
            for t in texts:   # Default ausserhalb der Achse: "Default = 0.992" im Bild
                if t[2].startswith("Default =") and x0 - 5 < t[0] < x0 + w and y0 < t[1] < y0 + h:
                    if not any(r["dataset"] == name and r["method"] == "Default" and r["figure"] == figure for r in rows):
                        value = float(t[2].split("=")[1])
                        rows += [{"figure": figure, "panel": panel, "metric": metric, "dataset": name,
                                  "method": "Default", "t": s, "value": value, "ci_low": value, "ci_high": value,
                                  "quelle": "Text"} for s in TRIALS]
    table = pd.DataFrame(rows)
    panels = table.drop_duplicates(["metric", "figure", "panel"])
    counts = panels.groupby(["metric", "dataset"]).size()
    table["doppelt"] = table.set_index(["metric", "dataset"]).index.map(lambda k: counts[k] > 1)
    return table


def _title(texts, clip):
    """Titel = Text knapp ueber dem Feld, dessen (geschaetzte) Mitte der Feldmitte am naechsten liegt."""
    x0, y0, w, h = clip
    candidates = [t for t in texts if y0 + h < t[1] < y0 + h + 40 and _number(t[2]) is None
                  and not t[2].startswith("Default") and t[2] not in METHODS]
    if not candidates:
        return "unbekannt"
    return min(candidates, key=lambda t: abs(t[0] + 3.9 * len(t[2]) - (x0 + w / 2)))[2]


def main(pdf_path, out):
    table = extract(pdf_path)
    table.to_csv(out, index=False)
    summary = table.groupby(["figure", "metric"]).agg(felder=("panel", "nunique"), namen=("dataset", "nunique"))
    print(summary.to_string())
    print("Methoden je Abbildung:")
    print(table.groupby("figure")["method"].unique().to_string())
    print("Geschrieben:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--out", default="paper_reference_values.csv")
    args = parser.parse_args()
    main(args.pdf, args.out)
