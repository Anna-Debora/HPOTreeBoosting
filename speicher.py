"""Speicherplan der grossen Laeufe (make_paper_jobs.py, watch_paper.py, refill.py).

Regel (1.11.2026): Speicher je Job = gemessene Spitze x FACTOR (1.3), mindestens MIN_GB (1 GB), aufgerundet auf die
naechste Stufe in MEM_STEPS. Unter dem Gastaccount es_mge war QOSMaxMemoryPerUser (Gesamtspeicher je Nutzer und
Partition) der Engpass; unter ls_math (seit 3.11.2026: 87 / 46 / 26 TB in normal.4h / 24h / 120h, siehe
watch_paper.QOS_LIMITS) ist er es kaum noch, knapper Speicher laesst aber weiterhin mehr Jobs auf einen Knoten.

Quelle der Spitze, in dieser Reihenfolge:
  1. Euler, gleiche Aufgabe und Methode (jobs_paper/speicher_euler.tsv aus watch_paper.py --speicher), wenn
     mindestens MIN_JOBS_METHOD Jobs dieser Methode mit MaxRSS beendet sind (COMPLETED oder TIMEOUT).
  2. Euler, gleiche Aufgabe, Maximum ueber alle HPO-Methoden (fuer Default: die Default-Messung, falls vorhanden).
     Bei weniger als MIN_JOBS_TASK gemessenen Jobs nicht allein: dann das Maximum mit der Mac-Messung.
  3. Mac (jobs_paper/speicher_mac.tsv, measure_memory.py: teuerster Versuch, num_leaves 1024) plus MAC_OVERHEAD_GB:
     measure_memory.py misst nur GPBoost, der echte Job hat mehr (Python-Pakete, Studien-Objekte, Vorhersagen).
     Abgleich Mac 1.11.2026, wine_quality (Mac-Spitze 0.22 GB), run_ablation.py --trials 12 --folds 1:
     random_search 0.44, SMAC 0.50, gp_bo 0.58 GB -> rund 0.35 GB Grundlast.
  4. Gar keine Messung: UNKNOWN_GB.
Ein OUT_OF_MEMORY auf Euler bei m GB (gleiche Aufgabe, HPO-Methoden bzw. Default) hebt den Plan auf mindestens die
Stufe ueber m. Wiederholungen nach OUT_OF_MEMORY eskalieren je Job weiter (watch_paper.escalate).

Lauf "extra" (EXTRA_METHODS: bohb, dehb, hebo, in .venv_extra): eigene Art "extra", damit ihre Messungen und OOM-Faelle
den Plan der sieben Methoden von paper/l1_270 nicht veraendern. Quelle fuer eine Extra-Methode:
  1. Euler, gleiche Aufgabe und Methode (mindestens MIN_JOBS_METHOD Jobs), wie oben.
  2. sonst wie oben fuer die HPO-Methoden der Aufgabe (Euler, sonst Mac) plus EXTRA_OVERHEAD_GB der Methode
     (zusaetzliche Pakete: torch/GPy fuer HEBO, KDE-Modelle fuer BOHB, Populationen fuer DEHB; Mac-Messung
     measure_extra.py), mindestens aber die Euler-Messung dieser Methode (auch mit weniger als 5 Jobs).
  OOM: das groessere von Extra- und HPO-OOM derselben Aufgabe.
"""
import math
from pathlib import Path

import pandas as pd

MEM_STEPS = (1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32)   # GB je Kern
FACTOR = 1.3
MIN_GB = 1
MAC_OVERHEAD_GB = 0.4
UNKNOWN_GB = 4
MIN_JOBS_METHOD = 5
MIN_JOBS_TASK = 3
EULER = Path("jobs_paper/speicher_euler.tsv")
MAC = Path("jobs_paper/speicher_mac.tsv")
EULER_COLUMNS = ["suite", "task", "name", "method", "jobs", "maxrss_gb", "oom", "oom_bei"]
EXTRA_METHODS = ("bohb", "dehb", "hebo")
# Zusaetzliche Spitze gegenueber Hyperband derselben Aufgabe (GB), Mac-Messung 10.11.2026 (measure_extra.py, 4 Aufgaben,
# Maximum): BOHB -0.03, DEHB +0.05, HEBO +0.56 (torch, GPy, GP-Modell); mit kleiner Reserve
EXTRA_OVERHEAD_GB = {"bohb": 0.1, "dehb": 0.1, "hebo": 0.6}


def kind_of(method):
    """'default', 'extra' (Lauf extra) oder 'hpo'."""
    return "default" if method == "default" else "extra" if method in EXTRA_METHODS else "hpo"


def step(gb):
    """Kleinste Stufe >= gb (hoechstens die groesste)."""
    return next((m for m in MEM_STEPS if m >= gb - 1e-9), MEM_STEPS[-1])


def next_step(gb):
    """Kleinste Stufe > gb (fuer OOM-Wiederholungen)."""
    return next((m for m in MEM_STEPS if m > gb + 1e-9), MEM_STEPS[-1])


def from_peak(peak):
    return step(max(MIN_GB, FACTOR * peak))


def sbatch_mem(gb):
    """--mem-per-cpu-Wert: ganze GB als '4G', sonst in MB ('1536M'; Slurm nimmt keine Dezimalzahlen)."""
    gb = float(gb)
    return f"{int(gb)}G" if gb == int(gb) else f"{int(math.ceil(gb * 1024))}M"


def fmt(gb):
    gb = float(gb)
    return f"{int(gb)}" if gb == int(gb) else f"{gb:g}"


def mac_peaks(path=MAC):
    """{(suite, task): Spitze in GB}; leer, wenn die Datei fehlt."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, sep="\t").dropna(subset=["peak_gb"])
    return {(int(r.suite), int(r.task)): float(r.peak_gb) for r in frame.itertuples()}


def euler_measurements(path=EULER):
    """speicher_euler.tsv (eine Zeile je Aufgabe und Methode) oder leerer DataFrame. Die alte Form (nur suite, task,
    name, mem_gb) wird nicht mehr gelesen; watch_paper.py --speicher schreibt die Datei neu."""
    if not Path(path).exists():
        return pd.DataFrame(columns=EULER_COLUMNS)
    frame = pd.read_csv(path, sep="\t")
    if "maxrss_gb" not in frame:
        print(f"{path}: alte Form ohne maxrss_gb, ignoriert (neu schreiben: python watch_paper.py --speicher)")
        return pd.DataFrame(columns=EULER_COLUMNS)
    return frame


class Plan:
    """plan(suite, task, method) -> GB je Kern, mit Quelle (plan.source(...))."""

    def __init__(self, mac=None, euler=None):
        self.mac = {k: v + MAC_OVERHEAD_GB for k, v in (mac_peaks() if mac is None else mac).items()}
        euler = euler_measurements() if euler is None else euler
        self.method, self.task, self.oom = {}, {}, {}
        for r in euler.itertuples():
            key = (int(r.suite), int(r.task))
            kind = kind_of(r.method)
            if int(r.jobs) > 0:
                self.method[(*key, r.method)] = (float(r.maxrss_gb), int(r.jobs))
                peak, n = self.task.get((*key, kind), (0.0, 0))
                self.task[(*key, kind)] = (max(peak, float(r.maxrss_gb)), n + int(r.jobs))
            if float(r.oom_bei or 0) > 0:
                self.oom[(*key, kind)] = max(self.oom.get((*key, kind), 0.0), float(r.oom_bei))

    def source(self, suite, task, method):
        """(Spitze in GB oder None, Quelle)."""
        if method in EXTRA_METHODS:
            return self._extra_source(suite, task, method)
        kind = "default" if method == "default" else "hpo"
        peak, n = self.method.get((suite, task, method), (None, 0))
        if peak is not None and n >= MIN_JOBS_METHOD:
            return peak, f"euler_methode({n})"
        peak, n = self.task.get((suite, task, kind), (None, 0))
        mac = self.mac.get((suite, task))
        if peak is not None and (n >= MIN_JOBS_TASK or mac is None):
            return peak, f"euler_aufgabe({n})"
        if peak is not None:
            return max(peak, mac), f"max(euler_aufgabe({n}), mac)"
        if mac is not None:
            return mac, "mac"
        return None, "unbekannt"

    def _extra_source(self, suite, task, method):
        peak, n = self.method.get((suite, task, method), (None, 0))
        if peak is not None and n >= MIN_JOBS_METHOD:
            return peak, f"euler_methode({n})"
        base, label = self.source(suite, task, "hpo")
        if base is None:
            return None, label
        peak = base + EXTRA_OVERHEAD_GB.get(method, 0.0)
        label = f"{label}+{EXTRA_OVERHEAD_GB.get(method, 0.0):g}"
        own, n = self.method.get((suite, task, method), (None, 0))
        if own is not None and own > peak:
            return own, f"euler_methode({n})"
        return peak, label
    def __call__(self, suite, task, method):
        suite, task = int(suite), int(task)
        peak, _ = self.source(suite, task, method)
        gb = UNKNOWN_GB if peak is None else from_peak(peak)
        if method in EXTRA_METHODS:
            oom = max(self.oom.get((suite, task, "extra"), 0.0), self.oom.get((suite, task, "hpo"), 0.0))
            return max(gb, next_step(oom)) if oom else gb
        oom = self.oom.get((suite, task, "default" if method == "default" else "hpo"), 0.0)
        return max(gb, next_step(oom)) if oom else gb
