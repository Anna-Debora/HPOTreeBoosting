"""Stand der grossen Laeufe (make_paper_jobs.py): pro Lauf und Welle fertig / laufend / wartend / unbekannt /
abgebrochen / fehlgeschlagen / fehlt, was refill.py noch nachfuellen muss (neu, Wiederholung, aufgegeben) und die
Laenge der Schlange.
Die Logik hier (Abgaben aus submitted.tsv, sacct, Einordnung je Job) benutzt auch refill.py.

Aufruf auf Euler (aus ~/HPOTreeBoosting, mit aktivierter .venv):
    python watch_paper.py                 # paper und l1_270, dazu extra (BOHB, DEHB, HEBO), sobald
                                          # jobs_paper/extra/jobs.tsv existiert (make_paper_jobs.py --run extra)
    python watch_paper.py --run extra     # nur extra
    python watch_paper.py --run paper     # nur einer
    python watch_paper.py --aufgaben      # zusaetzlich je Aufgabe und Welle: fertige Jobs / alle, "ok" = vollstaendig
    python watch_paper.py --speicher      # MaxRSS je Aufgabe und Methode aus sacct, OOM-Faelle, neuer Speicherplan,
                                          # Laufzeit gegen Zeitlimit; schreibt jobs_paper/speicher_euler.tsv, aus dem
                                          # refill.py den Speicher plant (speicher.py: MaxRSS x 1.3, Stufen ab 1 GB)
Am Ende eine Prognose je Lauf und Welle: verbrauchte Kernstunden (sacct), verbleibende (Schaetzung kosten.tsv),
mittlere Zahl gleichzeitig laufender Kerne der letzten 24 h (sacct) und voraussichtliches Fertigstellungsdatum.
Wartende Jobs mit anderem Speicher als im Plan meldet watch_paper.py mit scancel-Skript:
    zu wenig (wuerden wohl OOM):      bash jobs_paper/<lauf>/scancel_zu_wenig_speicher.sh
    zu viel (blockieren QOSMaxMemoryPerUser): bash jobs_paper/<lauf>/scancel_zu_viel_speicher.sh
Danach schickt refill.py sie mit dem Plan neu ab. Ein abgebrochener Job, der nie gelaufen ist (CANCELLED,
ElapsedRaw 0), zaehlt nicht als Fehlschlag.
laufend/wartend nach squeue (sonst sacct); unbekannt = sacct kennt die letzte Abgabe nicht, und sie steht nicht in
squeue; abgebrochen = CANCELLED. Beide reicht refill.py neu ein.
Ohne sacct (z. B. auf dem Mac nach dem Kopieren der Ergebnisse) nur der Dateistand (Abgaben dann 'unbekannt').

submitted.tsv (je Abgabe eine Zeile, ohne Kopf): Job-ID, idx-Datei, Zahl der Teiljobs, Zeit, Partition, Stunden,
Speicher GB. Aeltere Zeilen (submit.sh vor refill.py) haben nur die ersten vier Spalten; Partition, Stunden und
Speicher kommen dann aus arrays.tsv (eingefrorene Wellen-Arrays), aus dem Namen (retry_<part>_<mem>G_<h>h) oder
sind die der Probe. Array-Index i einer Abgabe = Zeile i+1 der idx-Datei = Zeilennummer in jobs.tsv.
"""
import argparse
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import quota
import speicher
from run_ablation import expected_rows

JOBS = Path("jobs_paper")
RUN_ORDER = ["paper", "l1_270", "extra"]      # Reihenfolge wie refill.py
COLUMNS = ["method", "seed", "suite", "task", "trials", "folder", "flags"]
SUBMITTED = ["id", "idx", "n", "zeit", "partition", "stunden", "mem"]
PARTITIONS = (("normal.4h", 4), ("normal.24h", 24), ("normal.120h", 120))
# Slurm-Account ls_math (seit 3.11.2026, vorher Gastaccount es_mge mit 64 Kernen je Partition und ~1000 Jobs):
# Limits je Nutzer, (Kerne, Speicher TB); normal.bulk wird nicht benutzt. Der Speicher zaehlt nur laufende Jobs.
ACCOUNT = "ls_math"
QOS_LIMITS = {"normal.4h": (26752, 87), "normal.24h": (14045, 46), "normal.120h": (8026, 26),
              "normal.bulk": (66880, 217)}
MAX_SUBMIT = 17000      # QOS MaxSubmit: wartende + laufende Jobs je Nutzer (jede Array-Aufgabe zaehlt)
QUEUE_LIMIT = 4000      # Standard fuer refill.py --limit
QUEUE_MAX = 15000       # groesster erlaubter Wert fuer refill.py --limit (Abstand zu MAX_SUBMIT)
MEM_STEPS = speicher.MEM_STEPS
FOLDS = 5
QUEUED = ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "COMPLETING", "CONFIGURING", "RESIZING", "REQUEUE_HOLD",
          "REQUEUE_FED", "SIGNALING", "STAGE_OUT")
ENDED_BAD = ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"]
# unbekannt: sacct kennt die letzte Abgabe nicht und squeue hat sie nicht (oder es gibt kein sacct/squeue);
# abgebrochen: CANCELLED (z. B. scancel eines wartenden Jobs); beide fuellt refill.py nach, wenn sie nicht in squeue sind
STATES = ["fertig", "laufend", "wartend", "unbekannt", "abgebrochen", "fehlgeschlagen", "fehlt"]


def file_state(job):
    """'fertig', 'angefangen (k/5)' oder 'fehlt'."""
    if job.method == "default":
        path = Path(job.folder) / "default" / "gpboost" / f"{job.suite}_{job.task}.csv"
        return "fertig" if path.exists() and path.stat().st_size > 0 else "fehlt"
    path = Path(job.folder) / job.method / f"seed_{job.seed}" / "gpboost" / f"{job.suite}_{job.task}__num_leaves.csv"
    if not path.exists():
        return "fehlt"
    try:
        sizes = pd.read_csv(path, usecols=["fold"])["fold"].value_counts()
    except Exception:
        return "angefangen (0/5)"
    n_rows = expected_rows(job.method, int(job.trials), "--paper_grid" in str(job.flags))
    done = sum(1 for n in sizes if n == (n_rows or sizes.max()))
    return "fertig" if done >= FOLDS else f"angefangen ({done}/5)"


# ---------------------------------------------------------------- Abgaben und Slurm
def resources_from_idx(run, idx):
    """(Partition, Stunden, Speicher) einer alten Abgabe ohne diese Spalten."""
    name = Path(idx).stem
    match = re.match(r"retry_(normal\.\d+h)_(\d+)G_(\d+)h$", name)
    if match:
        return match.group(1), int(match.group(3)), int(match.group(2))
    if name == "probe":
        return "normal.4h", 4, 4
    arrays = JOBS / run / "arrays.tsv"
    if arrays.exists():
        row = pd.read_csv(arrays, sep="\t").set_index("array")
        if name in row.index:
            return row.loc[name, "partition"], int(row.loc[name, "stunden"]), int(row.loc[name, "mem_gb"])
    return None, None, None


def submissions(run):
    """Abgaben eines Laufs in der Reihenfolge von submitted.tsv, mit Ressourcen und Zeilen der idx-Datei."""
    path = JOBS / run / "submitted.tsv"
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        fields = (raw.split("\t") + [""] * len(SUBMITTED))[:len(SUBMITTED)]
        sub = dict(zip(SUBMITTED, fields))
        sub["id"] = sub["id"].split(";")[0]
        if sub["partition"]:
            sub["stunden"], sub["mem"] = int(sub["stunden"]), float(sub["mem"])
        else:
            sub["partition"], sub["stunden"], sub["mem"] = resources_from_idx(run, sub["idx"])
        idx = Path(sub["idx"])
        sub["lines"] = [int(x) for x in idx.read_text().split()] if idx.exists() else []
        out.append(sub)
    return out


def run_sacct(ids, fmt="JobID,State,ElapsedRaw", steps=False):
    """sacct-Text fuer diese Job-IDs oder None (kein sacct, z. B. auf dem Mac)."""
    if not ids:
        return ""
    cmd = ["sacct", "-n", "-P", "-j", ",".join(ids), f"--format={fmt}"] + ([] if steps else ["-X"])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_sacct(text):
    """{(Job-ID, Array-Index): (Zustand, Sekunden)}; wartende Bereiche wie 123_[5-9,12%50] einzeln aufgeloest."""
    states = {}
    for row in (text or "").splitlines():
        job_id, state, elapsed = (row.split("|") + ["", ""])[:3]
        state = (state.split() or ["?"])[0]
        base, _, rest = job_id.partition("_")
        if rest.startswith("["):
            for part in rest.strip("[]").split("%")[0].split(","):
                start, _, end = part.partition("-")
                for i in range(int(start), int(end or start) + 1):
                    states[(base, i)] = (state, 0)
        elif rest.isdigit():
            states[(base, int(rest))] = (state, int(elapsed or 0))
    return states


def attempts(run, sacct_text=False):
    """{Zeile in jobs.tsv: [Versuche in Abgabe-Reihenfolge]}; ein Versuch = dict(id, index, partition, stunden,
    mem, state, sekunden); state None = sacct kennt ihn (noch) nicht. sacct_text=False: sacct selbst aufrufen.
    Gibt (Versuche, sacct verfuegbar?) zurueck."""
    subs = submissions(run)
    if sacct_text is False:
        sacct_text = run_sacct(sorted({s["id"] for s in subs}))
    states = parse_sacct(sacct_text)
    out = {}
    for sub in subs:
        for i, line in enumerate(sub["lines"][:int(sub["n"] or len(sub["lines"]))]):
            state, seconds = states.get((sub["id"], i), (None, 0))
            out.setdefault(line, []).append(dict(id=sub["id"], index=i, partition=sub["partition"],
                                                 stunden=sub["stunden"], mem=sub["mem"], state=state,
                                                 sekunden=seconds))
    return out, sacct_text is not None


def slurm_states(run):
    """{Zeile: (Zustand, Laufzeit, Job-ID, Array-Index)} der letzten Abgabe je Zeile; leer ohne sacct."""
    tries, available = attempts(run)
    if not available:
        return {}
    return {line: (t[-1]["state"] or "UNBEKANNT", t[-1]["sekunden"], t[-1]["id"], t[-1]["index"])
            for line, t in tries.items()}


def never_ran(attempt):
    """Abgebrochen, bevor er lief (z. B. scancel eines wartenden Jobs): kein Fehlschlag."""
    return attempt["state"] == "CANCELLED" and not attempt["sekunden"]


def escalate(tries, plan):
    """Ressourcen fuer die Wiederholung aus allen bisherigen Versuchen eines Jobs: nach TIMEOUT die naechstlaengere
    Partition ueber dem laengsten abgelaufenen Versuch, nach OUT_OF_MEMORY die naechste Speicherstufe ueber dem
    groessten OOM-Versuch, sonst der aktuelle Plan (kosten.tsv bzw. speicher_euler.tsv); nie weniger als der Plan.
    So bleibt eine Eskalation auch nach einem spaeteren anderen Fehler erhalten, und ein wartender Job, der wegen
    zu viel Speicher abgebrochen wurde, kommt mit dem kleineren Plan zurueck. plan: dict(partition, stunden, mem)."""
    part, hours, mem = plan["partition"], int(plan["stunden"]), float(plan["mem"])
    timeouts = [a["stunden"] or plan["stunden"] for a in tries if a["state"] == "TIMEOUT"]
    if timeouts:
        bigger = [p for p in PARTITIONS if p[1] > max(timeouts)]
        longer = bigger[0] if bigger else PARTITIONS[-1]
        if longer[1] > hours:
            part, hours = longer
    ooms = [a["mem"] or plan["mem"] for a in tries if a["state"] == "OUT_OF_MEMORY"]
    if ooms:
        mem = max(mem, speicher.next_step(max(ooms)))
    return dict(partition=part, stunden=int(hours), mem=mem)


def classify(jobs, tries, max_tries=3, queued=None):
    """Spalten 'slurm' (Zustand der letzten Abgabe laut sacct), 'squeue' (Zustand laut squeue, '' = nicht in der
    Schlange, None = squeue unbekannt), 'versuche' (beendet ohne Ergebnis), 'aktion' (fertig / in_schlange / neu /
    wiederholen / aufgegeben) und die Ressourcen der naechsten Abgabe (partition_neu, stunden_neu, mem_neu).
    jobs braucht line, datei, partition, zeitlimit, mem_gb. queued: {(Job-ID, Array-Index): Zustand} aus squeue
    (queued_tasks) oder None.
    in_schlange: letzte Abgabe steht in squeue oder ist laut sacct PENDING/RUNNING/...; ohne squeue zaehlt auch
    eine sacct unbekannte Abgabe als in der Schlange (nie doppelt). Mit squeue ist eine Abgabe, die weder in squeue
    steht noch laut sacct wartet/laeuft (unbekannt, CANCELLED, NODE_FAIL, ...), keine Abgabe mehr -> wiederholen."""
    slurm, live, failed, action, res = [], [], [], [], []
    for job in jobs.itertuples():
        t = tries.get(job.line, [])
        last = t[-1] if t else None
        state = last["state"] if last else None
        in_queue = None if queued is None else (queued.get((last["id"], last["index"]), "") if last else "")
        n_failed = sum(1 for a in t if a["state"] in ENDED_BAD + ["COMPLETED"] and not never_ran(a))
        plan = dict(partition=job.partition, stunden=int(job.zeitlimit), mem=float(job.mem_gb))
        slurm.append("nicht abgeschickt" if not t else (state or "UNBEKANNT"))
        live.append(in_queue)
        failed.append(n_failed)
        if job.datei == "fertig":
            action.append("fertig"), res.append(plan)
        elif not t:
            action.append("neu"), res.append(plan)
        elif in_queue or state in QUEUED or (state is None and queued is None):
            action.append("in_schlange"), res.append(plan)
        elif n_failed >= max_tries:
            action.append("aufgegeben"), res.append(plan)
        else:
            action.append("wiederholen"), res.append(escalate(t, plan))
    out = jobs.copy()
    out["slurm"], out["versuche"], out["aktion"] = slurm, failed, action
    out["squeue"] = pd.Series(live, index=out.index, dtype=object)
    out["partition_neu"] = [r["partition"] for r in res]
    out["stunden_neu"] = [r["stunden"] for r in res]
    out["mem_neu"] = [r["mem"] for r in res]
    return out


def queue():
    """Eigene Jobs in der Schlange (jede Array-Aufgabe einzeln): DataFrame id, name, partition, state; None ohne
    squeue."""
    try:
        text = subprocess.run(["squeue", "--me", "-h", "-r", "-o", "%i|%j|%P|%T"], capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = [(line.split("|") + ["", "", ""])[:4] for line in text.splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=["id", "name", "partition", "state"])


def queued_tasks(q):
    """{(Job-ID, Array-Index): Zustand} der Array-Aufgaben in der Schlange (squeue -r: '123_7'); None ohne squeue."""
    if q is None:
        return None
    out = {}
    for job_id, state in zip(q["id"], q["state"]):
        base, _, index = str(job_id).partition("_")
        if index.isdigit():
            out[(base, int(index))] = state
    return out


def load_jobs(run, with_files=True):
    """jobs.tsv mit Plan aus kosten.tsv (welle, partition, zeitlimit, mem_gb, reihenfolge) und Dateistand.
    mem_gb: aus kosten.tsv; gibt es jobs_paper/speicher_euler.tsv (Euler-Messung), neu nach speicher.Plan
    (Euler-Messung je Aufgabe und Methode, sonst Mac)."""
    jobs = pd.read_csv(JOBS / run / "jobs.tsv", sep="\t", header=None, names=COLUMNS, keep_default_na=False)
    costs = pd.read_csv(JOBS / run / "kosten.tsv", sep="\t", keep_default_na=False)
    for column in ("partition", "mem_gb", "kernstunden", "name", "stunden", "welle", "zeitlimit", "reihenfolge"):
        jobs[column] = costs[column] if column in costs else (costs["stunden"] if column == "zeitlimit" else 0)
    jobs["mem_gb"] = jobs["mem_gb"].astype(float)
    if speicher.EULER.exists():
        plan = speicher.Plan()
        jobs["mem_gb"] = [plan(s, t, m) for s, t, m in zip(jobs["suite"], jobs["task"], jobs["method"])]
    jobs["line"] = jobs.index + 1
    jobs["datei"] = [file_state(j) for j in jobs.itertuples()] if with_files else "fehlt"
    return jobs


# ---------------------------------------------------------------- Uebersichten
def status(datei, slurm, squeue=None):
    """Zustand eines Jobs fuer die Wellen-Uebersicht; squeue: Zustand laut squeue (Vorrang vor sacct), '' oder
    None = nicht in der Schlange bzw. unbekannt."""
    if datei == "fertig":
        return "fertig"
    if isinstance(squeue, str) and squeue:
        return "laufend" if squeue in ("RUNNING", "COMPLETING") else "wartend"
    if slurm in ("RUNNING", "COMPLETING"):
        return "laufend"
    if slurm in ("PENDING", "REQUEUED", "SUSPENDED"):
        return "wartend"
    if slurm == "UNBEKANNT":
        return "unbekannt"
    if slurm == "CANCELLED":
        return "abgebrochen"
    if slurm in ENDED_BAD or slurm == "COMPLETED":     # COMPLETED ohne fertige Datei zaehlt als fehlgeschlagen
        return "fehlgeschlagen"
    return "fehlt"


def wave_table(jobs):
    """Je Welle: Zahl der Jobs je Zustand, nachzufuellen (neu + wiederholen), aufgegeben, Aufgaben vollstaendig."""
    rows = []
    for welle, group in jobs.groupby("welle"):
        counts = group["status"].value_counts()
        done = group.groupby(["suite", "task"])["status"].apply(lambda s: (s == "fertig").all())
        row = {"welle": int(welle), "jobs": len(group), **{k: int(counts.get(k, 0)) for k in STATES}}
        if "aktion" in group:
            row["nachfuellen"] = int(group["aktion"].isin(["neu", "wiederholen"]).sum())
            row["aufgegeben"] = int((group["aktion"] == "aufgegeben").sum())
        row.update(aufgaben_fertig=int(done.sum()), aufgaben=len(done))
        rows.append(row)
    return pd.DataFrame(rows)


def task_table(jobs):
    """Je Aufgabe und Welle: 'fertig/alle', 'ok' wenn vollstaendig; Spalte 'bis' = letzte Welle, bis zu der die
    Aufgabe lueckenlos fertig ist (0 = nicht einmal Welle 1)."""
    frame = jobs.assign(fertig=jobs["status"] == "fertig").groupby(["suite", "task", "name", "welle"])["fertig"] \
        .agg(["sum", "size"]).reset_index()
    frame["text"] = [("ok" if s == n else f"{s}/{n}") for s, n in zip(frame["sum"], frame["size"])]
    table = frame.pivot_table(index=["suite", "task", "name"], columns="welle", values="text", aggfunc="first")
    table.columns = [f"welle{w}" for w in table.columns]
    table["bis"] = [next((i for i, v in enumerate(row, 1) if v != "ok"), len(row) + 1) - 1
                    for row in table.to_numpy()]
    order = jobs.groupby(["suite", "task", "name"])["kernstunden"].sum()     # kleine Aufgaben zuerst
    return table.loc[order.sort_values(kind="stable").index].reset_index()


def disk_text():
    """Kopfzeile Home/Scratch (quota.py: lquota, sonst df) mit Warnungen ueber 80 %."""
    info = quota.usage()
    _, warnings_ = quota.check(info)
    return "\n".join(["Speicher: " + quota.describe("home", info["home"]) + "; "
                      + quota.describe("scratch", info["scratch"])] + [w for w in warnings_ if w.startswith("WARN")])


def queue_text(limit=QUEUE_LIMIT):
    q = queue()
    if q is None:
        return "Schlange: unbekannt (kein squeue)"
    by_name = Counter(q["name"])
    states = Counter(q["state"])
    running = Counter(q.loc[q["state"] == "RUNNING", "partition"])
    return (f"Schlange: {len(q)} Jobs (Limit refill.py {limit}; QOS MaxSubmit {MAX_SUBMIT}), davon "
            + ", ".join(f"{k} {v}" for k, v in sorted(by_name.items())) + " | "
            + ", ".join(f"{k} {v}" for k, v in sorted(states.items()))
            + "\nLaufend je Partition (Kerne / QOS-Limit, Account " + ACCOUNT + "): "
            + (", ".join(f"{p} {n}/{QOS_LIMITS.get(p, ('?',))[0]}" for p, n in sorted(running.items())) or "keine"))


LEGACY_MEM = 4        # Abgaben ohne bekannten Speicher stammen aus der Zeit vor dem 1.11.2026: alle >= 4 GB
MIN_SECONDS = 300     # kuerzere Jobs (z. B. "SCHON FERTIG" nach --resume) sagen nichts ueber den Speicher


def parse_rss(value):
    """sacct-MaxRSS ('1234K', '1.5G', '0') in GB."""
    if not value:
        return 0.0
    unit = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}.get(value[-1], 1e-9)
    return float(value[:-1] if value[-1] in "KMGT" else value) * unit


def parse_steps(text):
    """{(Job-ID, Array-Index): dict(state, sekunden, rss, oom)} aus sacct mit Schritten (JobID|State|MaxRSS|
    ElapsedRaw): Zustand und Laufzeit aus der Hauptzeile, MaxRSS = Maximum ueber die Schritte (.batch, .extern)."""
    out = {}
    for row in (text or "").splitlines():
        job_id, state, maxrss, elapsed = (row.split("|") + ["", "", "", ""])[:4]
        base, _, rest = job_id.partition("_")
        index, _, step_name = rest.partition(".")
        if not index.isdigit():
            continue
        entry = out.setdefault((base, int(index)), dict(state=None, sekunden=0, rss=0.0, oom=False))
        state = (state.split() or ["?"])[0]
        if not step_name:
            entry["state"], entry["sekunden"] = state, int(elapsed or 0)
        entry["oom"] |= state.startswith("OUT_OF_ME")
        entry["rss"] = max(entry["rss"], parse_rss(maxrss))
    return out


def measurements(runs, sacct_text=None):
    """Ein Eintrag je beendetem Versuch mit Messung: lauf, line, suite, task, name, method, angefordert, state,
    sekunden, rss, oom, gezaehlt (COMPLETED/TIMEOUT, mindestens MIN_SECONDS), kernstunden, zeitlimit (Plan)."""
    rows = []
    for run in runs:
        jobs = load_jobs(run, with_files=False).set_index("line")
        subs = submissions(run)
        text = sacct_text if sacct_text is not None else run_sacct(
            sorted({s["id"] for s in subs}), fmt="JobID,State,MaxRSS,ElapsedRaw", steps=True)
        if text is None:
            return None
        steps = parse_steps(text)
        for sub in subs:
            for i, line in enumerate(sub["lines"][:int(sub["n"] or len(sub["lines"]))]):
                m = steps.get((sub["id"], i))
                if m is None or m["state"] in QUEUED or m["state"] is None:
                    continue
                job = jobs.loc[line]
                long_enough = m["sekunden"] >= MIN_SECONDS
                rows.append(dict(lauf=run, line=line, suite=int(job.suite), task=int(job.task), name=job["name"],
                                 method=job.method, angefordert=float(sub["mem"] or LEGACY_MEM), state=m["state"],
                                 sekunden=m["sekunden"], rss=m["rss"] if long_enough or m["oom"] else 0.0,
                                 oom=m["oom"], gezaehlt=long_enough and m["state"] in ("COMPLETED", "TIMEOUT"),
                                 kernstunden=float(job.kernstunden), zeitlimit=int(job.zeitlimit),
                                 limit_abgabe=int(sub["stunden"] or job.zeitlimit)))
    return pd.DataFrame(rows)


def euler_table(frame):
    """speicher_euler.tsv: je Aufgabe und Methode Zahl der gezaehlten Jobs, MaxRSS (Maximum ueber alle beendeten
    Versuche mit Messung, beide Laeufe), OOM-Faelle und groesster Speicher, bei dem ein OOM kam."""
    rows = []
    for (suite, task, name, method), g in frame.groupby(["suite", "task", "name", "method"]):
        rows.append(dict(suite=suite, task=task, name=name, method=method, jobs=int(g["gezaehlt"].sum()),
                         maxrss_gb=round(g.loc[g["gezaehlt"] | g["oom"], "rss"].max() if len(g) else 0.0, 3),
                         oom=int(g["oom"].sum()), oom_bei=float(g.loc[g["oom"], "angefordert"].max())
                         if g["oom"].any() else 0.0))
    table = pd.DataFrame(rows, columns=speicher.EULER_COLUMNS)
    table["maxrss_gb"] = table["maxrss_gb"].fillna(0.0)
    return table


def memory_report(runs, sacct_text=None, out=None):
    """MaxRSS je Aufgabe und Methode gegen angeforderten und neuen Speicher; schreibt speicher_euler.tsv (Grundlage
    fuer speicher.Plan und damit fuer refill.py) und prueft die Zeitlimits gegen die gemessenen Laufzeiten."""
    out = Path(out or speicher.EULER)
    frame = measurements(runs, sacct_text)
    if frame is None:
        print("kein sacct")
        return None
    if frame.empty or not (frame["gezaehlt"] | frame["oom"]).any():
        print("Noch keine beendeten Jobs mit MaxRSS.")
        return None
    table = euler_table(frame)
    table.to_csv(out, sep="\t", index=False)
    plan = speicher.Plan(euler=table)
    table["alt_gb"] = [frame.loc[(frame["suite"] == r.suite) & (frame["task"] == r.task)
                                 & (frame["method"] == r.method), "angefordert"].max() for r in table.itertuples()]
    table["neu_gb"] = [plan(r.suite, r.task, r.method) for r in table.itertuples()]
    table["quelle"] = [plan.source(r.suite, r.task, r.method)[1] for r in table.itertuples()]
    pd.set_option("display.width", 200)
    print("Je Aufgabe und Methode (jobs = gezaehlte Jobs mit Messung, MaxRSS in GB; neu = Plan nach speicher.py):")
    print(table.sort_values(["maxrss_gb"], ascending=False).drop(columns=["suite", "task"]).round(2)
          .to_string(index=False))
    hpo = table[table["method"] != "default"]
    per_task = hpo.groupby("name").agg(jobs=("jobs", "sum"), maxrss_gb=("maxrss_gb", "max"), oom=("oom", "sum"),
                                       alt_gb=("alt_gb", "max"), neu_min=("neu_gb", "min"),
                                       neu_max=("neu_gb", "max")).sort_values("maxrss_gb", ascending=False)
    print("\nJe Aufgabe (HPO-Methoden):")
    print(per_task.round(2).to_string())
    spread = per_task[per_task["neu_max"] > per_task["neu_min"]]
    print(f"\nAufgaben, in denen sich die Methoden in der Stufe unterscheiden: {len(spread)} von {len(per_task)}"
          + (" (" + ", ".join(spread.index[:10]) + ")" if len(spread) else ""))
    print(f"{out}: {len(table)} Zeilen (Aufgabe x Methode); refill.py und watch_paper.py planen damit den Speicher.")
    time_report(frame)
    return table


def time_report(frame):
    """Laufzeit der fertigen Jobs gegen Schaetzung (kernstunden) und neues Zeitlimit."""
    done = frame[(frame["state"] == "COMPLETED") & (frame["sekunden"] >= MIN_SECONDS)]
    timeouts = frame[frame["state"] == "TIMEOUT"]
    if done.empty and timeouts.empty:
        return
    done = done.assign(h=done["sekunden"] / 3600)
    done = done.assign(verhaeltnis=done["h"] / done["kernstunden"].clip(lower=0.01),
                       ausnutzung=done["h"] / done["zeitlimit"])
    print("\nLaufzeit fertiger Jobs (h = Wanduhr, verhaeltnis = h / Schaetzung, ausnutzung = h / neues Zeitlimit;"
          " Jobs mit --resume nach Abbruch laufen kuerzer):")
    print(done.groupby(["lauf", "method"]).agg(jobs=("h", "size"), h_median=("h", "median"), h_max=("h", "max"),
                                               verh_median=("verhaeltnis", "median"),
                                               verh_max=("verhaeltnis", "max"),
                                               ausn_max=("ausnutzung", "max")).round(2).to_string())
    over = done[done["h"] > done["zeitlimit"]]
    tight = done[done["ausnutzung"] > 0.8]
    print(f"Fertige Jobs, die das neue Zeitlimit ueberschritten haetten: {len(over)} von {len(done)}; ueber 80 % "
          f"des Limits: {len(tight)}"
          + ("" if over.empty else " (z. B. " + ", ".join(f"{r.name} {r.method} {r.h:.1f} h > {r.zeitlimit} h"
                                                           for r in over.head(5).itertuples()) + ")"))
    print(f"TIMEOUT: {len(timeouts)}" + ("" if timeouts.empty else " (z. B. " + ", ".join(
        f"{r.name} {r.method} Limit {r.limit_abgabe} h" for r in timeouts.head(5).itertuples()) + ")"))


SCANCEL = {"zu_wenig": "wartende Jobs mit weniger Speicher als im Plan (wuerden wohl OOM)",
           "zu_viel": "wartende Jobs mit mehr Speicher als im Plan (belegen unnoetig QOSMaxMemoryPerUser)"}


def pending_mismatch(run, jobs, tries):
    """Wartende Jobs, deren Abgabe weniger oder mehr Speicher hat als der Plan (jobs.mem_gb): je Richtung ein
    scancel-Skript jobs_paper/<lauf>/scancel_<richtung>_speicher.sh (alte Skripte werden geloescht, wenn nichts
    mehr abweicht). Gibt die Meldezeilen zurueck. Nach dem Abbrechen schickt refill.py die Jobs mit dem Plan neu ab;
    das Abbrechen eines wartenden Jobs zaehlt nicht als Fehlschlag (never_ran)."""
    found = {"zu_wenig": [], "zu_viel": []}
    for r in jobs[jobs["slurm"] == "PENDING"].itertuples():
        last = tries[r.line][-1]
        if last["mem"] is None:
            continue
        if float(last["mem"]) < float(r.mem_gb):
            found["zu_wenig"].append((last, float(r.mem_gb)))
        elif float(last["mem"]) > float(r.mem_gb):
            found["zu_viel"].append((last, float(r.mem_gb)))
    out = []
    for kind, hits in found.items():
        script = JOBS / run / f"scancel_{kind}_speicher.sh"
        if not hits:
            if script.exists():
                script.unlink()
            continue
        ids = [f"{a['id']}_{a['index']}" for a, _ in hits]
        script.write_text(f"#!/bin/bash\n# {SCANCEL[kind]}, geschrieben von watch_paper.py; danach schickt refill.py "
                          "sie mit dem Plan neu ab\n# (wartend abgebrochen = nie gelaufen: zaehlt nicht als "
                          "Fehlschlag)\n" + "".join(f"scancel {' '.join(ids[i:i + 200])}\n"
                                                    for i in range(0, len(ids), 200)))
        moves = Counter((speicher.fmt(a["mem"]), speicher.fmt(m), a["partition"]) for a, m in hits)
        freed = sum(float(a["mem"]) - m for a, m in hits)
        out.append(f"  {len(hits)} {SCANCEL[kind]}: " + ", ".join(
            f"{n}x {old}->{new} GB {part}" for (old, new, part), n in moves.most_common(6))
            + (f"; frei nach Neuabgabe {freed:.0f} GB" if kind == "zu_viel" else "") + f" -> bash {script}")
    return out


# ---------------------------------------------------------------- Prognose
JOB_NAMES = ("hpo-paper", "hpo-l1_270", "hpo-extra")
MIN_CALIBRATION_JOBS = 20     # so viele fertige Jobs braucht der Faktor tatsaechlich / Schaetzung


def progress(jobs, tries):
    """Spalten verbraucht_h (Laufzeit aller Versuche laut sacct, 1 Kern je Job) und rest_h (Schaetzung aus
    kosten.tsv fuer das, was fehlt: kernstunden x fehlende Folds / 5; fertig 0)."""
    used = [sum(a["sekunden"] for a in tries.get(line, [])) / 3600 for line in jobs["line"]]
    rest = []
    for datei, hours in zip(jobs["datei"], jobs["kernstunden"].astype(float)):
        match = re.search(r"\((\d)/5\)", datei)
        rest.append(0.0 if datei == "fertig" else hours * (FOLDS - int(match.group(1))) / FOLDS if match else hours)
    return jobs.assign(verbraucht_h=used, rest_h=rest)


def concurrent_cores(hours=24, now=None, sacct_text=None):
    """Mittlere Zahl gleichzeitig belegter Kerne der Rechenjobs (JOB_NAMES) in den letzten `hours` Stunden aus
    sacct (Start, End, NCPUS; laufende Jobs bis jetzt); None ohne sacct."""
    now = now or datetime.now()
    begin = now - timedelta(hours=hours)
    if sacct_text is None:
        cmd = ["sacct", "-n", "-P", "-X", "-S", begin.strftime("%Y-%m-%dT%H:%M:%S"), "-E",
               now.strftime("%Y-%m-%dT%H:%M:%S"), "--format=JobID,JobName,Start,End,NCPUS"]
        try:
            sacct_text = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    busy = 0.0
    for row in sacct_text.splitlines():
        _, name, start, end, ncpus = (row.split("|") + [""] * 5)[:5]
        if name not in JOB_NAMES:
            continue
        try:
            started = datetime.fromisoformat(start)
        except ValueError:                  # Unknown/None: nie gestartet (wartend)
            continue
        try:
            ended = datetime.fromisoformat(end)
        except ValueError:                  # Unknown: laeuft noch
            ended = now
        overlap = (min(ended, now) - max(started, begin)).total_seconds()
        if overlap > 0:
            busy += overlap * int(ncpus or 1)
    return busy / (hours * 3600)


def forecast(frames, cores, now=None):
    """frames: [(lauf, jobs mit welle, kernstunden, verbraucht_h, rest_h, datei)] in refill-Reihenfolge (paper vor
    l1_270). Je Lauf und Welle: verbraucht_h, rest_h, offene Jobs und fertig_am. Beide Laeufe teilen sich die Kerne,
    refill.py schickt Welle vor Welle und in einer Welle paper vor l1_270: fertig_am = jetzt + (kumulierter Rest bis
    zu dieser Welle und diesem Lauf) / cores, mindestens der laengste offene Job der Gruppe. fertig_am_kal: ebenso
    mit dem Rest mal Faktor (verbraucht / Schaetzung der fertigen Jobs), wenn es genug fertige Jobs gibt.
    Gibt (Tabelle, Faktor oder None) zurueck."""
    now = now or datetime.now()
    rows = []
    done_used = done_est = done_n = 0
    for nr, (run, jobs) in enumerate(frames):
        fertig = jobs["datei"] == "fertig"
        done_used += jobs.loc[fertig, "verbraucht_h"].sum()
        done_est += jobs.loc[fertig, "kernstunden"].astype(float).sum()
        done_n += int((fertig & (jobs["verbraucht_h"] > 0)).sum())
        for welle, g in jobs.groupby("welle"):
            rows.append(dict(lauf=run, welle=int(welle), _nr=nr, jobs=len(g), offen=int((g["rest_h"] > 0).sum()),
                             verbraucht_h=g["verbraucht_h"].sum(), rest_h=g["rest_h"].sum(),
                             _laengster=g["rest_h"].max()))
    table = pd.DataFrame(rows).sort_values(["welle", "_nr"], kind="stable").reset_index(drop=True)
    factor = done_used / done_est if done_n >= MIN_CALIBRATION_JOBS and done_est > 0 else None

    def dates(scale):
        if not cores or cores <= 0:
            return ["unbekannt"] * len(table)
        hours = np.maximum(table["rest_h"].cumsum() * scale / cores, table["_laengster"] * scale)
        return ["fertig" if r <= 0 else (now + timedelta(hours=float(h))).strftime("%Y-%m-%d %H:%M")
                for r, h in zip(table["rest_h"], hours)]
    table["fertig_am"] = dates(1.0)
    if factor is not None:
        table["fertig_am_kal"] = dates(factor)
    return table.drop(columns=["_nr", "_laengster"]), factor


def print_forecast(frames, cores):
    table, factor = forecast(frames, cores)
    print("===== Prognose (alle Laeufe teilen sich die Kerne; Reihenfolge wie refill.py: Welle, dann Lauf) =====")
    print("Kerne im Mittel der letzten 24 h (sacct, " + "/".join(JOB_NAMES) + "): "
          + ("unbekannt (kein sacct)" if cores is None else f"{cores:,.1f}"))
    out = table.assign(verbraucht_h=table["verbraucht_h"].round(0).astype(int),
                       rest_h=table["rest_h"].round(0).astype(int))
    print(out.rename(columns={"verbraucht_h": "verbraucht_kernh", "rest_h": "rest_kernh"}).to_string(index=False))
    print(f"Zusammen: verbraucht {table['verbraucht_h'].sum():,.0f} Kernstunden, verbleibend "
          f"{table['rest_h'].sum():,.0f} (Schaetzung kosten.tsv, angefangene Jobs anteilig nach fertigen Folds)")
    print("fertig_am = jetzt + kumulierter Rest / Kerne (mindestens der laengste offene Job); keine Wartezeit fuer "
          "Wiederholungen eingerechnet."
          + (f" fertig_am_kal: Rest x {factor:.2f} (verbraucht / Schaetzung der fertigen Jobs, inkl. "
             "Fehlversuche)." if factor is not None else
             f" (Kalibrierung ab {MIN_CALIBRATION_JOBS} fertigen Jobs mit sacct-Laufzeit)"))
    print()


def watch(run, per_task=False, max_tries=3, q=False):
    """q: squeue-DataFrame (queue()), None ohne squeue; False = selbst abfragen. squeue vor sacct, damit ein Job,
    der dazwischen endet, nicht als verschwunden gilt."""
    q = queue() if q is False else q
    jobs = load_jobs(run)
    tries, available = attempts(run)
    jobs = classify(jobs, tries, max_tries, queued_tasks(q))
    finished = jobs["datei"] == "fertig"
    print(f"===== Lauf {run}: {finished.sum()} von {len(jobs)} Jobs fertig ({finished.mean():.1%}) =====")
    print("Dateien:", dict(Counter(s.split(" (")[0] for s in jobs["datei"])))
    if available and tries:
        print("Slurm (letzte Abgabe je Job):", dict(Counter(jobs["slurm"])))
        used = sum(a["sekunden"] for t in tries.values() for a in t) / 3600
        print(f"Verbrauchte Kernstunden laut sacct: {used:,.0f} (Schaetzung fuer den ganzen Lauf: "
              f"{jobs['kernstunden'].sum():,.0f})")
    per_method = jobs.assign(fertig=finished).groupby("method")["fertig"].agg(["sum", "size"])
    print("Fertig je Methode:", ", ".join(f"{m} {r['sum']}/{r['size']}" for m, r in per_method.iterrows()))
    jobs["status"] = [status(d, s, l) for d, s, l in zip(jobs["datei"], jobs["slurm"], jobs["squeue"])]
    if not available:
        print("(ohne sacct: abgeschickte Jobs als 'unbekannt' gezaehlt, Wiederholungen nicht erkennbar)")
    waves = wave_table(jobs)
    print(waves.to_string(index=False))
    todo = jobs[jobs["aktion"].isin(["neu", "wiederholen"])]
    print(f"Noch nachzufuellen (refill.py): {len(todo)} Jobs (neu {int((todo['aktion'] == 'neu').sum())}, "
          f"Wiederholung {int((todo['aktion'] == 'wiederholen').sum())}), in der Schlange "
          f"{int((jobs['aktion'] == 'in_schlange').sum())}, aufgegeben {int((jobs['aktion'] == 'aufgegeben').sum())}")
    tasks = task_table(jobs)
    print("Aufgaben vollstaendig mit allen Seeds 1 bis 5w (fuer make_table5.py --seeds 1-<5w> --partial): "
          + ", ".join(f"w={w}: {int((tasks['bis'] >= w).sum())}/{len(tasks)}" for w in waves["welle"]))
    if per_task:
        print(tasks.drop(columns=["suite"]).to_string(index=False))
    else:
        open_waves = waves.loc[waves["fertig"] < waves["jobs"], "welle"]
        if len(open_waves):
            col = f"welle{int(open_waves.min())}"
            open_tasks = tasks[tasks[col] != "ok"]
            print(f"Erste unvollstaendige Welle {col[5:]}: {len(open_tasks)} Aufgaben offen, z. B. "
                  + ", ".join(f"{r.name} {getattr(r, col)}" for r in open_tasks.head(8).itertuples())
                  + "  (alle: --aufgaben)")
    bad = jobs[jobs["slurm"].isin(ENDED_BAD) & ~finished]
    for state, group in bad.groupby("slurm"):
        examples = ", ".join(f"Zeile {r.line} {r.method} {r.suite}_{r.task} seed {r.seed}"
                             for r in group.head(3).itertuples())
        print(f"  {state}: {len(group)} (z. B. {examples})")
    if len(bad):
        last = tries[bad.iloc[0]["line"]][-1]
        print(f"  Log des ersten: jobs_paper/logs/hpo-{run}_{last['id']}_{last['index']}.err")
    lost = jobs[(jobs["slurm"] == "UNBEKANNT") & (jobs["squeue"] == "") & ~finished]
    if available and len(lost):
        ids = sorted({tries[line][-1]["id"] for line in lost["line"]})
        print(f"  UNBEKANNT und nicht in squeue: {len(lost)} Jobs aus {len(ids)} Abgaben ({', '.join(ids[:5])}"
              f"{', ...' if len(ids) > 5 else ''}) -> refill.py reicht sie neu ein. Pruefen: sacct -X -P -j {ids[0]} "
              "--format=JobID,State,ElapsedRaw,Submit,End")
    for line in pending_mismatch(run, jobs, tries):
        print(line)
    gave_up = jobs[jobs["aktion"] == "aufgegeben"]
    if len(gave_up):
        print(f"  aufgegeben nach {max_tries} Versuchen (refill.py --max_versuche erhoehen oder Ursache beheben): "
              + ", ".join(f"Zeile {r.line} {r.method} {r.name} seed {r.seed} ({r.slurm})"
                          for r in gave_up.head(5).itertuples()))
    print()
    return progress(jobs, tries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=tuple(RUN_ORDER), action="append")
    parser.add_argument("--aufgaben", action="store_true", help="Stand je Aufgabe und Welle ausgeben")
    parser.add_argument("--speicher", action="store_true", help="MaxRSS je Aufgabe, schreibt speicher_euler.tsv")
    parser.add_argument("--max_versuche", type=int, default=3)
    args = parser.parse_args()
    runs = args.run or ["paper", "l1_270", *(["extra"] if (JOBS / "extra" / "jobs.tsv").exists() else [])]
    if args.speicher:
        memory_report(runs)
    else:
        print(disk_text())
        print(queue_text())
        runs = sorted(runs, key=RUN_ORDER.index)     # Reihenfolge wie refill.py (paper vor l1_270 vor extra)
        frames = [(run, watch(run, args.aufgaben, args.max_versuche)) for run in runs]   # je Lauf squeue vor sacct
        print_forecast(frames, concurrent_cores())
