"""Stand der grossen Laeufe (make_paper_jobs.py): pro Lauf und Welle fertig / laufend / wartend / fehlgeschlagen /
fehlt, was refill.py noch nachfuellen muss (neu, Wiederholung, aufgegeben) und die Laenge der Schlange.
Die Logik hier (Abgaben aus submitted.tsv, sacct, Einordnung je Job) benutzt auch refill.py.

Aufruf auf Euler (aus ~/HPOTreeBoosting, mit aktivierter .venv):
    python watch_paper.py                 # beide Laeufe
    python watch_paper.py --run paper     # nur einer
    python watch_paper.py --aufgaben      # zusaetzlich je Aufgabe und Welle: fertige Jobs / alle, "ok" = vollstaendig
    python watch_paper.py --speicher      # MaxRSS je Aufgabe aus sacct gegen den angeforderten Speicher, OOM-Faelle;
                                          # schreibt jobs_paper/speicher_euler.tsv (refill.py nimmt den hoeheren Wert)
Ohne sacct (z. B. auf dem Mac nach dem Kopieren der Ergebnisse) nur der Dateistand (laufend/wartend dann unbekannt).

submitted.tsv (je Abgabe eine Zeile, ohne Kopf): Job-ID, idx-Datei, Zahl der Teiljobs, Zeit, Partition, Stunden,
Speicher GB. Aeltere Zeilen (submit.sh vor refill.py) haben nur die ersten vier Spalten; Partition, Stunden und
Speicher kommen dann aus arrays.tsv (eingefrorene Wellen-Arrays), aus dem Namen (retry_<part>_<mem>G_<h>h) oder
sind die der Probe. Array-Index i einer Abgabe = Zeile i+1 der idx-Datei = Zeilennummer in jobs.tsv.
"""
import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from run_ablation import expected_rows

JOBS = Path("jobs_paper")
COLUMNS = ["method", "seed", "suite", "task", "trials", "folder", "flags"]
SUBMITTED = ["id", "idx", "n", "zeit", "partition", "stunden", "mem"]
PARTITIONS = (("normal.4h", 4), ("normal.24h", 24), ("normal.120h", 120))
MEM_STEPS = (4, 8, 12, 16, 24, 32)
FOLDS = 5
QUEUED = ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "COMPLETING", "CONFIGURING", "RESIZING", "REQUEUE_HOLD",
          "REQUEUE_FED", "SIGNALING", "STAGE_OUT")
ENDED_BAD = ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"]
STATES = ["fertig", "laufend", "wartend", "fehlgeschlagen", "fehlt"]


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
            sub["stunden"], sub["mem"] = int(sub["stunden"]), int(sub["mem"])
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


def escalate(last, state, plan):
    """Ressourcen fuer die Wiederholung: nach TIMEOUT naechstlaengere Partition, nach OUT_OF_MEMORY naechste
    Speicherstufe, sonst wie zuletzt; nie weniger als der aktuelle Plan (kosten.tsv). last/plan: dict(partition,
    stunden, mem)."""
    part, hours, mem = last["partition"] or plan["partition"], last["stunden"] or plan["stunden"], \
        last["mem"] or plan["mem"]
    if state == "TIMEOUT":
        bigger = [p for p in PARTITIONS if p[1] > hours]
        part, hours = bigger[0] if bigger else PARTITIONS[-1]
    if state == "OUT_OF_MEMORY":
        mem = next((m for m in MEM_STEPS if m > mem), MEM_STEPS[-1])
    if plan["stunden"] > hours:
        part, hours = plan["partition"], plan["stunden"]
    return dict(partition=part, stunden=int(hours), mem=int(max(mem, plan["mem"])))


def classify(jobs, tries, max_tries=3):
    """Spalten 'slurm' (Zustand der letzten Abgabe), 'versuche' (beendet ohne Ergebnis), 'aktion' (fertig /
    in_schlange / neu / wiederholen / aufgegeben) und die Ressourcen der naechsten Abgabe (partition_neu,
    stunden_neu, mem_neu). jobs braucht line, datei, partition, zeitlimit, mem_gb."""
    slurm, failed, action, res = [], [], [], []
    for job in jobs.itertuples():
        t = tries.get(job.line, [])
        last = t[-1] if t else None
        state = last["state"] if last else None
        n_failed = sum(1 for a in t if a["state"] in ENDED_BAD + ["COMPLETED"])
        plan = dict(partition=job.partition, stunden=int(job.zeitlimit), mem=int(job.mem_gb))
        slurm.append("nicht abgeschickt" if not t else (state or "UNBEKANNT"))
        failed.append(n_failed)
        if job.datei == "fertig":
            action.append("fertig"), res.append(plan)
        elif not t:
            action.append("neu"), res.append(plan)
        elif state is None or state in QUEUED:          # unbekannt zaehlt als in der Schlange (nie doppelt)
            action.append("in_schlange"), res.append(plan)
        elif n_failed >= max_tries:
            action.append("aufgegeben"), res.append(plan)
        else:
            action.append("wiederholen"), res.append(escalate(last, state, plan))
    out = jobs.copy()
    out["slurm"], out["versuche"], out["aktion"] = slurm, failed, action
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


def euler_memory(path=JOBS / "speicher_euler.tsv"):
    """{(suite, task): GB} aus watch_paper.py --speicher (nur auf Euler, nicht im Repository)."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, sep="\t")
    return {(int(r.suite), int(r.task)): int(r.mem_gb) for r in frame.itertuples()}


def load_jobs(run, with_files=True):
    """jobs.tsv mit Plan aus kosten.tsv (welle, partition, zeitlimit, mem_gb, reihenfolge) und Dateistand.
    mem_gb: hoeherer Wert aus kosten.tsv und jobs_paper/speicher_euler.tsv (nicht fuer Default)."""
    jobs = pd.read_csv(JOBS / run / "jobs.tsv", sep="\t", header=None, names=COLUMNS, keep_default_na=False)
    costs = pd.read_csv(JOBS / run / "kosten.tsv", sep="\t", keep_default_na=False)
    for column in ("partition", "mem_gb", "kernstunden", "name", "stunden", "welle", "zeitlimit", "reihenfolge"):
        jobs[column] = costs[column] if column in costs else (costs["stunden"] if column == "zeitlimit" else 0)
    override = euler_memory()
    if override:
        jobs["mem_gb"] = [m if meth == "default" else max(m, override.get((s, t), 0))
                          for m, meth, s, t in zip(jobs["mem_gb"], jobs["method"], jobs["suite"], jobs["task"])]
    jobs["line"] = jobs.index + 1
    jobs["datei"] = [file_state(j) for j in jobs.itertuples()] if with_files else "fehlt"
    return jobs


# ---------------------------------------------------------------- Uebersichten
def status(datei, slurm):
    """Zustand eines Jobs fuer die Wellen-Uebersicht."""
    if datei == "fertig":
        return "fertig"
    if slurm in ("RUNNING", "COMPLETING"):
        return "laufend"
    if slurm in ("PENDING", "REQUEUED", "SUSPENDED", "UNBEKANNT"):
        return "wartend"
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


def queue_text(limit=940):
    q = queue()
    if q is None:
        return "Schlange: unbekannt (kein squeue)"
    by_name = Counter(q["name"])
    states = Counter(q["state"])
    return (f"Schlange: {len(q)} Jobs (Limit refill.py {limit}; QOS ~1000), davon "
            + ", ".join(f"{k} {v}" for k, v in sorted(by_name.items())) + " | "
            + ", ".join(f"{k} {v}" for k, v in sorted(states.items())))


def memory_report(runs):
    """MaxRSS je Aufgabe (sacct, Schritt .batch) gegen den angeforderten Speicher; schreibt speicher_euler.tsv
    mit der Stufe ueber 1.3 x MaxRSS bzw. ueber dem Speicher eines OOM-Versuchs."""
    rows = []
    for run in runs:
        jobs = load_jobs(run, with_files=False).set_index("line")
        subs = submissions(run)
        text = run_sacct(sorted({s["id"] for s in subs}), fmt="JobID,State,MaxRSS,ReqMem", steps=True)
        if text is None:
            print("kein sacct")
            return
        rss, oom = {}, set()
        for row in text.splitlines():
            job_id, state, maxrss = (row.split("|") + ["", "", ""])[:3]
            base, _, rest = job_id.partition("_")
            index = rest.split(".")[0]
            if not index.isdigit():
                continue
            key = (base, int(index))
            if state.startswith("OUT_OF_ME"):
                oom.add(key)
            if maxrss:
                unit = {"K": 1e-6, "M": 1e-3, "G": 1.0}.get(maxrss[-1], 1e-9)
                value = float(maxrss[:-1] if maxrss[-1] in "KMG" else maxrss) * unit
                rss[key] = max(rss.get(key, 0.0), value)
        for sub in subs:
            for i, line in enumerate(sub["lines"]):
                if (sub["id"], i) in rss or (sub["id"], i) in oom:
                    job = jobs.loc[line]
                    rows.append(dict(lauf=run, suite=job.suite, task=job.task, name=job["name"], method=job.method,
                                     maxrss_gb=rss.get((sub["id"], i), 0.0), angefordert=sub["mem"],
                                     oom=(sub["id"], i) in oom))
    if not rows:
        print("Noch keine beendeten Jobs mit MaxRSS.")
        return
    frame = pd.DataFrame(rows)
    per_task = frame.groupby(["suite", "task", "name"]).agg(
        jobs=("maxrss_gb", "size"), maxrss_gb=("maxrss_gb", "max"), angefordert=("angefordert", "max"),
        oom=("oom", "sum"), oom_bei=("angefordert", lambda s: s[frame.loc[s.index, "oom"]].max())).reset_index()
    per_task["oom_bei"] = per_task["oom_bei"].fillna(0)
    per_task["vorschlag"] = [next((m for m in MEM_STEPS if m >= 1.3 * r.maxrss_gb and m > r.oom_bei),
                                  MEM_STEPS[-1]) for r in per_task.itertuples()]
    per_task = per_task.sort_values("maxrss_gb", ascending=False)
    print(per_task.round(2).to_string(index=False))
    out = per_task[per_task["vorschlag"] > 4][["suite", "task", "name", "vorschlag"]].rename(
        columns={"vorschlag": "mem_gb"})
    out.to_csv(JOBS / "speicher_euler.tsv", sep="\t", index=False)
    print(f"\n{JOBS / 'speicher_euler.tsv'}: {len(out)} Aufgaben ueber 4 GB (refill.py nimmt max(Plan, diese Werte))")


def watch(run, per_task=False, max_tries=3):
    jobs = load_jobs(run)
    tries, available = attempts(run)
    jobs = classify(jobs, tries, max_tries)
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
    jobs["status"] = [status(d, s) for d, s in zip(jobs["datei"], jobs["slurm"])]
    if not available:
        print("(ohne sacct: abgeschickte Jobs als 'wartend' gezaehlt, Wiederholungen nicht erkennbar)")
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
    low = [(r.line, tries[r.line][-1]) for r in jobs[jobs["slurm"] == "PENDING"].itertuples()
           if (tries[r.line][-1]["mem"] or 0) < r.mem_gb]
    if low:
        script = JOBS / run / "scancel_zu_wenig_speicher.sh"
        ids = [f"{a['id']}_{a['index']}" for _, a in low]
        script.write_text("#!/bin/bash\n# wartende Jobs mit weniger Speicher als im Plan; refill.py schickt sie danach "
                          "mit dem Plan neu ab\n" + "".join(f"scancel {' '.join(ids[i:i + 200])}\n"
                                                              for i in range(0, len(ids), 200)))
        print(f"  {len(low)} wartende Jobs mit weniger Speicher als im Plan (wuerden wohl OOM): bash {script}")
    gave_up = jobs[jobs["aktion"] == "aufgegeben"]
    if len(gave_up):
        print(f"  aufgegeben nach {max_tries} Versuchen (refill.py --max_versuche erhoehen oder Ursache beheben): "
              + ", ".join(f"Zeile {r.line} {r.method} {r.name} seed {r.seed} ({r.slurm})"
                          for r in gave_up.head(5).itertuples()))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("paper", "l1_270"), action="append")
    parser.add_argument("--aufgaben", action="store_true", help="Stand je Aufgabe und Welle ausgeben")
    parser.add_argument("--speicher", action="store_true", help="MaxRSS je Aufgabe, schreibt speicher_euler.tsv")
    parser.add_argument("--max_versuche", type=int, default=3)
    args = parser.parse_args()
    runs = args.run or ["paper", "l1_270"]
    if args.speicher:
        memory_report(runs)
    else:
        print(queue_text())
        for run in runs:
            watch(run, args.aufgaben, args.max_versuche)
