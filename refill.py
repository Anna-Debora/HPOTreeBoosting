"""Nachfueller fuer die grossen Laeufe: schickt die Jobs aus jobs_paper/<lauf>/jobs.tsv in Portionen ab und haelt
die eigene Schlange (squeue --me -h -r | wc -l, jede Array-Aufgabe einzeln) unter --limit (Standard 940; Euler-QOS
erlaubt ~1000 Jobs, sonst QOSMaxSubmitJobPerUserLimit).

Je Job (Zeile von jobs.tsv) entscheidet watch_paper.classify():
  fertig        Ergebnisdatei vollstaendig                                    -> nichts
  in_schlange   letzte Abgabe wartet/laeuft oder ist sacct (noch) unbekannt   -> nichts (nie doppelt)
  neu           nie abgeschickt                                               -> abschicken mit Plan (kosten.tsv)
  wiederholen   letzte Abgabe beendet ohne fertige Datei (TIMEOUT, OUT_OF_MEMORY, FAILED, CANCELLED, ...)
                -> abschicken, nach TIMEOUT naechstlaengere Partition, nach OUT_OF_MEMORY naechste Speicherstufe
                   (4, 8, 12, 16, 24, 32 GB), nie weniger als der Plan; angefangene Dateien laufen ab dem ersten
                   fehlenden Fold weiter (run_ablation.py --resume)
  aufgegeben    schon --max_versuche Mal ohne Ergebnis beendet                 -> nichts, wird gemeldet
Abgaben stehen in jobs_paper/<lauf>/submitted.tsv (auch die alten aus submit.sh); jede Portion bekommt eine eigene,
nie wieder geaenderte idx-Datei jobs_paper/<lauf>/arrays/teile/<zeit>_<k>.idx.

Reihenfolge: Welle 1 vor 2 vor 3 vor 4; innerhalb einer Welle zuerst Wiederholungen, dann die schnellen
Partitionen (normal.4h, 24h, 120h), dann paper vor l1_270, dann kleine Aufgaben zuerst (Spalte reihenfolge).
--reserve N (Standard 100): in der ersten offenen Welle bekommt jede Partition mit offenen Jobs zuerst bis zu N
Plaetze in der Schlange (wartend + laufend), damit nicht nur normal.4h-Jobs warten, waehrend die 64-Kerne-Grenze je
Partition die anderen Partitionen leer laesst.

Aufruf (Euler, ~/HPOTreeBoosting, .venv aktiv):
    python refill.py --run beide                 # einmal nachfuellen (taeglich 1-2x von Hand)
    DRY=1 python refill.py --run beide           # nur anzeigen (auch: --dry)
    python refill.py --run paper --limit 600     # nur paper, kleinere Schlange
    python refill.py --run beide --daemon        # selbstfortsetzender Slurm-Job: alle 10 min nachfuellen
    python refill.py --stop                      # Daemon beenden (scancel -n hpo-refill)
Exit-Code 0: in Ordnung; 3: nichts mehr zu tun (alles fertig oder aufgegeben, nichts in der Schlange);
1: Fehler (z. B. sbatch abgelehnt, unbekannte hpo-Jobs in der Schlange, kein sacct).
"""
import argparse
import fcntl
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import watch_paper as wp

RUNS = ("paper", "l1_270")
MAX_ARRAY = 1000
HOURS = dict(wp.PARTITIONS)
DAEMON = "hpo-refill"
DAEMON_SCRIPT = "jobs_paper/refill_daemon.sbatch"
NOTHING_LEFT = 3


def candidates(runs, max_tries):
    """Alle Jobs der Laeufe mit Einordnung; bricht ab, wenn es Abgaben gibt, sacct aber fehlt."""
    frames = []
    for number, run in enumerate(runs):
        jobs = wp.load_jobs(run)
        tries, available = wp.attempts(run)
        if tries and not available:
            raise SystemExit(f"{run}: sacct nicht verfuegbar, abgeschickte Jobs nicht pruefbar -> Abbruch")
        jobs = wp.classify(jobs, tries, max_tries)
        frames.append(jobs.assign(lauf=run, lauf_nr=number))
    return pd.concat(frames, ignore_index=True)


def order(todo):
    """Abgabe-Reihenfolge (siehe Kopf)."""
    return todo.assign(_retry=todo["aktion"] != "wiederholen", _part=todo["partition_neu"].map(HOURS)) \
        .sort_values(["welle", "_retry", "_part", "lauf_nr", "reihenfolge"], kind="stable")


def select(jobs, free, queued_per_partition=None, reserve=100):
    """Die naechsten hoechstens `free` Jobs (neu/wiederholen) in Abgabe-Reihenfolge, mit Reserve je Partition."""
    todo = order(jobs[jobs["aktion"].isin(["neu", "wiederholen"])])
    if free <= 0 or todo.empty:
        return todo.iloc[:0]
    picked = []
    if reserve > 0:
        first = todo[todo["welle"] == todo["welle"].min()]
        for part, group in first.groupby("partition_neu", sort=False):
            room = reserve - (queued_per_partition or {}).get(part, 0)
            picked += list(group.index[:max(0, room)])
        picked = [i for i in todo.index if i in set(picked)][:free]
    rest = [i for i in todo.index if i not in set(picked)]
    chosen = set(picked + rest[:free - len(picked)])
    return todo.loc[[i for i in todo.index if i in chosen]]


def portions(chosen):
    """[(lauf, partition, stunden, mem, [Zeilen])] je Welle, in Reihenfolge des ersten Jobs, je hoechstens MAX_ARRAY."""
    out = []
    keys = ["welle", "lauf", "partition_neu", "stunden_neu", "mem_neu"]     # Welle zuerst: Welle 2 nie vor Welle 1
    position = {i: k for k, i in enumerate(chosen.index)}
    for key, group in sorted(chosen.groupby(keys, sort=False), key=lambda kv: position[kv[1].index[0]]):
        lines = group["line"].tolist()
        for start in range(0, len(lines), MAX_ARRAY):
            out.append((*key[1:], lines[start:start + MAX_ARRAY]))
    return out


def sbatch_command(run, part, hours, mem, idx, n):
    return ["sbatch", "--parsable", f"--array=0-{n - 1}", f"--partition={part}", f"--time={int(hours)}:00:00",
            f"--mem-per-cpu={int(mem)}G", f"--job-name=hpo-{run}", "jobs_paper/run_paper_job.sbatch", run, str(idx)]


def submit(portion, stamp, k, dry):
    """Eine Portion abschicken; True bei Erfolg."""
    run, part, hours, mem, lines = portion
    idx = wp.JOBS / run / "arrays" / "teile" / f"{stamp}_{k}.idx"
    while idx.exists():                       # idx-Dateien sind in submitted.tsv referenziert: nie ueberschreiben
        k = f"{k}x"
        idx = idx.with_name(f"{stamp}_{k}.idx")
    cmd = sbatch_command(run, part, hours, mem, idx, len(lines))
    if dry:
        print("  DRY:", " ".join(cmd), f"  ({len(lines)} Jobs, Zeilen {lines[0]}...{lines[-1]})")
        return True
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text("\n".join(map(str, lines)) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        idx.unlink()
        print(f"  sbatch abgelehnt ({run}, {part}, {mem} GB, {len(lines)} Jobs): {result.stderr.strip()}")
        return False
    job_id = result.stdout.strip().split(";")[0]
    with open(wp.JOBS / run / "submitted.tsv", "a") as handle:
        handle.write("\t".join(map(str, [job_id, idx, len(lines), datetime.now().strftime("%F %T"), part,
                                         int(hours), int(mem)])) + "\n")
    print(f"  abgeschickt: Job {job_id}, {len(lines)} Jobs, {run} {part} {hours} h {mem} GB")
    return True


def unknown_jobs(q, runs):
    """hpo-Jobs in der Schlange, die in keiner submitted.tsv stehen (z. B. von Hand abgeschickt)."""
    known = {s["id"] for run in RUNS for s in wp.submissions(run)}
    names = {f"hpo-{run}" for run in RUNS}
    mine = q[q["name"].isin(names)]
    return sorted({i.split("_")[0] for i in mine["id"]} - known)


def refill(runs, limit, reserve, max_tries, dry):
    q = wp.queue()
    if q is None:
        if not dry:
            print("squeue nicht verfuegbar -> Abbruch (nur auf Euler moeglich; Vorschau: --dry)")
            return 1
        q = pd.DataFrame(columns=["id", "name", "partition", "state"])
    unknown = unknown_jobs(q, runs)
    if unknown:
        print(f"Unbekannte hpo-Jobs in der Schlange (nicht in submitted.tsv): {unknown[:10]} -> Abbruch, sonst "
              "koennten Jobs doppelt laufen. Pruefen mit: squeue --me")
        return 1
    jobs = candidates(runs, max_tries)
    queued_per_partition = q["partition"].value_counts().to_dict()
    free = limit - len(q)
    chosen = select(jobs, free, queued_per_partition, reserve)
    counts = jobs["aktion"].value_counts().to_dict()
    print(f"{datetime.now():%F %T} Schlange {len(q)}/{limit}, frei {max(free, 0)}; Jobs: "
          + ", ".join(f"{k} {counts.get(k, 0)}" for k in ("fertig", "in_schlange", "neu", "wiederholen",
                                                         "aufgegeben")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sent = 0
    for k, portion in enumerate(portions(chosen)):
        if not submit(portion, stamp, k, dry):
            return 1
        sent += len(portion[4])
    todo = int(jobs["aktion"].isin(["neu", "wiederholen"]).sum())
    print(f"Abgeschickt {sent}, danach noch nachzufuellen {todo - sent}"
          + (f" (Welle {int(chosen['welle'].min())}-{int(chosen['welle'].max())})" if sent else ""))
    gave_up = jobs[jobs["aktion"] == "aufgegeben"]
    if len(gave_up):
        print(f"Aufgegeben ({len(gave_up)}): " + ", ".join(f"{r.lauf} Zeile {r.line} {r.method} {r.name} ({r.slurm})"
                                                           for r in gave_up.head(5).itertuples()))
    if todo - sent == 0 and not (jobs["aktion"] == "in_schlange").any() and sent == 0:
        print("Nichts mehr zu tun.")
        return NOTHING_LEFT
    return 0


def daemon_running():
    q = wp.queue()
    return [] if q is None else q.loc[q["name"] == DAEMON, "id"].tolist()


def start_daemon(args):
    running = daemon_running()
    if running:
        print(f"Daemon laeuft schon oder wartet: {running} (beenden: python refill.py --stop)")
        return 1
    passed = ["--run", args.run, "--limit", str(args.limit), "--reserve", str(args.reserve),
              "--max_versuche", str(args.max_versuche)]
    cmd = ["sbatch", "--parsable", DAEMON_SCRIPT, *passed]
    if args.dry:
        print("DRY:", " ".join(cmd))
        return 0
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode == 0:
        print("Daemon abgeschickt; Log: jobs_paper/logs/refill_<Job-ID>.out; beenden: python refill.py --stop")
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=("paper", "l1_270", "beide"), default="beide")
    parser.add_argument("--limit", type=int, default=940, help="hoechstens so viele eigene Jobs in der Schlange")
    parser.add_argument("--reserve", type=int, default=100, help="Plaetze je Partition in der ersten offenen Welle")
    parser.add_argument("--max_versuche", type=int, default=3, help="Wiederholungen je Job bis 'aufgegeben'")
    parser.add_argument("--dry", action="store_true", default=bool(os.environ.get("DRY")), help="nur anzeigen")
    parser.add_argument("--daemon", action="store_true", help="als selbstfortsetzenden Slurm-Job starten")
    parser.add_argument("--stop", action="store_true", help="Daemon beenden")
    args = parser.parse_args(argv)
    if args.stop:
        ids = daemon_running()
        if ids:
            subprocess.run(["scancel", "-n", DAEMON])
        print(f"beendet: {ids}" if ids else "kein Daemon in der Schlange")
        return 0
    if args.daemon:
        return start_daemon(args)
    runs = list(RUNS) if args.run == "beide" else [args.run]
    if args.dry:
        return refill(runs, args.limit, args.reserve, args.max_versuche, dry=True)
    wp.JOBS.mkdir(exist_ok=True)
    with open(wp.JOBS / ".refill.lock", "w") as lock:
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)      # POSIX-Sperre, geht auch auf NFS ($HOME)
        except OSError:
            print("Ein anderer refill.py laeuft gerade (Sperre jobs_paper/.refill.lock) -> nichts getan")
            return 0
        return refill(runs, args.limit, args.reserve, args.max_versuche, dry=False)


if __name__ == "__main__":
    sys.exit(main())
