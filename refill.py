"""Nachfueller fuer die grossen Laeufe: schickt die Jobs aus jobs_paper/<lauf>/jobs.tsv in Portionen ab und haelt
die eigene Schlange (squeue --me -h -r | wc -l, jede Array-Aufgabe einzeln) unter --limit (Standard 4000, hoechstens
15000; QOS des Accounts ls_math: MaxSubmit 17000 Jobs je Nutzer, sonst QOSMaxSubmitJobPerUserLimit). Jeder sbatch
(Rechenjobs, Daemon, Nachfolger des Daemons) geht mit --account (Standard ls_math).

Je Job (Zeile von jobs.tsv) entscheidet watch_paper.classify():
  fertig        Ergebnisdatei vollstaendig                                    -> nichts
  in_schlange   letzte Abgabe steht in squeue oder ist laut sacct PENDING/RUNNING/...   -> nichts (nie doppelt)
  neu           nie abgeschickt                                               -> abschicken mit Plan
  wiederholen   letzte Abgabe nicht in squeue und beendet ohne fertige Datei (TIMEOUT, OUT_OF_MEMORY, FAILED,
                CANCELLED, NODE_FAIL, ...) oder sacct unbekannt (z. B. wartend abgebrochen und nie in sacct)
                -> abschicken mit Plan, nach TIMEOUT naechstlaengere Partition, nach OUT_OF_MEMORY naechste
                   Speicherstufe ueber dem groessten OOM-Versuch (speicher.MEM_STEPS: 1, 1.5, 2, 3, 4, 6, 8, 12, 16,
                   24, 32 GB); angefangene Dateien laufen ab dem ersten fehlenden Fold weiter (--resume)
  aufgegeben    schon --max_versuche Mal ohne Ergebnis beendet                 -> nichts, wird gemeldet
                (wartend abgebrochene Jobs, CANCELLED mit Laufzeit 0, zaehlen nicht: watch_paper.never_ran)
Plan: Partition und Zeitlimit aus kosten.tsv, Speicher je Aufgabe und Methode nach speicher.py (Euler-MaxRSS aus
jobs_paper/speicher_euler.tsv, sonst Mac-Messung). Je Portion ein Array mit gleicher Welle, Lauf, Partition,
Zeitlimit und Speicher; --mem-per-cpu in GB oder bei halben Stufen in MB (1.5 GB -> 1536M).
Abgaben stehen in jobs_paper/<lauf>/submitted.tsv (auch die alten aus submit.sh); jede Portion bekommt eine eigene,
nie wieder geaenderte idx-Datei jobs_paper/<lauf>/arrays/teile/<zeit>_<k>.idx.

Reihenfolge: Welle 1 vor 2 vor 3 vor 4; innerhalb einer Welle zuerst Wiederholungen, dann die schnellen
Partitionen (normal.4h, 24h, 120h), dann paper vor l1_270, dann kleine Aufgaben zuerst (Spalte reihenfolge).
--reserve N (Standard 100): in der ersten offenen Welle bekommt jede Partition mit offenen Jobs zuerst bis zu N
Plaetze in der Schlange (wartend + laufend), damit nicht nur normal.4h-Jobs warten, waehrend die Kerngrenze je
Partition (ls_math: 26752 / 14045 / 8026 Kerne in normal.4h / 24h / 120h) die anderen Partitionen leer laesst.

Lauf extra (BOHB, DEHB, HEBO; make_paper_jobs.py --run extra): --run extra oder --run alle (= paper, l1_270, extra).
"beide" bleibt paper + l1_270 (ein laufender Daemon mit --run beide schickt nichts von extra ab). Extra-Jobs gehen
mit jobs_paper/run_extra_job.sbatch (.venv_extra) und dem Namen hpo-extra ab; Schlangen-Limit, Reserve, Wellen,
Wiederholungen, Speicher- und Zeitregeln wie oben (eine gemeinsame Schlange fuer alle Laeufe). In einer Welle kommt
extra nach paper und l1_270.

Lauf gpb (korrekte Skalen ohne lambda_l1: TPE, GP-BO, SMAC, Hyperband; make_paper_jobs.py --run gpb): nur mit
--run gpb oder in einer Liste mit Komma (z. B. --run extra,gpb); "beide" und "alle" bleiben wie bisher, damit ein
laufender Daemon nach git pull nichts von gpb abschickt. gpb-Jobs gehen mit jobs_paper/run_paper_job.sbatch (.venv)
und dem Namen hpo-gpb ab; --l1 off und --scales gpboost stehen in jobs_paper/gpb/jobs.tsv (Spalte flags).

Aufruf (Euler, ~/HPOTreeBoosting, .venv aktiv):
    python refill.py --run beide                 # einmal nachfuellen (taeglich 1-2x von Hand)
    python refill.py --run extra                 # nur den Lauf extra
    python refill.py --run gpb                   # nur den Lauf gpb
    python refill.py --run alle,gpb --daemon     # Daemon fuer paper, l1_270, extra und gpb
    python refill.py --run alle --daemon         # Daemon fuer alle drei Laeufe (vorher: python refill.py --stop)
    DRY=1 python refill.py --run beide           # nur anzeigen (auch: --dry)
    python refill.py --run paper --limit 600     # nur paper, kleinere Schlange
    python refill.py --run beide --account es_mge  # anderer Slurm-Account
    python refill.py --run beide --daemon        # selbstfortsetzender Slurm-Job: alle 10 min nachfuellen,
                                                 # Standard normal.4h mit 4:00:00, reiht sich selbst neu ein
    python refill.py --run beide --daemon --daemon-partition normal.24h --daemon-zeit 24:00:00
    python refill.py --stop                      # alle hpo-refill-Jobs beenden (laufend und wartend, jede
                                                 # Partition), prueft danach, dass keiner mehr da ist
Speicherplatz (quota.py): vor jeder Einreichung lquota (sonst df). Home ueber 80 % der Soft Quota (Platz oder Dateien)
-> nichts einreichen, deutliche Warnung, Daemon beenden (jobs_paper/refill.stop, Exit-Code 4); Scratch ueber 80 %
-> nur Warnung. Unbekannte Belegung (kein lquota, kein df) blockiert nicht.
Exit-Code 0: in Ordnung; 3: nichts mehr zu tun (alles fertig oder aufgegeben, nichts in der Schlange);
4: Home-Quota fast voll (nichts eingereicht, Daemon beendet);
1: Fehler (z. B. sbatch abgelehnt, unbekannte hpo-Jobs in der Schlange, kein sacct).
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import quota
import speicher
import watch_paper as wp

RUNS = ("paper", "l1_270")                 # --run beide
ALL_RUNS = RUNS + ("extra",)               # --run alle
KNOWN_RUNS = ALL_RUNS + ("gpb",)           # alle Laeufe mit jobs_paper/<lauf>/submitted.tsv (gpb nur ausdruecklich)
RUN_GROUPS = {"beide": RUNS, "alle": ALL_RUNS}
SBATCH_SCRIPTS = {"extra": "jobs_paper/run_extra_job.sbatch"}      # sonst jobs_paper/run_paper_job.sbatch
MAX_ARRAY = 1000
HOURS = dict(wp.PARTITIONS)
DAEMON = "hpo-refill"
DAEMON_SCRIPT = "jobs_paper/refill_daemon.sbatch"
NOTHING_LEFT = 3
QUOTA_FULL = 4
DAEMON_PARTITION, DAEMON_TIME = "normal.4h", "4:00:00"
DAEMON_MARGIN_S = 20 * 60          # so lange vor dem Zeitlimit endet der Daemon (refill_daemon.sbatch)


def candidates(runs, max_tries, queued=None):
    """Alle Jobs der Laeufe mit Einordnung; bricht ab, wenn es Abgaben gibt, sacct aber fehlt. queued: Aufgaben in
    squeue (wp.queued_tasks), None = unbekannt (dann zaehlt eine sacct unbekannte Abgabe als in der Schlange)."""
    frames = []
    for number, run in enumerate(runs):
        jobs = wp.load_jobs(run)
        tries, available = wp.attempts(run)
        if tries and not available:
            raise SystemExit(f"{run}: sacct nicht verfuegbar, abgeschickte Jobs nicht pruefbar -> Abbruch")
        jobs = wp.classify(jobs, tries, max_tries, queued)
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


def sbatch_command(run, part, hours, mem, idx, n, account=wp.ACCOUNT):
    return ["sbatch", "--parsable", f"--account={account}", f"--array=0-{n - 1}", f"--partition={part}",
            f"--time={int(hours)}:00:00",
            f"--mem-per-cpu={speicher.sbatch_mem(mem)}", f"--job-name=hpo-{run}",
            SBATCH_SCRIPTS.get(run, "jobs_paper/run_paper_job.sbatch"), run, str(idx)]      # gpb: run_paper_job


def submit(portion, stamp, k, dry, account=wp.ACCOUNT):
    """Eine Portion abschicken; True bei Erfolg."""
    run, part, hours, mem, lines = portion
    idx = wp.JOBS / run / "arrays" / "teile" / f"{stamp}_{k}.idx"
    while idx.exists():                       # idx-Dateien sind in submitted.tsv referenziert: nie ueberschreiben
        k = f"{k}x"
        idx = idx.with_name(f"{stamp}_{k}.idx")
    cmd = sbatch_command(run, part, hours, mem, idx, len(lines), account)
    if dry:
        print("  DRY:", " ".join(cmd), f"  ({len(lines)} Jobs, Zeilen {lines[0]}...{lines[-1]})")
        return True
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text("\n".join(map(str, lines)) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        idx.unlink()
        print(f"  sbatch abgelehnt ({run}, {part}, {speicher.fmt(mem)} GB, {len(lines)} Jobs): "
              f"{result.stderr.strip()}")
        return False
    job_id = result.stdout.strip().split(";")[0]
    with open(wp.JOBS / run / "submitted.tsv", "a") as handle:
        handle.write("\t".join(map(str, [job_id, idx, len(lines), datetime.now().strftime("%F %T"), part,
                                         int(hours), speicher.fmt(mem)])) + "\n")
    print(f"  abgeschickt: Job {job_id}, {len(lines)} Jobs, {run} {part} {hours} h {speicher.fmt(mem)} GB")
    return True


def unknown_jobs(q, runs):
    """hpo-Jobs in der Schlange, die in keiner submitted.tsv stehen (z. B. von Hand abgeschickt). Geprueft werden
    die Namen von paper und l1_270 und der verlangten Laeufe (mit --run beide also wie bisher nur diese zwei)."""
    known = {s["id"] for run in KNOWN_RUNS for s in wp.submissions(run)}
    names = {f"hpo-{run}" for run in (*RUNS, *runs)}
    mine = q[q["name"].isin(names)]
    return sorted({i.split("_")[0] for i in mine["id"]} - known)


def disk_ok(dry):
    """Home/Scratch pruefen; bei vollem Home (nicht --dry) den Daemon ueber jobs_paper/refill.stop beenden (der
    laufende Daemon prueft die Datei vor jedem Durchgang und streicht seinen Nachfolger)."""
    ok, lines = quota.check()
    for line in lines:
        print(("!" * 100 + "\n" if line.startswith("WARNUNG") else "") + line, flush=True)
    if not ok and not dry:
        try:
            (wp.JOBS / "refill.stop").touch()
            print("Daemon wird beendet (jobs_paper/refill.stop); nichts eingereicht.\n" + "!" * 100, flush=True)
        except OSError as error:
            print(f"jobs_paper/refill.stop nicht schreibbar ({error}); Daemon von Hand beenden: "
                  "python refill.py --stop\n" + "!" * 100, flush=True)
    return ok


def refill(runs, limit, reserve, max_tries, dry, account=wp.ACCOUNT):
    if not disk_ok(dry):
        return QUOTA_FULL
    q = wp.queue()                       # vor sacct: ein Job, der dazwischen endet, gilt nicht als verschwunden
    no_squeue = q is None
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
    jobs = candidates(runs, max_tries, None if no_squeue else wp.queued_tasks(q))
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
        if not submit(portion, stamp, k, dry, account):
            return 1
        sent += len(portion[4])
    todo = int(jobs["aktion"].isin(["neu", "wiederholen"]).sum())
    print(f"Abgeschickt {sent}, danach noch nachzufuellen {todo - sent}"
          + (f" (Welle {int(chosen['welle'].min())}-{int(chosen['welle'].max())})" if sent else ""))
    lost = jobs[(jobs["aktion"] == "wiederholen") & jobs["slurm"].isin(["UNBEKANNT", "CANCELLED"])]
    if len(lost):
        print(f"Darunter nicht in squeue und sacct UNBEKANNT/CANCELLED (z. B. wartend abgebrochen): "
              + ", ".join(f"{run} {n}" for run, n in lost.groupby("lauf").size().items()))
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


def time_seconds(text):
    """Slurm-Zeit 'H:MM:SS' oder 'D-HH:MM:SS' in Sekunden; ValueError bei anderer Form (Slurm liest 'H:MM' als
    Minuten:Sekunden, deshalb nicht erlaubt)."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", text)
    if not match:
        raise ValueError(f"Zeitlimit {text!r}: erwartet H:MM:SS oder D-HH:MM:SS, z. B. 4:00:00")
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def daemon_command(args):
    """sbatch-Befehl fuer den Daemon; Partition und Zeitlimit gehen als Umgebung an das Skript und seine
    Nachfolger (die sich mit denselben Werten neu einreihen)."""
    seconds = time_seconds(args.daemon_zeit)
    longest = dict(wp.PARTITIONS).get(args.daemon_partition)
    if longest is None:
        raise ValueError(f"Partition {args.daemon_partition!r}: erlaubt {', '.join(dict(wp.PARTITIONS))}")
    if seconds > longest * 3600:
        raise ValueError(f"Zeitlimit {args.daemon_zeit} laenger als {args.daemon_partition} ({longest} h)")
    if seconds <= DAEMON_MARGIN_S + 600:
        raise ValueError(f"Zeitlimit {args.daemon_zeit} zu kurz (Daemon endet {DAEMON_MARGIN_S // 60} min vorher)")
    passed = ["--run", args.run, "--limit", str(args.limit), "--reserve", str(args.reserve),
              "--max_versuche", str(args.max_versuche), "--account", args.account]
    env = (f"ALL,DAEMON_PARTITION={args.daemon_partition},DAEMON_TIME={args.daemon_zeit},DAEMON_SECONDS={seconds},"
           f"DAEMON_ACCOUNT={args.account}")
    return ["sbatch", "--parsable", f"--account={args.account}", f"--partition={args.daemon_partition}",
            f"--time={args.daemon_zeit}", f"--export={env}", DAEMON_SCRIPT, *passed]


def start_daemon(args):
    ok, lines = quota.check()
    if not ok:
        print("\n".join(lines) + "\n-> Daemon nicht gestartet")
        return QUOTA_FULL
    running = daemon_running()
    if running:
        print(f"Daemon laeuft schon oder wartet: {running} (beenden: python refill.py --stop)")
        return 1
    try:
        cmd = daemon_command(args)
    except ValueError as error:
        print(error)
        return 1
    if args.dry:
        print("DRY:", " ".join(cmd))
        return 0
    stale = wp.JOBS / "refill.stop"
    if stale.exists():                       # sonst wuerde der neue Daemon sofort wieder enden
        stale.unlink()
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode == 0:
        print(f"Daemon abgeschickt ({args.account}, {args.daemon_partition}, {args.daemon_zeit}); Log: "
              "jobs_paper/logs/refill_<Job-ID>.out; beenden: python refill.py --stop")
    return result.returncode


def stop_daemon(rounds=3, pause=5):
    """Alle hpo-refill-Jobs (laufend und wartend, jede Partition) abbrechen, bis keiner mehr in der Schlange ist:
    ein Daemon, der gerade startet, kann zwischen squeue und scancel noch einen Nachfolger abschicken."""
    stopped = []
    for _ in range(rounds):
        ids = daemon_running()
        if not ids:
            break
        subprocess.run(["scancel", *ids])
        stopped += ids
        time.sleep(pause)
    left = daemon_running()
    if left:
        print(f"beendet: {stopped}; noch in der Schlange: {left} (nochmal: python refill.py --stop)")
        return 1
    print(f"beendet: {stopped}" if stopped else "kein Daemon in der Schlange")
    return 0


def queue_limit(text):
    """--limit: ganze Zahl von 1 bis wp.QUEUE_MAX."""
    value = int(text)
    if not 1 <= value <= wp.QUEUE_MAX:
        raise argparse.ArgumentTypeError(f"{value}: erlaubt 1 bis {wp.QUEUE_MAX} (QOS MaxSubmit {wp.MAX_SUBMIT})")
    return value


def parse_runs(text):
    """--run: ein Lauf, 'beide', 'alle' oder eine Liste mit Komma (z. B. 'alle,gpb') -> Laeufe ohne Doppelte."""
    runs = []
    for part in text.split(","):
        part = part.strip()
        if part not in (*KNOWN_RUNS, *RUN_GROUPS):
            raise argparse.ArgumentTypeError(
                f"{part!r}: erlaubt {', '.join((*KNOWN_RUNS, *RUN_GROUPS))} oder eine Liste mit Komma")
        runs += [r for r in RUN_GROUPS.get(part, (part,)) if r not in runs]
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="beide",
                        help="paper, l1_270, extra, gpb, beide = paper + l1_270 (Standard), alle = paper + l1_270 + "
                             "extra (ohne gpb); mehrere mit Komma, z. B. alle,gpb")
    parser.add_argument("--limit", type=queue_limit, default=wp.QUEUE_LIMIT,
                        help=f"hoechstens so viele eigene Jobs in der Schlange (Standard {wp.QUEUE_LIMIT}, "
                             f"1 bis {wp.QUEUE_MAX}; QOS MaxSubmit {wp.MAX_SUBMIT})")
    parser.add_argument("--account", default=wp.ACCOUNT,
                        help=f"Slurm-Account fuer alle sbatch-Aufrufe (Standard {wp.ACCOUNT})")
    parser.add_argument("--reserve", type=int, default=100, help="Plaetze je Partition in der ersten offenen Welle")
    parser.add_argument("--max_versuche", type=int, default=3, help="Wiederholungen je Job bis 'aufgegeben'")
    parser.add_argument("--dry", action="store_true", default=bool(os.environ.get("DRY")), help="nur anzeigen")
    parser.add_argument("--daemon", action="store_true", help="als selbstfortsetzenden Slurm-Job starten")
    parser.add_argument("--daemon-partition", dest="daemon_partition", default=DAEMON_PARTITION,
                        help=f"Partition des Daemons (Standard {DAEMON_PARTITION})")
    parser.add_argument("--daemon-zeit", dest="daemon_zeit", default=DAEMON_TIME,
                        help=f"Zeitlimit des Daemons, H:MM:SS (Standard {DAEMON_TIME}); er endet 20 min vorher, "
                             "der Nachfolger uebernimmt")
    parser.add_argument("--stop", action="store_true", help="Daemon beenden (laufend und wartend)")
    args = parser.parse_args(argv)
    try:
        runs = parse_runs(args.run)
    except argparse.ArgumentTypeError as error:
        parser.error(f"--run {error}")
    if args.stop:
        return stop_daemon()
    if args.daemon:
        return start_daemon(args)
    if args.dry:
        return refill(runs, args.limit, args.reserve, args.max_versuche, dry=True, account=args.account)
    wp.JOBS.mkdir(exist_ok=True)
    with open(wp.JOBS / ".refill.lock", "w") as lock:
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)      # POSIX-Sperre, geht auch auf NFS ($HOME)
        except OSError:
            print("Ein anderer refill.py laeuft gerade (Sperre jobs_paper/.refill.lock) -> nichts getan")
            return 0
        return refill(runs, args.limit, args.reserve, args.max_versuche, dry=False, account=args.account)


if __name__ == "__main__":
    sys.exit(main())
