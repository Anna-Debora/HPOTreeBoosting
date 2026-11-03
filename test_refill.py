"""test_refill.py -- refill.py mit nachgebildetem squeue/sacct/sbatch: Schlangen-Limit, Reihenfolge, nie doppelt,
Wiederholung mit mehr Speicher/Zeit, aufgegeben, sbatch-Fehler, Reserve je Partition, Vorschau, halbe
Speicherstufen, wartend abgebrochene Jobs (kein Fehlschlag, Neuabgabe mit kleinerem Plan).
Aufruf: python test_refill.py"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import refill

# Zeile: method, seed, welle, partition, zeitlimit, mem, reihenfolge
PLAN = [("default", 0, 1, "normal.4h", 1, 4, 1),          # 1: fertig (Datei da)
        ("tpe", 1, 1, "normal.4h", 4, 4, 2),              # 2
        ("SMAC", 1, 1, "normal.120h", 30, 4, 5),          # 3
        ("tpe", 2, 1, "normal.4h", 4, 4, 3),              # 4
        ("tpe", 6, 2, "normal.4h", 4, 4, 6),              # 5: Welle 2
        ("gp_bo", 1, 1, "normal.24h", 20, 12, 4)]         # 6


class FakeSlurm:
    def __init__(self):
        self.queue, self.sacct, self.calls, self.next_id, self.fail = [], {}, [], 500, False
        self.respawn = False

    def __call__(self, cmd, capture_output=True, text=True, check=False):
        self.calls.append(cmd)
        if cmd[0] == "squeue":
            return SimpleNamespace(stdout="".join(f"{q}\n" for q in self.queue), returncode=0)
        if cmd[0] == "sacct":
            return SimpleNamespace(stdout="".join(f"{k}|{v}|0\n" for k, v in self.sacct.items()), returncode=0)
        if cmd[0] == "sbatch":
            if self.fail:
                return SimpleNamespace(stdout="", stderr="QOSMaxSubmitJobPerUserLimit", returncode=1)
            self.next_id += 1
            opt = dict(c[2:].split("=", 1) for c in cmd if c.startswith("--") and "=" in c)
            if "array" not in opt:                                   # Daemon-Skript
                self.queue.append(f"{self.next_id}|hpo-refill|normal.24h|PENDING")
                return SimpleNamespace(stdout=f"{self.next_id}\n", stderr="", returncode=0)
            n = int(opt["array"].split("-")[1]) + 1
            self.queue += [f"{self.next_id}_{i}|{opt['job-name']}|{opt['partition']}|PENDING" for i in range(n)]
            for i in range(n):
                self.sacct[f"{self.next_id}_{i}"] = "PENDING"
            return SimpleNamespace(stdout=f"{self.next_id};euler\n", stderr="", returncode=0)
        if cmd[0] == "scancel":
            ids = set(cmd[1:])
            self.queue = [q for q in self.queue if q.split("|")[0] not in ids]
            if self.respawn:                     # laufender Daemon schickt gerade noch einen Nachfolger ab
                self.respawn = False
                self.next_id += 1
                self.queue.append(f"{self.next_id}|hpo-refill|normal.4h|PENDING")
            return SimpleNamespace(stdout="", returncode=0)
        raise AssertionError(cmd)

    def sbatches(self):
        return [c for c in self.calls if c[0] == "sbatch"]


class RefillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        folder = Path("jobs_paper/paper/arrays")
        folder.mkdir(parents=True)
        rows, costs = [], []
        for method, seed, welle, part, limit, mem, order in PLAN:
            rows.append([method, seed, 336, 1, 135, "R", ""])
            costs.append(dict(method=method, seed=seed, suite=336, task=1, name="t", kernstunden=1.0, partition=part,
                              stunden=limit, mem_gb=mem, welle=welle, zeitlimit=limit, reihenfolge=order, array=""))
        pd.DataFrame(rows).to_csv("jobs_paper/paper/jobs.tsv", sep="\t", header=False, index=False)
        pd.DataFrame(costs).to_csv("jobs_paper/paper/kosten.tsv", sep="\t", index=False)
        Path("R/default/gpboost").mkdir(parents=True)
        Path("R/default/gpboost/336_1.csv").write_text("x\n1\n")
        self.slurm = FakeSlurm()
        self.patch = mock.patch("subprocess.run", self.slurm)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def run_refill(self, *args):
        return refill.main(["--run", "paper", "--reserve", "0", *args])

    def submitted(self):
        return Path("jobs_paper/paper/submitted.tsv").read_text().splitlines()

    def lines_of(self, call):
        return [int(x) for x in Path(call[-1]).read_text().split()]

    def test_limit_order_and_no_doubles(self):
        self.slurm.queue = ["42|other|normal.4h|RUNNING"]                     # fremder Job zaehlt mit
        self.assertEqual(self.run_refill("--limit", "3"), 0)
        calls = self.slurm.sbatches()
        self.assertEqual(len(calls), 1)                                       # 2 Plaetze frei: beide normal.4h
        self.assertEqual(self.lines_of(calls[0]), [2, 4])                     # Welle 1, schnelle Partition, Reihenfolge
        self.assertIn("--mem-per-cpu=4G", calls[0])
        self.assertIn("--time=4:00:00", calls[0])
        self.assertEqual(self.submitted()[0].split("\t")[2:3] + self.submitted()[0].split("\t")[4:],
                         ["2", "normal.4h", "4", "4"])
        self.assertEqual(self.run_refill("--limit", "3"), 0)                  # Schlange voll: nichts
        self.assertEqual(len(self.slurm.sbatches()), 1)
        self.assertEqual(self.run_refill("--limit", "100"), 0)                # Rest, ohne 2 und 4 nochmal
        later = [self.lines_of(c) for c in self.slurm.sbatches()[1:]]
        self.assertEqual(later, [[6], [3], [5]])                              # 24h vor 120h, Welle 2 zuletzt
        self.assertIn("--mem-per-cpu=12G", self.slurm.sbatches()[1])
        self.assertEqual(self.run_refill("--limit", "100"), 0)
        self.assertEqual(len(self.slurm.sbatches()), 4)                       # alles in der Schlange

    def test_retry_escalates_and_gives_up(self):
        self.run_refill("--limit", "100")
        first = {self.lines_of(c)[0]: c for c in self.slurm.sbatches()}
        job_of_line3 = self.submitted()[[l for l in self.submitted()].index(
            next(r for r in self.submitted() if "normal.120h" in r))].split("\t")[0]
        self.slurm.queue = [q for q in self.slurm.queue if not q.startswith(job_of_line3)]
        self.slurm.sacct[f"{job_of_line3}_0"] = "OUT_OF_MEMORY"
        self.run_refill("--limit", "100")
        retry = self.slurm.sbatches()[-1]
        self.assertEqual(self.lines_of(retry), [3])
        self.assertIn("--mem-per-cpu=6G", retry)                          # Stufe ueber 4 GB
        self.assertIn("--partition=normal.120h", retry)
        for state, mem in (("TIMEOUT", None), ("FAILED", None)):          # 2. und 3. Fehlschlag
            last = self.submitted()[-1].split("\t")[0]
            self.slurm.queue = [q for q in self.slurm.queue if not q.startswith(last)]
            self.slurm.sacct[f"{last}_0"] = state
            n = len(self.slurm.sbatches())
            self.run_refill("--limit", "100")
            if state == "TIMEOUT":
                self.assertIn("--mem-per-cpu=6G", self.slurm.sbatches()[-1])   # Speicher bleibt erhoeht
                self.assertEqual(len(self.slurm.sbatches()), n + 1)
        self.assertEqual(len(self.slurm.sbatches()), n)                       # nach 3 Fehlschlaegen aufgegeben

    def set_plan_mem(self, line, mem):
        costs = pd.read_csv("jobs_paper/paper/kosten.tsv", sep="\t", keep_default_na=False)
        costs["mem_gb"] = costs["mem_gb"].astype(float)
        costs.loc[line - 1, "mem_gb"] = mem
        costs.to_csv("jobs_paper/paper/kosten.tsv", sep="\t", index=False)

    def test_half_steps_in_mb(self):
        self.set_plan_mem(2, 1.5)
        self.set_plan_mem(4, 1.5)
        self.run_refill("--limit", "100")
        first = self.slurm.sbatches()[0]
        self.assertEqual(self.lines_of(first), [2, 4])
        self.assertIn("--mem-per-cpu=1536M", first)
        self.assertEqual(self.submitted()[0].split("\t")[-1], "1.5")
        tries, _ = refill.wp.attempts("paper", sacct_text="")
        self.assertEqual(tries[2][0]["mem"], 1.5)

    def test_cancelled_pending_is_resubmitted_with_smaller_plan(self):
        self.run_refill("--limit", "100")                                     # alle mit 4 GB (Zeile 6: 12 GB)
        job = self.submitted()[0].split("\t")[0]                               # Zeilen 2 und 4, normal.4h
        self.set_plan_mem(2, 1.5)                                             # neuer, kleinerer Plan
        self.set_plan_mem(4, 1.5)
        for attempt in range(4):                                              # oefter als --max_versuche
            self.slurm.queue = [q for q in self.slurm.queue if not q.startswith(job)]
            for i in (0, 1):
                self.slurm.sacct[f"{job}_{i}"] = "CANCELLED by 1234"         # wartend abgebrochen, Laufzeit 0
            n = len(self.slurm.sbatches())
            self.run_refill("--limit", "100")
            self.assertEqual(len(self.slurm.sbatches()), n + 1)               # nie aufgegeben
            retry = self.slurm.sbatches()[-1]
            self.assertEqual(self.lines_of(retry), [2, 4])
            self.assertIn("--mem-per-cpu=1536M", retry)
            job = self.submitted()[-1].split("\t")[0]

    def test_vanished_jobs_are_resubmitted(self):
        """Wartend abgebrochen (scancel --state=PENDING), sacct kennt sie nicht oder als CANCELLED: weder in squeue
        noch PENDING/RUNNING/COMPLETED -> neu einreichen, ohne als Fehlschlag zu zaehlen; sacct-Luecke allein (Job
        noch in squeue) oder sacct PENDING reicht dagegen nicht."""
        self.run_refill("--limit", "100")
        rows = {self.lines_of(c)[0]: r.split("\t")[0] for c, r in zip(self.slurm.sbatches(), self.submitted())}
        lost, cancelled, lagging, pending = rows[2], rows[6], rows[3], rows[5]   # 2+4 / 6 / 3 / 5
        for _ in range(4):                                                  # oefter als --max_versuche
            gone = (lost, cancelled)
            self.slurm.queue = [q for q in self.slurm.queue if q.split("_")[0] not in gone + (pending,)]
            for key in [k for k in self.slurm.sacct if k.split("_")[0] in (lost, lagging)]:
                del self.slurm.sacct[key]                                   # UNBEKANNT (3 aber noch in squeue)
            self.slurm.sacct[f"{cancelled}_0"] = "NODE_FAIL" if _ == 0 else "CANCELLED by 1234"
            n = len(self.slurm.sbatches())
            self.assertEqual(self.run_refill("--limit", "100"), 0)
            new = sorted(l for c in self.slurm.sbatches()[n:] for l in self.lines_of(c))
            self.assertEqual(new, [2, 4, 6])                                # nicht 3 (in squeue), nicht 5 (PENDING)
            new_rows = self.submitted()[-2:]
            lost = next(r.split("\t")[0] for r in new_rows if "normal.4h" in r)
            cancelled = next(r.split("\t")[0] for r in new_rows if "normal.24h" in r)
        jobs = refill.candidates(["paper"], 3, refill.wp.queued_tasks(refill.wp.queue())).set_index("line")
        self.assertEqual(jobs.loc[3, "slurm"], "UNBEKANNT")
        self.assertEqual(jobs.loc[3, "aktion"], "in_schlange")
        self.assertEqual(jobs.loc[5, "aktion"], "in_schlange")
        self.assertEqual(jobs.loc[6, "versuche"], 1)                       # nur NODE_FAIL zaehlt
        # ohne squeue (Vorschau auf dem Mac): unbekannt zaehlt weiter als in der Schlange
        del self.slurm.sacct[f"{lost}_0"]
        self.assertEqual(refill.candidates(["paper"], 3, None).set_index("line").loc[2, "aktion"], "in_schlange")

    def test_sbatch_failure_stops_and_leaves_no_trace(self):
        self.slurm.fail = True
        self.assertEqual(self.run_refill("--limit", "100"), 1)
        self.assertFalse(Path("jobs_paper/paper/submitted.tsv").exists())
        self.assertEqual(list(Path("jobs_paper/paper/arrays/teile").glob("*.idx")), [])
        self.assertEqual(len(self.slurm.sbatches()), 1)                       # nach dem ersten Fehler Schluss

    def test_unknown_hpo_jobs_abort(self):
        self.slurm.queue = ["777_3|hpo-paper|normal.4h|PENDING"]
        self.assertEqual(self.run_refill("--limit", "100"), 1)
        self.assertEqual(self.slurm.sbatches(), [])

    def test_reserve_per_partition(self):
        refill.main(["--run", "paper", "--limit", "3", "--reserve", "1"])
        lines = sorted(l for c in self.slurm.sbatches() for l in self.lines_of(c))
        self.assertEqual(lines, [2, 3, 6])                 # je Partition einer statt 2, 4, 6

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self.run_refill("--limit", "100", "--dry"), 0)
        self.assertEqual(self.slurm.sbatches(), [])
        self.assertFalse(Path("jobs_paper/paper/submitted.tsv").exists())

    def test_nothing_left(self):
        self.run_refill("--limit", "100")
        for key in list(self.slurm.sacct):
            self.slurm.sacct[key] = "COMPLETED"
        self.slurm.queue = []
        with mock.patch("watch_paper.file_state", return_value="fertig"):
            self.assertEqual(self.run_refill("--limit", "100"), refill.NOTHING_LEFT)

    def test_daemon_refuses_second_instance(self):
        for queue in (["900|hpo-refill|normal.24h|RUNNING"], ["901|hpo-refill|normal.4h|PENDING"]):
            self.slurm.queue = queue
            self.assertEqual(refill.main(["--daemon"]), 1)
            self.assertEqual(self.slurm.sbatches(), [])
        self.slurm.queue = []
        self.assertEqual(refill.main(["--daemon", "--run", "beide"]), 0)
        self.assertEqual(refill.main(["--daemon", "--run", "beide"]), 1)      # der erste wartet jetzt
        self.assertEqual(len(self.slurm.sbatches()), 1)

    def test_daemon_default_is_4h(self):
        Path("jobs_paper/refill.stop").write_text("")                          # alter Rest: wird entfernt
        self.assertEqual(refill.main(["--daemon", "--run", "beide"]), 0)
        call = self.slurm.sbatches()[-1]
        self.assertIn("--partition=normal.4h", call)
        self.assertIn("--time=4:00:00", call)
        self.assertIn("--account=ls_math", call)
        self.assertIn("--export=ALL,DAEMON_PARTITION=normal.4h,DAEMON_TIME=4:00:00,DAEMON_SECONDS=14400,"
                      "DAEMON_ACCOUNT=ls_math", call)
        i = call.index(refill.DAEMON_SCRIPT)
        self.assertEqual(call[i + 1:i + 3], ["--run", "beide"])
        self.assertEqual(call[call.index("--limit") + 1], "4000")
        self.assertEqual(call[call.index("--account") + 1], "ls_math")       # auch fuer refill.py im Daemon
        self.assertFalse(Path("jobs_paper/refill.stop").exists())

    def test_daemon_partition_and_time(self):
        self.assertEqual(refill.main(["--daemon", "--daemon-partition", "normal.24h", "--daemon-zeit", "24:00:00"]),
                         0)
        call = self.slurm.sbatches()[-1]
        self.assertIn("--partition=normal.24h", call)
        self.assertIn("--time=24:00:00", call)
        self.assertIn("DAEMON_SECONDS=86400", next(c for c in call if c.startswith("--export=")))
        self.slurm.queue = []
        for bad in (["--daemon-zeit", "4:00"], ["--daemon-zeit", "5:00:00"], ["--daemon-zeit", "0:25:00"],
                    ["--daemon-partition", "bulk"]):
            self.assertEqual(refill.main(["--daemon", *bad]), 1, bad)
        self.assertEqual(len(self.slurm.sbatches()), 1)
        self.assertEqual(refill.time_seconds("1-02:03:04"), 93784)

    def test_default_limit_and_account(self):
        args = []
        with mock.patch.object(refill, "refill", side_effect=lambda *a, **kw: args.append((a, kw)) or 0):
            refill.main(["--run", "paper"])
            refill.main(["--run", "paper", "--limit", "15000", "--account", "es_mge"])
        self.assertEqual(args[0][0][1], 4000)
        self.assertEqual(args[0][1]["account"], "ls_math")
        self.assertEqual((args[1][0][1], args[1][1]["account"]), (15000, "es_mge"))
        for bad in ("15001", "0", "viele"):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                refill.main(["--limit", bad])

    def test_jobs_submitted_with_account(self):
        self.run_refill("--limit", "100")
        self.assertTrue(all("--account=ls_math" in c for c in self.slurm.sbatches()))
        self.slurm.queue = []
        self.slurm.calls.clear()
        Path("jobs_paper/paper/submitted.tsv").unlink()
        self.run_refill("--limit", "100", "--account", "es_mge")
        self.assertTrue(self.slurm.sbatches())
        self.assertTrue(all("--account=es_mge" in c for c in self.slurm.sbatches()))

    def test_stop_ends_both_variants(self):
        self.slurm.queue = ["19251384|hpo-refill|normal.24h|PENDING",              # alter 24h-Daemon
                            "600|hpo-refill|normal.4h|RUNNING", "601|hpo-refill|normal.4h|PENDING",
                            "700_1|hpo-paper|normal.4h|RUNNING"]
        self.slurm.respawn = True
        with mock.patch("time.sleep"):
            self.assertEqual(refill.main(["--stop"]), 0)
        self.assertEqual(self.slurm.queue, ["700_1|hpo-paper|normal.4h|RUNNING"])  # Rechenjobs bleiben
        cancels = [c for c in self.slurm.calls if c[0] == "scancel"]
        self.assertEqual(cancels[0][1:], ["19251384", "600", "601"])
        self.assertEqual(len(cancels), 2)                                      # zweite Runde: der Nachzuegler
        with mock.patch("time.sleep"):
            self.assertEqual(refill.main(["--stop"]), 0)                       # nichts mehr da


if __name__ == "__main__":
    unittest.main()
