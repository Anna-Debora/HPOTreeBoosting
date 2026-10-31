"""test_refill.py -- refill.py mit nachgebildetem squeue/sacct/sbatch: Schlangen-Limit, Reihenfolge, nie doppelt,
Wiederholung mit mehr Speicher/Zeit, aufgegeben, sbatch-Fehler, Reserve je Partition, Vorschau.
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
        self.assertIn("--mem-per-cpu=8G", retry)
        self.assertIn("--partition=normal.120h", retry)
        for state, mem in (("TIMEOUT", None), ("FAILED", None)):          # 2. und 3. Fehlschlag
            last = self.submitted()[-1].split("\t")[0]
            self.slurm.queue = [q for q in self.slurm.queue if not q.startswith(last)]
            self.slurm.sacct[f"{last}_0"] = state
            n = len(self.slurm.sbatches())
            self.run_refill("--limit", "100")
            if state == "TIMEOUT":
                self.assertIn("--mem-per-cpu=8G", self.slurm.sbatches()[-1])   # Speicher bleibt erhoeht
                self.assertEqual(len(self.slurm.sbatches()), n + 1)
        self.assertEqual(len(self.slurm.sbatches()), n)                       # nach 3 Fehlschlaegen aufgegeben

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
        self.slurm.queue = ["900|hpo-refill|normal.24h|RUNNING"]
        self.assertEqual(refill.main(["--daemon"]), 1)
        self.slurm.queue = []
        self.assertEqual(refill.main(["--daemon", "--run", "beide"]), 0)
        call = self.slurm.sbatches()[-1]
        self.assertEqual(call[2], refill.DAEMON_SCRIPT)
        self.assertEqual(call[3:5], ["--run", "beide"])


if __name__ == "__main__":
    unittest.main()
