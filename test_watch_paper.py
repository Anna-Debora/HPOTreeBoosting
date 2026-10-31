"""test_watch_paper.py -- watch_paper.py: sacct-Ausgabe auf Zeilen von jobs.tsv abbilden, Wiederholungslisten.
Aufruf: python test_watch_paper.py"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import watch_paper

SACCT = "\n".join(["900_0|COMPLETED|3600", "900_1|TIMEOUT|14400", "900_[2-3%10]|PENDING|0",
                   "901_0|OUT_OF_MEMORY|60", "901_1|RUNNING|120", "901_2|CANCELLED by 123|0"]) + "\n"


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        run = Path("jobs_paper/paper/arrays")
        run.mkdir(parents=True)
        (run / "a.idx").write_text("5\n6\n7\n8\n")
        (run / "b.idx").write_text("1\n2\n3\n")
        Path("jobs_paper/paper/submitted.tsv").write_text(
            "900\tjobs_paper/paper/arrays/a.idx\t4\t2026-10-30\n901\tjobs_paper/paper/arrays/b.idx\t3\t2026-10-30\n")

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_sacct_lines_map_to_job_lines(self):
        with mock.patch("subprocess.run", return_value=SimpleNamespace(stdout=SACCT)):
            states = watch_paper.slurm_states("paper")
        self.assertEqual(states[5][:2], ("COMPLETED", 3600))
        self.assertEqual(states[6][:2], ("TIMEOUT", 14400))
        self.assertEqual(states[7][0], "PENDING")         # Bereich [2-3] einzeln aufgeloest
        self.assertEqual(states[8][0], "PENDING")
        self.assertEqual(states[1][0], "OUT_OF_MEMORY")
        self.assertEqual(states[2][0], "RUNNING")
        self.assertEqual(states[3][0], "CANCELLED")
        self.assertEqual(states[6][2:], ("900", 1))         # Job-ID und Array-Index fuer den Log-Namen

    def test_retry_escalates_time_and_memory(self):
        job = SimpleNamespace(stunden=4, mem_gb=4, partition="normal.4h")
        self.assertEqual(watch_paper.retry_name(job, "TIMEOUT"), "retry_normal.24h_4G_24h")
        self.assertEqual(watch_paper.retry_name(job, "OUT_OF_MEMORY"), "retry_normal.4h_8G_4h")
        self.assertEqual(watch_paper.retry_name(job, "FAILED"), "retry_normal.4h_4G_4h")
        long = SimpleNamespace(stunden=120, mem_gb=8, partition="normal.120h")
        self.assertEqual(watch_paper.retry_name(long, "TIMEOUT"), "retry_normal.120h_8G_120h")

    def test_file_state_counts_complete_folds(self):
        folder = Path("R/tpe/seed_1/gpboost")
        folder.mkdir(parents=True)
        job = SimpleNamespace(method="tpe", seed=1, suite=336, task=1, trials=3, folder="R", flags="")
        self.assertEqual(watch_paper.file_state(job), "fehlt")
        pd.DataFrame({"fold": [0] * 3 + [1] * 3 + [2] * 2}).to_csv(folder / "336_1__num_leaves.csv", index=False)
        self.assertEqual(watch_paper.file_state(job), "angefangen (2/5)")
        pd.DataFrame({"fold": [f for f in range(5) for _ in range(3)]}).to_csv(folder / "336_1__num_leaves.csv",
                                                                                 index=False)
        self.assertEqual(watch_paper.file_state(job), "fertig")


class WaveTests(unittest.TestCase):
    def test_status_and_wave_tables(self):
        jobs = pd.DataFrame({
            "suite": [336] * 6 + [334] * 3, "task": [1] * 6 + [2] * 3, "name": ["a"] * 6 + ["b"] * 3,
            "welle": [1, 1, 1, 2, 2, 2, 1, 1, 2], "kernstunden": [1] * 6 + [5] * 3,
            "datei": ["fertig", "fertig", "fertig", "fertig", "angefangen (2/5)", "fehlt",
                      "fertig", "fehlt", "fehlt"],
            "slurm": ["COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED", "RUNNING", "PENDING",
                      "COMPLETED", "TIMEOUT", "nicht abgeschickt"]})
        jobs["status"] = [watch_paper.status(d, s) for d, s in zip(jobs["datei"], jobs["slurm"])]
        self.assertEqual(list(jobs["status"]), ["fertig"] * 4 + ["laufend", "wartend", "fertig", "fehlgeschlagen",
                                                                 "fehlt"])
        waves = watch_paper.wave_table(jobs).set_index("welle")
        self.assertEqual(waves.loc[1, "fertig"], 4)
        self.assertEqual(waves.loc[1, "fehlgeschlagen"], 1)
        self.assertEqual(waves.loc[1, "aufgaben_fertig"], 1)         # Aufgabe a ganz, b nicht
        self.assertEqual(waves.loc[2, "laufend"], 1)
        self.assertEqual(waves.loc[2, "aufgaben_fertig"], 0)
        tasks = watch_paper.task_table(jobs).set_index("name")
        self.assertEqual(tasks.loc["a", "welle1"], "ok")
        self.assertEqual(tasks.loc["a", "welle2"], "1/3")
        self.assertEqual(tasks.loc["a", "bis"], 1)
        self.assertEqual(tasks.loc["b", "welle1"], "1/2")
        self.assertEqual(tasks.loc["b", "bis"], 0)
        self.assertEqual(list(tasks.index), ["a", "b"])               # kleine Aufgabe zuerst


if __name__ == "__main__":
    unittest.main()
