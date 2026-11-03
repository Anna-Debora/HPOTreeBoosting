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

    def test_escalate_time_and_memory(self):
        plan = dict(partition="normal.4h", stunden=4, mem=4)
        a = lambda state, **kw: dict(dict(partition="normal.4h", stunden=4, mem=4, state=state, sekunden=10), **kw)
        esc = watch_paper.escalate
        self.assertEqual(esc([a("TIMEOUT")], plan), dict(partition="normal.24h", stunden=24, mem=4))
        self.assertEqual(esc([a("OUT_OF_MEMORY")], plan)["mem"], 6)
        self.assertEqual(esc([a("OUT_OF_MEMORY", mem=8)], plan)["mem"], 12)
        self.assertEqual(esc([a("OUT_OF_MEMORY", mem=1.5)], dict(plan, mem=1))["mem"], 2)
        self.assertEqual(esc([a("FAILED")], plan), plan)
        self.assertEqual(esc([a("TIMEOUT", partition="normal.120h", stunden=120)], plan)["stunden"], 120)
        self.assertEqual(esc([a("OUT_OF_MEMORY", mem=32)], plan)["mem"], 32)
        # Eskalation bleibt nach einem spaeteren anderen Fehler erhalten
        self.assertEqual(esc([a("OUT_OF_MEMORY"), a("TIMEOUT", mem=6)], plan),
                         dict(partition="normal.24h", stunden=24, mem=6))
        # Plan inzwischen kleiner, letzter Versuch abgebrochen: Plan, nicht der alte Speicher
        self.assertEqual(esc([a("CANCELLED", sekunden=0)], dict(plan, mem=1.5))["mem"], 1.5)
        # Plan hoeher: nie weniger als der Plan
        self.assertEqual(esc([a("FAILED")], dict(partition="normal.24h", stunden=21, mem=12)),
                         dict(partition="normal.24h", stunden=21, mem=12))

    def test_cancelled_while_pending_is_no_failure(self):
        jobs = pd.DataFrame(dict(line=[1, 2], datei=["fehlt", "fehlt"], partition="normal.4h", zeitlimit=4,
                                 mem_gb=[1.5, 1.5]))
        never = [dict(id=str(k), index=0, partition="normal.4h", stunden=4, mem=4, state="CANCELLED", sekunden=0)
                 for k in range(5)]
        ran = [dict(never[0], state="CANCELLED", sekunden=100)] * 3
        out = watch_paper.classify(jobs, {1: never, 2: ran}, max_tries=3).set_index("line")
        self.assertEqual(out.loc[1, "aktion"], "wiederholen")
        self.assertEqual(out.loc[1, "versuche"], 0)
        self.assertEqual(out.loc[1, "mem_neu"], 1.5)
        self.assertEqual(out.loc[2, "aktion"], "aufgegeben")       # lief schon: zaehlt

    def test_squeue_decides_unknown_jobs(self):
        jobs = pd.DataFrame(dict(line=[1, 2, 3, 4, 5], datei="fehlt", partition="normal.4h", zeitlimit=4, mem_gb=1.0))
        a = lambda state, index: [dict(id="7", index=index, partition="normal.4h", stunden=4, mem=1.0, state=state,
                                       sekunden=0)]
        tries = {1: a(None, 0), 2: a(None, 1), 3: a("CANCELLED", 2), 4: a("PENDING", 3), 5: a("NODE_FAIL", 4)}
        q = pd.DataFrame(dict(id=["7_1", "8"], name="hpo-paper", partition="normal.4h", state=["RUNNING", "PENDING"]))
        queued = watch_paper.queued_tasks(q)
        self.assertEqual(queued, {("7", 1): "RUNNING"})
        out = watch_paper.classify(jobs, tries, 3, queued).set_index("line")
        self.assertEqual(out["aktion"].tolist(), ["wiederholen", "in_schlange", "wiederholen", "in_schlange",
                                                  "wiederholen"])
        self.assertEqual(out["versuche"].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual([watch_paper.status(d, s, l) for d, s, l in zip(out["datei"], out["slurm"], out["squeue"])],
                         ["unbekannt", "laufend", "abgebrochen", "wartend", "fehlgeschlagen"])
        old = watch_paper.classify(jobs, tries, 3).set_index("line")      # ohne squeue: wie bisher
        self.assertEqual(old.loc[1, "aktion"], "in_schlange")
        self.assertEqual(watch_paper.status("fehlt", "UNBEKANNT", None), "unbekannt")

    def test_pending_mismatch_scripts(self):
        jobs = pd.DataFrame(dict(line=[1, 2, 3, 4], slurm=["PENDING", "PENDING", "PENDING", "RUNNING"],
                                 mem_gb=[1.5, 8.0, 4.0, 1.0]))
        mk = lambda i, mem: [dict(id="77", index=i, partition="normal.4h", stunden=4, mem=mem, state="PENDING",
                                  sekunden=0)]
        tries = {1: mk(0, 4), 2: mk(1, 4), 3: mk(2, 4), 4: mk(3, 4)}
        lines = watch_paper.pending_mismatch("paper", jobs, tries)
        self.assertEqual(len(lines), 2)
        high = Path("jobs_paper/paper/scancel_zu_viel_speicher.sh").read_text()
        low = Path("jobs_paper/paper/scancel_zu_wenig_speicher.sh").read_text()
        self.assertIn("scancel 77_0\n", high)                     # 4 GB angefordert, Plan 1.5
        self.assertIn("scancel 77_1\n", low)                      # 4 GB angefordert, Plan 8
        self.assertNotIn("77_2", high + low)                       # passt
        self.assertNotIn("77_3", high + low)                       # laeuft schon
        self.assertIn("1x 4->1.5 GB normal.4h", " ".join(lines))
        watch_paper.pending_mismatch("paper", jobs.assign(mem_gb=4.0), tries)
        self.assertFalse(Path("jobs_paper/paper/scancel_zu_viel_speicher.sh").exists())   # nichts mehr: weg

    def test_memory_report(self):
        Path("jobs_paper/paper/jobs.tsv").write_text("".join(
            f"{m}\t1\t336\t1\t135\tR\t\n" for m in ["tpe"] * 7 + ["SMAC"]))      # a.idx: 900_0-3 = Zeilen 5-8
        pd.DataFrame(dict(method=["tpe"] * 7 + ["SMAC"], partition="normal.4h", mem_gb=4,
                          kernstunden=1.0, name="t", stunden=2, welle=1, zeitlimit=2,
                          reihenfolge=range(1, 9))).to_csv("jobs_paper/paper/kosten.tsv", sep="\t", index=False)
        sacct = "\n".join([
            "900_0|COMPLETED|", "900_0.batch|COMPLETED|1200M|", "900_0.extern|COMPLETED|1K|",
            "900_1|TIMEOUT|", "900_1.batch|CANCELLED|1.1G|",
            "900_2|COMPLETED|", "900_2.batch|COMPLETED|50M|",              # zu kurz ("SCHON FERTIG")
            "900_3|OUT_OF_MEMORY|", "900_3.batch|OUT_OF_MEMORY|4G|",
            "900_[4-7]|PENDING|"])
        # Spalten JobID|State|MaxRSS|ElapsedRaw
        rows = []
        for row in sacct.splitlines():
            parts = row.split("|")
            elapsed = {"900_0": 4000, "900_1": 7300, "900_2": 5, "900_3": 600}.get(parts[0], 0)
            rows.append("|".join(parts[:3] + [str(elapsed)]))
        table = watch_paper.memory_report(["paper"], sacct_text="\n".join(rows) + "\n",
                                          out="jobs_paper/speicher_euler.tsv").set_index("method")
        self.assertEqual(table.loc["tpe", "jobs"], 2)              # 900_0, 900_1 (900_2 zu kurz)
        self.assertAlmostEqual(table.loc["tpe", "maxrss_gb"], 1.2)
        self.assertEqual(table.loc["SMAC", "oom"], 1)
        self.assertEqual(table.loc["SMAC", "oom_bei"], 4)
        self.assertEqual(table.loc["tpe", "neu_gb"], 6)            # OOM bei 4 GB in der Aufgabe hebt alle HPO
        saved = pd.read_csv("jobs_paper/speicher_euler.tsv", sep="\t")
        self.assertEqual(list(saved.columns), watch_paper.speicher.EULER_COLUMNS)

    def test_old_submission_resources(self):
        Path("jobs_paper/paper/arrays.tsv").write_text(
            "array\twelle\tpartition\tstunden\tmem_gb\tn\na\t1\tnormal.120h\t30\t4\t4\n")
        Path("jobs_paper/paper/arrays/retry_normal.120h_8G_30h.idx").write_text("5\n")
        with open("jobs_paper/paper/submitted.tsv", "a") as handle:
            handle.write("902\tjobs_paper/paper/arrays/retry_normal.120h_8G_30h.idx\t1\t2026-10-31\n")
            handle.write("903;euler\tjobs_paper/paper/arrays/b.idx\t2\t2026-10-31\tnormal.24h\t21\t12\n")
        subs = watch_paper.submissions("paper")
        self.assertEqual([(s["id"], s["partition"], s["stunden"], s["mem"]) for s in subs],
                         [("900", "normal.120h", 30, 4), ("901", None, None, None), ("902", "normal.120h", 30, 8),
                          ("903", "normal.24h", 21, 12)])
        tries, ok = watch_paper.attempts("paper", sacct_text="902_0|OUT_OF_MEMORY|10\n903_[0-1]|PENDING|0\n")
        self.assertTrue(ok)
        self.assertEqual([t["id"] for t in tries[5]], ["900", "902"])     # erst a.idx Index 0, dann der Retry
        self.assertEqual(tries[5][-1]["state"], "OUT_OF_MEMORY")
        self.assertIsNone(tries[5][0]["state"])                          # 900 kennt sacct hier nicht
        self.assertEqual(tries[2][-1]["state"], "PENDING")                # 903 nur Index 0-1: Zeilen 1, 2
        self.assertNotIn("903", [t["id"] for t in tries[3]])

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
        self.assertEqual(list(waves.columns[1:8]), ["fertig", "laufend", "wartend", "unbekannt", "abgebrochen",
                                                     "fehlgeschlagen", "fehlt"])
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


    def test_concurrent_cores(self):
        now = watch_paper.datetime(2026, 11, 3, 12, 0, 0)
        text = "\n".join([
            "1_0|hpo-paper|2026-11-02T12:00:00|2026-11-03T12:00:00|1",      # ganze 24 h: 1 Kern
            "1_1|hpo-l1_270|2026-11-03T06:00:00|Unknown|2",                 # laeuft seit 6 h mit 2 Kernen: 0.5
            "1_2|hpo-paper|2026-11-01T00:00:00|2026-11-02T18:00:00|1",      # nur 6 h im Fenster: 0.25
            "1_[3-9]|hpo-paper|Unknown|Unknown|1",                          # wartend
            "2|hpo-refill|2026-11-02T12:00:00|Unknown|1",                   # Daemon zaehlt nicht
            "3|anderes|2026-11-02T12:00:00|2026-11-03T12:00:00|64"]) + "\n"
        self.assertAlmostEqual(watch_paper.concurrent_cores(now=now, sacct_text=text), 1.75)
        with mock.patch("subprocess.run", side_effect=OSError):
            self.assertIsNone(watch_paper.concurrent_cores(now=now))

    def test_progress_and_forecast(self):
        def frame(states, welle, hours, used):
            jobs = pd.DataFrame(dict(line=range(1, len(states) + 1), datei=states, welle=welle, kernstunden=hours))
            tries = {i + 1: [dict(sekunden=u * 3600)] for i, u in enumerate(used) if u}
            return watch_paper.progress(jobs, tries)
        paper = frame(["fertig", "angefangen (2/5)", "fehlt", "fehlt"], [1, 1, 2, 2], [10.0, 10.0, 20.0, 5.0],
                      [12, 4, 0, 0])
        self.assertEqual(paper["rest_h"].tolist(), [0.0, 6.0, 20.0, 5.0])
        self.assertEqual(paper["verbraucht_h"].tolist(), [12, 4, 0, 0])
        l1 = frame(["fertig", "fehlt"], [1, 2], [8.0, 30.0], [8, 0])
        now = watch_paper.datetime(2026, 11, 3, 12, 0)
        table, factor = watch_paper.forecast([("paper", paper), ("l1_270", l1)], cores=2, now=now)
        self.assertIsNone(factor)                                   # zu wenige fertige Jobs
        self.assertEqual(list(zip(table["lauf"], table["welle"])),
                         [("paper", 1), ("l1_270", 1), ("paper", 2), ("l1_270", 2)])
        self.assertEqual(table["verbraucht_h"].tolist(), [16, 8, 0, 0])
        self.assertEqual(table["rest_h"].tolist(), [6.0, 0.0, 25.0, 30.0])
        # kumuliert 6 / 2 = 3 h; l1 Welle 1 fertig; 31 / 2 = 15.5 h -> aber laengster Job 20 h; 61 / 2 = 30.5 h
        self.assertEqual(table["fertig_am"].tolist(), ["2026-11-03 18:00", "fertig", "2026-11-04 08:00",
                                                       "2026-11-04 18:30"])
        self.assertEqual(watch_paper.forecast([("paper", paper)], cores=None, now=now)[0]["fertig_am"].iloc[0],
                         "unbekannt")
        many = frame(["fertig"] * 20 + ["fehlt"], [1] * 21, [1.0] * 20 + [10.0], [2] * 20 + [0])
        table, factor = watch_paper.forecast([("paper", many)], cores=10, now=now)
        self.assertAlmostEqual(factor, 2.0)
        self.assertEqual(table["fertig_am_kal"].iloc[0], "2026-11-04 08:00")   # 10 h x 2 = 20 h (laengster Job)


if __name__ == "__main__":
    unittest.main()
