"""test_check_results.py -- check_results.py: leere, abgeschnittene, falsch breite, unvollstaendige Ergebnisdateien
erkennen, angefangene (fertige Folds) nicht als Defekt, .tmp-Reste, Dateien laufender Jobs ueberspringen,
--loeschen loescht nur Defekte (nie smac3_output). Aufruf: python test_check_results.py"""
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import check_results
from run_experiments import RESULT_COLUMNS

JOBS = [("tpe", 1, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),      # 1 ok
        ("tpe", 2, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),      # 2 leer
        ("tpe", 3, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),      # 3 abgeschnitten
        ("tpe", 4, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),      # 4 angefangen (2 Folds)
        ("tpe", 5, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),      # 5 Fold mit 2 statt 3 Zeilen
        ("SMAC", 1, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),     # 6 laeuft (in squeue)
        ("tpe", 6, 336, 1, 3, "Results_paper", "--no_lambda_l1 --paper_grid"),      # 7 fehlt, aber .tmp-Rest
        ("default", 0, 336, 1, 135, "Results_paper", "")]                           # 8 Default ok


def hpo_frame(folds, rows=3):
    frame = pd.DataFrame([{c: 0.5 for c in RESULT_COLUMNS} for _ in range(rows * len(folds))])
    frame["fold"] = [f for f in folds for _ in range(rows)]
    frame["method"] = "tpe"
    return frame


class CheckResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("jobs_paper/paper").mkdir(parents=True)
        pd.DataFrame(JOBS).to_csv("jobs_paper/paper/jobs.tsv", sep="\t", header=False, index=False)
        Path("jobs_paper/paper/arrays").mkdir()
        Path("jobs_paper/paper/arrays/a.idx").write_text("6\n")
        Path("jobs_paper/paper/submitted.tsv").write_text("900\tjobs_paper/paper/arrays/a.idx\t1\t2026-11-04\n")
        self.paths = {}
        for line, job in enumerate(JOBS, 1):
            self.paths[line] = check_results.result_path(SimpleNamespace(
                method=job[0], seed=job[1], suite=job[2], task=job[3], trials=job[4], folder=job[5], flags=job[6]))[0]
            self.paths[line].parent.mkdir(parents=True, exist_ok=True)
        hpo_frame(range(5)).to_csv(self.paths[1], index=False)
        self.paths[2].write_text("")
        text = hpo_frame(range(5)).to_csv(index=False)
        self.paths[3].write_text(text[:-40])                                      # Abbruch mitten in der Zeile
        hpo_frame(range(2)).to_csv(self.paths[4], index=False)
        pd.concat([hpo_frame(range(4)), hpo_frame([4], rows=2)]).to_csv(self.paths[5], index=False)
        self.paths[6].write_text("halb")                                          # gerade am Schreiben
        Path(str(self.paths[7]) + ".tmp").write_text("x")
        default = pd.DataFrame(dict(iter=0, fold=[f for f in range(5) for _ in range(270)], method="default",
                                    task_id=1))
        default.to_csv(self.paths[8], index=False)
        Path("Results_paper/smac3_output/run").mkdir(parents=True)
        Path("Results_paper/smac3_output/run/runhistory.json").write_text("{}")
        Path("Results_paper/alt.csv").write_text("a,b\n1,2\n")

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def squeue(self, available=True):
        def run(cmd, **kw):
            if cmd[0] == "squeue" and available:
                return SimpleNamespace(stdout="900_0|hpo-paper|normal.4h|RUNNING\n", returncode=0)
            raise OSError("kein Slurm")
        return mock.patch("subprocess.run", run)

    def test_states(self):
        with self.squeue():
            table, _ = check_results.scan(["paper"], check_results.wp.queued_tasks(check_results.wp.queue()))
        states = {(r.zeile, r.pfad.endswith(".tmp")): (r.zustand, r.grund) for r in table.itertuples() if r.zeile}
        self.assertEqual(states[(1, False)][0], "ok")
        self.assertEqual(states[(2, False)], ("defekt", "leer"))
        self.assertIn("abgeschnitten", states[(3, False)][1])
        self.assertEqual(states[(4, False)], ("angefangen", "2/5 Folds"))
        self.assertEqual(states[(5, False)], ("defekt", "Zeilen je Fold: Fold 4 2 statt 3"))
        self.assertEqual(states[(6, False)][0], "laeuft")
        self.assertEqual(states[(7, False)][0], "fehlt")
        self.assertEqual(states[(7, True)][0], "tmp")
        self.assertEqual(states[(8, False)][0], "ok")
        self.assertIn("nicht in jobs.tsv", table.loc[table["zustand"] == "fremd", "grund"].iloc[0])

    def test_check_file_columns(self):
        path = Path("x.csv")
        path.write_text("a,b,c\n1,2,3\n1,2\n")
        self.assertEqual(check_results.check_file(path), ("defekt", "Zeile 3: 2 statt 3 Spalten"))
        path.write_text("a,b\n")
        self.assertEqual(check_results.check_file(path), ("defekt", "nur Kopfzeile"))
        path.write_text("fold,method\n0,tpe\n")
        self.assertIn("Spalten fehlen", check_results.check_file(path)[1])
        path.write_bytes(b"\xff\xfe\x00a\n")
        self.assertIn("unlesbar", check_results.check_file(path)[1])

    def test_list_and_delete(self):
        with self.squeue(), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(check_results.main(["--run", "paper"]), 1)
        self.assertIn("defekt 3", out.getvalue())
        self.assertIn("smac3_output: 1 Dateien", out.getvalue())
        self.assertTrue(self.paths[2].exists())                                  # ohne --loeschen nichts weg
        with self.squeue(), mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(check_results.main(["--run", "paper", "--loeschen"]), 0)
        for line in (2, 3, 5):
            self.assertFalse(self.paths[line].exists())
        self.assertFalse(Path(str(self.paths[7]) + ".tmp").exists())
        for line in (1, 4, 6, 8):                                                # ok, angefangen, laufend bleiben
            self.assertTrue(self.paths[line].exists())
        self.assertTrue(Path("Results_paper/smac3_output/run/runhistory.json").exists())
        self.assertTrue(Path("Results_paper/alt.csv").exists())                  # fremd: nur gemeldet

    def test_delete_refused_without_squeue(self):
        with self.squeue(available=False), mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(check_results.main(["--run", "paper", "--loeschen"]), 2)
        self.assertTrue(self.paths[2].exists())


if __name__ == "__main__":
    unittest.main()
