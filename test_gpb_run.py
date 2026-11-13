"""test_gpb_run.py -- Lauf "gpb" (korrekte Skalen ohne lambda_l1: TPE, GP-BO, SMAC, Hyperband) in der Infrastruktur.

  make_paper_jobs  4720 Jobs (4 Methoden x 59 Aufgaben x 20 Seeds), Flags --l1 off --scales gpboost in jeder Zeile,
                   Ordner Results_gpb, Wellen mit je 5 Seeds, Kostenfaktor nur fuer gpb, Probe 8 Jobs;
                   jobs_paper/gpb/jobs.tsv im Repo = neu erzeugt
  run_ablation     die Flags aus jobs.tsv ergeben scales=gpboost und lambda_l1 aus (nicht der Standard --l1 on)
  refill           --run gpb, Listen mit Komma, beide/alle unveraendert, sbatch-Skript run_paper_job.sbatch, hpo-gpb
  watch_paper      Reihenfolge, Job-Namen; check_results --run gpb
  make_table5      Ordner[methoden]-Filter; --gpb mit Results_paper als beiden Quellen = bisherige 7-Methoden-Tabelle
Aufruf: .venv/bin/python -m unittest test_gpb_run -v
"""
import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import make_paper_jobs as mpj
import refill
import watch_paper as wp

GPB = ("tpe", "gp_bo", "SMAC", "hyperband")


class MakePaperJobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_list = mpj.tasks()
        cls.sec = {(t.suite, t.task): 1.0 for t in cls.task_list.itertuples()}
        cls.jobs = mpj.plan(mpj.build("gpb", cls.sec, cls.task_list))

    def test_jobs(self):
        jobs = self.jobs
        self.assertEqual(len(jobs), 4720)
        self.assertEqual(set(jobs["method"]), set(GPB))
        self.assertEqual(set(jobs["folder"]), {"Results_gpb"})
        self.assertEqual(set(jobs["flags"]), {"--l1 off --scales gpboost"})
        self.assertEqual(set(jobs["trials"]), {135})
        for welle, group in jobs.groupby("welle"):
            self.assertEqual(sorted(set(group["seed"])), sorted(mpj.SEEDS[(welle - 1) * 5:welle * 5]))
            self.assertEqual(len(group), 1180)

    def test_cost_factor_only_for_gpb(self):
        paper = mpj.build("paper", self.sec, self.task_list)
        for method in GPB:
            p = paper.loc[paper["method"] == method, "kernstunden"].to_numpy()
            g = self.jobs.sort_index().loc[self.jobs["method"] == method, "kernstunden"].to_numpy()
            # kernstunden ist auf 3 Stellen gerundet
            self.assertAlmostEqual(g.sum() / p.sum(), mpj.GPB_COST_FACTOR[method], delta=0.01 * mpj.GPB_COST_FACTOR[method])
        self.assertNotIn("cost_factor", mpj.RUNS["paper"])
        self.assertNotIn("cost_factor", mpj.EXTRA_RUN)

    def test_repo_job_list_is_current(self):
        repo = pd.read_csv("jobs_paper/gpb/jobs.tsv", sep="\t", header=None, names=wp.COLUMNS, keep_default_na=False)
        self.assertEqual(len(repo), 4720)
        self.assertEqual(set(repo["flags"]), {"--l1 off --scales gpboost"})
        probe = [int(x) for x in Path("jobs_paper/gpb/arrays/probe.idx").read_text().split()]
        rows = repo.iloc[[i - 1 for i in probe]]
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows["method"]), set(GPB))
        self.assertEqual(set(zip(rows["suite"], rows["task"])), set(mpj.GPB_PROBE))
        self.assertEqual(set(rows["seed"]), {27225})

    def test_runs_unchanged(self):
        self.assertEqual(list(mpj.RUNS), ["paper", "l1_270"])
        self.assertEqual(mpj.RUNS["paper"]["flags"], "--no_lambda_l1 --paper_grid")


class RunAblationFlagsTests(unittest.TestCase):
    def test_flags_from_jobs_tsv(self):
        import run_ablation
        base = ["--method", "tpe", "--seed", "1", "--result_folder", "x"]
        args = run_ablation.parse_args(base + mpj.GPB_RUN["flags"].split())
        self.assertEqual(args.scales, "gpboost")
        self.assertTrue(args.no_lambda_l1)
        default = run_ablation.parse_args(base)              # ohne Flags: alter Standard
        self.assertEqual(default.scales, "paper")
        self.assertFalse(default.no_lambda_l1)
        self.assertTrue(run_ablation.parse_args(base + ["--no_lambda_l1"]).no_lambda_l1)
        with self.assertRaises(SystemExit):
            run_ablation.parse_args(base + ["--l1", "on", "--no_lambda_l1"])


class RefillTests(unittest.TestCase):
    def test_parse_runs(self):
        self.assertEqual(refill.parse_runs("beide"), ["paper", "l1_270"])
        self.assertEqual(refill.parse_runs("alle"), ["paper", "l1_270", "extra"])     # ohne gpb
        self.assertEqual(refill.parse_runs("gpb"), ["gpb"])
        self.assertEqual(refill.parse_runs("alle,gpb"), ["paper", "l1_270", "extra", "gpb"])
        self.assertEqual(refill.parse_runs("paper,beide"), ["paper", "l1_270"])
        with self.assertRaises(argparse.ArgumentTypeError):
            refill.parse_runs("gpb_l1")

    def test_sbatch_command(self):
        cmd = refill.sbatch_command("gpb", "normal.24h", 12, 2, "x.idx", 5)
        self.assertIn("jobs_paper/run_paper_job.sbatch", cmd)
        self.assertIn("--job-name=hpo-gpb", cmd)
        self.assertIn("--account=ls_math", cmd)
        self.assertEqual(cmd[-2:], ["gpb", "x.idx"])

    def test_bad_run_is_rejected(self):
        with self.assertRaises(SystemExit):
            refill.main(["--run", "unbekannt", "--dry"])

    def test_known_runs(self):
        self.assertIn("gpb", refill.KNOWN_RUNS)
        self.assertEqual(refill.ALL_RUNS, ("paper", "l1_270", "extra"))


class WatchCheckTests(unittest.TestCase):
    def test_watch_paper(self):
        self.assertEqual(wp.RUN_ORDER, ["paper", "l1_270", "extra", "gpb"])
        self.assertIn("hpo-gpb", wp.JOB_NAMES)
        jobs = wp.load_jobs("gpb", with_files=False)
        self.assertEqual(len(jobs), 4720)

    def test_check_results_paths(self):
        import check_results
        job = next(pd.read_csv("jobs_paper/gpb/jobs.tsv", sep="\t", header=None, names=wp.COLUMNS,
                               keep_default_na=False).itertuples())
        path, kind, rows = check_results.result_path(job)
        self.assertTrue(str(path).startswith("Results_gpb/"))
        self.assertEqual(kind, "hpo")
        expected = 135 if job.method in ("tpe", "gp_bo") else None
        self.assertEqual(rows, expected)


class Table5Tests(unittest.TestCase):
    def test_filter_and_merge(self):
        import make_table5 as t5
        import test_make_table5 as base
        with tempfile.TemporaryDirectory() as root:
            paper, gpb = f"{root}/paper", f"{root}/gpb"
            base.write_new_layout(paper, budget=3)
            # gpb: andere Werte fuer TPE und Co., dazu ein (falscher) Default, der herausgefiltert werden muss
            base.write_new_layout(gpb, budget=3, values={"tpe": (0.99, 0.1), "gp_bo": (0.2, 2.0), "SMAC": (0.5, 1.0),
                                                         "hyperband": (0.4, 1.1), "default": (0.0, 9.0)})
            data = t5.load_folders(t5.gpb_spec(paper, gpb))
            self.assertEqual(set(data["method"]), set(t5.METHODS))
            self.assertTrue((data.loc[data["method"] == "Default", "datei"].str.startswith(paper)).all())
            self.assertTrue((data.loc[data["method"] == "TPE", "datei"].str.startswith(gpb)).all())
            table, _ = t5.table5(data, 3)
            ranks = table[(table["block"] == "rank") & (table["metric"] == "R2")].set_index("method")["value"]
            self.assertEqual(ranks["TPE"], 1.0)
            # nur Results_paper als beide Quellen = bisherige Tabelle
            same, _ = t5.table5(t5.load_folders(t5.gpb_spec(paper, paper)), 3)
            pd.testing.assert_frame_equal(same, t5.table5(t5.load(paper), 3)[0])
            with self.assertRaises(ValueError):
                t5.load_folders(f"{paper}[tpee]")


if __name__ == "__main__":
    unittest.main()
