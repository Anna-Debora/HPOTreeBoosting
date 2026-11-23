"""test_gpb_alle.py -- Lauf "gpb_alle" (GPBoost, alle Methoden und 345 Datensaetze, Skalen gpboost, ohne lambda_l1).

  Jobs        59 alte Aufgaben nur BOHB/DEHB/HEBO (Rest wiederverwendet), 286 neue alle 10 Methoden:
              3540 + 51766 = 55306 Jobs, T = 135, Ordner Results_gpb_alle, lambda_l1 nie getunt
  Flags       Default ohne, Gitter/Zufallsgitter wie paper (135er Gitter), sonst --l1 off --scales gpboost
  Repo        jobs_paper/gpb_alle/jobs.tsv, probe.idx, submit_gpb_alle.sh, run_gpb_alle_job.sbatch = neu erzeugt;
              gpb, gpb_l1 byte-gleich
  Wiederverwendung  jobs.tsv von gpb/paper: gleiche Seeds, T, Flags
  Infrastruktur  refill (run_gpb_alle_job.sbatch, laengste zuerst), watch_paper, check_results, speicher (neue Aufgaben)
Aufruf: .venv/bin/python -m unittest test_gpb_alle -v
"""
import tempfile
import unittest
from pathlib import Path

import make_paper_jobs as mpj
import refill
import run_ablation
import speicher
import watch_paper as wp

RUN = "gpb_alle"


class JobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = wp.load_jobs(RUN, with_files=False)

    def test_counts(self):
        jobs = self.jobs
        self.assertEqual(len(jobs), 55306)
        self.assertEqual(len(jobs.drop_duplicates(["method", "seed", "suite", "task"])), 55306)
        old = jobs[jobs["suite"] < 1000]
        new = jobs[jobs["suite"] >= 1000]
        self.assertEqual(set(old["method"]), {"bohb", "dehb", "hebo"})
        self.assertEqual(len(old), 59 * 20 * 3)
        self.assertEqual(set(new["method"]), set(mpj.GPB_ALLE_METHODS))
        self.assertEqual(new["task"].nunique(), 286)
        self.assertEqual(len(new), 286 * (1 + 20 * 9))
        self.assertEqual(set(jobs["trials"]), {135})
        self.assertEqual(set(jobs["folder"]), {"Results_gpb_alle"})
        self.assertEqual(set(jobs.loc[jobs["method"] != "default", "seed"]), set(mpj.SEEDS))

    def test_flags_without_l1(self):
        for method, group in self.jobs.groupby("method"):
            flags = set(group["flags"])
            if method == "default":
                self.assertEqual(flags, {""})
            elif method in ("grid_search", "random_search"):
                self.assertEqual(flags, {"--no_lambda_l1 --paper_grid"})
            else:
                self.assertEqual(flags, {"--l1 off --scales gpboost"})
            for f in flags:
                if f:
                    args = run_ablation.parse_args(["--method", method, "--seed", "1", "--result_folder", "x"]
                                                   + f.split())
                    self.assertTrue(args.no_lambda_l1, (method, f))

    def test_repo_files_are_current(self):
        with tempfile.TemporaryDirectory() as out:
            mpj.main_gpb_alle(out, check=False)
            for name in ("gpb_alle/jobs.tsv", "gpb_alle/arrays/probe.idx", "submit_gpb_alle.sh",
                         "run_gpb_alle_job.sbatch", "aufgaben_alle.tsv"):
                self.assertEqual(Path(out, name).read_text(), Path("jobs_paper", name).read_text(), name)
            for run in ("gpb", "gpb_l1"):
                mpj.main_gpb(out, "jobs_paper/zeitmessung_mac.tsv", run=run)
                for name in (f"{run}/jobs.tsv", f"{run}/arrays/probe.idx", f"submit_{run}.sh"):
                    self.assertEqual(Path(out, name).read_text(), Path("jobs_paper", name).read_text(), name)

    def test_probe(self):
        lines = [int(x) for x in Path("jobs_paper/gpb_alle/arrays/probe.idx").read_text().split()]
        rows = self.jobs.iloc[[i - 1 for i in lines]]
        self.assertEqual(set(zip(rows["suite"], rows["task"])), set(mpj.GPB_ALLE_PROBE))
        self.assertEqual(len(rows), 2 * 10 + 3)
        script = Path("jobs_paper/submit_gpb_alle.sh").read_text()
        self.assertIn(f"--array=0-{len(rows) - 1}", script)
        self.assertIn("--job-name=hpo-gpb_alle jobs_paper/run_gpb_alle_job.sbatch gpb_alle", script)

    def test_sbatch_venv_per_method(self):
        text = Path("jobs_paper/run_gpb_alle_job.sbatch").read_text()
        self.assertIn("bohb|dehb|hebo) VENV=.venv_extra", text)
        self.assertIn("*)              VENV=.venv", text)
        self.assertIn("run_default.py", text)
        self.assertIn("--cpus-per-task=1", text)

    def test_reuse_settings(self):
        ok, lines = mpj.pruefe_wiederverwendung("jobs_paper", results=False)
        self.assertTrue(ok, "\n".join(lines))


class InfrastructureTests(unittest.TestCase):
    def test_refill(self):
        self.assertEqual(refill.parse_runs(RUN), [RUN])
        self.assertNotIn(RUN, refill.parse_runs("alle"))
        self.assertIn(RUN, refill.LONGEST_FIRST)
        cmd = refill.sbatch_command(RUN, "normal.24h", 24, 2, "x.idx", 5)
        self.assertIn("jobs_paper/run_gpb_alle_job.sbatch", cmd)
        self.assertIn("--job-name=hpo-gpb_alle", cmd)

    def test_watch_and_check(self):
        import check_results
        self.assertIn(RUN, wp.RUN_ORDER)
        self.assertIn("hpo-gpb_alle", wp.JOB_NAMES)
        with self.assertRaises(SystemExit):
            check_results.main(["--run", "gpb_allex"])

    def test_memory_new_tasks(self):
        peaks = speicher.mac_peaks()
        self.assertIn((1002, 930017), peaks)          # neue Aufgabe aus speicher_neu_mac.tsv
        self.assertIn((334, 361110), peaks)           # alte weiter aus speicher_mac.tsv
        self.assertNotIn((1002, 930017), speicher.mac_peaks("jobs_paper/../jobs_paper/speicher_mac.tsv"))
        plan = speicher.Plan()
        self.assertNotEqual(plan.source(1002, 930017, "tpe")[1], "unbekannt")


if __name__ == "__main__":
    unittest.main()
