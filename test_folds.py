"""test_folds.py -- Fold-Jobs im Lauf xgb_alle (run_ablation.py --fold, make_paper_jobs.split_folds).

  Jobs        nur xgb_alle: Jobs mit mehr als 240 echten min je Fold und Seed 27225 der Probe-Aufgaben als 5
              Fold-Jobs (flags + --fold 0..4), Default nie geteilt; gpb_alle, cat_alle ohne --fold und ohne Buendel
  Buendel     buendel.tsv: jede Zeile in genau einem Buendel, nie ueber Wellen, kurze zu 60-120 min, Speicher =
              Maximum, Zeitlimit <= 120 h; load_jobs je Buendel (fertig = alle Zeilen fertig), busy_lines
  Datei       write_fold: Bloecke nach Fold sortiert, Textzeilen unveraendert -> wie ein Job mit 5 Folds;
              gleichzeitige Fold-Jobs (Sperre) verlieren keinen Fold
  Status      watch_paper.file_state: Fold-Job fertig, wenn sein Fold vollstaendig ist; ohne --fold wie bisher
  check       check_results: Fold-Job ok bei fertigem Fold, .tmp eines laufenden Geschwisters wird nie geloescht
  Vergleich   vergleiche_folds.compare, xgb_alle_vergleich (2 Aufgaben x 9 Methoden, altes Format)
Aufruf: .venv/bin/python -m unittest test_folds -v
"""
import multiprocessing
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import check_results
import make_paper_jobs as mpj
import run_ablation
import vergleiche_folds
import watch_paper as wp

COLUMNS = ["iter", "val_score", "fold", "method"]


def block(fold, n=3, offset=0.0):
    return pd.DataFrame({"iter": range(n), "val_score": [0.1 * (i + 1) + offset + fold for i in range(n)],
                         "fold": fold, "method": "tpe"})


def old_format(folds, n=3):
    """Text wie ein Job mit 5 Folds (write_csv_atomic mit append)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d, "x.csv")
        for f in folds:
            run_ablation.job_io.write_csv_atomic(block(f, n), path, append=True)
        return path.read_text()


def write_one(args):
    path, fold = args
    run_ablation.write_fold(block(fold), Path(path), fold)


class WriteFoldTests(unittest.TestCase):
    def test_any_order_gives_old_format(self):
        for order in ([0, 1, 2, 3, 4], [4, 2, 0, 3, 1], [3, 1, 4, 0, 2]):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d, "x.csv")
                for f in order:
                    run_ablation.write_fold(block(f), path, f)
                self.assertEqual(path.read_text(), old_format(range(5)), order)
                self.assertTrue(run_ablation.fold_done(path, 2, 3))

    def test_same_fold_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "x.csv")
            run_ablation.write_fold(block(1, offset=9), path, 1)
            run_ablation.write_fold(block(0), path, 0)
            run_ablation.write_fold(block(1), path, 1)
            self.assertEqual(path.read_text(), old_format([0, 1]))

    def test_fold_done(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "x.csv")
            self.assertFalse(run_ablation.fold_done(path, 0, 3))
            run_ablation.write_fold(block(2), path, 2)
            self.assertTrue(run_ablation.fold_done(path, 2, 3))
            self.assertTrue(run_ablation.fold_done(path, 2, None))      # Hyperband/SMAC
            self.assertFalse(run_ablation.fold_done(path, 1, None))
            self.assertFalse(run_ablation.fold_done(path, 2, 4))

    def test_concurrent_folds_keep_all(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "x.csv")
            for _ in range(5):
                path.unlink(missing_ok=True)
                with multiprocessing.get_context("spawn").Pool(5) as pool:
                    pool.map(write_one, [(str(path), f) for f in (3, 0, 4, 1, 2)])
                self.assertEqual(path.read_text(), old_format(range(5)))

    def test_parse_fold(self):
        base = ["--method", "tpe", "--seed", "1", "--result_folder", "x"]
        self.assertIsNone(run_ablation.parse_args(base).fold)
        self.assertEqual(run_ablation.parse_args(base + ["--fold", "3"]).fold, 3)
        self.assertEqual(run_ablation.parse_args(base + ["--folds", "2"]).folds, 2)
        with self.assertRaises(SystemExit):
            run_ablation.parse_args(base + ["--fold", "5"])


class StateTests(unittest.TestCase):
    def job(self, folder, flags, method="tpe"):
        return next(pd.DataFrame([dict(method=method, seed=7, suite=1, task=2, trials=3, folder=folder,
                                       flags=flags)]).itertuples())

    def test_file_state(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "tpe", "seed_7", "xgboost", "1_2__num_leaves.csv")
            path.parent.mkdir(parents=True)
            for f in (0, 3):
                run_ablation.write_fold(block(f), path, f)
            cache = {}
            flags = "--library xgboost --l1 off --scales gpboost"
            self.assertEqual(wp.job_fold(self.job(d, flags)), None)
            self.assertEqual(wp.file_state(self.job(d, flags)), "angefangen (2/5)")      # ohne --fold wie bisher
            self.assertEqual(wp.file_state(self.job(d, flags + " --fold 3"), cache), "fertig")
            self.assertEqual(wp.file_state(self.job(d, flags + " --fold 1"), cache), "fehlt")
            self.assertEqual(list(cache), [path])                                      # einmal gelesen
            self.assertEqual(wp.file_state(self.job(d + "x", flags + " --fold 1")), "fehlt")

    def test_check_results(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "tpe", "seed_7", "xgboost", "1_2__num_leaves.csv")
            path.parent.mkdir(parents=True)
            frame = pd.concat([pd.DataFrame({c: [0] * 3 for c in check_results.RESULT_COLUMNS}).assign(fold=f)
                               for f in (0, 3)])
            path.write_text(frame.to_csv(index=False))
            flags = "--library xgboost"
            ok = check_results.check_file(path, "hpo", 3)
            self.assertEqual(ok[0], "angefangen")
            run = Path(d, "jobs_paper", "r")
            run.mkdir(parents=True)
            pd.DataFrame([dict(method="tpe", seed=7, suite=1, task=2, trials=3, folder=d, flags=f"{flags} --fold {k}")
                          for k in (0, 1)]).to_csv(run / "jobs.tsv", sep="\t", header=False, index=False)
            Path(str(path) + ".tmp").write_text("x")
            old_jobs, old_sub = wp.JOBS, wp.submissions
            wp.JOBS = Path(d, "jobs_paper")
            check_results.wp.submissions = lambda r: [dict(id="9", n=2, lines=[2])]
            try:
                frame, _ = check_results.scan(["r"], {("9", 0): "RUNNING"})
            finally:
                wp.JOBS, check_results.wp.submissions = old_jobs, old_sub
            states = dict(zip(zip(frame["zeile"], frame["pfad"].str.endswith(".tmp")), frame["zustand"]))
            self.assertEqual(states[(1, False)], "ok")              # Fold 0 fertig
            self.assertEqual(states[(2, False)], "laeuft")          # Fold 1 in squeue
            self.assertEqual(states[(1, True)], "laeuft")           # .tmp gehoert dem laufenden Fold 1


class JobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = wp.load_jobs("xgb_alle", with_files=False, lines=True)
        cls.costs = pd.read_csv("jobs_paper/xgb_alle/kosten.tsv", sep="\t", keep_default_na=False)
        cls.table = wp.bundles("xgb_alle")
        cls.settings = mpj.ALL_RUNS["xgb_alle"]

    def test_split_rule(self):
        jobs, costs = self.jobs, self.costs
        split = jobs["flags"].str.contains("--fold")
        self.assertFalse(split[jobs["method"] == "default"].any())
        probe = pd.Series([(s, t) in mpj.GPB_ALLE_PROBE for s, t in zip(jobs["suite"], jobs["task"])])
        self.assertTrue(split[probe & (jobs["seed"] == 27225) & (jobs["method"] != "default")].all())
        limit = self.settings["fold_split_minutes"]
        whole = ~split & (jobs["method"] != "default")
        self.assertLessEqual(costs.loc[whole, "minuten_echt"].max() / 5, limit)
        self.assertGreater(costs.loc[split & ~(probe & (jobs["seed"] == 27225)), "minuten_echt"].min(), limit - 0.1)
        self.assertAlmostEqual(costs["kernstunden"].sum(), 471302, delta=5)

    def test_bundles_cover_lines_once(self):
        table, jobs = self.table, self.jobs
        members = [z for zeilen in table["zeilen"] for z in zeilen]
        self.assertEqual(sorted(members), list(range(1, len(jobs) + 1)))
        self.assertEqual(list(table["buendel"]), list(range(1, len(table) + 1)))
        wave = jobs.set_index("line")["welle"]
        for zeilen, welle in zip(table["zeilen"], table["welle"]):
            self.assertEqual({int(wave[z]) for z in zeilen}, {welle})       # nie ueber Wellen hinweg
        self.assertEqual(sorted(table["reihenfolge"]), list(range(1, len(table) + 1)))

    def test_bundle_sizes_and_limits(self):
        table, costs = self.table, self.costs
        low, high = self.settings["bundle_minutes"]
        probe = table["buendel"].isin([int(x) for x in Path("jobs_paper/xgb_alle/arrays/probe.idx").read_text().split()])
        minutes = costs.set_index(costs.index + 1)["minuten_echt"]
        mem = costs.set_index(costs.index + 1)["mem_gb"]
        for row in table[~probe].itertuples():
            self.assertAlmostEqual(row.minuten_echt, float(minutes[row.zeilen].sum()), delta=0.2)
            self.assertEqual(row.mem_gb, float(mem[row.zeilen].max()))
            if len(row.zeilen) > 1:
                self.assertLessEqual(row.minuten_echt, high + 0.5)
        multi = table[~probe & (table["zeilen"].str.len() > 1)]
        self.assertGreater((multi["minuten_echt"] >= low).mean(), 0.97)    # nur Reste je Welle kuerzer
        self.assertLessEqual(table["zeitlimit"].max(), 120)
        self.assertEqual(set(table["partition"]) - {"normal.4h", "normal.24h", "normal.120h"}, set())

    def test_probe_bundles(self):
        ids = [int(x) for x in Path("jobs_paper/xgb_alle/arrays/probe.idx").read_text().split()]
        self.assertEqual(len(ids), 27)
        probe = self.table[self.table["buendel"].isin(ids)]
        lines = [z for zeilen in probe["zeilen"] for z in zeilen]
        self.assertEqual(len(lines), 3 + 27 * 5)
        rows = self.jobs.set_index("line").loc[lines]
        self.assertEqual(set(rows["seed"]), {0, 27225})
        for zeilen in probe["zeilen"]:
            folds = self.jobs.set_index("line").loc[zeilen]
            folds = folds[folds["method"] != "default"]
            self.assertEqual(sorted(folds["flags"].str[-1]), list("01234"))
            self.assertEqual(folds["method"].nunique(), 1)
        script = Path("jobs_paper/submit_xgb_alle.sh").read_text()
        self.assertIn("--array=0-26 ", script)
        self.assertIn("jobs_paper/run_buendel_job.sbatch xgb_alle", script)
        self.assertIn("vergleich)", script)
        self.assertIn("jobs_paper/run_alle_job.sbatch xgb_alle_vergleich", script)

    def test_vergleich(self):
        ver = pd.read_csv("jobs_paper/xgb_alle_vergleich/jobs.tsv", sep="\t", header=None, names=wp.COLUMNS)
        self.assertEqual(len(ver), 18)
        self.assertFalse(ver["flags"].str.contains("--fold").any())
        self.assertEqual(set(ver["folder"]), {"Results_xgb_alle_vergleich"})
        ids = [int(x) for x in Path("jobs_paper/xgb_alle/arrays/probe.idx").read_text().split()]
        lines = [z for zeilen in self.table[self.table["buendel"].isin(ids)]["zeilen"] for z in zeilen]
        probe = self.jobs.set_index("line").loc[lines]
        for row in ver.itertuples():           # jede Vergleichszeile hat ihre 5 Fold-Jobs in der Probe
            same = probe[(probe["method"] == row.method) & (probe["seed"] == row.seed) & (probe["task"] == row.task)]
            self.assertEqual(sorted(same["flags"]), [f"{row.flags} --fold {k}" for k in range(5)])

    def test_load_jobs_per_bundle(self):
        bundled = wp.load_jobs("xgb_alle", with_files=False)
        self.assertEqual(len(bundled), len(self.table))
        self.assertEqual(list(bundled["line"]), list(self.table["buendel"]))
        self.assertEqual(list(bundled["zeitlimit"]), list(self.table["zeitlimit"]))
        self.assertEqual(set(bundled["datei"]), {"fehlt"})
        self.assertIn("jobs_paper/run_buendel_job.sbatch",
                      __import__("refill").sbatch_command("xgb_alle", "normal.4h", 4, 1, "x", 2))
        self.assertIn("jobs_paper/run_alle_job.sbatch",
                      __import__("refill").sbatch_command("gpb_alle", "normal.4h", 4, 1, "x", 2))
        self.assertIsNone(wp.bundles("gpb_alle"))
        self.assertIsNone(wp.bundles("cat_alle"))

    def test_other_runs_without_fold(self):
        for run in ("gpb_alle", "cat_alle"):
            text = Path(f"jobs_paper/{run}/jobs.tsv").read_text()
            self.assertNotIn("--fold", text, run)
            self.assertFalse(Path(f"jobs_paper/{run}/buendel.tsv").exists())
        self.assertNotIn("bundle_minutes", mpj.ALL_RUNS["gpb_alle"])
        self.assertNotIn("bundle_minutes", mpj.ALL_RUNS["cat_alle"])


class BundleStateTests(unittest.TestCase):
    def test_bundle_state_and_busy_lines(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d, "jobs_paper", "r")
            (run / "arrays").mkdir(parents=True)
            flags = "--library xgboost"
            pd.DataFrame([dict(method="tpe", seed=7, suite=1, task=2, trials=3, folder=d, flags=f"{flags} --fold {k}")
                          for k in range(3)]).to_csv(run / "jobs.tsv", sep="\t", header=False, index=False)
            pd.DataFrame(dict(name="x", kernstunden=1, partition="normal.4h", stunden=1, mem_gb=1, welle=1,
                              zeitlimit=1, reihenfolge=[1, 2, 3])).to_csv(run / "kosten.tsv", sep="\t", index=False)
            pd.DataFrame(dict(buendel=[1, 2], zeilen=["1,2", "3"], welle=1, kernstunden=1, minuten_echt=60,
                              partition="normal.24h", stunden=8, zeitlimit=8, mem_gb=[3.0, 1.0],
                              reihenfolge=[1, 2])).to_csv(run / "buendel.tsv", sep="\t", index=False)
            path = Path(d, "tpe", "seed_7", "xgboost", "1_2__num_leaves.csv")
            path.parent.mkdir(parents=True)
            run_ablation.write_fold(block(0), path, 0)
            old_jobs = wp.JOBS
            wp.JOBS = Path(d, "jobs_paper")
            old_sub = wp.submissions
            try:
                jobs = wp.load_jobs("r")
                self.assertEqual(list(jobs["line"]), [1, 2])
                self.assertEqual(list(jobs["datei"]), ["angefangen (1/2)", "fehlt"])
                self.assertEqual(list(jobs["mem_gb"]), [3.0, 1.0])
                self.assertEqual(list(jobs["partition"]), ["normal.24h", "normal.24h"])
                run_ablation.write_fold(block(1), path, 1)
                self.assertEqual(list(wp.load_jobs("r")["datei"]), ["fertig", "fehlt"])
                wp.submissions = lambda r: [dict(id="9", n=1, lines=[1])]
                self.assertEqual(check_results.busy_lines("r", {("9", 0): "RUNNING"}), {1, 2})
            finally:
                wp.JOBS, wp.submissions = old_jobs, old_sub


class CompareTests(unittest.TestCase):
    def test_compare(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a.csv"), Path(d, "b.csv")
            frame = pd.DataFrame({"iter": [0, 1], "runtime": [1.5, 2.0], "val_score": [0.25, 0.5], "fold": [0, 0]})
            a.write_text(frame.to_csv(index=False))
            b.write_text(frame.assign(runtime=[9.0, 9.0]).to_csv(index=False))
            self.assertEqual(vergleiche_folds.compare(a, b), "gleich")
            b.write_text(frame.assign(val_score=[0.25, 0.51]).to_csv(index=False))
            self.assertIn("val_score", vergleiche_folds.compare(a, b))
            self.assertEqual(vergleiche_folds.compare(a, Path(d, "c.csv")), "fehlt im Fold-Format")


if __name__ == "__main__":
    unittest.main()
