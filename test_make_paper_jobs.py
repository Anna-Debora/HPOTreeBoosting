"""test_make_paper_jobs.py -- Plan in kosten.tsv (Wellen, Reihenfolge, Zeitlimit, Speicher) und die eingefrorenen
Wellen-Arrays. Arbeitet auf den erzeugten Dateien in jobs_paper/ (vorher: python make_paper_jobs.py).
Aufruf: python test_make_paper_jobs.py"""
import unittest
from pathlib import Path

import pandas as pd

import make_paper_jobs as mpj
from utils_modelling import seeds as SEEDS

HOURS = dict(mpj.PARTITIONS)


class PlanTests(unittest.TestCase):
    def costs(self, run):
        return pd.read_csv(Path("jobs_paper") / run / "kosten.tsv", sep="\t", keep_default_na=False)

    def check_run(self, run):
        costs = self.costs(run)
        jobs = pd.read_csv(Path("jobs_paper") / run / "jobs.tsv", sep="\t", header=None)
        self.assertEqual(len(costs), len(jobs))
        self.assertEqual(list(jobs[0]), list(costs["method"]))
        self.assertEqual(list(jobs[1]), list(costs["seed"]))
        for seed, welle in zip(costs["seed"], costs["welle"]):
            self.assertEqual(welle, 1 if seed == 0 else SEEDS.index(seed) // 5 + 1)
        self.assertEqual(sorted(costs["reihenfolge"]), list(range(1, len(costs) + 1)))
        ordered = costs.sort_values("reihenfolge")
        self.assertTrue(ordered["welle"].is_monotonic_increasing)
        for _, wave in ordered.groupby("welle"):
            self.assertTrue(wave["partition"].map(HOURS).is_monotonic_increasing)     # schnelle zuerst
        self.assertTrue((costs["zeitlimit"] >= costs["stunden"]).all())
        self.assertTrue((costs["zeitlimit"] <= costs["partition"].map(HOURS)).all())
        self.assertTrue(set(costs["mem_gb"]) <= set(mpj.MEM_STEPS))

    def test_paper(self):
        self.check_run("paper")

    def test_l1_270(self):
        self.check_run("l1_270")

    def test_memory_for_large_histograms(self):
        costs = self.costs("paper")
        mem = costs.groupby(["name", "method"])["mem_gb"].max()
        self.assertEqual(mem[("MiniBooNE", "random_search")], 12)        # OOM mit 4 GB auf Euler (Job 19145396)
        self.assertEqual(mem[("MiniBooNE", "default")], 4)
        self.assertGreaterEqual(mem[("jannis", "SMAC")], 12)
        self.assertEqual(mem[("wine_quality", "tpe")], 4)
        self.assertGreaterEqual(mem[("delays_zurich_transport", "tpe")], 8)
        self.assertEqual(mpj.mem_step(4.0), 4)
        self.assertEqual(mpj.mem_step(4.1), 8)
        self.assertEqual(mpj.memory_gb(337, 361068, "tpe", {(337, 361068): 6.1}), 12)
        self.assertEqual(mpj.memory_gb(337, 361068, "tpe", {}), 4)       # ohne Messung: nach Dateigroesse

    def test_frozen_arrays_untouched(self):
        # die Wellen-Arrays vom ersten Abschicken bleiben, wie sie sind (submitted.tsv verweist auf sie)
        for run in mpj.RUNS:
            folder = Path("jobs_paper") / run
            arrays = pd.read_csv(folder / "arrays.tsv", sep="\t")
            costs = self.costs(run)
            for a in arrays.itertuples():
                lines = [int(x) for x in (folder / "arrays" / f"{a.array}.idx").read_text().split()]
                self.assertEqual(len(lines), a.n)
                self.assertTrue((costs.iloc[[l - 1 for l in lines]]["array"] == a.array).all())
                self.assertTrue((costs.iloc[[l - 1 for l in lines]]["welle"] == a.welle).all())

    def test_wave_costs(self):
        waves = pd.read_csv("jobs_paper/kosten_wellen.tsv", sep="\t")
        self.assertEqual(set(waves["lauf"]), {"paper", "l1_270", "beide"})
        for r in waves.itertuples():
            self.assertGreaterEqual(r.tage_64_kerne, r.tage_110_kerne)


if __name__ == "__main__":
    unittest.main()
