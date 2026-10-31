"""test_make_paper_jobs.py -- Wellen: Seeds 1-5 / 6-10 / 11-15 / 16-20, Default in Welle 1, kleine Aufgaben zuerst,
Probe nicht in den Wellen, jede Zeile von jobs.tsv genau einmal. Arbeitet auf den erzeugten Dateien in jobs_paper/
(vorher: python make_paper_jobs.py). Aufruf: python test_make_paper_jobs.py"""
import unittest
from pathlib import Path

import pandas as pd

import make_paper_jobs as mpj
from utils_modelling import seeds as SEEDS


class WaveFileTests(unittest.TestCase):
    def check_run(self, run):
        folder = Path("jobs_paper") / run
        costs = pd.read_csv(folder / "kosten.tsv", sep="\t")
        arrays = pd.read_csv(folder / "arrays.tsv", sep="\t")
        probe = [int(x) for x in (folder / "arrays" / "probe.idx").read_text().split()]
        seen = []
        for a in arrays.itertuples():
            lines = [int(x) for x in (folder / "arrays" / f"{a.array}.idx").read_text().split()]
            self.assertEqual(len(lines), a.n)
            self.assertLessEqual(a.n, 1000)
            part = costs.iloc[[line - 1 for line in lines]]
            self.assertTrue((part["welle"] == a.welle).all(), a.array)
            self.assertTrue((part["partition"] == a.partition).all())
            self.assertTrue((part["mem_gb"] == a.mem_gb).all())
            self.assertLessEqual(part["stunden"].max(), a.stunden)
            self.assertTrue(a.array.startswith(f"welle{a.welle}."))
            # kleine Aufgaben zuerst: die Aufgaben erscheinen in der Reihenfolge ihrer Kosten je Versuch
            order = list(dict.fromkeys(zip(part["suite"], part["task"])))
            self.assertEqual(len(order), part[["suite", "task"]].drop_duplicates().shape[0],
                             "eine Aufgabe ist im Array nicht zusammenhaengend")
            seen += lines
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(sorted(seen + probe), list(range(1, len(costs) + 1)))
        for seed, welle in zip(costs["seed"], costs["welle"]):
            self.assertEqual(welle, 1 if seed == 0 else SEEDS.index(seed) // 5 + 1)
        self.assertEqual(costs.groupby("welle")["seed"].nunique().tolist(), [6, 5, 5, 5])   # Welle 1: + Default

    def test_paper(self):
        self.check_run("paper")

    def test_l1_270(self):
        self.check_run("l1_270")

    def test_small_tasks_first(self):
        timing = mpj.seconds_per_trial()
        folder = Path("jobs_paper/paper")
        costs = pd.read_csv(folder / "kosten.tsv", sep="\t")
        lines = [int(x) for x in (folder / "arrays" / "welle1.normal.4h_4G_0.idx").read_text().split()]
        tasks = list(dict.fromkeys(zip(costs.iloc[[l - 1 for l in lines]]["suite"],
                                       costs.iloc[[l - 1 for l in lines]]["task"])))
        secs = [timing[t] for t in tasks]
        self.assertEqual(secs, sorted(secs))
        first = costs.iloc[lines[0] - 1]
        self.assertEqual(first["method"], "default")       # erste Aufgabe: Default, dann Seed 1 alle Methoden

    def test_wave_costs(self):
        waves = pd.read_csv("jobs_paper/kosten_wellen.tsv", sep="\t")
        self.assertEqual(set(waves["lauf"]), {"paper", "l1_270", "beide"})
        for r in waves.itertuples():
            self.assertGreaterEqual(r.tage_64_kerne * 24, r.kernstunden / 64 - 2.5)
            self.assertGreaterEqual(r.tage_64_kerne, r.tage_110_kerne)
        self.assertLess(waves.loc[waves["lauf"] == "beide", "jobs"].sum(), mpj.MAX_SUBMIT)


if __name__ == "__main__":
    unittest.main()
