"""test_speicher.py -- Speicherplan (speicher.py): Stufen, sbatch-Werte, Quelle Euler je Methode / je Aufgabe / Mac,
OOM hebt den Plan. Aufruf: python test_speicher.py"""
import unittest

import pandas as pd

import speicher as sp


def euler(rows):
    return pd.DataFrame([dict(zip(sp.EULER_COLUMNS, r)) for r in rows], columns=sp.EULER_COLUMNS)


class StepTests(unittest.TestCase):
    def test_steps(self):
        self.assertEqual(sp.step(0.2), 1)
        self.assertEqual(sp.step(1.0), 1)
        self.assertEqual(sp.step(1.01), 1.5)
        self.assertEqual(sp.step(4.1), 6)
        self.assertEqual(sp.step(99), 32)
        self.assertEqual(sp.next_step(4), 6)
        self.assertEqual(sp.next_step(1.5), 2)
        self.assertEqual(sp.next_step(32), 32)
        self.assertEqual(sp.from_peak(0.3), 1)                  # mindestens 1 GB
        self.assertEqual(sp.from_peak(1.2), 2)                  # 1.56 -> 2
        self.assertEqual(sp.from_peak(1.5), 2)                  # 1.95 -> 2

    def test_sbatch_mem(self):
        self.assertEqual(sp.sbatch_mem(4), "4G")
        self.assertEqual(sp.sbatch_mem(4.0), "4G")
        self.assertEqual(sp.sbatch_mem(1.5), "1536M")
        self.assertEqual(sp.fmt(1.5), "1.5")
        self.assertEqual(sp.fmt(2.0), "2")


class PlanTests(unittest.TestCase):
    def test_sources(self):
        mac = {(1, 1): 0.6, (1, 2): 0.6}
        plan = sp.Plan(mac=mac, euler=euler([
            (1, 1, "a", "tpe", 6, 1.2, 0, 0), (1, 1, "a", "SMAC", 2, 2.5, 0, 0),
            (1, 2, "b", "tpe", 1, 0.5, 0, 0)]))
        self.assertEqual(plan(1, 1, "tpe"), 2)                  # eigene Methode, 6 Jobs: 1.2 x 1.3 = 1.56
        self.assertEqual(plan.source(1, 1, "tpe")[1], "euler_methode(6)")
        self.assertEqual(plan(1, 1, "SMAC"), 4)                 # nur 2 Jobs: Aufgabe, max 2.5 -> 3.25 -> 4
        self.assertEqual(plan(1, 1, "gp_bo"), 4)                # nicht gemessen: Aufgabe
        self.assertEqual(plan.source(1, 2, "tpe")[1], "max(euler_aufgabe(1), mac)")
        self.assertEqual(plan(1, 2, "tpe"), 1.5)                # max(0.5, 0.6 + 0.4) x 1.3 = 1.3
        self.assertEqual(plan(1, 3, "tpe"), sp.UNKNOWN_GB)
        self.assertEqual(plan(1, 1, "default"), 1.5)            # Default: keine Default-Messung -> Mac

    def test_oom_raises_plan(self):
        plan = sp.Plan(mac={}, euler=euler([(1, 1, "a", "tpe", 9, 1.0, 0, 0), (1, 1, "a", "SMAC", 0, 0.0, 1, 2)]))
        self.assertEqual(plan(1, 1, "tpe"), 3)                  # OOM bei 2 GB in SMAC: alle HPO-Methoden > 2
        self.assertEqual(plan(1, 1, "SMAC"), 3)
        self.assertEqual(plan(1, 1, "default"), sp.UNKNOWN_GB)


if __name__ == "__main__":
    unittest.main()
