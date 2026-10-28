"""test_make_repro_table.py -- Tabelle aus kuenstlichen Mini-Ergebnissen im echten Dateiformat.
wine_quality und MagicTelescope vollstaendig (Werte = Paper), sulfur nur ein abgebrochener Lauf (3 von 5 Folds),
alle anderen Aufgaben fehlen. Aufruf: python test_make_repro_table.py"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import make_repro_table as mrt
from test_compare_with_paper import _write_results


def _write_partial_sulfur(root):
    frames = [pd.DataFrame({"current_best_test_score": np.full(270, 0.8), "current_best_test_rmse": 0.01,
                            "current_best_test_log_loss": np.nan, "fold": fold, "method": method,
                            "try_num_leaves": True})
              for fold in range(3) for method in ("grid_search", "random_search", "tpe", "gp_bo")]
    folder = os.path.join(root, "Results", "seed_27225", "gpboost")
    pd.concat(frames).to_csv(os.path.join(folder, "336_361085__num_leaves.csv"), index=False)


class ReproTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _write_results(self.tmp.name)
        _write_partial_sulfur(self.tmp.name)
        self.out = os.path.join(self.tmp.name, "repro_table")
        self.table = mrt.main(self.tmp.name, 135, self.out)
        self.text = open(self.out + ".md").read()

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout(self):
        self.assertEqual(len(self.table), 7 * 2 * 7)  # Aufgaben x Kennzahlen x Methoden
        self.assertEqual(list(self.table["method"].unique()), mrt.METHODS)
        self.assertTrue(os.path.exists(self.out + ".csv"))
        for heading in ("## R2", "## RMSE", "## Accuracy", "## Log Loss"):
            self.assertIn(heading, self.text)

    def test_complete_tasks_match_paper(self):
        done = self.table[self.table["dataset"].isin(["wine_quality", "MagicTelescope"])
                          & (self.table["method"] != "Hyperband")]
        self.assertLess(done["differenz"].abs().max(), 1e-9)
        self.assertTrue((done.loc[done["method"] != "Default", "seeds"] == 2).all())
        self.assertTrue((done.loc[done["method"] == "Default", "seeds"] == 1).all())
        self.assertTrue((done["anmerkung"].fillna("") == "").all())

    def test_missing_and_partial_runs_are_empty_with_note(self):
        sulfur = self.table[(self.table["dataset"] == "sulfur") & (self.table["method"] != "Hyperband")]
        self.assertTrue(sulfur["wir"].isna().all())
        self.assertTrue((sulfur["seeds"] == 0).all())
        self.assertTrue(sulfur["anmerkung"].str.startswith(mrt.OPEN).all())
        self.assertIn("Lauf noch offen oder Zeitlimit", self.text)
        hyperband = self.table[self.table["method"] == "Hyperband"]
        self.assertTrue(hyperband["wir"].isna().all() and (hyperband["anmerkung"] == "nicht im Testlauf").all())

    def test_one_seed_counts_when_other_incomplete(self):
        root = self.tmp.name
        path = os.path.join(root, "Results", "seed_32244", "gpboost", "336_361076__num_leaves.csv")
        frame = pd.read_csv(path)
        frame[frame["fold"] < 2].to_csv(path, index=False)
        table = mrt.main(root, 135, self.out)
        tpe = table[(table["dataset"] == "wine_quality") & (table["method"] == "TPE")]
        self.assertTrue((tpe["seeds"] == 1).all())
        self.assertTrue(tpe["wir"].notna().all())
        self.assertTrue(tpe["anmerkung"].str.contains("Seed 32244 unvollstaendig").all())

    def test_no_results_at_all(self):
        with tempfile.TemporaryDirectory() as empty:
            table = mrt.main(empty, 135, os.path.join(empty, "t"))
        self.assertTrue(table["wir"].isna().all())


if __name__ == "__main__":
    unittest.main()
