"""test_make_table5.py -- Table 5 aus kuenstlichen Ergebnissen im echten Dateiformat, Werte von Hand gerechnet.
Aufruf: python test_make_table5.py"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import make_table5 as t5

REG, CLS = (336, 361076), (334, 361286)
# Endwerte je Methode (alle Seeds und Folds gleich): R2 bzw. Accuracy, dazu RMSE bzw. Log Loss
VALUES = {"default": (0.1, 2.0), "grid_search": (0.5, 0.9), "gp_bo": (0.6, 0.8), "hyperband": (0.4, 1.0),
          "random_search": (0.7, 0.7), "SMAC": (0.9, 0.5), "tpe": (0.8, 0.6)}
SEEDS = (11, 22)


def frame(method, score, loss, rows, classification, folds=5):
    loss_column = "current_best_test_log_loss" if classification else "current_best_test_rmse"
    other = "current_best_test_rmse" if classification else "current_best_test_log_loss"
    parts = []
    for fold in range(folds):
        scores = score if isinstance(score, list) else [score] * rows
        losses = loss if isinstance(loss, list) else [loss] * rows
        parts.append(pd.DataFrame({"iter": range(len(scores)), "current_best_test_score": scores,
                                   loss_column: losses, other: np.nan, "try_num_leaves": True, "fold": fold,
                                   "method": method}))
    return pd.concat(parts, ignore_index=True)


def write_new_layout(root, budget=3, values=VALUES, task=REG, seeds=SEEDS, classification=False):
    suite, task_id = task
    for method, (score, loss) in values.items():
        if method == "default":
            path = os.path.join(root, "default", "gpboost")
            os.makedirs(path, exist_ok=True)
            frame(method, score, loss, 270, classification).to_csv(f"{path}/{suite}_{task_id}.csv", index=False)
            continue
        rows = 2 if method == "hyperband" else budget
        for seed in seeds:
            path = os.path.join(root, method, f"seed_{seed}", "gpboost")
            os.makedirs(path, exist_ok=True)
            frame(method, score, loss, rows, classification).to_csv(f"{path}/{suite}_{task_id}__num_leaves.csv",
                                                                     index=False)


def value(table, block, metric, method):
    row = table[(table["block"] == block) & (table["metric"] == metric) & (table["method"] == method)]
    return float(row["value"].iloc[0])


class Table5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranks_and_relative_differences(self):
        write_new_layout(self.root)
        table, notes = t5.table5(t5.load(self.root), budget=3)
        self.assertEqual(notes, [])
        expected_rank = {"SMAC": 1, "TPE": 2, "Random Grid": 3, "GP-BO": 4, "Deterministic Grid": 5,
                         "Hyperband": 6, "Default": 7}
        for method, rank in expected_rank.items():
            self.assertEqual(value(table, "rank", "R2", method), rank)
            self.assertEqual(value(table, "rank", "RMSE", method), rank)   # RMSE: kleiner ist besser
        self.assertAlmostEqual(value(table, "rel", "R2", "Default"), 0.8 / 0.9)
        self.assertAlmostEqual(value(table, "rel", "R2", "TPE"), 0.1 / 0.9)
        self.assertAlmostEqual(value(table, "rel", "RMSE", "Default"), 1.5 / 0.5)
        self.assertAlmostEqual(value(table, "rel", "RMSE", "SMAC"), 0.0)

    def test_normalization_code_pool(self):
        # Pool R2: Default 0.1 x (3 Versuche x 6 Methoden) = 18, Hyperband 0.4 x 4, sonst je 6 Werte
        # -> 46 Werte, Q10 an Position 4.5 = 0.1, max 0.9 -> (x - 0.1) / 0.8
        write_new_layout(self.root)
        table, _ = t5.table5(t5.load(self.root), budget=3)
        for method, expected in {"Default": 0.0, "Hyperband": 0.375, "Deterministic Grid": 0.5, "GP-BO": 0.625,
                                 "Random Grid": 0.75, "TPE": 0.875, "SMAC": 1.0}.items():
            self.assertAlmostEqual(value(table, "norm", "R2", method), expected)
        # RMSE: min 0.5, Pool sortiert ..., 1.0 x 4, 2.0 x 18 -> Q90 an Position 40.5 = 2.0
        # -> (x - 0.5) / 1.5, oben bei 1 abgeschnitten
        self.assertAlmostEqual(value(table, "norm", "RMSE", "SMAC"), 0.0)
        self.assertAlmostEqual(value(table, "norm", "RMSE", "Deterministic Grid"), 0.4 / 1.5)
        self.assertAlmostEqual(value(table, "norm", "RMSE", "Default"), 1.0)

    def test_normalization_text_pool(self):
        # ohne Default: 34 Werte, Q10 an Position 3.3 -> 0.4 + 0.3 x 0.1 = 0.43
        write_new_layout(self.root)
        table, _ = t5.table5(t5.load(self.root), budget=3, pool="text")
        self.assertAlmostEqual(value(table, "norm", "R2", "Deterministic Grid"), (0.5 - 0.43) / (0.9 - 0.43))
        self.assertAlmostEqual(value(table, "norm", "R2", "Default"), 0.0)   # unter Q10 -> 0

    def test_budget_selects_row_and_skips_short_curves(self):
        values = dict(VALUES, random_search=([0.2, 0.5, 0.7, 0.95], [0.9, 0.8, 0.7, 0.45]))
        write_new_layout(self.root, budget=4, values=values)
        table, _ = t5.table5(t5.load(self.root), budget=3)
        self.assertEqual(value(table, "rank", "R2", "Random Grid"), 3)      # Zeile 3: 0.7
        table, notes = t5.table5(t5.load(self.root), budget=4)
        self.assertEqual(value(table, "rank", "R2", "Random Grid"), 1)      # Zeile 4: 0.95
        self.assertAlmostEqual(value(table, "rel", "R2", "SMAC"), 0.05 / 0.95)
        # nur 4 Zeilen bei Budget 4: alle anderen Methoden haben 4 Zeilen, Hyperband zaehlt seine Zeilen
        self.assertFalse([n for n in notes if n[2] == "Random Grid"])

    def test_incomplete_folds_and_trials_are_skipped(self):
        write_new_layout(self.root)
        path = os.path.join(self.root, "tpe", "seed_22", "gpboost", "336_361076__num_leaves.csv")
        frame("tpe", 0.0, 9.0, 3, False, folds=4).to_csv(path, index=False)   # nur 4 Folds, schlechte Werte
        table, notes = t5.table5(t5.load(self.root), budget=3)
        self.assertEqual(value(table, "rank", "R2", "TPE"), 2)                # Seed 22 zaehlt nicht
        self.assertIn(("R2", 336, 361076, "TPE", 22, "4 Folds"), notes)
        table, notes = t5.table5(t5.load(self.root), budget=5)               # zu wenige Versuche
        self.assertNotIn("TPE", set(table["method"]))
        self.assertTrue(any("3 < 5 Versuche" in n[-1] for n in notes))

    def test_classification_directions(self):
        write_new_layout(self.root, task=CLS, classification=True)
        table, _ = t5.table5(t5.load(self.root), budget=3)
        self.assertEqual(set(table["metric"]), {"Accuracy", "Log Loss"})
        self.assertEqual(value(table, "rank", "Accuracy", "SMAC"), 1)
        self.assertEqual(value(table, "rank", "Log Loss", "SMAC"), 1)
        self.assertEqual(value(table, "rank", "Log Loss", "Default"), 7)

    def test_ties_follow_paper_code_order(self):
        tie = dict(VALUES, tpe=(0.9, 0.5))          # TPE gleich gut wie SMAC
        write_new_layout(self.root, values=tie)
        table, _ = t5.table5(t5.load(self.root), budget=3)
        # stabile Sortierung in Spaltenreihenfolge: SMAC steht vor TPE, bekommt bei "hoeher besser" Rang 2
        self.assertEqual(value(table, "rank", "R2", "TPE"), 1)
        self.assertEqual(value(table, "rank", "R2", "SMAC"), 2)

    def test_repro_layout_gives_same_table(self):
        write_new_layout(self.root)
        new = t5.table5(t5.load(self.root), budget=3)[0]
        repro = os.path.join(self.root, "repro")
        suite, task = REG
        for seed in SEEDS:
            main = pd.concat([frame(m, *VALUES[m], 3, False) for m in ("grid_search", "random_search", "tpe", "gp_bo")])
            path = os.path.join(repro, "Results", f"seed_{seed}", "gpboost")
            os.makedirs(path)
            main.to_csv(f"{path}/{suite}_{task}__num_leaves.csv", index=False)
            for folder, method, rows in (("Results_SMAC", "SMAC", 3), ("Results_Hyperband", "hyperband", 2)):
                path = os.path.join(repro, folder, f"seed_{seed}", "gpboost")
                os.makedirs(path)
                frame(method, *VALUES[method], rows, False).to_csv(f"{path}/{suite}_{task}.csv", index=False)
        path = os.path.join(repro, "Results_Default_Run", "gpboost")
        os.makedirs(path)
        frame("default", *VALUES["default"], 270, False).to_csv(f"{path}/{suite}_{task}.csv", index=False)
        old = t5.table5(t5.load(repro), budget=3)[0]
        pd.testing.assert_frame_equal(new.reset_index(drop=True), old.reset_index(drop=True))

    def test_paper_table_matches_paper(self):
        paper = t5.paper_table()
        self.assertEqual(len(paper), 3 * 4 * 7)
        smac = paper[(paper["block"] == "norm") & (paper["metric"] == "R2") & (paper["method"] == "SMAC")]
        self.assertAlmostEqual(float(smac["paper"].iloc[0]), 0.898)
        rank = paper[(paper["block"] == "rank") & (paper["metric"] == "Log Loss") & (paper["method"] == "Hyperband")]
        self.assertAlmostEqual(float(rank["paper"].iloc[0]), 6.043)

    def test_main_writes_outputs(self):
        write_new_layout(self.root)
        out = os.path.join(self.root, "t5")
        t5.main([("Probe", self.root, [3])], out)
        for suffix in (".csv", ".md", ".pdf"):
            self.assertTrue(os.path.getsize(out + suffix) > 0)
        text = open(out + ".md").read()
        self.assertIn("| norm | R2 | 0.000 | 0.500 | 0.625 | 0.375 | 0.750 | 1.000 | 0.875 |", text)


if __name__ == "__main__":
    unittest.main()
