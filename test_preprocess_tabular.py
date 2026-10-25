"""test_preprocess_tabular.py -- Vorverarbeitung wie bei Grinsztajn. Aufruf: python test_preprocess_tabular.py"""
import unittest

import numpy as np
import pandas as pd

from preprocess_tabular import preprocess, selection_problem


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 300
        self.X = pd.DataFrame({
            "num_ok": rng.randn(n),
            "num_few": rng.randint(0, 4, n).astype(float),            # 4 Werte (<10) -> weg
            "num_binary": rng.randint(0, 2, n).astype(float),          # 2 Werte -> kategorial
            "cat_ok": rng.choice(list("abcde"), n),
            "cat_many": [f"lvl{i}" for i in rng.randint(0, 40, n)],    # > 20 Stufen -> weg
            "with_nan": rng.randn(n),
            "mostly_nan": np.where(rng.rand(n) < 0.8, np.nan, 1.0),    # > 50 % fehlend -> Spalte weg
        })
        self.X.loc[:9, "with_nan"] = np.nan                            # 10 Zeilen -> weg
        self.indicator = [False, False, False, True, True, False, False]

    def test_regression_rules(self):
        y = pd.Series(np.arange(300, dtype=float))
        X, y2, ind, info = preprocess(self.X, y, self.indicator, classification=False)
        self.assertEqual(list(X.columns), ["num_ok", "num_binary", "cat_ok", "with_nan"])
        self.assertEqual(len(X), 290)
        self.assertEqual(len(y2), 290)
        self.assertEqual(list(ind), [False, True, True, False])
        self.assertEqual(info["dropped_missing_columns"], 1)
        self.assertEqual(info["dropped_categorical"], 1)
        self.assertEqual(info["dropped_numeric"], 1)
        self.assertEqual(info["numeric_to_categorical"], 1)

    def test_classification_two_classes_balanced(self):
        y = pd.Series(["a"] * 150 + ["b"] * 100 + ["c"] * 50)
        X, y2, ind, info = preprocess(self.X, y, self.indicator, classification=True)
        self.assertEqual(set(y2.unique()), {0, 1})
        self.assertEqual(info["classes_kept"], "a|b")
        self.assertEqual(int((y2 == 0).sum()), int((y2 == 1).sum()))
        self.assertEqual(len(X), len(y2))
        self.assertEqual(len(X), 2 * 100)        # Klasse b hat 100 Zeilen, a wird auf 100 gekuerzt

    def test_classification_unbalanced_option(self):
        y = pd.Series(["a"] * 150 + ["b"] * 100 + ["c"] * 50)
        X, y2, _, _ = preprocess(self.X, y, self.indicator, classification=True, balance=False)
        self.assertEqual(len(X), 140 + 100)      # 10 NaN-Zeilen lagen alle in Klasse a

    def test_categorical_target_with_unused_categories(self):
        # wie bei OpenML: Ziel als Kategorie-Typ, c und d fallen weg bzw. kommen nie vor
        y = pd.Series(pd.Categorical(["a"] * 150 + ["b"] * 100 + ["c"] * 50, categories=["a", "b", "c", "d"]))
        X, y2, _, info = preprocess(self.X, y, self.indicator, classification=True)
        self.assertEqual(info["classes_kept"], "a|b")
        self.assertEqual(len(X), 2 * 100)        # frueher 0 Zeilen, weil c und d mit Anzahl 0 zaehlten
        self.assertEqual(int((y2 == 0).sum()), int((y2 == 1).sum()))

    def test_one_class_raises(self):
        with self.assertRaises(ValueError):
            preprocess(self.X, pd.Series(["a"] * 300), self.indicator, classification=True)

    def test_selection_criteria(self):
        self.assertIn("Merkmale", selection_problem({"rows_after": 5000, "cols_after": 3}))
        self.assertIn("Zeilen", selection_problem({"rows_after": 2999, "cols_after": 10}))
        self.assertIn("hochdimensional", selection_problem({"rows_after": 3000, "cols_after": 300}))
        self.assertIsNone(selection_problem({"rows_after": 3000, "cols_after": 10}))


if __name__ == "__main__":
    unittest.main()
