"""test_prepare_talent.py -- TALENT-Lader, Task-Register und Inhaltsvergleich. Aufruf: python test_prepare_talent.py"""
import json
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

import check_content_duplicates as content
import prepare_talent
from task_registry import register_tasks, unregister_range


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo boese",))


def _write_npy_with_pickle(path, payload):
    """Schreibt eine .npy-Datei mit Objekt-Dtype, deren Pickle-Teil `payload` ist."""
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "|O", "fortran_order": False, "shape": (1,)})
        pickle.dump(payload, f, protocol=2)


class SafeLoaderTests(unittest.TestCase):
    def test_object_array_of_strings_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "C_train.npy")
            np.save(path, np.array([["a", "b"], ["c", None]], dtype=object), allow_pickle=True)
            loaded = prepare_talent.load_npy(path)
            self.assertEqual(loaded.shape, (2, 2))
            self.assertEqual(loaded[1, 0], "c")
            self.assertIsNone(loaded[1, 1])

    def test_numeric_array_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "N_train.npy")
            np.save(path, np.arange(6.0).reshape(3, 2))
            np.testing.assert_array_equal(prepare_talent.load_npy(path), np.arange(6.0).reshape(3, 2))

    def test_foreign_pickle_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "y_train.npy")
            _write_npy_with_pickle(path, _Evil())
            with self.assertRaises(pickle.UnpicklingError):
                prepare_talent.load_npy(path)


class ReadDatasetTests(unittest.TestCase):
    def _make(self, tmp, task_type, y_parts):
        folder = os.path.join(tmp, "toy")
        os.makedirs(folder)
        sizes = [len(p) for p in y_parts]
        json.dump({"task_type": task_type, "train_size": sizes[0], "val_size": sizes[1], "test_size": sizes[2],
                   "num_feature_intro": {"x1": "x1", "x2": "x2"}, "cat_feature_intro": {"farbe": "farbe"}},
                  open(os.path.join(folder, "info.json"), "w"))
        rng = np.random.RandomState(0)
        for split, y in zip(("train", "val", "test"), y_parts):
            n = len(y)
            np.save(os.path.join(folder, f"N_{split}.npy"), rng.randn(n, 2))
            np.save(os.path.join(folder, f"C_{split}.npy"), np.array([["rot"], ["nan"]] * (n // 2), dtype=object),
                    allow_pickle=True)
            np.save(os.path.join(folder, f"y_{split}.npy"), np.asarray(y))
        return folder

    def test_splits_are_joined_and_types_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make(tmp, "regression", [np.arange(4.0), np.arange(2.0), np.arange(2.0)])
            X, y, indicator, info = prepare_talent.read_dataset(folder)
            self.assertEqual(list(X.columns), ["x1", "x2", "farbe"])
            self.assertEqual(indicator, [False, False, True])
            self.assertEqual(len(X), 8)
            self.assertEqual(y.dtype, float)
            self.assertTrue(X["farbe"].isna().sum() == 4)   # "nan" als Text gilt als fehlend

    def test_task_ids_per_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, kind in [("b", "binclass"), ("a", "regression"), ("c", "multiclass"), ("d", "regression")]:
                os.makedirs(os.path.join(tmp, name))
                json.dump({"task_type": kind}, open(os.path.join(tmp, name, "info.json"), "w"))
            ids = {folder: (suite, task) for suite, task, folder, _ in prepare_talent.task_list(tmp)}
            self.assertEqual(ids, {"a": (1001, 920000), "d": (1001, 920001), "b": (1002, 930000), "c": (1002, 930001)})


class RegistryTests(unittest.TestCase):
    def test_unregister_range_removes_only_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            register_tasks(1001, [(363672, "alt"), (920000, "t1"), (920005, "t2"), (900004, "bcco")], directory=tmp)
            removed = unregister_range(1001, 920000, 940000, directory=tmp)
            self.assertEqual(removed, 2)
            self.assertEqual(np.load(os.path.join(tmp, "1001_task_indices.npy")).tolist(), [363672, 900004])
            self.assertEqual(json.load(open(os.path.join(tmp, "1001_task_names.json"))), ["alt", "bcco"])


class ExcludedTests(unittest.TestCase):
    def test_prepare_new_datasets_reads_exclusions(self):
        import prepare_new_datasets
        excluded = prepare_new_datasets.excluded_tasks()
        for task in [(1001, 363705), (1001, 363686), (1002, 363627), (1002, 363673)]:
            self.assertIn(task, excluded)
        self.assertNotIn((1002, 3711), excluded)          # elevators bleibt (Variante)

    def test_registered_tasks_are_not_excluded(self):
        excluded = pd.read_csv("excluded_duplicates.csv")
        excluded = set(zip(excluded["suite_id"], excluded["task_id"]))
        for suite in (1001, 1002):
            ids = np.load(f"task_indices/{suite}_task_indices.npy").tolist()
            self.assertFalse(excluded & {(suite, t) for t in ids})


class ContentVerdictTests(unittest.TestCase):
    def _profiles(self, a, b, cls=False):
        return content.profile("a", a, cls, "a"), content.profile("b", b, cls, "b")

    def setUp(self):
        rng = np.random.RandomState(1)
        self.base = pd.DataFrame(rng.rand(2000, 5), columns=list("abcde"))
        self.base["__ziel__"] = rng.rand(2000)

    def test_same_data_is_identical(self):
        p, q = self._profiles(self.base.sample(frac=1, random_state=0), self.base)
        self.assertEqual(content.verdict(content.compare(p, q)), "identisch")

    def test_feature_subset_is_variant(self):
        p, q = self._profiles(self.base[["a", "b", "c", "__ziel__"]], self.base)
        self.assertEqual(content.verdict(content.compare(p, q)), "Variante")

    def test_other_target_is_variant(self):
        other = self.base.copy()
        other["__ziel__"] = np.random.RandomState(2).rand(2000)
        p, q = self._profiles(other, self.base)
        self.assertEqual(content.verdict(content.compare(p, q)), "Variante")

    def test_row_subset_regression_is_variant_but_classification_identical(self):
        p, q = self._profiles(self.base.iloc[:1200], self.base)
        result = content.compare(p, q)
        self.assertEqual(content.verdict(result, classification=False), "Variante")
        self.assertEqual(content.verdict(result, classification=True), "identisch")

    def test_unrelated_data_is_different(self):
        other = pd.DataFrame(np.random.RandomState(3).rand(2000, 6), columns=list("uvwxyz"))
        p, q = self._profiles(other, self.base)
        self.assertEqual(content.verdict(content.compare(p, q)), "verschieden")


if __name__ == "__main__":
    unittest.main()
