"""Handrechenbare Tests fuer Improvability und Elo."""

import unittest
import warnings

import numpy as np

from aggregation_methods import elo, improvability


def _results(default, methods, *, rmse=False, classification=False, normalized=False):
    default_array = np.asarray(default, dtype=float)
    result = {
        "Default": default_array,
        "rmse": rmse,
        "logloss": False,
        "classification": classification,
        "normalized": normalized,
    }
    for method, scores in methods.items():
        scores_array = np.asarray(scores, dtype=float)
        result[method] = scores_array[None, :, :, :]
    return result


class TestImprovability(unittest.TestCase):
    def test_rmse_is_lower_better_and_seed_then_task_means(self):
        default = [[10.0, 20.0], [10.0, 20.0]]
        best = np.array([[[5.0, 5.0], [10.0, 10.0]]])
        worse = np.array([[[10.0, 10.0], [40.0, 40.0]]])
        result = _results(default, {"random_search": best, "tpe": worse}, rmse=True)

        values = improvability(result)

        self.assertEqual(list(values["random_search"]), ["num_leaves"])
        np.testing.assert_allclose(values["random_search"]["num_leaves"], [0.0])
        np.testing.assert_allclose(values["tpe"]["num_leaves"], [0.625])
        np.testing.assert_allclose(values["Default"]["num_leaves"], [0.5])

    def test_accuracy_is_converted_to_error(self):
        default = [[0.5, 0.6], [0.5, 0.6]]
        better = np.array([[[0.75, 0.75], [0.7, 0.7]]])
        worse = np.array([[[0.5, 0.5], [0.3, 0.3]]])
        result = _results(default, {"random_search": better, "tpe": worse}, classification=True)

        values = improvability(result)

        self.assertAlmostEqual(values["Default"]["num_leaves"][0], 0.375)
        self.assertAlmostEqual(values["tpe"]["num_leaves"][0], 0.5357142857142857)

    def test_seed_aggregation_modes_differ(self):
        # Ein Task, zwei Seeds, RMSE. A=[1,3], B=[3,1], Default=2 in beiden Seeds.
        a = np.array([[[1.0, 3.0]]])
        b = np.array([[[3.0, 1.0]]])
        result = _results([[2.0]], {"a": a, "b": b}, rmse=True)

        # mean_error: Fehler je Methode 2, 2, 2 -> bestErr 2 -> alle 0.
        mean_error = improvability(result)
        for name in ("a", "b", "Default"):
            np.testing.assert_allclose(mean_error[name]["num_leaves"], [0.0])

        # per_seed: Seed 0: A 0, B 1-1/3, Default 1-1/2; Seed 1 gespiegelt.
        # Mittel: A = B = 1/3, Default = 1/2.
        per_seed = improvability(result, seed_aggregation="per_seed")
        np.testing.assert_allclose(per_seed["a"]["num_leaves"], [1 / 3])
        np.testing.assert_allclose(per_seed["b"]["num_leaves"], [1 / 3])
        np.testing.assert_allclose(per_seed["Default"]["num_leaves"], [0.5])

    def test_mean_error_default_with_unequal_methods(self):
        # Mittlere Fehler: A=2, B=4, Default=4 -> bestErr 2 -> A 0, B 0.5, Default 0.5.
        a = np.array([[[1.0, 3.0]]])
        b = np.array([[[2.0, 6.0]]])
        result = _results([[4.0]], {"a": a, "b": b}, rmse=True)

        values = improvability(result)

        np.testing.assert_allclose(values["a"]["num_leaves"], [0.0])
        np.testing.assert_allclose(values["b"]["num_leaves"], [0.5])
        np.testing.assert_allclose(values["Default"]["num_leaves"], [0.5])

    def test_unknown_seed_aggregation_raises(self):
        result = _results([[1.0]], {"a": np.array([[[1.0]]])}, rmse=True)
        with self.assertRaises(ValueError):
            improvability(result, seed_aggregation="median")

    def test_normalized_results_raise(self):
        result = _results([[1.0]], {"a": np.array([[[1.0]]])}, rmse=True, normalized=True)
        with self.assertRaises(ValueError):
            improvability(result)

    def test_r2_as_error_warns_but_rmse_and_classification_do_not(self):
        scores = {"a": np.array([[[0.5, 0.5]]])}
        with self.assertWarns(UserWarning):
            improvability(_results([[0.4]], scores))
        for kwargs in ({"rmse": True}, {"classification": True}):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                improvability(_results([[0.4]], scores, **kwargs))


class TestElo(unittest.TestCase):
    def test_default_is_anchored_and_scale_is_400(self):
        ten_wins_one_loss = np.array([[[[0.8] * 10 + [0.2]]]])
        result = {
            "Default": np.asarray([[0.5]]),
            "rmse": False,
            "logloss": False,
            "normalized": False,
            "random_search": ten_wins_one_loss,
        }

        values = elo(result, random_state=7)

        self.assertEqual(values["Default"]["num_leaves"][0], 1000.0)
        self.assertAlmostEqual(values["random_search"]["num_leaves"][0], 1400.0, places=3)

    def test_tied_methods_both_receive_anchor_rating(self):
        default = [[4.0, 9.0]]
        tied = np.array([[[4.0, 9.0], [4.0, 9.0]]])
        result = _results(default, {"random_search": tied}, rmse=True)

        values = elo(result, random_state=3)

        np.testing.assert_allclose(values["Default"]["num_leaves"], [1000.0])
        np.testing.assert_allclose(values["random_search"]["num_leaves"], [1000.0])


if __name__ == "__main__":
    unittest.main()