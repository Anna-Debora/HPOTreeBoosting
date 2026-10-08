"""Handrechenbare Tests fuer Improvability und Elo."""

import unittest

import numpy as np

from aggregation_methods import elo, improvability


def _results(default, methods, *, rmse=False):
    default_array = np.asarray(default, dtype=float)
    result = {
        "Default": default_array,
        "rmse": rmse,
        "logloss": False,
        "normalized": False,
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
        result = _results(default, {"random_search": better, "tpe": worse})

        values = improvability(result)

        self.assertAlmostEqual(values["Default"]["num_leaves"][0], 0.375)
        self.assertAlmostEqual(values["tpe"]["num_leaves"][0], 0.5357142857142857)


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