"""Run the three-library Hyperband check with runtime-accounting assertions."""

import time

import numpy as np

import methods
import test_hyperband_lib


_original_hyperband_method = methods.ParameterOptimization.hyperband_method
_observations = []


def _checked_hyperband_method(self, *args, **kwargs):
    original_train = self._train_model_for_validation
    measured_training_seconds = 0.0

    def _timed_train(*train_args, **train_kwargs):
        nonlocal measured_training_seconds
        started = time.perf_counter()
        try:
            return original_train(*train_args, **train_kwargs)
        finally:
            measured_training_seconds += time.perf_counter() - started

    self._train_model_for_validation = _timed_train
    try:
        result = _original_hyperband_method(self, *args, **kwargs)
    finally:
        self._train_model_for_validation = original_train

    assert "runtime" in result.columns
    assert result.columns.get_loc("runtime") == result.columns.get_loc("val_score") + 1
    assert result["runtime"].notna().all()
    assert (result["runtime"] > 0).all()
    np.testing.assert_allclose(result["runtime"].sum(), measured_training_seconds, rtol=0.1, atol=0.03)
    _observations.append((self.library, len(result), result["runtime"].sum()))
    return result


if __name__ == "__main__":
    methods.ParameterOptimization.hyperband_method = _checked_hyperband_method
    test_hyperband_lib.part1()
    assert {observation[0] for observation in _observations} == {"gpboost", "xgboost", "catboost"}, _observations
    print("Hyperband runtime validation passed:", _observations)
