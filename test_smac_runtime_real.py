"""Run test_smac_real with additional checks for measured objective runtimes."""

import time
import os

import numpy as np
import pandas as pd

import SMAC_method
import test_smac_real


_original_run_smac = SMAC_method.ParameterOptimizationSMAC.run_smac
_original_fill_reader_inputs = test_smac_real._fill_reader_inputs
_observations = []


def _fill_reader_inputs_with_hyperband_runtime(root):
    _original_fill_reader_inputs(root)
    path = os.path.join(
        root,
        "Results_Hyperband_2007",
        f"seed_{test_smac_real.SEED}",
        "gpboost",
        f"{test_smac_real.SUITE}_{test_smac_real.TASK}.csv",
    )
    frame = pd.read_csv(path)
    frame["val_score"] = 1.0
    frame.insert(frame.columns.get_loc("val_score") + 1, "runtime", 0.01)
    frame.to_csv(path, index=False)


def _checked_run_smac(self, *args, **kwargs):
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
        result = _original_run_smac(self, *args, **kwargs)
    finally:
        self._train_model_for_validation = original_train

    assert "lambda_l1" in result.columns
    assert result.columns.get_loc("runtime") == result.columns.get_loc("val_score") + 1
    assert result["runtime"].notna().all()
    assert (result["runtime"] > 0).all()
    assert result["lambda_l1"].between(0, 100).all()
    assert len(result) == 20
    np.testing.assert_allclose(result["runtime"].sum(), measured_training_seconds, rtol=0.1, atol=0.03)
    _observations.append((self.try_num_leaves, self.joint_tuning_depth_leaves, result["runtime"].sum()))
    return result


if __name__ == "__main__":
    SMAC_method.ParameterOptimizationSMAC.run_smac = _checked_run_smac
    test_smac_real._fill_reader_inputs = _fill_reader_inputs_with_hyperband_runtime
    test_smac_real.main()
    assert len(_observations) == 3, _observations
    print("SMAC runtime validation passed:", _observations)
