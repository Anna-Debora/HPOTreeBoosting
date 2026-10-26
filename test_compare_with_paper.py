"""test_compare_with_paper.py -- Vergleichsskript mit kuenstlichen Ergebnissen im echten Dateiformat.
Aufruf: python test_compare_with_paper.py"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import compare_with_paper as cmp

PAPER = pd.read_csv("paper_reference_values.csv")
TASKS = [(336, 361076, "wine_quality", "R2"), (337, 361065, "MagicTelescope", "Accuracy")]
SEEDS = (27225, 32244)


def _paper_value(name, metric, method, t=135):
    row = PAPER[(PAPER.dataset == name) & (PAPER.metric == metric) & (PAPER.method == method) & (PAPER.t == t)
                & ~PAPER.doppelt]
    return float(row["value"].iloc[0])


def _curve(final, n=270):
    """Steigt bis Versuch 135 auf `final` und bleibt dann dort (current_best)."""
    t = np.arange(1, n + 1)
    return final - 0.01 * np.clip(135 - t, 0, None) / 135


def _write_results(root, shift_tpe=0.0):
    for suite, task, name, metric in TASKS:
        rmse_metric = "RMSE" if metric == "R2" else "Log Loss"
        for seed in SEEDS:
            frames = []
            for fold in range(5):
                for method, paper_name in (("grid_search", "Deterministic Grid"), ("random_search", "Random Grid"),
                                           ("tpe", "TPE"), ("gp_bo", "GP-BO")):
                    final = _paper_value(name, metric, paper_name) + (shift_tpe if method == "tpe" else 0)
                    other = _paper_value(name, rmse_metric, paper_name)
                    score = _curve(final) if method != "grid_search" else np.full(270, final)
                    frames.append(pd.DataFrame({
                        "current_best_test_score": score,
                        "current_best_test_rmse": other if metric == "R2" else np.nan,
                        "current_best_test_log_loss": other if metric != "R2" else np.nan,
                        "fold": fold, "method": method, "try_num_leaves": True}))
            folder = os.path.join(root, "Results", f"seed_{seed}", "gpboost")
            os.makedirs(folder, exist_ok=True)
            pd.concat(frames).to_csv(os.path.join(folder, f"{suite}_{task}__num_leaves.csv"), index=False)
            smac = pd.concat([pd.DataFrame({
                "current_best_test_score": _curve(_paper_value(name, metric, "SMAC")),
                "current_best_test_rmse": _paper_value(name, rmse_metric, "SMAC") if metric == "R2" else np.nan,
                "current_best_test_log_loss": _paper_value(name, rmse_metric, "SMAC") if metric != "R2" else np.nan,
                "fold": fold, "method": "SMAC", "try_num_leaves": True}) for fold in range(5)])
            folder = os.path.join(root, "Results_SMAC", f"seed_{seed}", "gpboost")
            os.makedirs(folder, exist_ok=True)
            smac.to_csv(os.path.join(folder, f"{suite}_{task}.csv"), index=False)
        default = pd.concat([pd.DataFrame({
            "test_score": np.full(270, _paper_value(name, metric, "Default")),
            "test_rmse": _paper_value(name, rmse_metric, "Default") if metric == "R2" else np.nan,
            "test_log_loss": _paper_value(name, rmse_metric, "Default") if metric != "R2" else np.nan,
            "fold": fold, "method": "default"}) for fold in range(5)])
        folder = os.path.join(root, "Results_Default_Run", "gpboost")
        os.makedirs(folder, exist_ok=True)
        default.to_csv(os.path.join(folder, f"{suite}_{task}.csv"), index=False)


class CompareTests(unittest.TestCase):
    def _run(self, shift_tpe=0.0):
        with tempfile.TemporaryDirectory() as tmp:
            _write_results(tmp, shift_tpe)
            out = os.path.join(tmp, "vergleich")
            cmp.main(tmp, 135, out)
            return pd.read_csv(out + ".csv"), open(out + ".md").read()

    def test_identical_results_agree(self):
        table, text = self._run()
        self.assertEqual(set(table["method"]), {"Default", "Deterministic Grid", "Random Grid", "TPE", "GP-BO", "SMAC"})
        self.assertEqual(set(table["metric"]), {"R2", "RMSE", "Accuracy", "Log Loss"})
        self.assertTrue((table["urteil"] == "ok").all())
        self.assertLess(table["differenz"].abs().max(), 1e-9)
        self.assertTrue((table.loc[table["method"] != "Default", "seeds_wir"] == 2).all())
        self.assertIn("Gleicher Gewinner wie im Paper: 100%", text)

    def test_grid_compared_at_its_end(self):
        table, _ = self._run()
        self.assertTrue((table.loc[table["method"] == "Deterministic Grid", "budget_wir"] == 270).all())
        self.assertTrue((table.loc[table["method"] == "TPE", "budget_wir"] == 135).all())

    def test_large_shift_is_flagged(self):
        table, _ = self._run(shift_tpe=-0.05)
        tpe = table[(table["method"] == "TPE") & table["metric"].isin(["R2", "Accuracy"])]
        self.assertTrue((tpe["urteil"] == "auffaellig").all())
        others = table[(table["method"] != "TPE")]
        self.assertTrue((others["urteil"] == "ok").all())


if __name__ == "__main__":
    unittest.main()
