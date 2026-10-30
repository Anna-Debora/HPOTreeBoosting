"""test_run_ablation_resume.py -- run_ablation.py --resume: abgebrochen und fortgesetzt = am Stueck gerechnet.
Braucht data/336_361076 (wine_quality, python prepare_one_dataset.py --suite_id 336 --task_id 361076).
Aufruf: python test_run_ablation_resume.py"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SUITE, TASK = "336", "361076"
DATA = Path(f"data/{SUITE}_{TASK}/{SUITE}_{TASK}_X.csv")
CASES = (("random_search", 6, ["--no_lambda_l1", "--paper_grid"]), ("tpe", 6, []), ("SMAC", 6, ["--no_lambda_l1"]),
         ("hyperband", 6, ["--hyperband_R", "30"]))


def run(method, trials, root, folds, flags, resume=True):
    command = [sys.executable, "run_ablation.py", "--method", method, "--seed", "27225", "--suite_id", SUITE,
               "--task_id", TASK, "--trials", str(trials), "--folds", str(folds), "--result_folder", root, *flags]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
    return subprocess.run(command + (["--resume"] if resume else []), capture_output=True, text=True, env=env)


def output(root, method):
    return Path(root) / method / "seed_27225" / "gpboost" / f"{SUITE}_{TASK}__num_leaves.csv"


def without_runtime(path):
    return pd.read_csv(path).drop(columns=["runtime"]).reset_index(drop=True)


@unittest.skipUnless(DATA.exists(), f"{DATA} fehlt")
class ResumeTests(unittest.TestCase):
    def test_resume_equals_single_run(self):
        for method, trials, flags in CASES:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                self.assertEqual(run(method, trials, a, 2, flags, resume=False).returncode, 0)
                # Lauf B: erst nur Fold 0 (wie ein Zeitlimit nach dem ersten Fold), dann fortsetzen
                self.assertEqual(run(method, trials, b, 1, flags).returncode, 0)
                second = run(method, trials, b, 2, flags)
                self.assertEqual(second.returncode, 0, second.stderr[-2000:])
                self.assertIn("WEITER", second.stdout)
                pd.testing.assert_frame_equal(without_runtime(output(a, method)), without_runtime(output(b, method)))

    def test_finished_file_is_skipped_and_broken_fold_redone(self):
        method, trials, flags = CASES[0]
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(run(method, trials, root, 2, flags).returncode, 0)
            path = output(root, method)
            full = without_runtime(path)
            again = run(method, trials, root, 2, flags)
            self.assertIn("SCHON FERTIG", again.stdout)
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-2]) + "\n")       # Fold 1 nur halb geschrieben
            redo = run(method, trials, root, 2, flags)
            self.assertIn("unvollstaendige Folds [1] verworfen", redo.stdout)
            pd.testing.assert_frame_equal(full, without_runtime(path))
            no_resume = run(method, trials, root, 2, flags, resume=False)
            self.assertEqual(no_resume.returncode, 3)          # ohne --resume wie bisher: nie ueberschreiben


if __name__ == "__main__":
    unittest.main()
