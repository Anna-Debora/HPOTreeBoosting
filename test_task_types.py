"""test_task_types.py -- zentrale Klassifikations-Erkennung. Aufruf: python test_task_types.py"""
import re
import unittest
from pathlib import Path

from task_types import is_classification, NEW_CLASSIFICATION_SUITE, NEW_REGRESSION_SUITE


class TaskTypeTests(unittest.TestCase):
    def test_old_suites(self):
        self.assertTrue(is_classification(334))
        self.assertTrue(is_classification(337))
        self.assertFalse(is_classification(335))
        self.assertFalse(is_classification(336))

    def test_new_suites(self):
        self.assertTrue(is_classification(NEW_CLASSIFICATION_SUITE))
        self.assertFalse(is_classification(NEW_REGRESSION_SUITE))

    def test_unknown_suite_raises(self):
        with self.assertRaises(ValueError):
            is_classification(999)

    def test_no_hardcoded_list_left(self):
        for name in ("methods.py", "run_default.py", "run_experiment_H_SMAC.py",
                     "run_experiments.py", "make_mini_experiment.py"):
            text = Path(name).read_text()
            code = [l for l in text.splitlines() if not l.strip().startswith("#")]
            self.assertFalse(any("[334, 337]" in l for l in code), name)

    def test_suites_for(self):
        import os, tempfile
        from task_types import suites_for, NEW_CLASSIFICATION_SUITE
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(suites_for(True, tmp), [334, 337])
            self.assertEqual(suites_for(False, tmp), [335, 336])
            open(os.path.join(tmp, f"{NEW_CLASSIFICATION_SUITE}_task_indices.npy"), "w").close()
            self.assertEqual(suites_for(True, tmp), [334, 337, NEW_CLASSIFICATION_SUITE])
            self.assertEqual(suites_for(False, tmp), [335, 336])

    def test_report_knows_new_suites(self):
        text = Path("report_data.py").read_text()
        self.assertNotIn("(334, 335, 336, 337)", text)
        self.assertNotIn("(334, 337)", text)


if __name__ == "__main__":
    unittest.main()
