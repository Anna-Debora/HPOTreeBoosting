"""test_job_io.py -- job_io.py: atomares Schreiben (Abbruch hinterlaesst keine halbe Datei und keinen .tmp-Rest),
Anhaengen ohne zweite Kopfzeile, leises Logging (optuna WARNING, gpboost-Warnungen und -Infos weg).
Aufruf: python test_job_io.py"""
import io
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import optuna
import pandas as pd

import job_io


class AtomicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "r.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_append(self):
        job_io.write_csv_atomic(pd.DataFrame(dict(fold=[0, 0], x=[1, 2])), self.path, append=True)   # neu: mit Kopf
        job_io.write_csv_atomic(pd.DataFrame(dict(fold=[1], x=[3])), self.path, append=True)
        self.assertEqual(self.path.read_text(), "fold,x\n0,1\n0,2\n1,3\n")
        job_io.write_csv_atomic(pd.DataFrame(dict(fold=[2], x=[4])), self.path)                 # ersetzen
        self.assertEqual(self.path.read_text(), "fold,x\n2,4\n")
        self.assertEqual(os.listdir(self.tmp.name), ["r.csv"])

    def test_abort_keeps_old_file(self):
        self.path.write_text("fold,x\n0,1\n")
        frame = pd.DataFrame(dict(fold=[1], x=[2]))
        with mock.patch("os.fsync", side_effect=OSError(122, "Disk quota exceeded")):
            with self.assertRaises(OSError):
                job_io.write_csv_atomic(frame, self.path, append=True)
        self.assertEqual(self.path.read_text(), "fold,x\n0,1\n")
        self.assertEqual(os.listdir(self.tmp.name), ["r.csv"])                   # kein .tmp-Rest
        with mock.patch("os.fsync", side_effect=KeyboardInterrupt):              # auch bei Signal/Abbruch
            with self.assertRaises(KeyboardInterrupt):
                job_io.write_csv_atomic(frame, Path(self.tmp.name) / "neu.csv")
        self.assertEqual(os.listdir(self.tmp.name), ["r.csv"])


class QuietTests(unittest.TestCase):
    def test_quiet_logging(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            job_io.quiet_logging()
            warnings.warn("categorical_column in param dict is overridden.")
            warnings.warn("Overriding the parameters from Reference Dataset.")
            warnings.warn("etwas anderes")
        messages = [str(w.message) for w in caught]
        self.assertIn("etwas anderes", messages)
        self.assertFalse([m for m in messages if "overridden" in m or "Reference Dataset" in m])
        self.assertEqual(optuna.logging.get_verbosity(), optuna.logging.WARNING)
        import gpboost.basic
        out = io.StringIO()
        with redirect_stdout(out):
            for chunk in ("[GPBoost] [Info] ", "Total Bins 21", "\n", "[GPBoost] [Fatal] kaputt", "\n"):
                gpboost.basic._log_native(chunk)              # die C++-Bibliothek liefert Stuecke, "\n" schliesst ab
        self.assertEqual(out.getvalue(), "[GPBoost] [Fatal] kaputt\n")


if __name__ == "__main__":
    unittest.main()
