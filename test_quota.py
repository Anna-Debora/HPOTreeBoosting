"""test_quota.py -- quota.py: lquota-Tabelle und df lesen, Schwelle 80 % fuer Home (Platz und Dateien), Scratch nur
Warnung, fehlende Befehle blockieren nicht. Aufruf: python test_quota.py"""
import os
import unittest
from unittest import mock

import quota

LQUOTA = """+-----------------------------+-------------+------------------+------------------+------------------+
| Storage location:           | Quota type: | Used:            | Soft quota:      | Hard quota:      |
+-----------------------------+-------------+------------------+------------------+------------------+
| /cluster/home/anna          | space       |         41.00 GB |         50.00 GB |         55.00 GB |
| /cluster/home/anna          | files       |             8961 |           160000 |           200000 |
+-----------------------------+-------------+------------------+------------------+------------------+
| /cluster/scratch/anna       | space       |         16.38 kB |          2.50 TB |          2.70 TB |
| /cluster/scratch/anna       | files       |                 2 |          1000000 |          1500000 |
+-----------------------------+-------------+------------------+------------------+------------------+
"""
DF = """Filesystem     1024-blocks     Used Available Capacity Mounted on
nas:/home/anna    52428800 20971520  31457280      40% /cluster/home/anna
"""
ENV = dict(HOME="/cluster/home/anna", SCRATCH="/cluster/scratch/anna", USER="anna")


class QuotaTests(unittest.TestCase):
    def test_parse(self):
        rows = quota.parse_lquota(LQUOTA)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], dict(ort="/cluster/home/anna", art="space", used=41e9, soft=50e9, hard=55e9))
        self.assertAlmostEqual(rows[2]["used"], 16380)
        self.assertEqual(quota.parse_size("1.5 TiB"), 1.5 * 2 ** 40)
        self.assertIsNone(quota.parse_size("viel"))
        self.assertEqual(quota.parse_lquota("lquota: command not found"), [])
        self.assertEqual(quota.parse_df(DF, "/x")[0]["used"], 20971520 * 1024)

    @mock.patch.dict(os.environ, ENV)
    def test_home_over_limit_blocks(self):
        info = quota.usage(lquota_text=LQUOTA)
        self.assertAlmostEqual(info["home"]["anteil"], 0.82)
        self.assertEqual(info["home"]["quelle"], "lquota")
        ok, lines = quota.check(info)
        self.assertFalse(ok)
        self.assertIn("WARNUNG: Home: 41.0 GB / 50.0 GB (82%)", lines[0])
        self.assertIn("Dateien 8961 / 160000", quota.describe("home", info["home"]))

    @mock.patch.dict(os.environ, ENV)
    def test_files_count_and_scratch_warning_only(self):
        text = LQUOTA.replace("41.00 GB", "10.00 GB").replace("             8961", "           150000") \
            .replace("16.38 kB", "2.40 TB")
        ok, lines = quota.check(quota.usage(lquota_text=text))
        self.assertFalse(ok)                                       # 150000 / 160000 Dateien
        text = LQUOTA.replace("41.00 GB", "10.00 GB").replace("16.38 kB", "2.40 TB")
        ok, lines = quota.check(quota.usage(lquota_text=text))
        self.assertTrue(ok)                                        # Scratch 96 %: nur Warnung
        self.assertEqual(len(lines), 1)
        self.assertIn("Scratch", lines[0])

    @mock.patch.dict(os.environ, ENV)
    def test_fallback_df_and_unknown(self):
        info = quota.usage(lquota_text=None, df_texts={"home": DF, "scratch": None})
        self.assertEqual(info["home"]["quelle"], "df")
        self.assertAlmostEqual(info["home"]["anteil"], 0.4)
        self.assertIsNone(info["scratch"]["anteil"])
        ok, lines = quota.check(info)
        self.assertTrue(ok)
        self.assertIn("unbekannt", quota.describe("scratch", info["scratch"]))
        with mock.patch("subprocess.run", side_effect=OSError):            # weder lquota noch df
            ok, lines = quota.check()
        self.assertTrue(ok)
        self.assertIn("nicht geprueft", lines[0])


if __name__ == "__main__":
    unittest.main()
