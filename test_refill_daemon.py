"""test_refill_daemon.py -- jobs_paper/refill_daemon.sbatch mit nachgebildeten module/squeue/sbatch/scancel/python:
Nachfolger in gleicher Partition und mit gleichem Zeitlimit (afterany), kein zweiter Daemon, wartender Nachfolger
wird nicht verdoppelt, Exit-Code 3 beendet die Kette. Aufruf: python test_refill_daemon.py (braucht bash)"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "jobs_paper" / "refill_daemon.sbatch"
FAKES = {
    "module": "exit 0",
    "squeue": """state=PENDING; for a in "$@"; do case "$a" in RUNNING) state=RUNNING;; esac; done
f="$FAKE/$state"; [ -f "$f" ] && cat "$f"; exit 0""",
    "sbatch": 'echo "sbatch $*" >> "$FAKE/calls"; echo "777;euler"',
    "scancel": 'echo "scancel $*" >> "$FAKE/calls"',
    "python": 'echo "python $*" >> "$FAKE/calls"; exit "${FAKE_RC:-3}"',
}


class DaemonScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fake, self.home = root / "fake", root / "home"
        (self.home / "HPOTreeBoosting" / ".venv" / "bin").mkdir(parents=True)
        (self.home / "HPOTreeBoosting" / ".venv" / "bin" / "activate").write_text("")
        (self.home / "HPOTreeBoosting" / "jobs_paper").mkdir()
        shutil.copy(SCRIPT, self.home / "HPOTreeBoosting" / "jobs_paper" / "refill_daemon.sbatch")
        (self.fake / "bin").mkdir(parents=True)
        for name, body in FAKES.items():
            path = self.fake / "bin" / name
            path.write_text("#!/bin/bash\n" + body + "\n")
            path.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, running="", pending="", **env):
        (self.fake / "RUNNING").write_text(running)
        (self.fake / "PENDING").write_text(pending)
        environ = dict(os.environ, HOME=str(self.home), FAKE=str(self.fake), SLURM_JOB_ID="600",
                       PATH=f"{self.fake / 'bin'}:{os.environ['PATH']}", REFILL_INTERVAL="1", **env)
        result = subprocess.run(["bash", str(SCRIPT), "--run", "beide"], env=environ, capture_output=True,
                                text=True, timeout=60)
        calls = (self.fake / "calls").read_text().splitlines() if (self.fake / "calls").exists() else []
        return result, calls

    def test_successor_same_partition_and_time(self):
        result, calls = self.run_script(running="600\n", DAEMON_PARTITION="normal.4h", DAEMON_TIME="4:00:00",
                                        DAEMON_SECONDS="14400")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0], "sbatch --parsable --partition=normal.4h --time=4:00:00 --export=ALL "
                                   "--dependency=afterany:600 jobs_paper/refill_daemon.sbatch --run beide")
        self.assertEqual(calls[1], "python refill.py --run beide")
        self.assertEqual(calls[2], "scancel 777")                 # Exit-Code 3: Kette zu Ende, Nachfolger weg

    def test_defaults_without_environment(self):
        result, calls = self.run_script(running="600\n")
        self.assertIn("--partition=normal.4h --time=4:00:00", calls[0])
        self.assertIn("in normal.4h, Zeitlimit 4:00:00, laeuft bis 13200 s", result.stdout)

    def test_second_daemon_exits(self):
        result, calls = self.run_script(running="600\n599\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls, [])                               # kein Nachfolger, kein refill.py
        self.assertIn("anderer Daemon laeuft schon: 599", result.stdout)

    def test_pending_successor_not_doubled_and_ends_before_limit(self):
        result, calls = self.run_script(running="600\n", pending="650\n", FAKE_RC="0", DAEMON_SECONDS="1203",
                                        DAEMON_TIME="0:20:03")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(c.startswith("sbatch") for c in calls))
        self.assertFalse(any(c.startswith("scancel") for c in calls))  # Nachfolger 650 uebernimmt
        self.assertIn("Nachfolger 650 uebernimmt", result.stdout)


if __name__ == "__main__":
    unittest.main()
