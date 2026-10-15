"""Mini end-to-end check of the single-job calls (default/hyperband/smac) and the report.

Runs abalone (335/361288), gpboost, one seed, with heavily shortened budgets (process-local
patches only). The default/hyperband/smac calls are parsed from the generated sbatch scripts.
Usage: python test_single_jobs_mini.py   (about 3-6 minutes)
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

import methods
import SMAC_method
import run_experiments
from make_euler_jobs import generate_jobs
from make_report import run_report

PROJECT = Path(__file__).resolve().parent
ROOT = Path("/tmp/mini_single")
SUITE, TASK, SEED, LIBRARY = 335, 361288, 27225, "gpboost"
TRIALS = {"grid_search": 10, "random_search": 10, "tpe": 10, "gp_bo": 10, "SMAC": 10}


def _patch():
    orig = {
        "grid": methods.ParameterOptimization.grid_search_method,
        "opt": optuna.study.Study.optimize,
        "gp": methods.gp_minimize,
        "hb": methods.ParameterOptimization.hyperband_method,
        "val": methods.ParameterOptimization._train_model_for_validation,
        "sval": SMAC_method.ParameterOptimizationSMAC._train_model_for_validation,
        "scenario": SMAC_method.Scenario,
    }

    def grid(self, *a, num_try_random=None, **k):
        n = TRIALS["grid_search"] if num_try_random in (None, 270) else min(num_try_random, TRIALS["random_search"])
        return orig["grid"](self, *a, num_try_random=n, **k)

    def optimize(self, func, *a, **k):
        k["n_trials"] = min(k.get("n_trials", TRIALS["tpe"]), TRIALS["tpe"])
        return orig["opt"](self, func, *a, **k)

    def gp(func, dimensions, *a, **k):
        k["n_calls"] = TRIALS["gp_bo"]
        return orig["gp"](func, dimensions, *a, **k)

    def hb(self, *a, **k):
        k["R"] = 9
        return orig["hb"](self, *a, **k)

    def bounded(original, is_smac=False):
        def run(self, X_train, y_train, X_val, y_val, params, num_boost_round=1000):
            if not is_smac:  # SMAC passes a Configuration object that must stay intact
                params = dict(params)
                if "n_iter" in params:
                    params["n_iter"] = min(int(params["n_iter"]), 80)
            return original(self, X_train, y_train, X_val, y_val, params, num_boost_round=min(num_boost_round, 80))
        return run

    methods.ParameterOptimization.grid_search_method = grid
    optuna.study.Study.optimize = optimize
    methods.gp_minimize = gp
    methods.ParameterOptimization.hyperband_method = hb
    methods.ParameterOptimization._train_model_for_validation = bounded(orig["val"])
    SMAC_method.ParameterOptimizationSMAC._train_model_for_validation = bounded(orig["sval"], is_smac=True)
    SMAC_method.Scenario = lambda *a, **k: orig["scenario"](*a, **{**k, "n_trials": TRIALS["SMAC"]})


def _job_argv(kind, tasks_file, results):
    """Return the python argv of job 0, parsed from the generated sbatch script."""
    script, manifest, _, _ = generate_jobs(tasks_file, [SEED], [LIBRARY], results, ROOT / f"jobs_{kind}",
                                           project_dir=ROOT / "proj", kind=kind)
    text = script.read_text()
    block = re.search(r"^python (.*?)(?=\n\S|\Z)", text.split("read -r")[1], re.S | re.M).group(0)
    fields = re.search(r"read -r (.*?) <<<", text).group(1).split()
    values = dict(zip(fields, manifest.read_text().splitlines()[0].split("\t")))
    line = re.sub(r"\\\n", " ", block)
    for name, value in values.items():
        line = line.replace(f'"${name}"', value)
    return shlex.split(line)


def main():
    if os.environ.get("FRESH", "1") == "1":
        shutil.rmtree(ROOT, ignore_errors=True)
    (ROOT / "proj").mkdir(parents=True, exist_ok=True)
    if not (ROOT / "proj" / "data").exists():
        (ROOT / "proj" / "data").symlink_to(PROJECT / "data", target_is_directory=True)
    results = ROOT / "res"
    tasks_file = ROOT / "tasks.txt"
    tasks_file.write_text(f"{SUITE} {TASK}\n")
    assert not list((ROOT / "proj").glob("suite_*_tasks.txt"))

    _patch()
    os.chdir(ROOT / "proj")
    sys.path.insert(0, str(PROJECT))
    # Main methods are not part of the new single-job calls; run them with the existing runner.
    main_csv = results / "Results" / f"seed_{SEED}" / LIBRARY / f"{SUITE}_{TASK}.csv"
    if not main_csv.exists():
        run_experiments.main(argparse.Namespace(suite_id=str(SUITE), task_id=str(TASK), seed=str(SEED),
                                                result_folder=str(results / "Results"), library=LIBRARY))
    for kind, script in (("default", "run_default.py"), ("hyperband", "run_experiment_H_SMAC.py"),
                         ("smac", "run_experiment_H_SMAC.py")):
        argv = _job_argv(kind, tasks_file, results)
        print("RUN", " ".join(argv), flush=True)
        assert argv[:2] == ["python", script], argv
        sys.argv = [str(PROJECT / script), *argv[2:]]
        runpy.run_path(str(PROJECT / script), run_name="__main__")

    default_path = results / "Results_Default_Run" / LIBRARY / f"{SUITE}_{TASK}.csv"
    frame = pd.read_csv(default_path)
    frame.groupby("fold", sort=False).head(max(TRIALS.values())).to_csv(default_path, index=False)

    out = ROOT / "report"
    data = run_report(results, [LIBRARY], out, budgets=[10])
    bad = [w for w in data["warnings"] if "Fehlt" in w or "create_scores_dict" in w]
    table = pd.read_csv(out / "tables" / "overall_metrics.csv")
    norm = table["normalized_score"]
    print("warnings:", *data["warnings"], sep="\n  ")
    print("missing files:", data["missing_files"])
    print("normalized_score non-NaN:", int(norm.notna().sum()), "of", len(norm))
    assert not bad, bad
    assert not data["missing_files"]
    assert norm.notna().any()
    print("OK")


if __name__ == "__main__":
    main()
