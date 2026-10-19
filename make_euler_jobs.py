"""Generate (but never submit) Slurm array jobs for run_experiments.py."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

STRATEGIES = ("max_depth", "num_leaves", "joint", "num_iter")
KINDS = ("main", "default", "hyperband", "smac")

RUNTIME_ESTIMATES = {
    "main": (
        "main: per job (one task, seed, library, strategy) 5 folds x (random 270 + TPE 270 + GP-BO 270 "
        "[+ grid <=270 for max_depth/num_leaves]) validation fits, each followed by a test refit, i.e. about "
        "8k-11k fits of <=1000 rounds (early stopping 20). Unmeasured guess: 1-4 h per job on 1 core."
    ),
    "default": (
        "default: 5 folds x 1 fit (default_method), result repeated 270x. Unmeasured guess: seconds to a few minutes per job."
    ),
    "hyperband": (
        "hyperband (one job per task/seed/library): R=2150, factor=2.8 -> s_max=7, i.e. 8 brackets (not 11); about "
        "3.6k config trainings and ~124k boosting rounds per fold and strategy; x5 folds x3 strategies = ~1.9M "
        "rounds plus one test refit per surviving config. Guess: 0.5-2.5 h per job at 1-5 ms/round; "
        "large tasks may hit the 4 h limit and then lose the whole task (CSV is written per strategy)."
    ),
    "smac": (
        "smac (one job per task/seed/library): 270 trials x 5 folds x 3 strategies = 4050 validation fits + "
        "4050 test refits of <=1000 rounds. Guess: 0.5-2 h per job for gpboost; xgboost/catboost per-round cost "
        "is unmeasured (catboost is typically slower), so large tasks may hit the 4 h limit."
    ),
}


def _sbatch_header(job_count, logs):
    return f"""#!/bin/bash
#SBATCH --job-name=hpo-tree-boosting
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-{job_count - 1}
#SBATCH --output={shlex.quote(str((logs / '%x_%A_%a.out').resolve()))}
#SBATCH --error={shlex.quote(str((logs / '%x_%A_%a.err').resolve()))}
#SBATCH --export=ALL

set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
"""


def generate_other_jobs(kind, tasks_file, seeds, libraries, results, out, project_dir="."):
    """Jobs for run_default.py (kind=default) or run_experiment_H_SMAC.py (hyperband/smac)."""
    tasks = read_tasks(tasks_file)
    libraries = list(libraries)
    results_root = Path(results).resolve()
    out = Path(out)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    project_shell = shlex.quote(str(Path(project_dir).resolve()))
    manifest = out / "jobs.tsv"

    if kind == "default":
        rows = [(suite, task, library) for suite, task in tasks for library in libraries]
        body = f"""IFS=$'\\t' read -r SUITE_ID TASK_ID LIBRARY <<< "$JOB_LINE"
python run_default.py \\
    --suite_id "$SUITE_ID" \\
    --task_id "$TASK_ID" \\
    --library "$LIBRARY" \\
    --result_folder {shlex.quote(str(results_root / 'Results_Default_Run'))}
"""
    else:
        if not seeds:
            raise ValueError(f"--seeds is required for kind={kind}")
        rows = [(suite, task, int(seed), library) for suite, task in tasks
                for seed in seeds for library in libraries]
        folder_flag = ("--result_folder_H", "Results_Hyperband_2007") if kind == "hyperband" \
            else ("--result_folder_SMAC", "Results_SMAC")
        body = f"""IFS=$'\\t' read -r SUITE_ID TASK_ID SEED LIBRARY <<< "$JOB_LINE"
python run_experiment_H_SMAC.py \\
    --suite_id "$SUITE_ID" \\
    --task_id "$TASK_ID" \\
    --seed "$SEED" \\
    --library "$LIBRARY" \\
    --only {kind} \\
    {folder_flag[0]} {shlex.quote(str(results_root / folder_flag[1]))}
"""
    manifest.write_text("".join("\t".join(map(str, row)) + "\n" for row in rows))
    manifest_shell = shlex.quote(str(manifest.resolve()))
    script = _sbatch_header(len(rows), logs) + f"""cd {project_shell}
source .venv/bin/activate

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {manifest_shell})
""" + body
    script_path = out / f"run_{kind}_array.sbatch"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path, manifest, f"sbatch {shlex.quote(str(script_path.resolve()))}", len(rows)


def read_tasks(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text())
        tasks = []
        for record in records:
            if isinstance(record, dict):
                tasks.append((int(record.get("suite_id", record.get("suite", 335))),
                              int(record.get("task_id", record.get("task", record.get("id"))))))
            else:
                suite, task = record[:2]
                tasks.append((int(suite), int(task)))
        return tasks
    if path.suffix.lower() == ".csv":
        frame = __import__("pandas").read_csv(path)
        return [(int(row.get("suite_id", row.get("suite", 335))),
                 int(row.get("task_id", row.get("task", row.get("id")))))
                for _, row in frame.iterrows()]

    tasks = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        fields = value.replace(",", " ").split()
        if len(fields) == 1:
            tasks.append((335, int(fields[0])))
        else:
            tasks.append((int(fields[0]), int(fields[1])))
    if not tasks:
        raise ValueError(f"No task IDs found in {path}")
    return tasks


def generate_jobs(tasks_file, seeds, libraries, results, out, project_dir=".", kind="main"):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if kind != "main":
        return generate_other_jobs(kind, tasks_file, seeds, libraries, results, out, project_dir)
    tasks = read_tasks(tasks_file)
    seeds = [int(seed) for seed in seeds]
    libraries = list(libraries)
    jobs = [(suite_id, task_id, seed, library)
            for suite_id, task_id in tasks
            for seed in seeds
            for library in libraries]
    jobs = [(*job, strategy) for job in jobs for strategy in STRATEGIES]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = out / "jobs.tsv"
    manifest.write_text("".join(
        f"{suite_id}\t{task_id}\t{seed}\t{library}\t{strategy}\n"
        for suite_id, task_id, seed, library, strategy in jobs
    ))

    manifest_shell = shlex.quote(str(manifest.resolve()))
    results_shell = shlex.quote(str(Path(results).resolve()))
    project_shell = shlex.quote(str(Path(project_dir).resolve()))
    script = f"""#!/bin/bash
#SBATCH --job-name=hpo-tree-boosting
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-{len(jobs) - 1}
#SBATCH --output={shlex.quote(str((logs / '%x_%A_%a.out').resolve()))}
#SBATCH --error={shlex.quote(str((logs / '%x_%A_%a.err').resolve()))}
#SBATCH --export=ALL

set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
module load stack/2024-06 gcc/12.2.0 python/3.9.18
cd {project_shell}
source .venv/bin/activate

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {manifest_shell})
IFS=$'\\t' read -r SUITE_ID TASK_ID SEED LIBRARY STRATEGY <<< "$JOB_LINE"
python run_experiments.py \\
    --suite_id "$SUITE_ID" \\
    --task_id "$TASK_ID" \\
    --seed "$SEED" \\
    --library "$LIBRARY" \\
    --strategy "$STRATEGY" \\
    --result_folder {results_shell}

# run_experiments.py writes the task CSV only after all strategies complete.
# If the job reaches the 4-hour limit, partial results are not saved and are lost.
"""
    script_path = out / "run_hpo_array.sbatch"
    script_path.write_text(script)
    script_path.chmod(0o755)
    command = f"sbatch {shlex.quote(str(script_path.resolve()))}"
    return script_path, manifest, command, len(jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--kind", choices=KINDS, default="main")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--libraries", nargs="+", choices=("gpboost", "xgboost", "catboost"),
                        default=("gpboost", "xgboost", "catboost"))
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args(argv)
    if args.kind != "default" and not args.seeds:
        parser.error(f"--seeds is required for --kind {args.kind}")
    script, manifest, command, count = generate_jobs(
        args.tasks_file, args.seeds, args.libraries, args.results, args.out,
        project_dir=args.project_dir, kind=args.kind,
    )
    print(f"Generated {count} array jobs")
    print(f"Manifest: {manifest}")
    print(f"Slurm script: {script}")
    print("Submit manually with:")
    print(command)
    print("Runtime estimate (from code, not measured):")
    print(RUNTIME_ESTIMATES[args.kind])
    if args.kind == "main":
        print("Warning: task CSV files are written only at the end; hitting the 4 h limit loses that job's results.")


if __name__ == "__main__":
    main()
