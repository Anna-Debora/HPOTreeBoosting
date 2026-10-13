"""Generate (but never submit) Slurm array jobs for run_experiments.py."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


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


def generate_jobs(tasks_file, seeds, libraries, results, out):
    tasks = read_tasks(tasks_file)
    seeds = [int(seed) for seed in seeds]
    libraries = list(libraries)
    jobs = [(suite_id, task_id, seed, library)
            for suite_id, task_id in tasks
            for seed in seeds
            for library in libraries]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = out / "jobs.tsv"
    manifest.write_text("".join(
        f"{suite_id}\t{task_id}\t{seed}\t{library}\n"
        for suite_id, task_id, seed, library in jobs
    ))

    manifest_shell = shlex.quote(str(manifest.resolve()))
    results_shell = shlex.quote(str(Path(results).resolve()))
    script = f"""#!/bin/bash
#SBATCH --job-name=hpo-tree-boosting
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --array=0-{len(jobs) - 1}
#SBATCH --output={shlex.quote(str((logs / '%x_%A_%a.out').resolve()))}
#SBATCH --error={shlex.quote(str((logs / '%x_%A_%a.err').resolve()))}
#SBATCH --export=ALL

set -euo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.9.18
source .venv/bin/activate

JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {manifest_shell})
IFS=$'\\t' read -r SUITE_ID TASK_ID SEED LIBRARY <<< "$JOB_LINE"
python run_experiments.py \\
    --suite_id "$SUITE_ID" \\
    --task_id "$TASK_ID" \\
    --seed "$SEED" \\
    --library "$LIBRARY" \\
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
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--libraries", nargs="+", choices=("gpboost", "xgboost", "catboost"),
                        default=("gpboost", "xgboost", "catboost"))
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    script, manifest, command, count = generate_jobs(
        args.tasks_file, args.seeds, args.libraries, args.results, args.out
    )
    print(f"Generated {count} array jobs")
    print(f"Manifest: {manifest}")
    print(f"Slurm script: {script}")
    print("Submit manually with:")
    print(command)
    print("Warning: task CSV files are written only at the end; hitting the 4 h limit loses that job's results.")


if __name__ == "__main__":
    main()
