"""Build plots and tables from experiment result folders.

Example:
    python make_report.py --results synthetic_results --libraries gpboost xgboost catboost \
        --out report --budgets 10 30 90 270
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from report_data import load_report_data
from report_plots import make_all_plots
from report_tables import make_all_tables


def _write_readme(path, args, data):
    lines = [
        "HPO comparison report",
        f"Date: {date.today().isoformat()}",
        f"Results: {Path(args.results).resolve()}",
        f"Output: {Path(args.out).resolve()}",
        f"Libraries: {', '.join(args.libraries)}",
        f"Budgets (trials): {', '.join(map(str, args.budgets))}",
        f"Language: {args.language}",
        f"Runtime selection: {args.selection}",
        f"Tasks discovered: {len(data['tasks'])}",
        f"Maximum iteration axis: {data['iterations']}",
        "",
        "Tasks and seed counts by library:",
    ]
    for library in args.libraries:
        lines.append(
            f"- {library}: {len(data['tasks'])} tasks; "
            f"{len(data['seeds'].get(library, []))} seeds "
            f"({', '.join(map(str, data['seeds'].get(library, []))) or 'none'})"
        )
    lines.extend(["", "Missing methods/files and warnings:"])
    if data["warnings"]:
        lines.extend(f"- {warning}" for warning in data["warnings"])
    else:
        lines.append("- None")
    if data["missing_files"]:
        lines.append("\nMissing result files:")
        lines.extend(f"- {file_path}" for file_path in data["missing_files"])
    Path(path).write_text("\n".join(lines) + "\n")


def run_report(results, libraries, out, tasks_file=None, budgets=None, language="en", selection="val"):
    budgets = list(budgets or (10, 30, 90, 270))
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("budgets must be positive iteration counts")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must not contain duplicates")
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    data = load_report_data(results, libraries, tasks_file=tasks_file)
    unavailable = [budget for budget in budgets if budget > data["iterations"]]
    if unavailable:
        data["warnings"].append(
            f"Budgets außerhalb der verfügbaren Iterationsachse übersprungen: {unavailable}"
        )
        budgets = [budget for budget in budgets if budget <= data["iterations"]]
        if not budgets:
            budgets = [data["iterations"]]
            data["warnings"].append(
                f"Kein angefordertes Budget verfügbar; verwende Endbudget {data['iterations']}."
            )
    make_all_plots(data, out_path, language=language, selection=selection)
    make_all_tables(data, out_path, budgets)
    args = argparse.Namespace(results=results, libraries=libraries, out=out,
                              budgets=budgets, language=language, selection=selection)
    _write_readme(out_path / "README.txt", args, data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Root of the result directory tree")
    parser.add_argument("--libraries", nargs="+", required=True,
                        choices=("gpboost", "xgboost", "catboost"))
    parser.add_argument("--out", required=True, help="Output directory for report artifacts")
    parser.add_argument("--tasks-file", help="Optional CSV, JSON or whitespace task list")
    parser.add_argument("--budgets", nargs="+", type=int, default=(10, 30, 90, 270))
    parser.add_argument("--language", choices=("en", "de"), default="en")
    parser.add_argument("--selection", choices=("val", "oracle"), default="val")
    args = parser.parse_args(argv)
    data = run_report(args.results, args.libraries, args.out, args.tasks_file,
                      args.budgets, args.language, args.selection)
    print(f"Report written to {Path(args.out).resolve()}")
    print(f"Tasks: {len(data['tasks'])}; iteration axis: {data['iterations']}")
    print(f"Warnings: {len(data['warnings'])}; missing files: {len(data['missing_files'])}")


if __name__ == "__main__":
    main()
