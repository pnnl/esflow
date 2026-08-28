#!/usr/bin/env python3
"""Write a CSV queue for human silent-versus-obvious review.

Reads a benchmark/results/{mode}_report.json (written by run_benchmark.py)
and queues every case whose StructuralGrade score is "undetermined" (0.5)
for manual review.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_evals.reporting import EvaluationReportAdapter

from benchmark.common import RESULTS_DIR
from benchmark.datasets import parse_case_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="{mode}_report.json")
    parser.add_argument("--mode", required=True, choices=("protocol", "baseline"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or RESULTS_DIR / f"manual_review_{args.mode}.csv"

    report = EvaluationReportAdapter.validate_json(args.report.read_bytes())
    rows = [
        case for case in report.cases
        if case.scores.get("StructuralGrade") is not None
        and (case.scores["StructuralGrade"].reason or "").startswith("undetermined:")
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "idx", "mode", "task", "model", "run", "auto_reason", "output_dir",
            "manual_grade", "notes",
        ])
        writer.writeheader()
        for idx, case in enumerate(rows):
            model, task, run = parse_case_name(case.name)
            score = case.scores["StructuralGrade"]
            writer.writerow({
                "idx": idx, "mode": args.mode, "task": task, "model": model, "run": run,
                "auto_reason": score.reason or "", "output_dir": str(case.output),
                "manual_grade": "", "notes": "",
            })
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
