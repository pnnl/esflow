#!/usr/bin/env python3
"""Merge manual silent/obvious/success labels into structural grades.

Attaches the resolved final grade to each ReportCase as a label
("final_grade") instead of writing a separate scores_final_*.json, then
re-serializes the same EvaluationReport in place.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_evals.evaluators.evaluator import EvaluationResult
from pydantic_evals.reporting import EvaluationReportAdapter

from benchmark.common import RESULTS_DIR
from benchmark.datasets import parse_case_name


def resolve_final_grade(auto_grade: str, manual_grade: str | None) -> str:
    """Resolve final_grade: manual override > auto success/crash > undetermined.

    Only "success" and "crash" auto grades are trusted without a human label
    (they're deterministic per benchmark/grading.py); "undetermined" rows
    fall back to the manual label if one was recorded, else stay
    "undetermined".
    """
    automatic = {"success": "success", "crash": "crash"}.get(auto_grade)
    return manual_grade or automatic or "undetermined"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="{mode}_report.json")
    parser.add_argument("--mode", required=True, choices=("protocol", "single_agent"))
    parser.add_argument("--manual", type=Path, default=None)
    args = parser.parse_args()
    manual_path = args.manual or RESULTS_DIR / f"manual_review_{args.mode}.csv"

    manual: dict[tuple[str, str, int], str | None] = {}
    with manual_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["task"], int(row["run"]))
            manual[key] = row["manual_grade"].strip().lower() or None

    report = EvaluationReportAdapter.validate_json(args.report.read_bytes())
    for case in report.cases:
        score = case.scores.get("StructuralGrade")
        if score is None:
            continue
        model, task, run = parse_case_name(case.name)
        auto_grade = (score.reason or "").split(":", 1)[0].strip()
        final_grade = resolve_final_grade(auto_grade, manual.get((model, task, run)))
        case.labels["final_grade"] = EvaluationResult(
            name="final_grade", value=final_grade, reason=None, source=score.source,
        )

    args.report.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
    print(f"updated {args.report}")


if __name__ == "__main__":
    main()
