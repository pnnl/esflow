#!/usr/bin/env python3
"""Merge multiple EvaluationReport JSON files into one.

Supports the chunked benchmark workflow in benchmark/RUNBOOK.md: each chunk
(e.g. one model) is run as a separate `run_benchmark.py --output <path>`
invocation to keep each run small enough to finish within a single
foreground session (see RUNBOOK.md for why). This script concatenates the
per-chunk reports' cases/failures/report_evaluator_failures back into a
single combined report at the standard benchmark/results/{mode}_report.json
path, so the existing grading pipeline (make_manual_queue.py,
merge_grades.py, self_debug_crashes.py) can run against it unmodified --
those scripts only ever read report.cases and don't care how the report was
assembled.

`analyses` (experiment-wide report-evaluator output, not per-case) isn't
combined here: none of v2's benchmark evaluators currently populate it (only
the per-case StructuralGrade evaluator is used), so there's nothing
meaningful to merge. If a future report-level evaluator is added, this
script will need to decide how to combine its output across chunks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter

from benchmark.common import RESULTS_DIR


def merge_reports(reports: list[EvaluationReport], name: str) -> EvaluationReport:
    """Concatenate cases/failures/report_evaluator_failures across reports."""
    merged = EvaluationReport(name=name, cases=[], failures=[], report_evaluator_failures=[])
    for report in reports:
        merged.cases.extend(report.cases)
        merged.failures.extend(report.failures)
        merged.report_evaluator_failures.extend(report.report_evaluator_failures)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("protocol", "single_agent"))
    parser.add_argument(
        "--reports", type=Path, nargs="+", required=True,
        help="per-chunk {mode}_report_*.json files to combine (shell-glob these)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="combined report path (default: benchmark/results/{mode}_report.json)",
    )
    args = parser.parse_args()

    missing = [p for p in args.reports if not p.exists()]
    if missing:
        raise SystemExit(f"report file(s) not found: {', '.join(str(p) for p in missing)}")

    reports = [
        EvaluationReportAdapter.validate_json(path.read_bytes()) for path in args.reports
    ]
    merged = merge_reports(reports, name=f"{args.mode}_benchmark")

    output = args.output or (RESULTS_DIR / f"{args.mode}_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(EvaluationReportAdapter.dump_json(merged, indent=2))
    print(f"merged {len(args.reports)} report(s) ({len(merged.cases)} case(s)) -> {output}")


if __name__ == "__main__":
    main()
