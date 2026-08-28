#!/usr/bin/env python3
"""Run the v2 protocol and v1-style free-code benchmark conditions.

Builds a pydantic_evals Dataset of (model, task, run) Cases graded inline by
the shared StructuralGrade evaluator (benchmark/grading.py), then writes one
EvaluationReport per mode to benchmark/results/{mode}_report.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_evals.reporting import EvaluationReportAdapter

from common.config import MODELS
from benchmark.common import PILOT_MODELS, RESULTS_DIR, TASKS
from benchmark.datasets import make_dataset, run_baseline_case, run_protocol_case


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("protocol", "baseline"), required=True)
    p.add_argument("--models", nargs="+", default=PILOT_MODELS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--timeout", type=int, default=300, help="baseline wall timeout")
    p.add_argument("--skip-execution", action="store_true")
    p.add_argument("--sandbox-image", default=None)
    p.add_argument(
        "--concurrency", type=int, default=None,
        help="max concurrent cases (default: 2 for baseline, unlimited for protocol)",
    )
    return p


def validate_args(args: argparse.Namespace) -> None:
    unknown_models = sorted(set(args.models) - set(MODELS))
    unknown_tasks = sorted(set(args.tasks) - set(TASKS))
    if unknown_models:
        raise SystemExit(f"unknown v2 model labels: {', '.join(unknown_models)}")
    if unknown_tasks:
        raise SystemExit(f"unknown benchmark tasks: {', '.join(unknown_tasks)}")
    if args.runs < 1:
        raise SystemExit("--runs must be positive")


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    dataset = make_dataset(args.mode, args.models, args.tasks, args.runs)

    if args.mode == "protocol":
        async def task_fn(label: str) -> Path:
            return await run_protocol_case(label, skip_execution=args.skip_execution)
        concurrency = args.concurrency
    else:
        from benchmark.common import SANDBOX_IMAGE
        sandbox_image = args.sandbox_image or SANDBOX_IMAGE

        async def task_fn(label: str) -> Path:
            return await run_baseline_case(
                label, timeout=args.timeout, sandbox_image=sandbox_image,
                skip_execution=args.skip_execution,
            )
        concurrency = args.concurrency if args.concurrency is not None else 2

    report = dataset.evaluate_sync(task_fn, max_concurrency=concurrency)
    report.print(include_reasons=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.mode}_report.json"
    path.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
