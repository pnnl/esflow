#!/usr/bin/env python3
"""Run the v2 protocol and single-agent benchmark conditions.

Builds a pydantic_evals Dataset of (model, task, run) Cases graded inline by
the shared StructuralGrade evaluator (benchmark/grading.py), then writes one
EvaluationReport per mode to benchmark/results/{mode}_report.json (or to
--output, if given -- see benchmark/RUNBOOK.md for chunked, per-model
invocations that write distinct report files and are combined afterward with
merge_reports.py).

Two arms:
  --mode protocol      -- v2's supervisor/planner architecture
  --mode single_agent  -- a single, undelegated LLM call given the tool
                           catalog, matching v1's original architecture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.retries import RetryConfig
from pydantic_evals.reporting import EvaluationReportAdapter
from tenacity import retry_if_exception, stop_after_attempt, wait_exponential

from common.config import MODELS
from benchmark.common import PILOT_MODELS, RESULTS_DIR, TASKS
from benchmark.datasets import make_dataset, run_planned_case

# Gateway 5xx (seen in practice as sporadic 504 Gateway Time-out from the PNNL
# Depot gateway, especially for Gemini), read/connect timeouts on the raw
# HTTP transport (httpx.TimeoutException, seen surfacing as an unwrapped
# httpx.ReadTimeout from google-genai's own client, not a pydantic-ai
# exception), and outright connection failures (httpx.ConnectError, seen
# live as "Temporary failure in name resolution" -- a transient local/network
# DNS blip, not anything wrong with the request itself) are transient infra
# flakiness, not a model- or code-quality signal -- retry the whole
# plan+execute case a few times with backoff rather than letting one bad
# gateway response or network hiccup sink an otherwise-valid case.
#
# UsageLimitExceeded (the planner's tool-calling loop needing more requests
# than oneshot_planner.py's configured limit) is included here too: unlike a
# workflow that's genuinely invalid (wrong param, hallucinated reference --
# real model-quality signal that should be graded as a crash), how many
# turns the planner needs to converge is itself noisy per-attempt --
# confirmed live by re-running a case that hit the limit in one attempt and
# having it resolve within budget on the very next attempt with no other
# change. Retrying gives the model another independent attempt rather than
# letting one unlucky number of turns silently drop the case from
# report.cases into report.failures (excluded from grading entirely).
_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}


def _is_transient_failure(exc: BaseException) -> bool:
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _TRANSIENT_STATUS_CODES
    if isinstance(exc, UsageLimitExceeded):
        return True
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


RETRY_TRANSIENT_FAILURES: RetryConfig = {
    "retry": retry_if_exception(_is_transient_failure),
    "stop": stop_after_attempt(4),
    "wait": wait_exponential(multiplier=2, min=2, max=30),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("protocol", "single_agent"), required=True)
    p.add_argument("--models", nargs="+", default=PILOT_MODELS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--skip-execution", action="store_true")
    p.add_argument("--concurrency", type=int, default=None, help="max concurrent cases")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path (default: benchmark/results/{mode}_report.json); "
        "use a distinct path per chunk (e.g. per model) to avoid overwriting "
        "another chunk's report, then combine with merge_reports.py",
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

    async def task_fn(label: str) -> Path:
        return await run_planned_case(args.mode, label, skip_execution=args.skip_execution)

    report = dataset.evaluate_sync(
        task_fn, max_concurrency=args.concurrency, retry_task=RETRY_TRANSIENT_FAILURES
    )
    report.print(include_reasons=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = args.output or (RESULTS_DIR / f"{args.mode}_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
