#!/usr/bin/env python3
"""
Workflow execution + numerical tolerance evaluation.

Grade taxonomy
--------------
The paper uses four grades: crash, success, silent failure, obvious failure.
Two can be assigned deterministically; the rest need human review.

  Step 1 - Crash detection
      CRASH iff the required final deliverable is missing:
        T1               -> a summary-statistics CSV
        T2 .. T7         -> at least one PNG figure

  Step 2 - Success detection
      SUCCESS iff the key data file is numerically identical to the
      reference within float64 precision (rtol=1e-12, atol=1e-15).

  Step 3 - Manual review
      UNDETERMINED runs are flagged for human inspection.

Evaluator
---------
``StructuralGrade`` combines Steps 1 and 2 and returns a numeric score:
  crash        -> 0.0
  success      -> 1.0
  undetermined -> 0.5

For each (model, task) pair, the evaluation task function:
  1. Reads the task prompt from protocol_prompts/{task}.txt
  2. Generates a workflow using the oneshot planner
  3. Executes the workflow via workflow_runner
  4. Returns the output directory path

The evaluation uses the in-file ``StructuralGrade`` evaluator together with
the reusable helpers from ``evals.structural_grading_helpers`` to score each run.

Usage:
    python evals/workflow_execution_numerical_tolerance_eval.py
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from agents.planner.oneshot_planner import plan_workflow_one_shot
from common.workflow import Settings
from common.logging_setup import configure_console_logging
from common.workflow_runner import run_workflow_definition
from evals.structural_grading_helpers import (
    find_key_csv, find_key_nc,
    has_deliverable, protocol_matches_reference,
)


configure_console_logging()

GRADE_CRASH = 0.0
GRADE_UNDETERMINED = 0.5
GRADE_SUCCESS = 1.0

MODELS = [
    "GPT 5.4",
    "Gemini 3.5 Flash"
]
TASKS = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    "task_03_et_benchmark",
    "task_04_streamflow_fdc",
    "task_05_basin_streamflow",
    "task_06_water_balance",
    "task_07_integrated_diagnostic"
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REF_DIR = Path(__file__).resolve().parent / "reference_workflows" / "results"
PROMPTS_DIR = Path(__file__).resolve().parent / "protocol_prompts"


@dataclass
class StructuralGrade(Evaluator[str, Path]):
    """Combined Steps 1 + 2 returning a numeric score.

    Scores:
      crash        -> 0.0
      success      -> 1.0
      undetermined -> 0.5  (needs manual review)
    """
    task: str
    ref_csv: Path | None = None
    ref_nc: Path | None = None

    def evaluate(self, ctx: EvaluatorContext[str, Path]) -> EvaluationReason:
        ok, reason = has_deliverable(ctx.output, self.task)
        if not ok:
            return EvaluationReason(value=GRADE_CRASH, reason=f"crash: {reason}")

        match, why = protocol_matches_reference(
            ctx.output, self.task, self.ref_csv, self.ref_nc
        )
        if match:
            return EvaluationReason(value=GRADE_SUCCESS, reason=f"success: {why}")
        return EvaluationReason(value=GRADE_UNDETERMINED, reason=f"undetermined: {why}")


def make_dataset() -> Dataset[str, Path, None]:
    """Build a pydantic-evals dataset of (model, task) cases."""
    cases: list[Case[str, Path, None]] = []
    for model in MODELS:
        for task in TASKS:
            ref_dir = REF_DIR / task
            cases.append(Case(
                name=f"{model}_protocol_{task}",
                inputs=f"{model}/{task}",
                expected_output=None,
                evaluators=[StructuralGrade(
                    task=task,
                    ref_csv=find_key_csv(ref_dir, task),
                    ref_nc=find_key_nc(ref_dir, task),
                )],
            ))
    return Dataset(name="structural_grading_protocol", cases=cases)


def run_task(label: str) -> Path:
    """Generate a workflow, execute it, and return the output directory.

    Called once per evaluation repetition. A fresh UUID is generated each
    call so repeated runs are written to separate directories.

    Args:
        label: "<model>/<task>" identifying the case

    Returns:
        Path to the output directory containing the workflow's outputs
    """
    model, task = label.split("/")

    user_goal = (PROMPTS_DIR / f"{task}.txt").read_text()
    output_dir = RESULTS_DIR / f"{model}_protocol" / task / uuid4().hex[:8]
    settings = Settings(data_dir="./data/sample", output_dir=str(output_dir))

    workflow = asyncio.run(plan_workflow_one_shot(user_goal, settings))
    context = run_workflow_definition(workflow.model_dump(), verbose=False)

    return Path(context["output_dir"])


if __name__ == "__main__":
    dataset = make_dataset()
    report = dataset.evaluate_sync(run_task, repeat=4, max_concurrency=2)
    report.print(include_reasons=True)
