"""pydantic_evals Dataset/Case construction for the v2 benchmark.

Each (model, task, run) triple becomes its own Case -- make_dataset() builds
one Case per run directly, rather than using pydantic_evals' repeat= kwarg,
so each run keeps its own artifact/output directory and case name. Grading
happens inline via the shared ``StructuralGrade`` evaluator
(``benchmark/grading.py``) instead of a second pass over JSON files.

Two arms:
  "protocol"     -- v2's supervisor/planner architecture
                    (agents/planner/oneshot_planner.py)
  "single_agent" -- a single, undelegated LLM call given the tool catalog,
                    matching v1's original single-agent architecture
                    (agents/planner/single_agent_planner.py)

Both produce structured Workflow YAML and execute identically via
run_workflow_definition() -- the only difference between run_protocol_case()
and run_single_agent_case() is which planner produced the artifact.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Awaitable, Callable

from pydantic_evals import Case, Dataset

from agents.planner.oneshot_planner import plan_workflow_one_shot
from agents.planner.single_agent_planner import plan_workflow_single_agent
from common.config import MODELS
from common.workflow import Settings, Workflow
from common.workflow_runner import run_workflow_definition
from common.workflow_validation import validate_workflow

from benchmark.common import REFERENCE_DIR, SAMPLE_CASE_NAME, output_dir, prompt_path, run_dir
from benchmark.grading import StructuralGrade

PlannerFn = Callable[[str, Settings, object], Awaitable[Workflow]]

PLANNERS: dict[str, PlannerFn] = {
    "protocol": plan_workflow_one_shot,
    "single_agent": plan_workflow_single_agent,
}


def _collect_step_errors(context: dict | None) -> str:
    """Return "step_id: error message" lines for every step that raised.

    run_workflow_definition() swallows per-step exceptions internally and
    keeps executing (it only returns None for a pre-execution validation
    failure), so a workflow can "complete" while one or more of its steps
    silently failed and the required deliverable never got produced. This
    is the real error signal for that case -- has_deliverable()'s generic
    reason string alone tells a repair model nothing about which step or
    tool actually failed.
    """
    if not context:
        return ""
    lines = []
    for step_id, step_state in context.items():
        if step_id in ("settings", "output_dir"):
            continue
        error = (step_state or {}).get("result", {}).get("error")
        if error:
            lines.append(f"{step_id}: {error}")
    return "\n".join(lines)


def case_name(model: str, task: str, run: int) -> str:
    return f"{model}/{task}/run{run}"


def parse_case_name(name: str) -> tuple[str, str, int]:
    model, task, run = name.split("/")
    return model, task, int(run.removeprefix("run"))


async def run_planned_case(
    mode: str, label: str, *, skip_execution: bool = False
) -> Path:
    """Plan (via the mode's planner), validate, and execute a run.

    Returns the output dir. Catalog validation failures and runner
    rejections are *expected* LLM output-quality failure modes -- not
    exceptional -- so this returns out_dir normally for them (with the
    reason written to .error.txt) and lets StructuralGrade's
    has_deliverable() check grade the run crash, since no deliverable was
    ever produced. This keeps every run in report.cases, where
    make_manual_queue.py/merge_grades.py/self_debug_crashes.py look,
    instead of report.failures, which they don't.

    Only genuinely unexpected exceptions (e.g. an Agent.run() call failing
    due to a network/API error, or a bug in the harness itself) still
    propagate, landing in pydantic_evals' report.failures -- those are
    infra/tooling failures, not model-quality signal, and shouldn't be
    counted as a model "crash" in the benchmark stats.
    """
    planner = PLANNERS[mode]
    model_name, task, run = parse_case_name(label)
    artifact = run_dir(mode, model_name, task) / f"run{run}.yaml"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir(mode, model_name, task, run)
    prompt = prompt_path(mode, task).read_text(encoding="utf-8")
    settings = Settings(
        case_name=SAMPLE_CASE_NAME, data_dir="./data/sample", output_dir=str(out_dir)
    )

    try:
        workflow = await planner(prompt, settings, MODELS[model_name])
        workflow.write_to_file(artifact)
        errors = validate_workflow(workflow.to_yaml_dict())
        if errors:
            artifact.with_suffix(".error.txt").write_text(
                "catalog validation failed: " + "; ".join(errors), encoding="utf-8"
            )
            return out_dir
        if skip_execution:
            return out_dir
        context = run_workflow_definition(workflow.to_yaml_dict(), verbose=False)
        if context is None:
            artifact.with_suffix(".error.txt").write_text("runner rejected workflow", encoding="utf-8")
            return out_dir
        step_errors = _collect_step_errors(context)
        if step_errors:
            artifact.with_suffix(".error.txt").write_text(step_errors, encoding="utf-8")
        return out_dir
    except Exception:
        artifact.with_suffix(".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


async def run_protocol_case(label: str, *, skip_execution: bool = False) -> Path:
    """Plan, validate, and execute a protocol (supervisor/planner) run."""
    return await run_planned_case("protocol", label, skip_execution=skip_execution)


async def run_single_agent_case(label: str, *, skip_execution: bool = False) -> Path:
    """Plan, validate, and execute a single-agent (v1-style) run."""
    return await run_planned_case("single_agent", label, skip_execution=skip_execution)


def make_dataset(mode: str, models: list[str], tasks: list[str], runs: int) -> Dataset[str, Path]:
    """Build a Dataset of (model, task, run) Cases graded by StructuralGrade."""
    cases: list[Case[str, Path, None]] = []
    for model in models:
        for task in tasks:
            for run in range(1, runs + 1):
                name = case_name(model, task, run)
                cases.append(Case(
                    name=name,
                    inputs=name,
                    evaluators=[StructuralGrade(mode=mode, task=task, reference_dir=REFERENCE_DIR)],
                ))
    return Dataset(name=f"{mode}_benchmark", cases=cases)
