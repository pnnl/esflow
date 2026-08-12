from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .oneshot_planner import oneshot_planner
from .planner_tools import check_completeness, run_workflow_validation
from common import WorkflowState
from common.config import load_prompt, model
from common.workflow import Settings, Workflow
from common.workflow_runner import (
    ExecutionStepStatus,
    run_workflow_definition,
    step_statuses_from_execution_context,
)


ONESHOT_PLANNER_EXECUTOR_PROMPT = (
    "You are the one-shot Workflow planner-executor. "
    "Complete the request in a single pass with no follow-up turn. "
    "Use the available tools in a single pass and do not rely on a second turn to gather missing details. "
    "First use the plan_with_planner tool to create or revise the workflow; it delegates to the one-shot planner. "
    "The planner must produce a valid executable workflow before execution is attempted. "
    "Always return a PlannerExecutorResult object. "
    "If planning reports missing information or returns a clarifying question, return a PlannerExecutorResult "
    "with status='needs_input', that message, and the current workflow. Do not execute anything. "
    "Treat underspecified requests as needing input rather than guessing placeholder values. "
    "After planning succeeds, call execute_planned_workflow in the same pass. "
    "Return the PlannerExecutorResult from execute_planned_workflow as the final answer."
)


class PlannerExecutorResult(BaseModel):
    """Structured result for one-shot planning and optional execution."""

    status: Literal["needs_input", "planned", "execution_blocked", "executed"]
    message: str
    workflow: Workflow | None = None
    step_statuses: list[ExecutionStepStatus] = Field(default_factory=list)
    output_dir: str | None = None
    workflow_file: str | None = None
    plot_images: list[str] = Field(default_factory=list)


# The static mount in app.py exposes this dir by name, so a plot written to
# ``output/<run>/plot.png`` is reachable at ``/output/<run>/plot.png``.
_SERVED_OUTPUT_ROOT = "output"


def _plot_url(path: Path) -> str | None:
    """Return the browser URL for a plot file, or None if it isn't served.

    Only files under the mounted output root (relative to the process CWD) are
    reachable; anything else can't be shown inline and is reported by path only.
    """
    try:
        rel = path.resolve().relative_to(Path.cwd())
    except ValueError:
        return None
    if not rel.parts or rel.parts[0] != _SERVED_OUTPUT_ROOT:
        return None
    return "/" + rel.as_posix()


def _collect_plot_urls(workflow: Workflow, execution_context: dict) -> list[str]:
    """Map this run's rendered image outputs to served URLs.

    Uses the paths recorded in the execution context for each step, not a
    directory glob, so re-runs into the same output dir show only the current
    run's plots rather than every image accumulated there.
    """
    urls: list[str] = []
    seen: set[str] = set()
    for step in workflow.steps:
        step_outputs = execution_context.get(step.id, {}).get("outputs", {})
        for value in step_outputs.values():
            if not isinstance(value, str) or not value.lower().endswith(".png"):
                continue
            url = _plot_url(Path(value))
            if url is not None and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


async def plan_with_planner(ctx: RunContext[WorkflowState], task: str) -> str:
    """Use the planner agent as a subagent to create or revise the workflow."""
    result = await oneshot_planner.run(task, deps=ctx.deps, output_type=Workflow)
    if isinstance(result.output, Workflow):
        ctx.deps.workflow = result.output
        return (
            f"Planner updated the workflow to {len(result.output.steps)} step(s). "
            "Call execute_planned_workflow to run it."
        )
    return result.output


async def execute_planned_workflow(ctx: RunContext[WorkflowState]) -> PlannerExecutorResult:
    """Execute the currently planned workflow via the shared workflow runner."""
    if not ctx.deps.workflow.steps:
        return PlannerExecutorResult(
            status="execution_blocked",
            message="No workflow steps exist yet. Call plan_with_planner first.",
            workflow=ctx.deps.workflow,
        )

    gaps = await check_completeness(ctx)
    if gaps:
        return PlannerExecutorResult(
            status="execution_blocked",
            message="Workflow was not executed because required values are missing:\n- " + "\n- ".join(gaps),
            workflow=ctx.deps.workflow,
        )

    errors = await run_workflow_validation(ctx)
    if errors:
        return PlannerExecutorResult(
            status="execution_blocked",
            message="Workflow was not executed because validation failed:\n- " + "\n- ".join(errors),
            workflow=ctx.deps.workflow,
        )

    # Persist the plan next to its outputs; the runner only holds it in memory.
    output_dir = Path(ctx.deps.workflow.settings.output_dir)
    workflow_path = output_dir / "workflow.yaml"
    ctx.deps.workflow.write_to_file(workflow_path)

    # The runner is synchronous and CPU/IO-bound; run it off the event loop so the
    # web UI stays responsive while the workflow executes.
    execution_context = await asyncio.to_thread(
        run_workflow_definition,
        ctx.deps.workflow.to_yaml_dict(),
        workflow_path=workflow_path,
    )
    if execution_context is None:
        return PlannerExecutorResult(
            status="execution_blocked",
            message="Workflow execution did not start due to validation or setup errors.",
            workflow=ctx.deps.workflow,
        )

    resolved_output_dir = Path(execution_context.get("output_dir", output_dir))
    plot_urls = _collect_plot_urls(ctx.deps.workflow, execution_context)

    message = f"Executed workflow with {len(ctx.deps.workflow.steps)} step(s)."
    if plot_urls:
        images_md = "\n\n".join(
            f"![{Path(url).name}]({url})" for url in plot_urls
        )
        # Provide the exact Markdown so the agent can echo it verbatim; the URL is
        # short enough to reproduce reliably, unlike an inlined image.
        message += "\n\nPlot(s) — include this Markdown verbatim in your reply:\n" + images_md

    return PlannerExecutorResult(
        status="executed",
        message=message,
        workflow=ctx.deps.workflow,
        step_statuses=step_statuses_from_execution_context(ctx.deps.workflow, execution_context),
        output_dir=str(resolved_output_dir),
        workflow_file=str(workflow_path),
        plot_images=plot_urls,
    )


ONESHOT_PLANNER_EXECUTOR_TOOLS = [
    plan_with_planner,
    execute_planned_workflow,
]


oneshot_planner_executor: Agent[WorkflowState, PlannerExecutorResult] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=PlannerExecutorResult,
    tools=ONESHOT_PLANNER_EXECUTOR_TOOLS,
    instructions=load_prompt(ONESHOT_PLANNER_EXECUTOR_PROMPT),
)


async def plan_and_execute_workflow(user_goal: str, settings: Settings) -> PlannerExecutorResult:
    """Plan and execute an ESMFlow workflow from a user goal via the one-shot planner-executor."""
    state = WorkflowState(
        workflow=Workflow(
            name="Planned Workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )
    )
    result = await oneshot_planner_executor.run(
        user_goal,
        deps=state,
        output_type=PlannerExecutorResult,
    )
    return result.output
