from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .oneshot_planner import oneshot_planner
from .planner_tools import check_completeness, run_workflow_validation
from common import WorkflowState
from common.config import load_prompt, model
from common.workflow import Settings, Workflow
from common.workflow_runner import run_workflow_definition


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


class ExecutionStepStatus(BaseModel):
    """Structured execution status for one workflow step."""

    step_id: str
    status: Literal["completed", "failed", "reused", "skipped"]
    error: str | None = None


class PlannerExecutorResult(BaseModel):
    """Structured result for one-shot planning and optional execution."""

    status: Literal["needs_input", "planned", "execution_blocked", "executed"]
    message: str
    workflow: Workflow | None = None
    step_statuses: list[ExecutionStepStatus] = Field(default_factory=list)
    output_dir: str | None = None


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


def _step_statuses_from_execution_context(
    workflow: Workflow, execution_context: dict
) -> list[ExecutionStepStatus]:
    """Convert workflow runner context into structured per-step statuses."""
    step_statuses: list[ExecutionStepStatus] = []
    for step in workflow.steps:
        step_state = execution_context.get(step.id, {})
        result = step_state.get("result", {})
        if result.get("error"):
            step_statuses.append(
                ExecutionStepStatus(
                    step_id=step.id,
                    status="failed",
                    error=result["error"],
                )
            )
        elif result.get("reused"):
            step_statuses.append(ExecutionStepStatus(step_id=step.id, status="reused"))
        elif result.get("skipped"):
            step_statuses.append(ExecutionStepStatus(step_id=step.id, status="skipped"))
        else:
            step_statuses.append(ExecutionStepStatus(step_id=step.id, status="completed"))
    return step_statuses


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

    execution_context = run_workflow_definition(
        ctx.deps.workflow.to_yaml_dict(),
        workflow_path="planned_workflow.yaml",
    )
    if execution_context is None:
        return PlannerExecutorResult(
            status="execution_blocked",
            message="Workflow execution did not start due to validation or setup errors.",
            workflow=ctx.deps.workflow,
        )

    return PlannerExecutorResult(
        status="executed",
        message=f"Executed workflow with {len(ctx.deps.workflow.steps)} step(s).",
        workflow=ctx.deps.workflow,
        step_statuses=_step_statuses_from_execution_context(ctx.deps.workflow, execution_context),
        output_dir=execution_context.get("output_dir", ctx.deps.workflow.settings.output_dir),
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
