from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from agents.planner import data_discovery, diagnostics, extraction, visualization
from common.workflow import Settings, Workflow
from common.workflow_runner import run_workflow_definition

from agents.planner import water_cycle
from common import WorkflowState
from common.config import MODELS, model, load_prompt
from common.workflow_validation import validate_workflow


PLANNER_ROUTING_PROMPT = (
    "You are the Workflow planner. "
    "First choose how to respond:\n"
    "- If the user's message is a greeting, a question, ambiguous, or missing information "
    "required to compose steps (e.g. variable, model case, years, data location), reply in "
    "plain text: briefly ask a focused clarifying question. Do NOT fabricate a workflow or "
    "invent placeholder settings or params.\n"
    "- Only when the request is a complete, groundable analysis task, compose the workflow "
    "and return the structured Workflow object (do not describe it in prose).\n"
    "When composing, decide which subagents are needed based on missing information in the "
    "user request and current workflow state. Do not call a subagent unless it adds required steps. "
    "Use this conditional order only when needed: "
    "data discovery if required external data or metadata is missing; "
    "extraction if variables, fields, or timeseries must be produced; "
    "diagnostics or water cycle if derived metrics are requested; "
    "visualization only if plots are requested. "
    "If required inputs are already available from settings or prior step outputs, skip data discovery. "
    "Before returning any structured Workflow, call check_completeness and run_workflow_validation. "
    "If check_completeness returns gaps, ask the user to supply those values instead of guessing or "
    "emitting null/placeholder params. If either tool returns issues, ask the user for the missing "
    "information or corrections in plain text "
    "instead of returning an invalid workflow. "
    "If no new steps are needed, return the workflow unchanged."
)

ONESHOT_PLANNER_PROMPT = (
    "You are the one-shot Workflow planner. "
    "Produce the best complete Workflow in a single pass with no follow-up turn. "
    "Return only the structured Workflow object, never prose. "
    "Never ask clarifying questions and never return plain text. "
    "Do not invent placeholder values, nulls, or unknown tokens for required params. "
    "If a required value is not explicitly provided, use an existing ${settings.*} value or "
    "a valid ${step_id.outputs.*} reference from prior steps; if neither is possible, omit that step. "
    "Plan minimally: include only steps required to satisfy the request. "
    "Select subagents only when they add needed steps, in this order when applicable: "
    "data discovery for missing external datasets/metadata; "
    "extraction for needed fields/timeseries; "
    "diagnostics or water cycle for derived analyses; "
    "visualization only when plots are explicitly requested. "
    "Preserve validity: unique step IDs, known tool names only, and valid output references. "
    "If required inputs already exist from settings or prior outputs, do not regenerate them. "
    "Before returning, call check_completeness and run_workflow_validation. "
    "Only return a Workflow when both tools report no issues. "
    "If no new steps are needed, return the workflow unchanged."
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

def default_settings() -> Settings:
    """Build server-side default workflow settings for web requests."""
    return Settings(
        data_dir=os.getenv("ESFLOW_DATA_DIR", "./data/e3sm"),
        output_dir=os.getenv("ESFLOW_OUTPUT_DIR", "./outputs")
    )

def _bootstrap_workflow_state() -> WorkflowState:
    """Create initial workflow state used by web-mode tool calls."""
    return WorkflowState(
        workflow=Workflow(
            name="Generated Workflow",
            description="Web workflow request",
            settings=default_settings(),
            steps=[],
        )
    )

_PLACEHOLDER_TOKENS = {"UNKNOWN", "<UNKNOWN>", "TBD", "N/A", "NONE", "NULL"}


async def check_completeness(ctx: RunContext[WorkflowState]) -> list[str]:
    """List unfilled/placeholder params and dangling output references in the current workflow.

    Returns an empty list when the workflow is fully specified.
    """
    gaps: list[str] = []
    known_outputs = {
        f"{step.id}.outputs.{key}"
        for step in ctx.deps.workflow.steps
        for key in step.outputs
    }
    for step in ctx.deps.workflow.steps:
        for key, val in step.params.items():
            if val is None:
                gaps.append(f"{step.id}.{key} is null")
            elif isinstance(val, str):
                s = val.strip()
                if s == "" or s.upper() in _PLACEHOLDER_TOKENS:
                    gaps.append(f"{step.id}.{key} is empty or a placeholder")
                elif s.startswith("${") and s.endswith("}"):
                    ref = s[2:-1]
                    if not ref.startswith("settings.") and ref not in known_outputs:
                        gaps.append(f"{step.id}.{key} references unknown output '{val}'")
    return gaps


async def run_workflow_validation(ctx: RunContext[WorkflowState]) -> list[str]:
    """Validate the current workflow against the tool catalog and return errors."""
    return validate_workflow(ctx.deps.workflow.to_yaml_dict())


_OUTPUT_REF = re.compile(r"\$\{(\w+)\.outputs\.\w+\}")


def _iter_param_strings(value):
    """Yield every string found in a param value (handles nested lists/dicts)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_param_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_param_strings(v)


async def render_dag(ctx: RunContext[WorkflowState]) -> str:
    """Render the current workflow as a Mermaid flowchart.

    Nodes are steps; edges are derived from ``${step_id.outputs.key}`` references
    in each step's params, so the diagram reflects the real data-flow dependencies.
    """
    steps = ctx.deps.workflow.steps
    if not steps:
        return "_(no steps in the workflow yet)_"

    step_ids = {step.id for step in steps}
    lines = ["```mermaid", "flowchart TD"]
    for step in steps:
        lines.append(f'    {step.id}["{step.id}: {step.tool}"]')

    seen_edges = set()
    for step in steps:
        parents = set()
        for value in step.params.values():
            for text in _iter_param_strings(value):
                parents.update(_OUTPUT_REF.findall(text))
        for parent in parents:
            edge = (parent, step.id)
            if parent in step_ids and parent != step.id and edge not in seen_edges:
                seen_edges.add(edge)
                lines.append(f"    {parent} --> {step.id}")

    lines.append("```")
    return "\n".join(lines)


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


PLANNER_TOOLS = [
    data_discovery.call_data_discovery,
    extraction.call_extraction,
    diagnostics.call_diagnostics,
    water_cycle.call_water_cycle_synthesis,
    visualization.call_visualization,
    render_dag,
    check_completeness,
    run_workflow_validation,
]

ONESHOT_PLANNER_EXECUTOR_TOOLS = [
    plan_with_planner,
    execute_planned_workflow,
]


planner: Agent[WorkflowState, Workflow | str] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=[Workflow, str],
    tools=PLANNER_TOOLS,
    instructions=load_prompt(PLANNER_ROUTING_PROMPT),
)

oneshot_planner: Agent[WorkflowState, Workflow | str] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=[Workflow],
    tools=PLANNER_TOOLS,
    instructions=load_prompt(ONESHOT_PLANNER_PROMPT),
)

oneshot_planner_executor: Agent[WorkflowState, PlannerExecutorResult] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=PlannerExecutorResult,
    tools=ONESHOT_PLANNER_EXECUTOR_TOOLS,
    instructions=load_prompt(ONESHOT_PLANNER_EXECUTOR_PROMPT),
)

async def plan_workflow_one_shot(user_goal: str, settings: Settings) -> Workflow:
    """Plan an ESMFlow workflow from a user goal using the planner chain."""
    state = WorkflowState(
        workflow=Workflow(
            name="Planned Workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )
    )

    # Force a structured Workflow on the programmatic path (evals depend on this).
    # The plain-text conversational path is only enabled for the to_web chat UI.
    result = await oneshot_planner.run(user_goal, deps=state, output_type=Workflow)
    return result.output


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

app = planner.to_web(
    deps=_bootstrap_workflow_state(),
    models=MODELS,
    html_source=Path(__file__).parent / "web_ui.html",
    instructions=(
        "You are chatting interactively. Greet the user briefly and explain you compose "
        "ESM analysis workflows from a validated tool library. "
        "After each change to the workflow, call render_dag and include its Mermaid diagram "
        "in your reply so the user can see the current plan."
    ),
)

# Serve ESFlow branding assets (logo, favicons) referenced by web_ui.html.
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)