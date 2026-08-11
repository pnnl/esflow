"""Stateless stdio MCP server for planning and executing ESMFlow workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from agents.planner.oneshot_planner import plan_workflow_one_shot
from agents.planner.oneshot_planner_executor import (
    PlannerExecutorResult,
    plan_and_execute_workflow as _plan_and_execute_workflow,
)
from agents.planner.settings import default_settings
from common.logging_setup import configure_console_logging
from common.workflow import Settings, Workflow
from common.workflow_runner import (
    ExecutionStepStatus,
    run_workflow_definition,
    step_statuses_from_execution_context,
)
from common.workflow_validation import validate_workflow as _validate_workflow


configure_console_logging()


mcp = FastMCP(
    "ESMFlow",
    instructions=(
        "Plan, validate, and execute Earth System Model analysis workflows. "
        "Calls are stateless: pass the complete workflow returned by plan_workflow "
        "to validate_workflow or execute_workflow."
    ),
)


class WorkflowExecutionResult(BaseModel):
    """Structured result from executing an existing workflow."""

    output_dir: str
    workflow_file: str
    step_statuses: list[ExecutionStepStatus] = Field(default_factory=list)
    outputs: dict[str, dict[str, str]] = Field(default_factory=dict)


def _settings(data_dir: str | None, output_dir: str | None) -> Settings:
    """Build settings from explicit values or the server's environment defaults."""
    defaults = default_settings()
    return Settings(
        data_dir=data_dir or defaults.data_dir,
        output_dir=output_dir or defaults.output_dir,
    )


@mcp.tool
def validate_workflow(workflow: Workflow) -> list[str]:
    """Validate a workflow against the ESMFlow tool catalog; an empty list means valid."""
    return _validate_workflow(workflow.to_yaml_dict())


@mcp.tool
async def plan_workflow(
    user_goal: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
) -> Workflow:
    """Plan an ESM analysis workflow from a natural-language goal without executing it."""
    return await plan_workflow_one_shot(user_goal, _settings(data_dir, output_dir))


@mcp.tool
async def execute_workflow(
    workflow: Workflow,
    start_from: str | None = None,
    reuse: bool = False,
) -> WorkflowExecutionResult:
    """Execute a validated workflow and return per-step statuses and output file paths."""
    errors = _validate_workflow(workflow.to_yaml_dict())
    if errors:
        raise ValueError("Workflow failed catalog validation:\n- " + "\n- ".join(errors))

    output_dir = Path(workflow.settings.output_dir)
    workflow_file = output_dir / "workflow.yaml"
    workflow.write_to_file(workflow_file)

    context = await asyncio.to_thread(
        run_workflow_definition,
        workflow.to_yaml_dict(),
        workflow_path=workflow_file,
        start_from=start_from,
        reuse=reuse,
    )
    if context is None:
        raise ValueError("Workflow execution did not start; check the requested start_from step.")

    outputs = {
        step.id: {
            key: value
            for key, value in context.get(step.id, {}).get("outputs", {}).items()
            if isinstance(value, str)
        }
        for step in workflow.steps
    }
    return WorkflowExecutionResult(
        output_dir=str(context.get("output_dir", output_dir)),
        workflow_file=str(workflow_file),
        step_statuses=step_statuses_from_execution_context(workflow, context),
        outputs=outputs,
    )


@mcp.tool
async def plan_and_execute_workflow(
    user_goal: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
) -> PlannerExecutorResult:
    """Plan and execute an ESM workflow in one call, with completeness and catalog validation gates."""
    return await _plan_and_execute_workflow(user_goal, _settings(data_dir, output_dir))


if __name__ == "__main__":
    mcp.run()
