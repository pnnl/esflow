from __future__ import annotations

import re

from pydantic_ai import RunContext

from agents.domain import data_discovery, diagnostics, extraction, visualization
from agents.domain import water_cycle
from agents.domain.extensions import EXTENSION_PLANNER_TOOLS
from agents.planner.onboarding_advisor import ONBOARDING_ADVISOR_TOOLS
from agents.planner.onboarding_bridge import ONBOARDING_BRIDGE_TOOLS
from common import WorkflowState
from common.workflow_validation import (
    check_completeness as _check_completeness,
    validate_workflow,
)


async def check_completeness(ctx: RunContext[WorkflowState]) -> list[str]:
    """List unfilled/placeholder params and dangling output references in the current workflow.

    Returns an empty list when the workflow is fully specified.
    """
    return _check_completeness(ctx.deps.workflow.to_yaml_dict())


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


ONESHOT_PLANNER_TOOLS = [
    data_discovery.call_data_discovery,
    extraction.call_extraction,
    diagnostics.call_diagnostics,
    water_cycle.call_water_cycle_synthesis,
    visualization.call_visualization,
    # Subagents for user-onboarded categories, built at import time from
    # extensions/registry.yaml. Empty unless the user created a new category.
    *EXTENSION_PLANNER_TOOLS,
    check_completeness,
    run_workflow_validation,
]


PLANNER_TOOLS = [
    *ONESHOT_PLANNER_TOOLS,
    render_dag,
    # Read-only onboarding advice, plus one tool that delegates a turn to the
    # onboarding agent so a user can register code without leaving the chat.
    # Interactive stacks only: ONESHOT_PLANNER_TOOLS is what the benchmark and
    # MCP server run, and must stay unchanged and non-mutating.
    *ONBOARDING_ADVISOR_TOOLS,
    *ONBOARDING_BRIDGE_TOOLS,
]
