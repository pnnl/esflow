from __future__ import annotations

import re

from pydantic_ai import RunContext

from agents.domain import data_discovery, diagnostics, extraction, visualization
from agents.domain import water_cycle
from agents.domain import spatial_bias, extremes, model_comparison, obs_survey
from agents.domain.extensions import EXTENSION_PLANNER_TOOLS
from agents.planner.onboarding_advisor import ONBOARDING_ADVISOR_TOOLS
from agents.planner.onboarding_bridge import ONBOARDING_BRIDGE_TOOLS
from common import WorkflowState, filter_stale_steps
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


async def prune_stale_steps(ctx: RunContext[WorkflowState]) -> str:
    """Remove stale/orphaned steps from the workflow that are not connected to any output.

    A step is stale when it has empty params AND empty outputs AND no other step
    references it.  Call this after subagent calls to keep the workflow clean.

    Returns a summary of how many steps were pruned (or confirms nothing was removed).
    """
    before = len(ctx.deps.workflow.steps)
    pruned = filter_stale_steps(ctx.deps.workflow)
    removed = before - len(pruned.steps)
    original_ids = {s.id for s in ctx.deps.workflow.steps}
    ctx.deps.workflow = pruned
    if removed == 0:
        return "No stale steps found — workflow is clean."
    removed_ids = sorted(original_ids - {s.id for s in pruned.steps})
    return f"Pruned {removed} stale step(s): {removed_ids}"


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


async def validate_data(
    ctx: RunContext[WorkflowState],
    data_dir: str = None,
    case_name: str = None,
    years: list = None,
) -> str:
    """Validate that user-provided data is in the correct format before composing workflows.

    This checks data structure against catalog requirements: gauge metadata, streamflow CSVs,
    basin polygons, E3SM model output, and observation files.

    Args:
        data_dir: Root data directory to inspect (default: from workflow settings /
                  ``ESFLOW_DATA_DIR`` env var)
        case_name: E3SM case name (default: from workflow settings /
                   ``ESFLOW_CASE_NAME`` env var)
        years: Years to check for data coverage (default: ``ESFLOW_YEARS`` env var,
               then empty list)

    Returns: Formatted validation report with feasible/blocked tools.
    """
    # Lazy imports — pandas/numpy are heavy; only load when actually validating.
    from agents import data_preflight  # noqa: PLC0415
    from common.config import runtime_config  # noqa: PLC0415

    workflow = ctx.deps.workflow
    data_dir = data_dir or workflow.settings.data_dir or str(runtime_config.resolved_data_dir())
    case_name = case_name or workflow.settings.case_name or runtime_config.ESFLOW_CASE_NAME or ""

    # Resolve the tool catalog from env/config so no path is hard-coded here.
    catalog_file = runtime_config.resolved_catalog_file()
    if not catalog_file.exists():
        return (
            f"✗ tool_catalog.yaml not found at {catalog_file}. "
            "Set ESFLOW_CATALOG_FILE in your .env to point at the correct path."
        )

    # Auto-detect case_name and years from E3SM filenames when not supplied
    # via args, workflow settings, or .env.  This means users never need to
    # configure ESFLOW_CASE_NAME / ESFLOW_YEARS manually.
    if not case_name or years is None:
        from agents.data_preflight import _autodetect_e3sm_context  # noqa: PLC0415
        detected = _autodetect_e3sm_context(data_dir)
        if not case_name:
            case_name = detected['case_name']
        if years is None:
            years = runtime_config.resolved_years() or detected['years']

    report = data_preflight.preflight_check(
        data_dir=data_dir,
        catalog_file=str(catalog_file),
        case_name=case_name,
        years=years,
    )

    lines = [
        "📊 Data Validation Report",
        "=" * 70,
        report.summary,
        "",
    ]

    if report.has_errors:
        lines += [
            "⚠️  ERROR: Some data is missing or malformed.",
            "   Fix these issues before composing workflows:",
            "",
        ]
        for error in report.errors:
            lines.append(f"   • {error}")

    # Always list tool names explicitly so the planner can filter subagent routing.
    if report.feasible:
        lines += [
            "",
            f"✓ FEASIBLE ({len(report.feasible)} tools — data requirements met, safe to include in workflow):",
        ]
        for t in sorted(report.feasible):
            lines.append(f"   ✓ {t}")

    if report.partial:
        lines += [
            "",
            f"⚠  PARTIAL ({len(report.partial)} tools — may run with degraded results):",
        ]
        for t in sorted(report.partial):
            lines.append(f"   ⚠ {t}")

    if report.blocked:
        lines += [
            "",
            f"✗ BLOCKED ({len(report.blocked)} tools — required data is MISSING, do NOT include in workflow):",
        ]
        for t in sorted(report.blocked):
            lines.append(f"   ✗ {t}")

    lines += [
        "",
        "PLANNER INSTRUCTION: Only compose workflow steps that use tools listed under FEASIBLE or PARTIAL above.",
        "Do NOT call any subagent that would produce steps using a BLOCKED tool unless the user explicitly overrides.",
        "If the user's request requires only blocked tools, inform them of the missing data and do not compose a workflow.",
    ]

    return "\n".join(lines)


ONESHOT_PLANNER_TOOLS = [
    validate_data,
    data_discovery.call_data_discovery,
    extraction.call_extraction,
    diagnostics.call_diagnostics,
    water_cycle.call_water_cycle_synthesis,
    visualization.call_visualization,
    spatial_bias.call_spatial_bias,
    extremes.call_extremes,
    model_comparison.call_model_comparison,
    obs_survey.call_obs_survey,
    # Subagents for user-onboarded categories, built at import time from
    # extensions/registry.yaml. Empty unless the user created a new category.
    *EXTENSION_PLANNER_TOOLS,
    check_completeness,
    run_workflow_validation,
    prune_stale_steps,
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
