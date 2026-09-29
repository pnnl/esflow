"""Shared state and utilities for planner and subagents."""

import re
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Any

from pydantic_ai.models import Model

from common.workflow import Workflow

_OUTPUT_REF = re.compile(r"\$\{(\w+)\.outputs\.\w+\}")


_model_override: ContextVar[Model | None] = ContextVar("model_override", default=None)


@dataclass
class WorkflowState:
    """Shared mutable workflow state used by the planner and subagents."""

    workflow: Workflow

    #: Scratch slot for a delegated onboarding session, so the planner can hand
    #: turns to the onboarding agent without a separate process. Deliberately
    #: untyped: importing ``agents.onboarding`` here would be a circular import,
    #: and nothing in the planning path may depend on the onboarding package.
    onboarding: Any = None

    def context_summary(self) -> str:
        """Summarize workflow settings and existing steps with output key->filename mappings.

        The settings block is always emitted first so every subagent sees the
        concrete data_dir / case_name / output_dir values that were resolved from
        the environment at startup.  This prevents the LLM from asking the user
        for values that are already known.
        """
        s = self.workflow.settings
        lines = [
            "Workflow settings (resolved from environment — use ${settings.<key>} in params):",
            f"  data_dir:   {s.data_dir}",
            f"  output_dir: {s.output_dir}",
            f"  case_name:  {s.case_name or '(auto-detected from E3SM filenames)'}",
        ]

        if not self.workflow.steps:
            lines.append("\nNo steps yet. You are creating the beginning of the workflow.")
            return "\n".join(lines)

        lines.append("\nExisting steps. Use their outputs with ${step_id.outputs.key}:")
        for step in self.workflow.steps:
            outputs_text = ", ".join(f"{k}: {v}" for k, v in step.outputs.items())
            lines.append(f"- {step.id} (tool={step.tool}) -> {outputs_text}")

        return "\n".join(lines)


def _with_context(state: WorkflowState, task: str) -> str:
    """Format task with current workflow context (settings + existing steps)."""
    return f"{state.context_summary()}\n\nTask:\n{task}"


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


def _get_step_dependencies(workflow: Workflow) -> dict:
    """Build a dependency graph: step_id -> set of step_ids it references."""
    dependencies = {step.id: set() for step in workflow.steps}
    step_ids = {step.id for step in workflow.steps}

    for step in workflow.steps:
        for value in _iter_param_strings(step.params):
            for dep_step_id in _OUTPUT_REF.findall(value):
                if dep_step_id in step_ids:
                    dependencies[step.id].add(dep_step_id)

    return dependencies


def _get_reachable_steps(workflow: Workflow) -> set:
    """Return the set of step IDs that are part of the live workflow graph.

    A step is considered reachable if it produces outputs that matter (non-empty
    outputs dict) OR is referenced by at least one other step.  All ancestors of
    reachable steps are included via a backwards BFS/DFS walk.
    """
    if not workflow.steps:
        return set()

    dependencies = _get_step_dependencies(workflow)
    reverse_deps: dict = {step.id: set() for step in workflow.steps}
    for step_id, deps in dependencies.items():
        for dep in deps:
            reverse_deps[dep].add(step_id)

    # Leaves: steps with real outputs OR referenced by another step OR explicitly configured params
    leaves = {
        step.id for step in workflow.steps
        if step.outputs or reverse_deps[step.id] or step.params
    }

    reachable: set = set()
    to_visit = list(leaves)
    while to_visit:
        current = to_visit.pop()
        if current in reachable:
            continue
        reachable.add(current)
        to_visit.extend(dependencies[current] - reachable)

    return reachable


def filter_stale_steps(workflow: Workflow) -> Workflow:
    """Return a new Workflow containing only the reachable (non-stale) steps.

    Steps that have empty params AND empty outputs AND are not referenced by any
    other step are considered stale and dropped.  If nothing is stale, the
    original workflow object is returned unchanged.
    """
    reachable = _get_reachable_steps(workflow)

    if len(reachable) == len(workflow.steps):
        return workflow  # nothing to prune

    clean_steps = [s for s in workflow.steps if s.id in reachable]
    return Workflow(
        name=workflow.name,
        description=workflow.description,
        settings=workflow.settings,
        steps=clean_steps,
    )
