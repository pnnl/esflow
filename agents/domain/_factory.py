"""Shared construction for category-constrained workflow subagents."""

from typing import Type

from pydantic_ai import Agent, RunContext

from common import WorkflowState, _model_override, _with_context
from common.config import load_prompt, model
from common.workflow import Step


def make_domain_subagent(
    step_type: Type[Step],
    subagent_name: str,
    result_label: str,
    tool_name: str,
    description: str,
    instructions: str | None = None,
) -> tuple[Agent, callable]:
    """Create a category agent and its workflow-mutating tool function."""
    agent = Agent(
        model,
        output_type=list[step_type],
        instructions=load_prompt(
            instructions
            or f"You are the {subagent_name} subagent. "
            "Return only new steps for your category. "
            "Rely on the typed output schema to enforce allowed tool names."
        ),
    )

    async def call(ctx: RunContext[WorkflowState], task: str) -> str:
        run_kwargs = {}
        if _model_override.get() is not None:
            run_kwargs["model"] = _model_override.get()
        result = await agent.run(_with_context(ctx.deps, task), **run_kwargs)
        # Replace, don't duplicate: the planner's only way to "fix" a
        # duplicate-step-id validation error is to re-ask a subagent for a
        # step with that same id (e.g. "retain exactly one step with id X"),
        # but extending the list unconditionally would just add yet another
        # copy of that id, growing the duplicate count instead of resolving
        # it -- an unrecoverable loop that eventually exhausts the request
        # budget. Existing steps sharing an id with a returned step are
        # dropped in place (preserving position) before the new step(s) are
        # appended, so re-adding a step with a known id is an update, not an
        # append.
        new_ids = {step.id for step in result.output}
        ctx.deps.workflow.steps = [
            step for step in ctx.deps.workflow.steps if step.id not in new_ids
        ]
        ctx.deps.workflow.steps.extend(result.output)
        return f"Added {len(result.output)} {result_label} step(s): {[s.id for s in result.output]}"

    call.__name__ = tool_name
    call.__doc__ = description
    return agent, call
