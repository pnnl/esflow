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
) -> tuple[Agent, callable]:
    """Create a category agent and its workflow-mutating tool function."""
    agent = Agent(
        model,
        output_type=list[step_type],
        instructions=load_prompt(
            f"You are the {subagent_name} subagent. "
            "Return only new steps for your category. "
            "Rely on the typed output schema to enforce allowed tool names."
        ),
    )

    async def call(ctx: RunContext[WorkflowState], task: str) -> str:
        run_kwargs = {}
        if _model_override.get() is not None:
            run_kwargs["model"] = _model_override.get()
        result = await agent.run(_with_context(ctx.deps, task), **run_kwargs)
        ctx.deps.workflow.steps.extend(result.output)
        return f"Added {len(result.output)} {result_label} step(s): {[s.id for s in result.output]}"

    call.__name__ = tool_name
    call.__doc__ = description
    return agent, call
