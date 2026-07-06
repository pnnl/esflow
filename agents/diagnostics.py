"""ESM Diagnostics and Skill Metrics subagent."""

from typing import List

from pydantic_ai import Agent, RunContext

from common.workflow import DiagnosticsAndSkillMetricsStep

from common import WorkflowState, _with_context
from common.config import model, load_prompt

agent = Agent(
    model,
    output_type=List[DiagnosticsAndSkillMetricsStep],
    instructions=load_prompt(
        "You are the ESM Diagnostics and Skill Metrics subagent. "
        "Return only new steps for your category. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)


async def call_diagnostics(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create diagnostics and metric computation steps."""
    result = await agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} diagnostics step(s): {[s.id for s in result.output]}"
