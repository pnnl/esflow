"""ESM Diagnostic Visualization subagent."""

from typing import List

from pydantic_ai import Agent, RunContext

from common.workflow import DiagnosticVisualizationStep

from common import WorkflowState, _with_context
from common.config import model, load_prompt

agent = Agent(
    model,
    output_type=List[DiagnosticVisualizationStep],
    instructions=load_prompt(
        "You are the ESM Diagnostic Visualization subagent. "
        "Return only new steps for your category. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)


async def call_visualization(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create visualization steps for diagnostics outputs."""
    result = await agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} visualization step(s): {[s.id for s in result.output]}"
