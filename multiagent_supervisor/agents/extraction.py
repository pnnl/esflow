"""ESM Spatial-Temporal Extraction subagent."""

from typing import List

from pydantic_ai import Agent, RunContext

from workflow import SpatialTemporalExtractionStep

from common import WorkflowState, _with_context
from common.config import model, load_prompt

agent = Agent(
    model,
    output_type=List[SpatialTemporalExtractionStep],
    instructions=load_prompt(
        "You are the ESM Spatial-Temporal Extraction subagent. "
        "Return only new steps for your category. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)


async def call_extraction(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create extraction and matching steps."""
    result = await agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} extraction step(s): {[s.id for s in result.output]}"
