"""ESM Data Discovery and Intake subagent."""

from typing import List

from pydantic_ai import Agent, RunContext

from workflow import DataDiscoveryStep

from common import WorkflowState, _with_context
from common.config import model, load_prompt

agent = Agent(
    model,
    output_type=List[DataDiscoveryStep],
    instructions=load_prompt(
        "You are the ESM Data Discovery and Intake subagent. "
        "Return only new steps for your category. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)


async def call_data_discovery(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create data intake steps (fetch/load metadata)."""
    result = await agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} data discovery step(s): {[s.id for s in result.output]}"
