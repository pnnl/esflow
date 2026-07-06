"""Basin-Scale Water Cycle Synthesis subagent."""

from typing import List

from pydantic_ai import Agent, RunContext

from common.workflow import BasinScaleWaterCycleSynthesisStep

from common import WorkflowState, _with_context
from common.config import model, load_prompt

agent = Agent(
    model,
    output_type=List[BasinScaleWaterCycleSynthesisStep],
    instructions=load_prompt(
        "You are the Basin-Scale Water Cycle Synthesis subagent. "
        "Return only new steps for your category. "
        "Rely on the typed output schema to enforce allowed tool names."
    ),
)


async def call_water_cycle_synthesis(ctx: RunContext[WorkflowState], task: str) -> str:
    """Create basin-scale water cycle synthesis steps."""
    result = await agent.run(_with_context(ctx.deps, task))
    ctx.deps.workflow.steps.extend(result.output)
    return f"Added {len(result.output)} water cycle step(s): {[s.id for s in result.output]}"
