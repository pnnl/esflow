from __future__ import annotations

from pydantic_ai import Agent, RunContext

from agents import data_discovery, diagnostics, extraction, visualization
from workflow import Settings, Workflow

from agents import water_cycle
from common import WorkflowState
from common.config import model, load_prompt


async def get_workflow_tool(ctx: RunContext[WorkflowState]) -> str:
    """Return the currently assembled workflow as JSON."""
    return ctx.deps.workflow.model_dump_json(indent=2)


supervisor: Agent[WorkflowState, Workflow] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=Workflow,
    tools=[
        data_discovery.call_data_discovery,
        extraction.call_extraction,
        diagnostics.call_diagnostics,
        water_cycle.call_water_cycle_synthesis,
        visualization.call_visualization,
        get_workflow_tool,
    ],
    instructions=load_prompt(
        "You are the Workflow Planning and Routing supervisor. "
        "Break requests into subproblems and call subagents in dependency order: "
        "data -> extraction -> diagnostics/water cycle -> visualization. "
        "After composing steps, return the complete structured Workflow object."
    ),
)


async def build_workflow(user_goal: str, settings: Settings) -> Workflow:
    """Generate an ESMFlow workflow from a user goal using the supervisor chain."""
    state = WorkflowState(
        workflow=Workflow(
            name="Generated Workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )
    )

    result = await supervisor.run(user_goal, deps=state)
    return result.output


app = supervisor.to_web()
