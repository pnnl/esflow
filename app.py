from __future__ import annotations

import os

from pydantic_ai import Agent

from agents import data_discovery, diagnostics, extraction, visualization
from workflow import Settings, Workflow

from agents import water_cycle
from common import WorkflowState
from common.config import model, load_prompt


def default_settings() -> Settings:
    """Build server-side default workflow settings for web requests."""
    return Settings(
        data_dir=os.getenv("ESFLOW_DATA_DIR", "./data/e3sm"),
        output_dir=os.getenv("ESFLOW_OUTPUT_DIR", "./outputs")
    )

def _bootstrap_workflow_state() -> WorkflowState:
    """Create initial workflow state used by web-mode tool calls."""
    return WorkflowState(
        workflow=Workflow(
            name="Generated Workflow",
            description="Web workflow request",
            settings=default_settings(),
            steps=[],
        )
    )

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
    ],
    instructions=load_prompt(
        "You are the Workflow Planning and Routing supervisor. "
        "Decide which subagents are needed based on missing information in the user request and current workflow state. "
        "Do not call a subagent unless it adds required steps. "
        "Use this conditional order only when needed: "
        "data discovery if required external data or metadata is missing; "
        "extraction if variables, fields, or timeseries must be produced; "
        "diagnostics or water cycle if derived metrics are requested; "
        "visualization only if plots are requested. "
        "If required inputs are already available from settings or prior step outputs, skip data discovery. "
        "If no new steps are needed, return the workflow unchanged. "
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

app = supervisor.to_web(deps=_bootstrap_workflow_state())