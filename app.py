from __future__ import annotations

import os
from pathlib import Path

from pydantic_ai import Agent, RunContext
import yaml

from agents import data_discovery, diagnostics, extraction, visualization
from workflow import Settings, Workflow

from agents import water_cycle
from common import WorkflowState
from common.config import model, load_prompt


def _default_settings() -> Settings:
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
            settings=_default_settings(),
            steps=[],
        )
    )

async def serialize_workflow_to_yaml(ctx: RunContext[WorkflowState]) -> str:
    """Serialize the currently assembled workflow to a YAML string."""
    workflow_dict = ctx.deps.workflow.to_yaml_dict()
    return yaml.safe_dump(workflow_dict, sort_keys=False)


async def write_yaml_string_to_output(
    ctx: RunContext[WorkflowState],
    yaml_content: str,
    file_name: str = "generated_workflow.yaml",
) -> str:
    """Write a YAML string to a file in the configured output directory."""
    output_dir = Path(ctx.deps.workflow.settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file_name).name
    if not safe_name.endswith((".yaml", ".yml")):
        safe_name = f"{safe_name}.yaml"

    workflow_path = output_dir / safe_name
    workflow_path.write_text(yaml_content, encoding="utf-8")

    return str(workflow_path)

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
        serialize_workflow_to_yaml,
        write_yaml_string_to_output,
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

app = supervisor.to_web(deps=_bootstrap_workflow_state())