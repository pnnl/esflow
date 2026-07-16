from __future__ import annotations

from pathlib import Path

from starlette.staticfiles import StaticFiles

from agents.planner.multistep_planner import planner
from agents.planner.multistep_planner_executor import multistep_planner_executor
from agents.planner.settings import default_settings
from common import WorkflowState
from common.config import MODELS, WebAgentMode, runtime_config
from common.workflow import Workflow


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


def _build_web_app():
    """Build the configured chat app for the selected web agent mode."""
    web_agent_mode = runtime_config.WEB_AGENT_MODE

    if web_agent_mode is WebAgentMode.PLANNER_EXECUTOR:
        web_agent = multistep_planner_executor
        instructions = (
            "You are chatting interactively. Greet the user briefly and explain you can plan and execute "
            "ESM analysis workflows from a validated tool library. "
            "Use plan_with_multistep_planner to keep the workflow updated across turns. "
            "After each workflow change, call render_dag and include its Mermaid diagram "
            "in your reply so the user can see the current plan. "
            "Only call execute_planned_workflow when the user asks to run the workflow or clearly confirms execution. "
            "After execute_planned_workflow returns, always send a plain-language summary to the user. "
            "Format that summary in this exact section order: Status:, Step Results:, Output:, Next Action:. "
            "In Status:, explain whether execution succeeded, was blocked, or needs more input. "
            "In Step Results:, summarize step outcomes. "
            "In Output:, mention the output directory and the saved workflow YAML path when available, "
            "otherwise state that no output directory is available yet. "
            "If the execution result includes plot image Markdown, reproduce that Markdown verbatim in the "
            "Output section so the plot renders inline in the chat. Do not alter or drop the image URLs. "
            "In Next Action:, tell the user the next useful thing to do. "
            "Do not stop at the raw tool result; convert it into a clear conversational response."
        )
    elif web_agent_mode is WebAgentMode.PLANNER:
        web_agent = planner
        instructions = (
            "You are chatting interactively. Greet the user briefly and explain you compose "
            "ESM analysis workflows from a validated tool library. "
            "After each change to the workflow, call render_dag and include its Mermaid diagram "
            "in your reply so the user can see the current plan."
        )
    else:
        raise ValueError(f"Unsupported WEB_AGENT_MODE: {web_agent_mode}")

    return web_agent.to_web(
        deps=_bootstrap_workflow_state(),
        models=MODELS,
        html_source=Path(__file__).parent / "web_ui.html",
        instructions=instructions,
    )


app = _build_web_app()

# Serve ESFlow branding assets (logo, favicons) referenced by web_ui.html.
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

# Serve generated workflow outputs so plots can be shown inline in the chat.
# Mounted at a URL matching the dir name, so a plot written to
# ``output/<run>/map.png`` is reachable at ``/output/<run>/map.png`` — a short,
# stable URL the agent can embed as a Markdown image.
_OUTPUT_DIR = Path(__file__).parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/output",
    StaticFiles(directory=str(_OUTPUT_DIR)),
    name="output",
)
