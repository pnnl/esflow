from __future__ import annotations

from pathlib import Path

from starlette.staticfiles import StaticFiles

from agents.planner.multistep_planner import planner
from agents.planner.settings import default_settings
from common import WorkflowState
from common.config import MODELS
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


app = planner.to_web(
    deps=_bootstrap_workflow_state(),
    models=MODELS,
    html_source=Path(__file__).parent / "web_ui.html",
    instructions=(
        "You are chatting interactively. Greet the user briefly and explain you compose "
        "ESM analysis workflows from a validated tool library. "
        "After each change to the workflow, call render_dag and include its Mermaid diagram "
        "in your reply so the user can see the current plan."
    ),
)

# Serve ESFlow branding assets (logo, favicons) referenced by web_ui.html.
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)
