from __future__ import annotations

from pathlib import Path

from starlette.staticfiles import StaticFiles

from agents.onboarding import OnboardingState, onboarding_agent
from agents.planner.multistep_planner import planner
from agents.planner.multistep_planner_executor import multistep_planner_executor
from agents.planner.settings import default_settings
from common import WorkflowState
from common.config import MODELS, WebAgentMode, runtime_config
from common.logging_setup import configure_console_logging
from common.workflow import Workflow


configure_console_logging()


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
    # Every mode but ONBOARDING plans workflows and therefore shares
    # WorkflowState; the onboarding agent mutates capability drafts instead.
    #
    # The planner modes no longer *need* the ONBOARDING mode: they carry
    # delegate_to_onboarding, which runs the onboarding agent in this same
    # process on its own OnboardingState (see agents/planner/onboarding_bridge.py).
    # ONBOARDING remains for a dedicated, planner-free onboarding session.
    deps = _bootstrap_workflow_state()

    if web_agent_mode is WebAgentMode.PLANNER_EXECUTOR:
        web_agent = multistep_planner_executor
        instructions = (
            "You are chatting interactively. Greet the user briefly and explain you can plan and execute "
            "ESM analysis workflows from a validated tool library. "
            "DATA VALIDATION RULE — MANDATORY: Before composing any workflow steps for ANY analysis "
            "request, you MUST call validate_data first. Do not skip this under any circumstances — "
            "it checks which data files are present for each tool and determines which tools are "
            "FEASIBLE, PARTIAL, or BLOCKED. "
            "Only compose workflow steps for FEASIBLE or PARTIAL tools. "
            "If tools the user needs are BLOCKED, tell them exactly which data files are missing "
            "before attempting to plan. Do not compose a workflow for BLOCKED tools unless the user "
            "explicitly acknowledges the missing data and asks to proceed anyway. "
            "WORKFLOW STATE RULE — CRITICAL: The workflow you built in a previous turn is preserved in "
            "memory. When the user asks to run/execute/go/proceed, do NOT call plan_with_multistep_planner "
            "again — the workflow steps are already in place. Call execute_planned_workflow directly. "
            "Only call plan_with_multistep_planner when the user wants to ADD, CHANGE, or REVISE steps. "
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
            "Do not stop at the raw tool result; convert it into a clear conversational response. "
            "If the user wants an analysis the tool library does not cover, or mentions their own Python "
            "code for it, call explain_onboarding, show_onboarding_example if they ask what format their "
            "code needs or where to put it, and preview_user_code_as_tool once they give you a "
            "file path. Those three only read files. When they want to register it for good, call "
            "delegate_to_onboarding and keep routing their onboarding replies through it — no mode "
            "switch or restart is needed to onboard. Registration is permanent, but this process cannot "
            "plan with the new tool until it restarts, so do not add it to the current workflow."
        )
    elif web_agent_mode is WebAgentMode.PLANNER:
        web_agent = planner
        instructions = (
            "You are chatting interactively. Greet the user briefly and explain you compose "
            "ESM analysis workflows from a validated tool library. "
            "DATA VALIDATION RULE — MANDATORY: Before composing any workflow steps for ANY analysis "
            "request, call validate_data first. Do not skip this — it checks which data files are "
            "present for each tool and returns FEASIBLE, PARTIAL, and BLOCKED tool lists. "
            "Only compose workflow steps using tools from the FEASIBLE or PARTIAL lists. "
            "If the user's request requires BLOCKED tools, report the missing data files and do not "
            "compose those steps unless the user explicitly asks to proceed anyway. "
            "After each change to the workflow, call render_dag and include its Mermaid diagram "
            "in your reply so the user can see the current plan. "
            "If the user wants an analysis the tool library does not cover, or mentions their own Python "
            "code for it, call explain_onboarding, show_onboarding_example if they ask what format their "
            "code needs or where to put it, and preview_user_code_as_tool once they give you a "
            "file path. Those three only read files. When they want to register it for good, call "
            "delegate_to_onboarding and keep routing their onboarding replies through it — no mode "
            "switch or restart is needed to onboard. Registration is permanent, but this process cannot "
            "plan with the new tool until it restarts, so do not add it to the current workflow."
        )
    elif web_agent_mode is WebAgentMode.ONBOARDING:
        web_agent = onboarding_agent
        deps = OnboardingState()
        instructions = (
            "You are chatting interactively in a dedicated onboarding session. Greet the user briefly "
            "and explain that you turn their own Python analysis code into ESMFlow "
            "capabilities the planner can use, and that you will never write anything "
            "to the repository without their approval. "
            "Ask for the path to the Python file they want to onboard, then call "
            "scan_user_code. Walk them through the introspector's notes before "
            "registering anything, and finish by showing the verification report and a "
            "runnable workflow snippet."
        )
    else:
        raise ValueError(f"Unsupported WEB_AGENT_MODE: {web_agent_mode}")

    return web_agent.to_web(
        deps=deps,
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
