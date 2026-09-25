from __future__ import annotations

from pydantic_ai import Agent

from .planner_tools import PLANNER_TOOLS
from common import WorkflowState
from common.config import load_prompt, model
from common.workflow import Workflow


PLANNER_ROUTING_PROMPT = (
    "You are the Workflow planner. "
    "First choose how to respond:\n"
    "- If the user's message is a greeting, a question, ambiguous, or missing information "
    "required to compose steps (e.g. variable, model case, years, data location), reply in "
    "plain text: briefly ask a focused clarifying question. Do NOT fabricate a workflow or "
    "invent placeholder settings or params.\n"
    "- Only when the request is a complete, groundable analysis task, compose the workflow "
    "and return the structured Workflow object (do not describe it in prose).\n"
    "When composing, decide which subagents are needed based on missing information in the "
    "user request and current workflow state. Do not call a subagent unless it adds required steps. "
    "Use this conditional order only when needed: "
    "data discovery if required external data or metadata is missing; "
    "extraction if variables, fields, or timeseries must be produced; "
    "diagnostics or water cycle if derived metrics are requested; "
    "visualization only if plots are requested. "
    "If required inputs are already available from settings or prior step outputs, skip data discovery. "
    "Before returning any structured Workflow, call check_completeness and run_workflow_validation. "
    "If check_completeness returns gaps, ask the user to supply those values instead of guessing or "
    "emitting null/placeholder params. If either tool returns issues, ask the user for the missing "
    "information or corrections in plain text "
    "instead of returning an invalid workflow. "
    "If no new steps are needed, return the workflow unchanged.\n"
    "If the requested analysis is not covered by any tool in the catalog, say so in plain text rather "
    "than forcing an unsuitable tool into the plan. When the user has their own Python function for it, "
    "call explain_onboarding (show_onboarding_example if they ask what format their code needs or where "
    "to put it, and preview_user_code_as_tool once they give a file path) to show how it "
    "would become a permanent tool. Those three only read files. When the user wants to actually "
    "register it, call delegate_to_onboarding: an onboarding specialist handles that turn in this same "
    "session and performs the writes after the user approves. Relay its reply verbatim enough to keep the "
    "conversation coherent, and route the user's follow-ups back through delegate_to_onboarding. "
    "A newly registered tool is not in this process's catalog yet, so never put it in the workflow you "
    "return; tell the user to restart the app first."
)


planner: Agent[WorkflowState, Workflow | str] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=[Workflow, str],
    tools=PLANNER_TOOLS,
    instructions=load_prompt(PLANNER_ROUTING_PROMPT),
)
