from __future__ import annotations

from pydantic_ai import Agent, RunContext

from .multistep_planner import planner
from .onboarding_advisor import ONBOARDING_ADVISOR_TOOLS
from .onboarding_bridge import ONBOARDING_BRIDGE_TOOLS
from .oneshot_planner_executor import execute_planned_workflow
from .planner_tools import render_dag, validate_data
from common import WorkflowState
from common.config import load_prompt, model
from common.workflow import Settings, Workflow


MULTISTEP_PLANNER_EXECUTOR_PROMPT = (
    "You are the interactive Workflow planner-executor. "
    "Handle the conversation over multiple turns when needed. "
    "DATA VALIDATION RULE — MANDATORY: Before composing any workflow steps for ANY analysis "
    "request, you MUST call validate_data first. Do not skip this under any circumstances. "
    "validate_data checks which data files are present for each tool and returns: "
    "FEASIBLE tools (data requirements met), PARTIAL tools (degraded results), and BLOCKED tools "
    "(required data files are missing — these tools cannot run). "
    "After validate_data returns, apply these rules strictly:\n"
    "  1. Pass the validate_data result to plan_with_multistep_planner so the planner can "
    "     restrict workflow composition to only FEASIBLE and PARTIAL tools.\n"
    "  2. If validate_data reports BLOCKED tools that the user's request requires, tell the user "
    "     exactly which data files are missing (from the BLOCKED section) before attempting to plan.\n"
    "  3. Do not compose or execute any workflow step that relies on a BLOCKED tool unless the "
    "     user explicitly acknowledges the missing data and asks to proceed anyway.\n"
    "  4. If ALL tools needed for the request are BLOCKED, do not call plan_with_multistep_planner. "
    "     Instead, report the missing data and ask the user to provide it.\n"
    "WORKFLOW STATE RULE — CRITICAL: The workflow built in a previous turn is preserved in memory "
    "across conversation turns. When the user says 'run', 'execute', 'go', 'proceed', or similar, "
    "do NOT call plan_with_multistep_planner again — the workflow steps are already in place. "
    "Call execute_planned_workflow directly. "
    "Only call plan_with_multistep_planner when the user explicitly wants to ADD, CHANGE, or REVISE steps. "
    "After validate_data clears (or user approves partial run), call plan_with_multistep_planner "
    "to create or revise the workflow; it delegates to the multistep planner. "
    "If the planner asks a clarifying question or reports missing information, return that message "
    "as the final answer and wait for the user's next turn. "
    "Do not guess placeholder values for required params. "
    "When the workflow changes, call render_dag and include the Mermaid diagram in your response. "
    "When the workflow is complete and the user wants execution, call execute_planned_workflow directly "
    "without re-planning. "
    "Always end every turn with a plain-language assistant message, never a structured object. "
    "After execute_planned_workflow returns, format the response in this exact section order: "
    "Status:, Step Results:, Output:, Next Action:. "
    "In Status:, say whether execution succeeded, was blocked, or needs more input. "
    "In Step Results:, summarize which steps completed, failed, were reused, or were skipped. "
    "In Output:, provide the output directory and the saved workflow YAML path when available, "
    "otherwise say no output directory is available yet. "
    "If the execution result includes plot image Markdown, reproduce that Markdown verbatim in the "
    "Output section so the plot renders inline in the chat. Do not alter or drop the image URLs. "
    "In Next Action:, tell the user the next useful thing to do, especially if execution was blocked or more input is needed. "
    "If the user needs an analysis no catalogued tool covers, or says they already have Python code that does it, "
    "call explain_onboarding, call show_onboarding_example if they ask what format their code needs or where to "
    "put it, and call preview_user_code_as_tool when they give you a path to their file. "
    "Those three only read files. When the user wants to go ahead and register it, call "
    "delegate_to_onboarding with their request: an onboarding specialist takes over that turn, in this same "
    "session, and writes the tool once the user approves. Relay its reply and pass the user's follow-up "
    "messages back through delegate_to_onboarding so the onboarding conversation stays continuous. "
    "A newly registered tool is permanent but cannot be planned with until the app restarts, so do not add "
    "it to the current workflow."
)


async def plan_with_multistep_planner(ctx: RunContext[WorkflowState], task: str) -> str:
    """Use the multistep planner as a subagent to create or revise the workflow.

    Always passes the current workflow context so the planner can build on existing steps
    rather than restarting from scratch.
    """
    from common import _with_context  # noqa: PLC0415

    task_with_context = _with_context(ctx.deps, task)
    result = await planner.run(task_with_context, deps=ctx.deps)
    if isinstance(result.output, Workflow):
        # Preserve the session's settings across the update -- the model's
        # final structured-output turn is free to rewrite `settings` from the
        # task text, discarding the actual output_dir/data_dir. See the
        # matching fix in plan_workflow_one_shot().
        result.output.settings = ctx.deps.workflow.settings
        ctx.deps.workflow = result.output
        return (
            f"Planner updated the workflow to {len(result.output.steps)} step(s). "
            "If the user wants execution, call execute_planned_workflow."
        )
    # Guard against None output (e.g. empty LLM response or pydantic-ai parse edge case)
    return result.output or "Planner returned no response. Please rephrase your request."


MULTISTEP_PLANNER_EXECUTOR_TOOLS = [
    validate_data,
    plan_with_multistep_planner,
    render_dag,
    execute_planned_workflow,
    # Read-only: lets this agent answer "can I use my own function?" in-place
    # instead of dead-ending the user.
    *ONBOARDING_ADVISOR_TOOLS,
    # Delegates a turn to the onboarding agent, which owns every repo write.
    # This agent still cannot register anything itself.
    *ONBOARDING_BRIDGE_TOOLS,
]


multistep_planner_executor: Agent[WorkflowState, str] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=str,
    tools=MULTISTEP_PLANNER_EXECUTOR_TOOLS,
    instructions=load_prompt(MULTISTEP_PLANNER_EXECUTOR_PROMPT),
)


async def plan_and_execute_workflow_multistep(
    user_goal: str, settings: Settings
) -> str:
    """Plan and optionally execute an ESMFlow workflow via the multistep planner-executor."""
    state = WorkflowState(
        workflow=Workflow(
            name="Planned Workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )
    )
    result = await multistep_planner_executor.run(user_goal, deps=state)
    return result.output
