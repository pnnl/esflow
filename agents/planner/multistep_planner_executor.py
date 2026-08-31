from __future__ import annotations

from pydantic_ai import Agent, RunContext

from .multistep_planner import planner
from .oneshot_planner_executor import execute_planned_workflow
from .planner_tools import render_dag
from common import WorkflowState
from common.config import load_prompt, model
from common.workflow import Settings, Workflow


MULTISTEP_PLANNER_EXECUTOR_PROMPT = (
    "You are the interactive Workflow planner-executor. "
    "Handle the conversation over multiple turns when needed. "
    "First use the plan_with_multistep_planner tool to create or revise the workflow; it delegates to the multistep planner. "
    "If the planner asks a clarifying question or reports missing information, return that message as the final answer and wait for the user's next turn. "
    "Do not guess placeholder values for required params. "
    "When the workflow changes, call render_dag and include the Mermaid diagram in your response. "
    "When the workflow is complete and the user wants execution, call execute_planned_workflow. "
    "Always end every turn with a plain-language assistant message, never a structured object. "
    "After execute_planned_workflow returns, format the response in this exact section order: "
    "Status:, Step Results:, Output:, Next Action:. "
    "In Status:, say whether execution succeeded, was blocked, or needs more input. "
    "In Step Results:, summarize which steps completed, failed, were reused, or were skipped. "
    "In Output:, provide the output directory and the saved workflow YAML path when available, "
    "otherwise say no output directory is available yet. "
    "If the execution result includes plot image Markdown, reproduce that Markdown verbatim in the "
    "Output section so the plot renders inline in the chat. Do not alter or drop the image URLs. "
    "In Next Action:, tell the user the next useful thing to do, especially if execution was blocked or more input is needed."
)
async def plan_with_multistep_planner(ctx: RunContext[WorkflowState], task: str) -> str:
    """Use the multistep planner as a subagent to create or revise the workflow."""
    result = await planner.run(task, deps=ctx.deps)
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
    return result.output


MULTISTEP_PLANNER_EXECUTOR_TOOLS = [
    plan_with_multistep_planner,
    render_dag,
    execute_planned_workflow,
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
