from __future__ import annotations

from pydantic_ai import Agent

from .planner_tools import ONESHOT_PLANNER_TOOLS
from common import WorkflowState
from common.config import load_prompt, model
from common.workflow import Settings, Workflow


ONESHOT_PLANNER_PROMPT = (
    "You are the one-shot Workflow planner. "
    "Produce the best complete Workflow in a single pass with no follow-up turn. "
    "Return only the structured Workflow object, never prose. "
    "Never ask clarifying questions and never return plain text. "
    "Do not invent placeholder values, nulls, or unknown tokens for required params. "
    "If a required value is not explicitly provided, use an existing ${settings.*} value or "
    "a valid ${step_id.outputs.*} reference from prior steps; if neither is possible, omit that step. "
    "Plan minimally: include only steps required to satisfy the request. "
    "Select subagents only when they add needed steps, in this order when applicable: "
    "data discovery for missing external datasets/metadata; "
    "extraction for needed fields/timeseries; "
    "diagnostics or water cycle for derived analyses; "
    "visualization only when plots are explicitly requested. "
    "Preserve validity: unique step IDs, known tool names only, and valid output references. "
    "If required inputs already exist from settings or prior outputs, do not regenerate them. "
    "Before returning, call check_completeness and run_workflow_validation. "
    "Only return a Workflow when both tools report no issues. "
    "If no new steps are needed, return the workflow unchanged."
)

oneshot_planner: Agent[WorkflowState, Workflow | str] = Agent(
    model,
    deps_type=WorkflowState,
    output_type=[Workflow],
    tools=ONESHOT_PLANNER_TOOLS,
    instructions=load_prompt(ONESHOT_PLANNER_PROMPT),
)


async def plan_workflow_one_shot(user_goal: str, settings: Settings) -> Workflow:
    """Plan an ESMFlow workflow from a user goal using the planner chain."""
    state = WorkflowState(
        workflow=Workflow(
            name="Planned Workflow",
            description=user_goal,
            settings=settings,
            steps=[],
        )
    )

    # Force a structured Workflow on the programmatic path (evals depend on this).
    # The plain-text conversational path is only enabled for the to_web chat UI.
    result = await oneshot_planner.run(user_goal, deps=state, output_type=Workflow)
    return result.output
