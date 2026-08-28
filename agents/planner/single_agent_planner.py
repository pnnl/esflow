"""Single-agent baseline planner: one LLM call, full tool catalog inlined,
no subagent delegation and no self-check tools.

This is the benchmark's actual baseline arm: v1's original architecture was
a single LLM call given the tool catalog as a static system prompt,
producing YAML directly with no supervisor/subagent orchestration and no
in-loop validation tools. plan_workflow_single_agent() reproduces that
architecture against v2's own tool catalog/data, so the benchmark measures
v2's supervisor/planner architecture (agents/planner/oneshot_planner.py)
against a single-agent baseline, not against free-form LLM-generated code.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

from common.config import load_prompt
from common.workflow import Settings, Workflow


SINGLE_AGENT_PROMPT = (
    "You are given a complete catalog of analysis tools below. "
    "Produce a complete Workflow that satisfies the user's request in a single pass. "
    "You have no subagents, no delegation, and no tools to check or validate your own "
    "output -- this response is final and will be validated afterward by a separate process. "
    "Return only the structured Workflow object, never prose. "
    "Choose tool names only from the catalog below; do not invent tools or params. "
    "Do not invent placeholder values, nulls, or unknown tokens for required params. "
    "If a required value is not explicitly provided, use an existing ${settings.*} value or "
    "a valid ${step_id.outputs.*} reference from a step earlier in the same workflow; "
    "if neither is possible, omit that step. "
    "Plan minimally: include only steps required to satisfy the request. "
    "Preserve validity: unique step IDs, known tool names only, and valid output references."
)

single_agent_planner: Agent[None, Workflow] = Agent(
    None,
    output_type=[Workflow],
    instructions=load_prompt(SINGLE_AGENT_PROMPT),
)


async def plan_workflow_single_agent(
    user_goal: str, settings: Settings, model_override: Model | None = None
) -> Workflow:
    """Plan an ESMFlow workflow with a single, undelegated LLM call.

    Unlike plan_workflow_one_shot() (agents/planner/oneshot_planner.py), this
    has no subagent tools and no check_completeness/run_workflow_validation
    self-check loop -- the model gets one shot at the full Workflow, with no
    in-loop feedback, matching v1's single-agent call_llm architecture.
    """
    if model_override is None:
        raise ValueError("plan_workflow_single_agent requires an explicit model_override")
    result = await single_agent_planner.run(
        user_goal,
        model=model_override,
        output_type=Workflow,
        deps=None,
    )
    workflow = result.output
    workflow.settings = settings
    return workflow
