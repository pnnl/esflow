"""The planner must be able to onboard code without a mode switch — via delegation.

``delegate_to_onboarding`` is the only planner tool that can cause a repo write,
and it causes it *indirectly*, by running :data:`agents.onboarding.onboarding_agent`
with its own ``OnboardingState``. These tests pin the properties that make that
safe and useful:

* the planner module itself still imports no write functions (the boundary the
  user asked for: "keep repo writes inside the onboarding agent");
* the delegated conversation persists across planner turns, so onboarding is
  genuinely multi-turn in one session rather than restarting on every call;
* the delegated run inherits the caller's model and usage, so the web UI's model
  picker applies and delegated tokens are accounted for;
* the benchmarked one-shot stack is untouched.

The delegated agent runs on ``TestModel(call_tools=[])``: TestModel otherwise
calls *every* tool it is offered, which for the onboarding agent includes
``register_draft`` — that would write adapters into ``tools/`` from a unit test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agents.onboarding import OnboardingState, onboarding_agent
from agents.planner.onboarding_bridge import (
    ONBOARDING_BRIDGE_TOOLS,
    OnboardingSession,
    delegate_to_onboarding,
)
from agents.planner.planner_tools import ONESHOT_PLANNER_TOOLS, PLANNER_TOOLS
from common import WorkflowState


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SOURCE = REPO_ROOT / "agents" / "planner" / "onboarding_bridge.py"

#: Names that mutate the repo. The bridge must reach none of them directly.
WRITE_NAMES = {
    "register_capability",
    "register_subagent",
    "remove_capability",
    "save_registry",
    "regenerate_catalog",
    "write_adapter",
    "remove_adapter",
}


@pytest.fixture
def ctx(workflow_state) -> RunContext[WorkflowState]:
    """A real RunContext, so the bridge's ctx.model/ctx.usage plumbing is exercised."""
    return RunContext(
        deps=workflow_state, model=TestModel(call_tools=[]), usage=RunUsage()
    )


def test_bridge_module_imports_no_write_functions():
    """Writes must stay behind the onboarding agent, not leak into planner code."""
    tree = ast.parse(BRIDGE_SOURCE.read_text())

    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert not imported & WRITE_NAMES, (
        "onboarding_bridge.py imported a write function directly; delegate to the "
        f"onboarding agent instead: {sorted(imported & WRITE_NAMES)}"
    )


async def test_delegation_returns_the_specialists_reply(ctx):
    reply = await delegate_to_onboarding(ctx, "I want to onboard my own function")

    assert isinstance(reply, str) and reply


async def test_delegation_creates_an_onboarding_session_on_the_workflow_state(ctx):
    assert ctx.deps.onboarding is None

    await delegate_to_onboarding(ctx, "onboard my code")

    session = ctx.deps.onboarding
    assert isinstance(session, OnboardingSession)
    assert isinstance(session.state, OnboardingState)
    # The delegated exchange is retained, which is what makes it multi-turn.
    assert session.history


async def test_delegation_reuses_one_session_across_turns(ctx):
    """Drafts from an earlier turn must survive; otherwise onboarding restarts."""
    await delegate_to_onboarding(ctx, "onboard examples/user_code/snow_metrics.py")
    session = ctx.deps.onboarding
    session.state.last_source = "sentinel"
    first_turn_messages = len(session.history)

    await delegate_to_onboarding(ctx, "yes, go ahead")

    assert ctx.deps.onboarding is session, "a new session replaced the live one"
    assert session.state.last_source == "sentinel", "draft state was discarded"
    assert len(session.history) > first_turn_messages, "history did not accumulate"


async def test_delegation_inherits_the_callers_model(ctx, monkeypatch):
    """The web UI's model picker must apply to the delegated run too."""
    seen = {}
    original = onboarding_agent.run

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(onboarding_agent, "run", spy)

    await delegate_to_onboarding(ctx, "onboard my code")

    assert seen["model"] is ctx.model
    assert seen["usage"] is ctx.usage
    assert isinstance(seen["deps"], OnboardingState)


async def test_delegation_warns_that_registered_tools_need_a_restart(ctx):
    """A tool registered mid-session is permanent but not yet plannable."""
    await delegate_to_onboarding(ctx, "onboard my code")
    # Simulate the specialist having registered something this session.
    ctx.deps.onboarding.state.registered.append("compute_snow_season_metrics")

    reply = await delegate_to_onboarding(ctx, "did that work?")

    assert "compute_snow_season_metrics" in reply
    assert "restart" in reply.lower()


def test_delegation_is_available_to_the_interactive_planner():
    assert set(ONBOARDING_BRIDGE_TOOLS).issubset(set(PLANNER_TOOLS))


def test_delegation_stays_out_of_the_benchmarked_oneshot_stack():
    """ONESHOT_PLANNER_TOOLS backs the benchmark and MCP server: no mutation there."""
    assert not set(ONBOARDING_BRIDGE_TOOLS) & set(ONESHOT_PLANNER_TOOLS)
