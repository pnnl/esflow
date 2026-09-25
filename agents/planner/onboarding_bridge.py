"""Let the planner hand a turn to the onboarding agent, in the same process.

Why this exists
---------------
Onboarding used to be a separate ``WEB_AGENT_MODE``: a user who discovered
mid-analysis that they needed their own Python function had to stop, restart the
app with ``WEB_AGENT_MODE=onboarding``, redo the conversation, and restart again.
That mode switch was never a technical requirement — it was just an ``if/elif``
in ``app.py`` binding one agent per process.

What this does *not* change
---------------------------
Repo writes still live entirely in the onboarding agent. The planner does not
gain ``register_capability``/``write_adapter``; it gains exactly one tool that
*delegates a turn* to :data:`agents.onboarding.onboarding_agent`, which keeps its
own instructions, its own approval discipline and its own ``OnboardingState``.
This is pydantic-ai agent delegation, so the onboarding agent's tool-call loop
runs to completion and only its final text comes back to the planner.

Session continuity
------------------
Onboarding is inherently multi-turn (scan, review notes, correct, register). A
single ``run`` would lose the drafts, so the session — ``OnboardingState`` plus
the delegated message history — is parked on ``WorkflowState.onboarding`` and
reused on every subsequent call. The user therefore keeps one continuous
onboarding conversation *inside* their planning conversation.

Catalog staleness
-----------------
Registering mid-session writes the adapter and the catalog correctly, but the
running process cannot plan with the new tool yet: ``common.workflow`` builds
``STEP_CLASSES`` (whose ``tool`` field is a ``Literal`` allow-list) at import
time, and the planner's ``output_type`` schema is derived from those classes at
construction. A new *category* is worse still, since its subagent tool would
have to be spliced into an already-constructed ``Agent``. So a restart is still
needed before the tool is usable — but only once, at the end, instead of twice
in the middle. :func:`delegate_to_onboarding` says so explicitly in its result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage

from agents.onboarding import OnboardingState, onboarding_agent
from common import WorkflowState


@dataclass
class OnboardingSession:
    """A delegated onboarding conversation, carried across planner turns."""

    state: OnboardingState = field(default_factory=OnboardingState)
    history: List[ModelMessage] = field(default_factory=list)


def _session(deps: WorkflowState) -> OnboardingSession:
    """Return the live onboarding session, creating it on first use."""
    if not isinstance(getattr(deps, "onboarding", None), OnboardingSession):
        deps.onboarding = OnboardingSession()
    return deps.onboarding


async def delegate_to_onboarding(ctx: RunContext[WorkflowState], request: str) -> str:
    """Hand this turn to the onboarding specialist, which can register user code.

    Use when the user wants their own Python function turned into a reusable
    ESMFlow tool and has agreed to go ahead — the specialist scans their file,
    walks them through what it inferred, and writes the adapter and catalog entry
    once they approve. Preview-only questions are cheaper via
    ``preview_user_code_as_tool``.

    The onboarding conversation is continuous: pass the user's latest message
    each time and previous drafts are still there.

    Args:
        request: The user's onboarding request, verbatim where possible,
            including the path to their .py file if they gave one.

    Returns:
        The specialist's reply, to relay to the user.
    """
    session = _session(ctx.deps)

    result = await onboarding_agent.run(
        request,
        deps=session.state,
        message_history=session.history,
        # Inherit the caller's model so the web UI's model picker applies, and
        # share usage so delegated tokens count against the parent run's limits.
        model=ctx.model,
        usage=ctx.usage,
    )
    session.history = result.all_messages()

    reply = result.output
    if session.state.registered:
        registered = ", ".join(session.state.registered)
        reply += (
            f"\n\n(Registered this session: {registered}. These are written to the "
            "catalog and available in every future session, but this running "
            "process still holds the tool list it loaded at startup — restart the "
            "app before planning with them.)"
        )
    return reply


#: Interactive planner stacks only. Deliberately kept out of
#: ONESHOT_PLANNER_TOOLS: that list backs the benchmark and the MCP server, which
#: must stay non-mutating and comparable across runs.
ONBOARDING_BRIDGE_TOOLS = [delegate_to_onboarding]
