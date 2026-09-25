"""The demonstration must be safe to run from a chat turn, always.

Two properties are tested here, both about the *agent layer* rather than the
demo engine (that is ``tests/onboarding/test_demo.py``):

1. **The demo always cleans up.** ``demo_onboarding_walkthrough`` is the only
   onboarding tool that writes to the repo without the user naming a file, so
   it must call ``end_demo`` on every path -- including when the run stage
   raises. The stage functions are stubbed here so the assertion is about
   control flow, not about the real registry.
2. **The explainers never write.** ``explain_onboarding_format`` (onboarding
   agent) and ``show_onboarding_example`` (planner advisor) exist so a user can
   ask "what does my code have to look like?" without anything happening on
   disk. Every write entry point is replaced with a fail-fast stub, mirroring
   ``tests/agents/test_onboarding_advisor.py``.

Plus a wiring check: the tool names quoted in the agent's phase-0 instructions
must actually be registered, or the model will hallucinate calls to tools that
do not exist.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest

# ``agents.onboarding.__init__`` re-exports the *Agent object* under the name
# ``onboarding_agent``, which shadows the submodule attribute on the package, so
# `import agents.onboarding.onboarding_agent as m` would bind the Agent instead
# of the module. Go through sys.modules to get the real module to monkeypatch.
agent_mod = importlib.import_module("agents.onboarding.onboarding_agent")
from agents.onboarding.onboarding_agent import (
    ONBOARDING_INSTRUCTIONS,
    ONBOARDING_TOOLS,
    demo_onboarding_walkthrough,
    demo_status,
    end_onboarding_demo,
    explain_onboarding_format,
)
from agents.onboarding.state import OnboardingState
from agents.planner.onboarding_advisor import (
    ONBOARDING_ADVISOR_TOOLS,
    show_onboarding_example,
)
from common import WorkflowState
from common.workflow import Settings, Workflow
from onboarding.demo import (
    DEMO_TOOL,
    DemoEndResult,
    DemoError,
    DemoRunResult,
    DemoStartResult,
    DemoStatus,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Ctx:
    """Minimal stand-in for RunContext: these tools only ever touch ``deps``."""

    deps: object


@pytest.fixture
def ctx() -> _Ctx:
    return _Ctx(deps=OnboardingState())


@pytest.fixture
def planner_ctx(tmp_path) -> _Ctx:
    return _Ctx(
        deps=WorkflowState(
            workflow=Workflow(
                name="t",
                description="t",
                settings=Settings(data_dir=str(tmp_path), output_dir=str(tmp_path)),
                steps=[],
            )
        )
    )


@pytest.fixture
def no_writes(monkeypatch):
    """Make every onboarding write path explode if a read-only tool calls it."""

    def _forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"read-only tool called write path {name!r}")

        return _raise

    import onboarding.registry as registry
    import onboarding.scaffold as scaffold

    for module, name in (
        (registry, "register_capability"),
        (registry, "register_subagent"),
        (registry, "remove_capability"),
        (registry, "save_registry"),
        (registry, "regenerate_catalog"),
        (scaffold, "write_adapter"),
        (scaffold, "remove_adapter"),
    ):
        monkeypatch.setattr(module, name, _forbidden(name))


def _start_result() -> DemoStartResult:
    return DemoStartResult(
        tool_name=DEMO_TOOL,
        subagent="analysis",
        adapter_path=f"tools/analyzers/{DEMO_TOOL}.py",
    )


def _end_result(**overrides) -> DemoEndResult:
    result = DemoEndResult(
        removed_capability=True,
        removed_adapter=True,
        regenerated_catalog=True,
        adapter_path=f"tools/analyzers/{DEMO_TOOL}.py",
    )
    return result.model_copy(update=overrides)


@pytest.fixture
def stages(monkeypatch):
    """Stub the three demo stages and record the order they were called in."""

    calls: list[str] = []

    def _start(*args, **kwargs):
        calls.append("start")
        return _start_result()

    def _run(*args, **kwargs):
        calls.append("run")
        return DemoRunResult(ok=True, scalars={"total_degree_days": 1.0})

    def _end(*args, **kwargs):
        calls.append("end")
        return _end_result()

    monkeypatch.setattr(agent_mod, "start_demo", _start)
    monkeypatch.setattr(agent_mod, "run_demo", _run)
    monkeypatch.setattr(agent_mod, "end_demo", _end)
    return calls


# ---------------------------------------------------------------------------
# The walkthrough is self-cleaning
# ---------------------------------------------------------------------------


async def test_the_walkthrough_registers_runs_then_offboards(ctx, stages):
    reply = await demo_onboarding_walkthrough(ctx)

    assert stages == ["start", "run", "end"]
    # All three stages are narrated back, so the user sees the whole arc.
    assert DEMO_TOOL in reply
    assert "Offboarded the demo capability" in reply


async def test_the_demo_is_offboarded_even_when_the_run_stage_fails(
    ctx, stages, monkeypatch
):
    """A broken run must not leave a registered capability behind."""

    def _boom(*args, **kwargs):
        stages.append("run")
        raise DemoError("subprocess died")

    monkeypatch.setattr(agent_mod, "run_demo", _boom)

    reply = await demo_onboarding_walkthrough(ctx)

    assert stages == ["start", "run", "end"]
    assert "The run stage failed: subprocess died" in reply
    assert "Offboarded the demo capability" in reply


async def test_a_failed_cleanup_is_reported_with_the_manual_command(
    ctx, stages, monkeypatch
):
    """If even offboarding fails the user is told how to finish the job."""

    def _boom(*args, **kwargs):
        stages.append("end")
        raise DemoError("registry is locked")

    monkeypatch.setattr(agent_mod, "end_demo", _boom)

    reply = await demo_onboarding_walkthrough(ctx)

    assert "WARNING" in reply
    assert "python -m onboarding.cli demo end" in reply


async def test_a_failed_start_is_reported_and_nothing_else_runs(
    ctx, stages, monkeypatch
):
    """A pre-existing foreign capability aborts the demo conversationally."""

    def _boom(*args, **kwargs):
        stages.append("start")
        raise DemoError("a capability with that name is NOT ours")

    monkeypatch.setattr(agent_mod, "start_demo", _boom)

    reply = await demo_onboarding_walkthrough(ctx)

    # No run, and critically no end_demo: we must not delete someone else's tool.
    assert stages == ["start"]
    assert "Could not start the demonstration" in reply


async def test_end_onboarding_demo_is_safe_to_call_when_already_clean(
    ctx, monkeypatch
):
    monkeypatch.setattr(
        agent_mod, "end_demo", lambda *a, **k: _end_result(was_already_clean=True)
    )

    reply = await end_onboarding_demo(ctx)

    assert "Nothing to offboard" in reply


async def test_end_onboarding_demo_reports_errors_conversationally(ctx, monkeypatch):
    def _boom(*args, **kwargs):
        raise DemoError("not ours")

    monkeypatch.setattr(agent_mod, "end_demo", _boom)

    reply = await end_onboarding_demo(ctx)

    assert reply.startswith("Could not clean up the demonstration")


async def test_demo_status_reports_the_engine_status(ctx, monkeypatch):
    monkeypatch.setattr(
        agent_mod,
        "_demo_status",
        lambda *a, **k: DemoStatus(
            registered=False, adapter_exists=False, in_catalog=False, adapter_path=""
        ),
    )

    reply = await demo_status(ctx)

    assert "not active" in reply.lower()


async def test_demo_status_reports_errors_conversationally(ctx, monkeypatch):
    def _boom(*args, **kwargs):
        raise DemoError("unreadable registry")

    monkeypatch.setattr(agent_mod, "_demo_status", _boom)

    assert "Could not determine" in await demo_status(ctx)


# ---------------------------------------------------------------------------
# The explainers are read-only
# ---------------------------------------------------------------------------


async def test_explain_onboarding_format_teaches_placement_without_writing(
    ctx, no_writes
):
    reply = await explain_onboarding_format(ctx)

    # Where the user's code goes, and what is generated for them.
    assert "examples/user_code" in reply
    assert "tools/" in reply
    # The workflow step is shown, brace-heavy references intact.
    assert f"tool: {DEMO_TOOL}" in reply
    # The introspector's reading of the exemplar.
    assert "gdd_file" in reply


async def test_explain_onboarding_format_can_include_the_source(ctx, no_writes):
    without = await explain_onboarding_format(ctx)
    with_source = await explain_onboarding_format(ctx, include_source=True)

    assert "```python" not in without
    assert "def compute_growing_degree_days" in with_source
    assert len(with_source) > len(without)


async def test_explain_onboarding_format_leaves_the_repository_untouched(
    ctx, no_writes
):
    watched = [
        REPO_ROOT / "extensions" / "registry.yaml",
        REPO_ROOT / "tools" / "tool_catalog.yaml",
    ]
    before = {path: path.read_bytes() for path in watched if path.exists()}

    await explain_onboarding_format(ctx, include_source=True)

    for path, content in before.items():
        assert path.read_bytes() == content, f"{path} was modified by a read-only tool"


async def test_the_advisor_example_shows_the_format_and_offers_the_live_demo(
    planner_ctx, no_writes
):
    reply = await show_onboarding_example(planner_ctx)

    assert "examples/user_code" in reply
    assert f"tool: {DEMO_TOOL}" in reply
    # The live demo is a write, so it is handed to the onboarding specialist.
    assert "delegate_to_onboarding" in reply
    assert "def compute_growing_degree_days" in reply


async def test_the_advisor_example_can_omit_the_source(planner_ctx, no_writes):
    reply = await show_onboarding_example(planner_ctx, include_source=False)

    assert "def compute_growing_degree_days" not in reply
    assert "delegate_to_onboarding" in reply


async def test_the_advisor_example_reports_a_missing_exemplar_conversationally(
    planner_ctx, monkeypatch, no_writes
):
    import agents.planner.onboarding_advisor as advisor_mod

    def _boom():
        raise DemoError("exemplar not found")

    monkeypatch.setattr(advisor_mod, "demo_draft", _boom)

    reply = await show_onboarding_example(planner_ctx)

    assert "Could not read the bundled onboarding example" in reply


# ---------------------------------------------------------------------------
# Wiring: the prompts may only name tools that exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "explain_onboarding_format",
        "demo_onboarding_walkthrough",
        "demo_status",
        "end_onboarding_demo",
    ],
)
def test_every_demo_tool_named_in_the_prompt_is_registered(name):
    assert name in ONBOARDING_INSTRUCTIONS, (
        f"'{name}' is no longer mentioned in the onboarding instructions; the "
        "model will not know the demonstration exists."
    )
    assert name in {tool.__name__ for tool in ONBOARDING_TOOLS}, (
        f"the prompt tells the model to call '{name}' but it is not in "
        "ONBOARDING_TOOLS, so the call would fail."
    )


def test_the_advisor_example_is_offered_to_the_planner():
    assert show_onboarding_example in ONBOARDING_ADVISOR_TOOLS
