"""The planner's onboarding advisor must inform, never write.

These tests are the enforcement mechanism for the design rule stated in
``agents/planner/onboarding_advisor.py``: the *advisor* tools can describe
onboarding and preview a user function, but they never mutate the repo. Writes
happen only inside the onboarding agent, which the planner reaches through the
separate ``delegate_to_onboarding`` tool (see
``tests/agents/test_onboarding_bridge.py``). Every write entry point is patched
with a fail-fast stub here, so if someone later reaches for one from an advisor
tool the test breaks rather than silently shipping a "preview" that edits
``tools/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agents.planner.onboarding_advisor import (
    ONBOARDING_ADVISOR_TOOLS,
    explain_onboarding,
    list_onboarded_capabilities,
    preview_user_code_as_tool,
)
from agents.planner.planner_tools import ONESHOT_PLANNER_TOOLS, PLANNER_TOOLS
from common import WorkflowState
from common.workflow import Settings, Workflow


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = "examples/user_code/snow_metrics.py"


@dataclass
class _Ctx:
    """Minimal stand-in for RunContext: the advisor tools only need ``deps``."""

    deps: WorkflowState


@pytest.fixture
def ctx(tmp_path) -> _Ctx:
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
    """Make every onboarding write path explode if an advisor tool calls it."""

    def _forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"advisor tool called write path {name!r}")

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


async def test_explain_onboarding_describes_requirements_and_handoff(ctx, no_writes):
    reply = await explain_onboarding(ctx)

    assert "onboarding.cli register" in reply
    # In-session delegation is the primary path; no mode switch is advertised.
    assert "delegate_to_onboarding" in reply
    assert "WEB_AGENT_MODE=onboarding" not in reply
    # Existing categories are listed so the user can pick one.
    assert "diagnostics" in reply


async def test_explain_onboarding_still_warns_that_use_needs_a_restart(ctx, no_writes):
    """Registration is in-session, but the plannable tool list is fixed at startup."""
    reply = await explain_onboarding(ctx)

    assert "restart" in reply.lower()


async def test_preview_reports_the_tool_it_would_create(ctx, no_writes):
    reply = await preview_user_code_as_tool(
        ctx, EXAMPLE, function_name="compute_snow_season_metrics"
    )

    assert "compute_snow_season_metrics" in reply
    # A copy-pasteable step, so the user sees the payoff before committing.
    assert "tool: compute_snow_season_metrics" in reply
    assert "nothing has been written" in reply


async def test_preview_leaves_the_repository_untouched(ctx, no_writes):
    before = {
        path: path.read_bytes()
        for path in [
            REPO_ROOT / "extensions" / "registry.yaml",
            REPO_ROOT / "tools" / "tool_catalog.yaml",
        ]
    }

    await preview_user_code_as_tool(ctx, EXAMPLE)

    for path, content in before.items():
        assert path.read_bytes() == content, f"{path.name} was modified by a preview"


async def test_preview_does_not_import_the_scanned_module(ctx, tmp_path, no_writes):
    """Scanning is AST-only, so import-time side effects must not fire."""
    marker = tmp_path / "imported.txt"
    source = tmp_path / "explosive.py"
    source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('boom')\n"
        "\n"
        "def do_thing(input_file: str) -> str:\n"
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        input_file: Path to a CSV.\n"
        '    """\n'
        "    return input_file\n"
    )

    reply = await preview_user_code_as_tool(ctx, str(source))

    assert "do_thing" in reply
    assert not marker.exists(), "the scanned module was imported and ran its side effect"


async def test_preview_reports_unreadable_sources_conversationally(ctx, no_writes):
    reply = await preview_user_code_as_tool(ctx, "examples/user_code/not_here.py")

    assert "Could not read" in reply
    assert ".py file" in reply


async def test_list_onboarded_capabilities_warns_about_the_restart(ctx, no_writes):
    reply = await list_onboarded_capabilities(ctx)

    assert "restart" in reply.lower()


def test_advisor_tools_are_available_to_the_interactive_planner():
    assert set(ONBOARDING_ADVISOR_TOOLS).issubset(set(PLANNER_TOOLS))


def test_advisor_tools_stay_out_of_the_benchmarked_oneshot_stack():
    """ONESHOT_PLANNER_TOOLS is what the benchmark and MCP server score."""
    assert not set(ONBOARDING_ADVISOR_TOOLS) & set(ONESHOT_PLANNER_TOOLS)
