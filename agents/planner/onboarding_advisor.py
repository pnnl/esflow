"""Read-only onboarding advice for the planner.

A user who discovers mid-analysis that they need their own Python function would
otherwise hit a dead end, because the planner has no idea onboarding exists.
These tools let it explain and *preview* onboarding without moving repo-write
authority into the planner. Committing is a separate, explicit step: the planner
delegates a turn to the onboarding agent via
``agents.planner.onboarding_bridge.delegate_to_onboarding``.

Every function here is strictly read-only:

* ``preview_user_code_as_tool`` parses the user's file with
  :func:`onboarding.introspect.scan_source`, which uses ``ast`` and never
  imports (so it is safe on modules with import-time side effects), then
  formats the drafts. It does **not** persist drafts to ``ctx.deps`` — planner
  tools are typed ``RunContext[WorkflowState]``, which has nowhere to put them,
  and keeping them stateless is what makes "preview" honest.
* Nothing in this module calls ``register_capability``, ``register_subagent``,
  ``write_adapter`` or ``regenerate_catalog``. ``tests/agents/test_onboarding_advisor.py``
  asserts that by patching those names and failing if they are touched.

Registration itself — writing adapters under ``tools/``, editing
``extensions/registry.yaml`` and regenerating the catalog — belongs to the
onboarding agent (or ``onboarding.cli``) and never happens here. Note that a tool
registered mid-session is permanent but not yet *plannable*: ``common.workflow``
builds its ``Step`` classes, whose ``tool`` field is a ``Literal`` allow-list, at
import time, and the planner's output schema is derived from them at
construction. That needs a restart. See ``AGENTS.md`` for the full contract.
"""

from __future__ import annotations

from typing import Optional

from pydantic_ai import RunContext

from common import WorkflowState
from common.tool_categories import category_specs
from onboarding.demo import (
    DemoError,
    demo_draft,
    demo_overview,
    demo_source_text,
    demo_workflow_snippet,
    where_code_goes,
)
from onboarding.introspect import scan_source
from onboarding.registry import registry_summary
from onboarding.scaffold import render_workflow_snippet, validate_draft


#: Shown when the user asks how to add their own code. Kept in one place so the
#: prose the planner repeats stays consistent with the CLI and the docs.
_HANDOFF = (
    "To make it permanent, say the word and I will hand this turn to the "
    "onboarding specialist (delegate_to_onboarding) — no mode switch, no "
    "restart, same conversation. It walks you through what it inferred and "
    "writes the tool once you approve.\n"
    "Outside the chat, the equivalent is:\n"
    "  python -m onboarding.cli register <path to your .py> "
    "--function <name> --subagent <category>\n"
    "Either way the tool is written into the catalog and stays available in all "
    "future sessions. One caveat: this running process fixed its plannable tool "
    "list at startup, so a newly registered tool can only be used in a plan "
    "after the app restarts."
)


async def explain_onboarding(ctx: RunContext[WorkflowState]) -> str:
    """Explain how the user can add their own Python function as a new tool.

    Call this whenever the user asks for an analysis no catalogued tool covers,
    or mentions that they already have code (a script, notebook cell or
    function) that does it. Returns the requirements their function must meet
    and the exact commands to register it.
    """
    categories = ", ".join(sorted(category_specs()))
    return (
        "Your own Python can become a first-class ESMFlow tool that the planner "
        "reuses in every future session.\n\n"
        "What the introspector needs from the function:\n"
        "  - module-level, public (no leading underscore) and not async\n"
        "  - type hints on the arguments, so params get the right types\n"
        "  - a docstring; its summary becomes the tool description and its "
        "Args: entries become param descriptions\n"
        "  - return a path/DataFrame/Dataset/figure for a single output, or a "
        "dict for several named outputs\n\n"
        f"Existing planner categories a tool can join: {categories}\n"
        "A brand-new category (its own subagent) can also be created.\n\n"
        "I can preview the result right now with preview_user_code_as_tool — "
        "that only reads your file.\n\n"
        f"{_HANDOFF}"
    )


async def preview_user_code_as_tool(
    ctx: RunContext[WorkflowState],
    source: str,
    function_name: Optional[str] = None,
) -> str:
    """Preview how a user Python file would be registered as ESMFlow tools.

    Read-only: the file is parsed, never imported, and nothing is written to the
    repository. Use this to show the user the tool name, params, outputs and a
    runnable workflow snippet *before* they commit in onboarding mode.

    Args:
        source: Path to a .py file, e.g. 'examples/user_code/snow_metrics.py'.
        function_name: Preview only this function instead of all public ones.
    """
    try:
        result = scan_source(source, function_name=function_name)
    except (FileNotFoundError, SyntaxError, ValueError) as exc:
        return (
            f"Could not read {source!r}: {exc}\n\n"
            "Give me a path to a .py file inside the repository."
        )

    if not result.drafts:
        return (
            f"{result.summary()}\n\nNothing onboardable was found. Only public, "
            "non-async, module-level functions are eligible."
        )

    blocks = []
    for draft in result.drafts:
        block = [draft.summary()]
        problems = validate_draft(draft)
        if problems:
            block.append(
                "Needs your input before registering:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
        block.append(
            "Workflow step it would enable:\n" + render_workflow_snippet(draft)
        )
        blocks.append("\n\n".join(block))

    return (
        f"{result.summary()}\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\nThis was a preview only — nothing has been written.\n\n"
        + _HANDOFF
    )


async def show_onboarding_example(
    ctx: RunContext[WorkflowState], include_source: bool = True
) -> str:
    """Show the worked onboarding example: the required code format and where it goes.

    Call this when the user asks what their code has to look like, where to put
    it, or for an example to copy. Returns the annotated exemplar, how the
    introspector reads it, a directory map (what the user writes versus what is
    generated) and the workflow step it produces.

    Read-only: the exemplar is parsed, never imported, and nothing is written.
    The *live* demonstration -- which registers the exemplar, runs it on
    synthesized data and then un-registers it -- is a repo write, so it belongs
    to the onboarding specialist; offer to hand the turn over for that.

    Args:
        include_source: Include the exemplar's full annotated source. Set False
            for a shorter answer when the user only wants the rules.
    """
    try:
        draft = demo_draft()
    except DemoError as exc:
        return f"Could not read the bundled onboarding example: {exc}"

    blocks = [
        "Here is a worked example of a correctly formatted tool.",
        where_code_goes(),
        "How the introspector reads that function:\n" + draft.summary(),
        "The workflow step it produces:\n" + demo_workflow_snippet(),
    ]
    if include_source:
        try:
            blocks.append(
                "The exemplar source (examples/user_code/demo_growing_degree_days.py):"
                "\n\n```python\n" + demo_source_text() + "\n```"
            )
        except DemoError as exc:
            blocks.append(f"(Could not read the exemplar source: {exc})")
    blocks.append(
        "I can also have the onboarding specialist run this live end to end:\n"
        + demo_overview()
        + "\nThat is a repo write, so say the word and I will hand the turn over "
        "(delegate_to_onboarding) and ask it for the onboarding demonstration. "
        "It removes the demo capability again when it finishes, so it leaves "
        "your capability set exactly as it was."
    )
    return "\n\n".join(blocks)


async def list_onboarded_capabilities(ctx: RunContext[WorkflowState]) -> str:
    """List capabilities and subagents that were already onboarded from user code.

    Useful when the user asks whether their tool is available yet, or wants to
    know what previous onboarding sessions added.
    """
    summary = registry_summary()
    return (
        f"{summary}\n\nOnboarded tools are in the catalog and usable in plans "
        "like any built-in tool. A tool registered during this session shows up "
        "here immediately but cannot be put in a workflow until the app "
        "restarts, because the plannable tool list was fixed at startup."
    )


#: Read-only advisor tools. Spliced into the interactive planner stacks only —
#: deliberately *not* into ONESHOT_PLANNER_TOOLS, which the benchmark and the
#: MCP server run, so scored behaviour is unchanged.
ONBOARDING_ADVISOR_TOOLS = [
    explain_onboarding,
    show_onboarding_example,
    preview_user_code_as_tool,
    list_onboarded_capabilities,
]
