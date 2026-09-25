"""Conversational agent that onboards user code as ESMFlow capabilities.

Design notes
------------
* **Nothing is written without approval.** ``scan_user_code`` only reads source
  (via AST -- the user's module is never imported), and produces drafts held in
  :class:`agents.onboarding.state.OnboardingState`. Only ``register_draft``
  touches the repo.
* **Every guess is surfaced.** The introspector records a note for each inferred
  type, output name and subagent routing decision; the prompt requires the agent
  to show those notes and get confirmation before registering.
* **Verification is not optional.** After registration the agent runs
  ``verify_capability``, which re-checks the adapter, catalog and planner
  allow-list in a fresh subprocess, and then offers a runnable workflow snippet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext

from agents.onboarding.state import OnboardingState
from common.config import load_prompt, model
from common.tool_categories import BUILTIN_CATEGORY_NAMES, category_specs
from onboarding.demo import (
    DemoError,
    demo_draft,
    demo_overview,
    demo_source_text,
    demo_status as _demo_status,
    end_demo,
    run_demo,
    start_demo,
    where_code_goes,
)
from onboarding.introspect import category_for_subagent, scan_source
from onboarding.models import OutputDraft, ParamDraft
from onboarding.registry import (
    RegistryError,
    register_capability,
    register_subagent,
    registry_summary,
    remove_capability,
    subagent_draft_for,
)
from onboarding.scaffold import (
    ScaffoldError,
    render_adapter,
    render_workflow_snippet,
    validate_draft,
)
from onboarding.verify import verify_capability as _verify_capability

ONBOARDING_INSTRUCTIONS = """
You are the ESMFlow **onboarding agent**. Your job is to turn a scientist's own
Python code into first-class ESMFlow capabilities (tools) and, where warranted,
new planner subagents (categories) -- without them having to learn the internal
tool contract.

Work through five phases. Do not skip ahead.

**0. Orient (optional but offer it).** If the user is new to onboarding, unsure
what format their code needs to be in, or asks for an example, run the guided
demonstration instead of describing it abstractly:
  - `explain_onboarding_format` -- the format rules, the annotated exemplar and
    a directory map of where user code goes versus what is generated. No writes.
  - then, with the user's agreement, `demo_onboarding_walkthrough` -- this
    registers the bundled exemplar, runs it on synthesized sample data, and then
    **un-registers it again**, so it is a demonstration and not a lasting change.
Say clearly that the demo tool is removed at the end, which is why it can be
shown to anyone at any time. Use `demo_status` if you need to know whether a
previous demonstration left anything behind, and `end_onboarding_demo` to clean
up. Never present the demo capability as one of the user's own capabilities.

**1. Discover.** Ask for the path to the Python file (or module) they want to
onboard. Call `scan_user_code`. Report what was found, including skipped
functions and why.

**2. Confirm.** For each candidate, show `inspect_draft` output and walk the user
through the *notes* section. The notes are guesses, and guesses are exactly what
breaks at runtime. Specifically confirm:
  - the tool name (planner-visible, must be a valid Python identifier);
  - the description (the planner selects tools using only this text, so it must
    say what the tool does, what it needs and what it produces);
  - each input's type -- `path` for filenames, `int`/`float`/`bool`/`str`, or
    `list[str]`/`list[int]`;
  - each output's name and type. Output types `csv`, `png` and `netcdf` become
    real files on disk; `int`, `float`, `str` and `dict` are returned inline.
Apply corrections with `update_draft`. If a required argument of the user's
function is missing from the inputs, it must be added or the tool cannot run.

**3. Route.** Every capability belongs to one planner subagent. Prefer an
existing category:
  - `data_discovery` -- finding/fetching/loading datasets and metadata
  - `extraction` -- pulling timeseries, fields or basin means out of datasets
  - `diagnostics` -- metrics, statistics, climatologies, skill scores, indices
  - `water_cycle` -- basin-scale water budget closure and synthesis
  - `visualization` -- figures
Only propose a *new* subagent (via `create_subagent`) when the work is a
genuinely new kind of scientific step that the existing five do not describe,
and say plainly that a new subagent requires an app restart to appear in the
planner. Explain that registering into an existing category needs no restart of
the planner's tool list, but the regenerated catalog does require a restart to
be seen by a running app.

**4. Register.** Only after explicit approval, call `register_draft`. This writes
`tools/<category>/<tool_name>.py`, records the capability in
`extensions/registry.yaml` and regenerates `tools/tool_catalog.yaml`. Report the
adapter path. If registration fails, relay the error verbatim and fix the draft
rather than retrying blindly.

**5. Verify and demonstrate.** Call `verify_capability` and show the report. If
anything fails, diagnose it: a missing catalog entry means the generator did not
pick the file up; a rejected tool name means the allow-list was not widened.
Finish with `suggest_workflow_snippet` so the user has a concrete workflow step
they can run, and remind them to restart a running app so the new catalog is
picked up.

Rules of engagement:
- Never invent a description or a type to move faster; ask.
- The demonstration is not the user's work. After a demo, if they want to
  onboard their own function, start again at phase 1 with their file.
- Never claim something is registered or verified unless the tool said so.
- Keep the user's science code untouched. If it cannot be adapted as-is, explain
  the smallest change to *their* function that would make it onboardable
  (usually: return a dict of named results, or add a return type annotation).
- `output_dir` is provided by the framework; it is never a declared input.
""".strip()


async def scan_user_code(
    ctx: RunContext[OnboardingState], source: str, function_name: Optional[str] = None
) -> str:
    """Introspect a user Python file/module and create capability drafts.

    Args:
        source: Path to a .py file (e.g. 'examples/user_code/snow_metrics.py')
            or a dotted module path.
        function_name: Optional single function to scan instead of all public ones.

    The user's code is parsed, never imported, so scanning is side-effect free.
    """
    try:
        result = scan_source(source, function_name=function_name)
    except (FileNotFoundError, SyntaxError, ValueError) as exc:
        return f"Could not scan {source!r}: {exc}"

    ctx.deps.add_drafts(result.drafts)
    ctx.deps.last_source = source
    if not result.drafts:
        return (
            f"{result.summary()}\n\nNothing onboardable was found. Public, "
            "non-async, module-level functions are eligible."
        )
    details = "\n\n".join(draft.summary() for draft in result.drafts)
    return f"{result.summary()}\n\n{details}"


async def list_drafts(ctx: RunContext[OnboardingState]) -> str:
    """List the capability drafts currently under review and their status."""
    return ctx.deps.context_summary()


async def inspect_draft(ctx: RunContext[OnboardingState], tool_name: str) -> str:
    """Show one draft in full, including the introspector's open notes."""
    try:
        draft = ctx.deps.get(tool_name)
    except KeyError as exc:
        return str(exc)
    problems = validate_draft(draft)
    text = draft.summary()
    if problems:
        text += "\n\nblocking problems:\n" + "\n".join(f"  - {p}" for p in problems)
    else:
        text += "\n\nThis draft is complete and can be registered."
    return text


async def update_draft(
    ctx: RunContext[OnboardingState],
    tool_name: str,
    new_tool_name: Optional[str] = None,
    description: Optional[str] = None,
    subagent: Optional[str] = None,
    category: Optional[str] = None,
    inputs: Optional[List[Dict[str, Any]]] = None,
    outputs: Optional[List[Dict[str, Any]]] = None,
    clear_notes: bool = False,
) -> str:
    """Apply user-confirmed corrections to a draft.

    Args:
        tool_name: The draft to modify.
        new_tool_name: Rename the tool (must be a valid Python identifier).
        description: Replacement planner-facing description.
        subagent: One of data_discovery, extraction, diagnostics, water_cycle,
            visualization, or a registered custom subagent. Setting this also
            updates the tools/<category>/ directory unless 'category' is given.
        category: Explicit tools/<category>/ directory override.
        inputs: Full replacement list of inputs, each a dict with keys
            name, type, required, default, description.
        outputs: Full replacement list of outputs, each a dict with keys
            name, type, description.
        clear_notes: Drop the introspector's notes once they have been resolved.
    """
    try:
        draft = ctx.deps.get(tool_name)
    except KeyError as exc:
        return str(exc)

    changes: List[str] = []
    if description is not None:
        draft.description = description
        changes.append("description")
    if subagent is not None:
        known = set(category_specs()) | set(BUILTIN_CATEGORY_NAMES)
        if subagent not in known:
            return (
                f"Unknown subagent {subagent!r}. Known: {', '.join(sorted(known))}. "
                "Use create_subagent first if this is a new category."
            )
        draft.subagent = subagent
        draft.category = category or category_for_subagent(subagent)
        changes.append(f"subagent -> {subagent} (category {draft.category})")
    elif category is not None:
        draft.category = category
        changes.append(f"category -> {category}")
    if inputs is not None:
        try:
            draft.inputs = [ParamDraft(**item) for item in inputs]
        except Exception as exc:
            return f"Could not apply inputs: {exc}"
        changes.append(f"{len(draft.inputs)} input(s)")
    if outputs is not None:
        try:
            draft.outputs = [OutputDraft(**item) for item in outputs]
        except Exception as exc:
            return f"Could not apply outputs: {exc}"
        changes.append(f"{len(draft.outputs)} output(s)")
    if clear_notes:
        draft.notes = []
        changes.append("cleared notes")
    if new_tool_name is not None and new_tool_name != tool_name:
        if not new_tool_name.isidentifier():
            return f"{new_tool_name!r} is not a valid Python identifier."
        ctx.deps.rename(tool_name, new_tool_name)
        changes.append(f"renamed to {new_tool_name}")
        draft = ctx.deps.get(new_tool_name)

    if not changes:
        return "No changes requested."
    return f"Updated ({'; '.join(changes)}).\n\n{draft.summary()}"


async def preview_adapter(ctx: RunContext[OnboardingState], tool_name: str) -> str:
    """Show the adapter source that registration would write, without writing it."""
    try:
        draft = ctx.deps.get(tool_name)
    except KeyError as exc:
        return str(exc)
    try:
        return f"```python\n{render_adapter(draft)}\n```"
    except ScaffoldError as exc:
        return str(exc)


async def create_subagent(
    ctx: RunContext[OnboardingState],
    name: str,
    description: str = "",
    display_name: str = "",
) -> str:
    """Register a brand new planner category (subagent).

    Only do this when no builtin category fits. The new subagent becomes visible
    to the planner after the app is restarted.
    """
    try:
        draft = register_subagent(
            subagent_draft_for(name, description=description, display_name=display_name)
        )
    except RegistryError as exc:
        return f"Could not create subagent: {exc}"
    return (
        f"Registered subagent '{draft.name}' (step class {draft.step_class}, planner "
        f"tool {draft.call_tool_name}). Capabilities can now be routed to it. "
        "Restart the app for the planner to gain the new delegation tool."
    )


async def register_draft(
    ctx: RunContext[OnboardingState], tool_name: str, overwrite: bool = False
) -> str:
    """Write the adapter, record the capability and regenerate the tool catalog.

    Call this only after the user has explicitly approved the draft.
    """
    try:
        draft = ctx.deps.get(tool_name)
    except KeyError as exc:
        return str(exc)
    try:
        entry, catalog_output = register_capability(draft, overwrite=overwrite)
    except RegistryError as exc:
        return f"Registration failed: {exc}"
    ctx.deps.registered.append(entry.tool_name)
    return (
        f"Registered '{entry.tool_name}'.\n"
        f"  adapter:  {entry.adapter_path}\n"
        f"  subagent: {entry.subagent}\n"
        f"  source:   {entry.source_module}.{entry.source_function}\n"
        f"{catalog_output}".rstrip()
    )


async def verify_capability(ctx: RunContext[OnboardingState], tool_name: str) -> str:
    """Re-check a registered capability end to end in a fresh subprocess."""
    return _verify_capability(tool_name).summary()


async def list_capabilities(ctx: RunContext[OnboardingState]) -> str:
    """Show every onboarded capability and subagent in the extension registry."""
    return registry_summary()


async def remove_registered_capability(
    ctx: RunContext[OnboardingState], tool_name: str, keep_adapter: bool = False
) -> str:
    """Un-register a capability and delete its generated adapter.

    Only onboarding-generated adapters can be deleted; hand-written builtin tools
    are protected.
    """
    try:
        entry = remove_capability(tool_name, delete_adapter=not keep_adapter)
    except RegistryError as exc:
        return f"Could not remove '{tool_name}': {exc}"
    kept = " (adapter left in place)" if keep_adapter else ""
    return f"Removed '{entry.tool_name}' from the registry{kept}."


async def suggest_workflow_snippet(
    ctx: RunContext[OnboardingState], tool_name: str
) -> str:
    """Produce a copy-pasteable workflow step exercising the new capability."""
    try:
        draft = ctx.deps.get(tool_name)
    except KeyError as exc:
        return str(exc)
    return (
        "Add this step to a workflow YAML (fill in the placeholder params, or "
        "reference an upstream step with ${step_id.outputs.key}):\n\n"
        f"```yaml\n{render_workflow_snippet(draft)}\n```"
    )


async def explain_onboarding_format(
    ctx: RunContext[OnboardingState], include_source: bool = False
) -> str:
    """Explain the required code format and where user code lives. Read-only.

    Shows the format rules, the introspector's reading of the bundled exemplar
    and a directory/workflow map. Nothing is written and no module is imported.
    Set ``include_source`` to also return the annotated exemplar source.
    """
    try:
        draft = demo_draft()
    except DemoError as exc:
        return f"Could not read the bundled exemplar: {exc}"
    parts = [demo_overview(), where_code_goes(), "How the exemplar is read:", draft.summary()]
    if include_source:
        try:
            parts.append("The exemplar source:\n\n```python\n" + demo_source_text() + "\n```")
        except DemoError as exc:
            parts.append(f"(Could not read the exemplar source: {exc})")
    return "\n\n".join(parts)


async def demo_onboarding_walkthrough(ctx: RunContext[OnboardingState]) -> str:
    """Register the bundled exemplar, run it on sample data, then un-register it.

    This is a self-cleaning demonstration: the capability it creates is always
    removed again before this tool returns, so it can be shown at any time and
    never becomes part of the user's own capability set.
    """
    try:
        started = start_demo()
    except DemoError as exc:
        return f"Could not start the demonstration: {exc}"
    stages = [started.summary()]
    try:
        stages.append(run_demo().summary())
    except DemoError as exc:
        stages.append(f"The run stage failed: {exc}")
    finally:
        # Non-negotiable: the demo must not outlive this call.
        try:
            stages.append(end_demo().summary())
        except DemoError as exc:
            stages.append(
                f"WARNING: could not fully clean up the demonstration: {exc}. "
                "Run `python -m onboarding.cli demo end` to finish removing it."
            )
    return "\n\n".join(stages)


async def demo_status(ctx: RunContext[OnboardingState]) -> str:
    """Report whether a previous demonstration left anything registered on disk."""
    try:
        return _demo_status().summary()
    except DemoError as exc:
        return f"Could not determine the demonstration status: {exc}"


async def end_onboarding_demo(ctx: RunContext[OnboardingState]) -> str:
    """Remove every trace of the demonstration capability. Safe to call twice."""
    try:
        return end_demo().summary()
    except DemoError as exc:
        return f"Could not clean up the demonstration: {exc}"


ONBOARDING_TOOLS = [
    explain_onboarding_format,
    demo_onboarding_walkthrough,
    demo_status,
    end_onboarding_demo,
    scan_user_code,
    list_drafts,
    inspect_draft,
    update_draft,
    preview_adapter,
    create_subagent,
    register_draft,
    verify_capability,
    list_capabilities,
    remove_registered_capability,
    suggest_workflow_snippet,
]


onboarding_agent: Agent[OnboardingState, str] = Agent(
    model,
    deps_type=OnboardingState,
    output_type=str,
    tools=ONBOARDING_TOOLS,
    instructions=load_prompt(ONBOARDING_INSTRUCTIONS),
)


@onboarding_agent.instructions
def _current_state(ctx: RunContext[OnboardingState]) -> str:
    """Keep the model aware of draft state without re-listing it by hand."""
    return ctx.deps.context_summary()
