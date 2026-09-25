"""Subagents generated from ``extensions/registry.yaml``.

Builtin categories each have a hand-written module in this package
(``data_discovery.py``, ``extraction.py``, ...).  User-onboarded *categories*
cannot, since they do not exist until the user creates them.  This module builds
one subagent per registry-declared category at import time using the same
:func:`agents.domain._factory.make_domain_subagent` used by the builtins, so an
extension subagent is indistinguishable from a builtin one at the planner level.

``EXTENSION_PLANNER_TOOLS`` is spliced into ``ONESHOT_PLANNER_TOOLS`` by
:mod:`agents.planner.planner_tools`.  When the registry has no custom subagents
(the default) this list is empty and nothing changes.

Note: this is import-time work, so a newly registered subagent only becomes
visible to the planner after a restart -- the same restart the generated tool
catalog already requires because ``common.workflow_validation`` caches it for the
lifetime of the process.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from pydantic_ai import Agent

from agents.domain._factory import make_domain_subagent
from common.tool_categories import extension_category_specs
from common.workflow import STEP_CLASSES

#: category name -> the generated Agent, for introspection and tests
EXTENSION_AGENTS: Dict[str, Agent] = {}

#: category name -> the planner-facing tool function
EXTENSION_CALLS: Dict[str, Callable] = {}

#: flat list to splice into the planner's tool list
EXTENSION_PLANNER_TOOLS: List[Callable] = []


def _build() -> None:
    for name, spec in extension_category_specs().items():
        step_class = STEP_CLASSES.get(name)
        if step_class is None:
            # Should not happen: common.workflow builds a Step class for every
            # spec returned by category_specs(). Skip rather than break imports.
            continue
        agent, call = make_domain_subagent(
            step_class,
            spec.display_name,
            spec.result_label,
            spec.call_tool_name,
            spec.description,
        )
        EXTENSION_AGENTS[name] = agent
        EXTENSION_CALLS[name] = call
        EXTENSION_PLANNER_TOOLS.append(call)


_build()


def extension_tool_summary() -> str:
    """One line per generated subagent, for logging and agent replies."""

    if not EXTENSION_CALLS:
        return "No user-onboarded subagents are active."
    specs = extension_category_specs()
    lines = ["Active user-onboarded subagents:"]
    for name, call in EXTENSION_CALLS.items():
        spec = specs[name]
        tools = ", ".join(spec.tools) or "(no tools registered yet)"
        lines.append(f"  - {call.__name__} -> {spec.step_class}: {tools}")
    return "\n".join(lines)
