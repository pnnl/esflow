"""Persist onboarded capabilities and subagents to ``extensions/registry.yaml``.

The registry is the single place that records what the user added.  It is read
by :mod:`common.tool_categories` (to widen the planner's ``Literal`` allow-lists
and to synthesize new subagents) and by :mod:`agents.domain.extensions` (to
build the actual pydantic-ai agents).

Registration is deliberately a three-step, all-or-nothing sequence:

1. write the adapter under ``tools/<category>/<tool_name>.py``
2. append a ``capabilities`` entry (and a ``subagents`` entry if new)
3. regenerate ``tools/tool_catalog.yaml``

If any step fails the earlier ones are rolled back so the repo never ends up in
a half-onboarded state.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from common.tool_categories import (
    BUILTIN_CATEGORY_NAMES,
    REGISTRY_PATH,
    _pascal_case,
    load_registry,
)
from onboarding.models import CapabilityDraft, RegistryEntry, SubagentDraft
from onboarding.scaffold import (
    ScaffoldError,
    adapter_path_for,
    relative_adapter_path,
    remove_adapter,
    validate_draft,
    write_adapter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "tools" / "tool_catalog.yaml"

REGISTRY_HEADER = """\
# ESMFlow extension registry
#
# Written by the onboarding agent (onboarding/registry.py).  Editing by hand is
# fine, but after any change run:
#
#     python tools/generate_catalog.py --overwrite
#
# and restart the app: common.workflow_validation caches the catalog for the
# lifetime of the process.
"""


class RegistryError(RuntimeError):
    """Raised for invalid or conflicting registry operations."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registry_path(path: Optional[Path] = None) -> Path:
    return Path(path) if path else REGISTRY_PATH


def save_registry(registry: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Write ``registry`` back to disk, preserving the explanatory header."""

    target = registry_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(registry, sort_keys=False, default_flow_style=False)
    target.write_text(f"{REGISTRY_HEADER}\n{body}")
    return target


def list_capabilities(path: Optional[Path] = None) -> List[RegistryEntry]:
    """All registered capabilities as typed entries."""

    registry = load_registry(registry_path(path))
    entries: List[RegistryEntry] = []
    for raw in registry["capabilities"]:
        if isinstance(raw, dict) and raw.get("tool_name"):
            entries.append(RegistryEntry(**raw))
    return entries


def list_subagents(path: Optional[Path] = None) -> List[SubagentDraft]:
    """All user-onboarded subagents as typed drafts."""

    registry = load_registry(registry_path(path))
    drafts: List[SubagentDraft] = []
    for raw in registry["subagents"]:
        if isinstance(raw, dict) and raw.get("name"):
            drafts.append(SubagentDraft(**raw))
    return drafts


def find_capability(
    tool_name: str, path: Optional[Path] = None
) -> Optional[RegistryEntry]:
    for entry in list_capabilities(path):
        if entry.tool_name == tool_name:
            return entry
    return None


def subagent_draft_for(name: str, description: str = "", display_name: str = "") -> SubagentDraft:
    """Fill in the conventional defaults for a brand new subagent."""

    pascal = _pascal_case(name)
    shown = display_name or f"{pascal} (user-onboarded)"
    return SubagentDraft(
        name=name,
        display_name=shown,
        description=description or f"Delegate {shown} work to the {name} subagent.",
        step_class=f"{pascal}Step",
        result_label=name.replace("_", " "),
        call_tool_name=f"call_{name}",
    )


def register_subagent(
    draft: SubagentDraft, path: Optional[Path] = None
) -> SubagentDraft:
    """Add a new planner category. Idempotent; builtins are rejected."""

    if not draft.name or not draft.name.isidentifier():
        raise RegistryError(f"subagent name {draft.name!r} must be an identifier")
    if draft.name in BUILTIN_CATEGORY_NAMES:
        raise RegistryError(
            f"'{draft.name}' is a builtin category; register capabilities into it "
            "instead of redefining it"
        )
    filled = subagent_draft_for(draft.name, draft.description, draft.display_name)
    merged = filled.model_copy(
        update={k: v for k, v in draft.model_dump().items() if v}
    )
    target = registry_path(path)
    registry = load_registry(target)
    for existing in registry["subagents"]:
        if isinstance(existing, dict) and existing.get("name") == merged.name:
            return SubagentDraft(**existing)
    registry["subagents"].append(merged.model_dump())
    save_registry(registry, target)
    return merged


def regenerate_catalog() -> str:
    """Run ``tools/generate_catalog.py --overwrite`` in a subprocess.

    A subprocess is used on purpose: the generator imports every tool module and
    the current process may already hold a stale ``TOOL_REGISTRY`` and the
    cached catalog singleton.
    """

    proc = subprocess.run(
        [sys.executable, "tools/generate_catalog.py", "--overwrite"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RegistryError(
            "catalog regeneration failed:\n"
            f"{proc.stdout}\n{proc.stderr}".strip()
        )
    return proc.stdout.strip()


def register_capability(
    draft: CapabilityDraft,
    path: Optional[Path] = None,
    tools_root: Optional[Path] = None,
    overwrite: bool = False,
    regenerate: bool = True,
) -> Tuple[RegistryEntry, str]:
    """Write the adapter, record the capability and refresh the catalog.

    Returns ``(entry, catalog_output)``.  Rolls back the adapter file and the
    registry edit if catalog regeneration fails.
    """

    problems = validate_draft(draft)
    if problems:
        raise RegistryError(
            f"draft for {draft.tool_name!r} is incomplete:\n  - "
            + "\n  - ".join(problems)
        )

    target = registry_path(path)
    registry = load_registry(target)

    known_categories = {c.name for c in list_subagents(target)} | set(
        BUILTIN_CATEGORY_NAMES
    )
    if draft.subagent not in known_categories:
        raise RegistryError(
            f"unknown subagent {draft.subagent!r}; register the subagent first "
            f"(known: {', '.join(sorted(known_categories))})"
        )

    existing = find_capability(draft.tool_name, target)
    if existing and not overwrite:
        raise RegistryError(
            f"capability {draft.tool_name!r} is already registered "
            f"(adapter: {existing.adapter_path}); pass overwrite=True to replace it"
        )

    adapter_file = adapter_path_for(draft, tools_root)
    previous_source = adapter_file.read_text() if adapter_file.exists() else None
    previous_registry = (
        target.read_text() if target.exists() else None
    )

    try:
        written = write_adapter(draft, tools_root=tools_root, overwrite=True)
    except ScaffoldError as exc:  # pragma: no cover - surfaced to the agent
        raise RegistryError(str(exc)) from exc

    entry = RegistryEntry(
        tool_name=draft.tool_name,
        category=draft.category,
        subagent=draft.subagent,
        source_module=draft.source_module,
        source_function=draft.source_function,
        adapter_path=relative_adapter_path(written),
        onboarded_at=_now(),
    )

    registry["capabilities"] = [
        raw
        for raw in registry["capabilities"]
        if not (isinstance(raw, dict) and raw.get("tool_name") == draft.tool_name)
    ]
    registry["capabilities"].append(entry.model_dump())
    save_registry(registry, target)

    catalog_output = ""
    if regenerate:
        try:
            catalog_output = regenerate_catalog()
        except RegistryError:
            # Roll back so a failed registration leaves no trace.
            if previous_source is None:
                written.unlink(missing_ok=True)
            else:
                written.write_text(previous_source)
            if previous_registry is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(previous_registry)
            raise

    return entry, catalog_output


def remove_capability(
    tool_name: str,
    path: Optional[Path] = None,
    tools_root: Optional[Path] = None,
    delete_adapter: bool = True,
    regenerate: bool = True,
) -> RegistryEntry:
    """Un-register a capability and (by default) delete its generated adapter."""

    target = registry_path(path)
    entry = find_capability(tool_name, target)
    if entry is None:
        raise RegistryError(f"capability {tool_name!r} is not registered")

    registry = load_registry(target)
    registry["capabilities"] = [
        raw
        for raw in registry["capabilities"]
        if not (isinstance(raw, dict) and raw.get("tool_name") == tool_name)
    ]
    save_registry(registry, target)

    if delete_adapter:
        try:
            remove_adapter(tool_name, entry.category, tools_root=tools_root)
        except ScaffoldError as exc:
            raise RegistryError(str(exc)) from exc

    if regenerate:
        regenerate_catalog()

    return entry


def registry_summary(path: Optional[Path] = None) -> str:
    """Human-readable listing for agent replies and the CLI."""

    capabilities = list_capabilities(path)
    subagents = list_subagents(path)
    if not capabilities and not subagents:
        return "No onboarded capabilities or subagents yet."

    lines: List[str] = []
    if subagents:
        lines.append("User-onboarded subagents:")
        for sub in subagents:
            lines.append(f"  - {sub.name} ({sub.display_name}) -> {sub.step_class}")
    if capabilities:
        lines.append("Onboarded capabilities:")
        for entry in capabilities:
            lines.append(
                f"  - {entry.tool_name}  [{entry.subagent}/{entry.category}]  "
                f"<- {entry.source_module}.{entry.source_function}"
            )
    return "\n".join(lines)
