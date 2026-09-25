"""Guard tests for the *committed* ``extensions/registry.yaml``.

The demo guards in ``test_demo.py`` check that one specific tool name is absent.
That is not enough. An interactive or aborted onboarding session writes to the
real registry, and anything it leaves behind is loaded at import time by
``common.tool_categories.category_specs()`` and turned into a live planner
category and delegation tool. A junk entry therefore ships as a user-visible
feature while every other test stays green -- which is exactly what happened
with a leftover subagent named ``a`` (``AStep`` / ``call_a``).

So these tests pin the whole file: the exact set of capabilities we intend to
ship, the internal consistency of each one, and the absence of any extension
subagent. Adding a real fifth-and-onwards exemplar means updating
``EXPECTED_CAPABILITIES`` deliberately, in the same commit as the adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_FILE = REPO_ROOT / "extensions" / "registry.yaml"

# The capabilities the repository intentionally ships, mapped to their subagent.
# Keep this in sync with docs/onboarding_design.md section 4 and AGENTS.md.
EXPECTED_CAPABILITIES: dict[str, str] = {
    "compute_standardized_anomaly_index": "diagnostics",
    "compute_snow_season_metrics": "diagnostics",
    "compute_gridded_trend": "diagnostics",
    "plot_month_year_heatmap": "visualization",
    "compute_flow_exceedance_thresholds": "diagnostics",
}


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load(REGISTRY_FILE.read_text()) or {}


@pytest.fixture(scope="module")
def capabilities(registry) -> list[dict]:
    return list(registry.get("capabilities") or [])


def test_the_committed_registry_holds_exactly_the_expected_capabilities(capabilities):
    """An unexpected name means a stray session wrote to the real registry."""

    found = sorted(entry.get("tool_name") for entry in capabilities)

    assert found == sorted(EXPECTED_CAPABILITIES), (
        "extensions/registry.yaml does not match the intended exemplar set.\n"
        f"  expected: {sorted(EXPECTED_CAPABILITIES)}\n"
        f"  found:    {found}\n"
        "If you deliberately onboarded a new permanent exemplar, add it to "
        "EXPECTED_CAPABILITIES here, to docs/onboarding_design.md section 4 and "
        "to AGENTS.md. Otherwise run `python -m onboarding.cli remove <tool>`."
    )


def test_the_committed_registry_declares_no_extension_subagents(registry):
    """A junk subagent becomes a real planner category and delegation tool.

    ``category_specs()`` merges registry subagents with the builtins at import
    time, so an accidental entry silently adds a Step class and a ``call_*``
    tool to every chat session.
    """

    subagents = registry.get("subagents") or []
    names = [entry.get("name") for entry in subagents]

    assert names == [], (
        f"extensions/registry.yaml declares extension subagents {names}. Each "
        "one adds a planner category and a call_* tool to every session. If "
        "these are leftovers from a test or interactive run, delete them and "
        "regenerate the catalog."
    )


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_CAPABILITIES))
def test_each_committed_capability_routes_where_the_docs_claim(tool_name, capabilities):
    """The registry is the source of truth the docs must agree with."""

    entry = next(e for e in capabilities if e.get("tool_name") == tool_name)

    assert entry.get("subagent") == EXPECTED_CAPABILITIES[tool_name]


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_CAPABILITIES))
def test_each_committed_capability_has_its_adapter_and_source_on_disk(
    tool_name, capabilities
):
    """A registry entry pointing at a deleted file breaks planner construction."""

    entry = next(e for e in capabilities if e.get("tool_name") == tool_name)

    adapter = REPO_ROOT / entry["adapter_path"]
    assert adapter.is_file(), f"missing generated adapter for {tool_name}: {adapter}"

    source = REPO_ROOT / Path(*entry["source_module"].split(".")).with_suffix(".py")
    assert source.is_file(), f"missing user code for {tool_name}: {source}"


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_CAPABILITIES))
def test_each_committed_capability_is_in_the_generated_catalog(tool_name):
    """The catalog is what the planner actually sees; drift makes tools unusable."""

    catalog_file = REPO_ROOT / "tools" / "tool_catalog.yaml"
    catalog = yaml.safe_load(catalog_file.read_text()) or {}
    names = {tool.get("name") for tool in (catalog.get("tools") or [])}

    assert tool_name in names, (
        f"'{tool_name}' is registered but absent from tools/tool_catalog.yaml; "
        "run `python tools/generate_catalog.py --overwrite`."
    )
