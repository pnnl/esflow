"""Tests for ``extensions/registry.yaml`` persistence and registration safety.

Every test points ``path=`` and ``tools_root=`` at ``tmp_path`` and passes
``regenerate=False``, so the real registry, the real ``tools/`` tree and the real
``tool_catalog.yaml`` are never touched. Catalog regeneration itself shells out to
``tools/generate_catalog.py`` and is exercised by the end-to-end flow rather than
here.

The behaviour worth pinning down is the all-or-nothing contract: a registration
that fails at the catalog step must leave *no* trace -- neither a stray adapter
file nor a registry entry pointing at a tool the catalog has never heard of.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from common.tool_categories import BUILTIN_CATEGORY_NAMES, category_specs
from onboarding.models import CapabilityDraft, OutputDraft, ParamDraft, SubagentDraft
from onboarding.registry import (
    RegistryError,
    find_capability,
    list_capabilities,
    list_subagents,
    register_capability,
    register_subagent,
    registry_summary,
    remove_capability,
    save_registry,
    subagent_draft_for,
)


@pytest.fixture
def registry_file(tmp_path) -> Path:
    """An empty-but-valid registry file inside tmp_path."""

    path = tmp_path / "registry.yaml"
    save_registry({"version": 1, "capabilities": [], "subagents": []}, path)
    return path


@pytest.fixture
def tools_root(tmp_path) -> Path:
    return tmp_path / "tools"


def _draft(**overrides) -> CapabilityDraft:
    payload: Dict[str, Any] = dict(
        tool_name="compute_my_index",
        description="Compute a standardized index from a timeseries.",
        subagent="diagnostics",
        category="analyzers",
        source_module="examples.user_code.my_index",
        source_function="compute_my_index",
        inputs=[ParamDraft(name="input_file", type="path", required=True)],
        outputs=[OutputDraft(name="index_file", type="csv")],
    )
    payload.update(overrides)
    return CapabilityDraft(**payload)


def _register(draft, registry_file, tools_root, **kwargs):
    return register_capability(
        draft,
        path=registry_file,
        tools_root=tools_root,
        regenerate=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_register_then_remove_leaves_no_trace(registry_file, tools_root):
    draft = _draft()

    entry, catalog_output = _register(draft, registry_file, tools_root)

    assert catalog_output == ""  # regeneration was skipped
    assert entry.tool_name == "compute_my_index"
    assert entry.subagent == "diagnostics"
    assert entry.category == "analyzers"
    assert entry.source_module == "examples.user_code.my_index"
    assert entry.onboarded_at  # timestamped
    adapter = tools_root / "analyzers" / "compute_my_index.py"
    assert adapter.exists()

    assert [e.tool_name for e in list_capabilities(registry_file)] == [
        "compute_my_index"
    ]
    assert find_capability("compute_my_index", registry_file) is not None

    removed = remove_capability(
        "compute_my_index",
        path=registry_file,
        tools_root=tools_root,
        regenerate=False,
    )

    assert removed.tool_name == "compute_my_index"
    assert not adapter.exists()
    assert list_capabilities(registry_file) == []


def test_the_registry_file_stays_readable_yaml_with_its_header(
    registry_file, tools_root
):
    _register(_draft(), registry_file, tools_root)
    text = registry_file.read_text()

    assert text.startswith("# ESMFlow extension registry")
    assert "python tools/generate_catalog.py --overwrite" in text

    parsed = yaml.safe_load(text)
    assert parsed["version"] == 1
    assert parsed["capabilities"][0]["tool_name"] == "compute_my_index"


def test_registering_a_second_capability_keeps_the_first(registry_file, tools_root):
    _register(_draft(), registry_file, tools_root)
    _register(
        _draft(tool_name="plot_my_index", subagent="visualization", category="plotters"),
        registry_file,
        tools_root,
    )

    assert {e.tool_name for e in list_capabilities(registry_file)} == {
        "compute_my_index",
        "plot_my_index",
    }


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_duplicate_registration_is_rejected_unless_overwritten(
    registry_file, tools_root
):
    _register(_draft(), registry_file, tools_root)

    with pytest.raises(RegistryError) as excinfo:
        _register(_draft(), registry_file, tools_root)
    assert "already registered" in str(excinfo.value)
    assert "overwrite=True" in str(excinfo.value)

    # Overwriting replaces rather than duplicates the entry.
    _register(
        _draft(description="An improved description."),
        registry_file,
        tools_root,
        overwrite=True,
    )
    entries = list_capabilities(registry_file)
    assert len(entries) == 1
    adapter = tools_root / "analyzers" / "compute_my_index.py"
    assert "An improved description." in adapter.read_text()


def test_an_unknown_subagent_is_rejected_before_anything_is_written(
    registry_file, tools_root
):
    with pytest.raises(RegistryError) as excinfo:
        _register(_draft(subagent="not_a_subagent"), registry_file, tools_root)

    message = str(excinfo.value)
    assert "unknown subagent 'not_a_subagent'" in message
    assert "diagnostics" in message  # the known set is listed for the user
    assert list_capabilities(registry_file) == []
    assert not tools_root.exists()


def test_an_incomplete_draft_is_rejected(registry_file, tools_root):
    with pytest.raises(RegistryError) as excinfo:
        _register(_draft(outputs=[]), registry_file, tools_root)

    assert "is incomplete" in str(excinfo.value)
    assert list_capabilities(registry_file) == []


def test_removing_an_unregistered_capability_is_an_error(registry_file, tools_root):
    with pytest.raises(RegistryError) as excinfo:
        remove_capability(
            "never_registered",
            path=registry_file,
            tools_root=tools_root,
            regenerate=False,
        )
    assert "is not registered" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_a_failed_catalog_regeneration_rolls_everything_back(
    registry_file, tools_root, monkeypatch
):
    import onboarding.registry as registry_module

    def _boom() -> str:
        raise RegistryError("catalog regeneration failed: simulated")

    monkeypatch.setattr(registry_module, "regenerate_catalog", _boom)
    before = registry_file.read_text()

    with pytest.raises(RegistryError) as excinfo:
        register_capability(
            _draft(), path=registry_file, tools_root=tools_root, regenerate=True
        )
    assert "simulated" in str(excinfo.value)

    # No adapter left behind, and the registry is byte-identical to before.
    assert not (tools_root / "analyzers" / "compute_my_index.py").exists()
    assert registry_file.read_text() == before
    assert list_capabilities(registry_file) == []


def test_rollback_restores_the_previous_adapter_on_overwrite(
    registry_file, tools_root, monkeypatch
):
    import onboarding.registry as registry_module

    _register(_draft(description="Original description."), registry_file, tools_root)
    adapter = tools_root / "analyzers" / "compute_my_index.py"
    original_source = adapter.read_text()
    original_registry = registry_file.read_text()

    def _boom() -> str:
        raise RegistryError("catalog regeneration failed: simulated")

    monkeypatch.setattr(registry_module, "regenerate_catalog", _boom)

    with pytest.raises(RegistryError):
        register_capability(
            _draft(description="Broken replacement."),
            path=registry_file,
            tools_root=tools_root,
            overwrite=True,
            regenerate=True,
        )

    assert adapter.read_text() == original_source
    assert registry_file.read_text() == original_registry


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------


def test_subagent_draft_fills_in_the_naming_conventions():
    draft = subagent_draft_for("snow_hydrology")

    assert draft.step_class == "SnowHydrologyStep"
    assert draft.call_tool_name == "call_snow_hydrology"
    assert draft.result_label == "snow hydrology"
    assert "SnowHydrology" in draft.display_name
    assert draft.description  # never blank; the planner needs it


def test_register_subagent_is_idempotent(registry_file):
    first = register_subagent(subagent_draft_for("snow_hydrology"), path=registry_file)
    second = register_subagent(
        subagent_draft_for("snow_hydrology", description="Different text."),
        path=registry_file,
    )

    assert first.name == second.name == "snow_hydrology"
    assert [s.name for s in list_subagents(registry_file)] == ["snow_hydrology"]
    # The stored definition wins, so an existing category is never silently
    # redefined out from under workflows that already reference it.
    assert second.description == first.description


@pytest.mark.parametrize("builtin", sorted(BUILTIN_CATEGORY_NAMES))
def test_builtin_categories_cannot_be_redefined(builtin, registry_file):
    with pytest.raises(RegistryError) as excinfo:
        register_subagent(subagent_draft_for(builtin), path=registry_file)

    assert "is a builtin category" in str(excinfo.value)
    assert list_subagents(registry_file) == []


def test_a_non_identifier_subagent_name_is_rejected(registry_file):
    with pytest.raises(RegistryError) as excinfo:
        register_subagent(
            SubagentDraft(name="snow hydrology"), path=registry_file
        )
    assert "must be an identifier" in str(excinfo.value)


def test_a_capability_can_be_registered_into_a_new_subagent(registry_file, tools_root):
    """The full extension path: new category, then a tool inside it."""

    register_subagent(subagent_draft_for("snow_hydrology"), path=registry_file)
    _register(
        _draft(
            tool_name="compute_snow_season_metrics",
            subagent="snow_hydrology",
            category="snow_hydrology",
        ),
        registry_file,
        tools_root,
    )

    entry = find_capability("compute_snow_season_metrics", registry_file)
    assert entry is not None and entry.subagent == "snow_hydrology"

    # ...and the planner picks the pairing up from the same file.
    specs = category_specs(registry_file)
    assert specs["snow_hydrology"].tools == ("compute_snow_season_metrics",)
    assert specs["snow_hydrology"].builtin is False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_reports_emptiness_and_then_contents(registry_file, tools_root):
    assert registry_summary(registry_file) == (
        "No onboarded capabilities or subagents yet."
    )

    register_subagent(subagent_draft_for("snow_hydrology"), path=registry_file)
    _register(_draft(), registry_file, tools_root)

    summary = registry_summary(registry_file)
    assert "User-onboarded subagents:" in summary
    assert "snow_hydrology" in summary
    assert "SnowHydrologyStep" in summary
    assert "Onboarded capabilities:" in summary
    assert "compute_my_index" in summary
    assert "[diagnostics/analyzers]" in summary
    assert "examples.user_code.my_index.compute_my_index" in summary
