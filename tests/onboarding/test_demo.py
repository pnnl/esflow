"""Tests for the self-cleaning onboarding demonstration.

Two kinds of test live here.

*Guard tests* assert the repository's committed state: the demo capability must
never be present in ``extensions/registry.yaml``, ``tools/tool_catalog.yaml`` or
as a generated adapter under ``tools/``. That is the whole promise of the
feature -- a demonstration that can be shown at any time because it always
un-registers itself -- and a stray commit would silently break it.

*Engine tests* exercise the stages. The pure/read-only stages (scan, sample
data, workflow rendering, guidance text) run directly. The stages that would
write into the real ``tools/`` tree are driven with monkeypatched registry
primitives, so the suite never mutates the repo. The genuinely end-to-end path
(register -> run in a subprocess -> offboard) is covered by
``python -m onboarding.cli demo all``; it is deliberately not run here because
it regenerates the real catalog.
"""

import csv

import pytest
import yaml

from onboarding import demo as demo_mod
from onboarding.demo import (
    DEMO_FUNCTION,
    DEMO_MODULE,
    DEMO_SOURCE,
    DEMO_TOOL,
    DemoEndResult,
    DemoError,
    DemoStartResult,
    DemoStatus,
    demo_draft,
    demo_overview,
    demo_source_text,
    demo_status,
    demo_workflow,
    demo_workflow_snippet,
    end_demo,
    run_demo,
    start_demo,
    synthesize_sample_data,
    where_code_goes,
)
from onboarding.models import RegistryEntry
from onboarding.scaffold import adapter_path_for, validate_draft

REPO_ROOT = demo_mod.REPO_ROOT


# ---------------------------------------------------------------------------
# Guard tests: the demo must leave no committed trace
# ---------------------------------------------------------------------------


def test_the_demo_capability_is_not_in_the_committed_registry():
    """A demo left registered would make the next demonstration a no-op."""

    registry_file = REPO_ROOT / "extensions" / "registry.yaml"
    registry = yaml.safe_load(registry_file.read_text()) or {}
    names = [
        entry.get("tool_name") for entry in (registry.get("capabilities") or [])
    ]

    assert DEMO_TOOL not in names, (
        f"'{DEMO_TOOL}' is registered in extensions/registry.yaml. The "
        "demonstration must un-register itself; run "
        "`python -m onboarding.cli demo end` and commit the result."
    )


def test_the_demo_tool_is_not_in_the_committed_catalog():
    catalog_file = REPO_ROOT / "tools" / "tool_catalog.yaml"
    catalog = yaml.safe_load(catalog_file.read_text()) or {}
    names = [tool.get("name") for tool in (catalog.get("tools") or [])]

    assert DEMO_TOOL not in names, (
        f"'{DEMO_TOOL}' is in tools/tool_catalog.yaml; regenerate the catalog "
        "without the demo capability."
    )


def test_no_generated_demo_adapter_is_committed():
    adapter = adapter_path_for(demo_draft())

    assert not adapter.exists(), (
        f"a generated demo adapter is present at {adapter}; only the exemplar "
        "under examples/user_code/ should be committed."
    )


def test_demo_status_reports_the_repository_as_clean():
    status = demo_status()

    assert status.clean
    assert not status.foreign_capability
    assert "not active" in status.summary()


# ---------------------------------------------------------------------------
# Stage 1: scan. The exemplar must keep inferring exactly.
# ---------------------------------------------------------------------------


def test_the_exemplar_is_introspectable_and_needs_no_corrections():
    draft = demo_draft()

    assert draft.tool_name == DEMO_TOOL
    assert draft.source_module == DEMO_MODULE
    assert draft.source_function == DEMO_FUNCTION
    assert validate_draft(draft) == [], (
        "the exemplar is the canonical 'correctly formatted' tool, so it must "
        "register without any user corrections"
    )


def test_the_exemplar_output_types_are_inferred_exactly():
    """The explicit int()/float()/str() coercions are load-bearing.

    The introspector only trusts literals and explicit coercions; bare variable
    names get guessed as float. If someone 'tidies away' the casts in the
    exemplar's returned dict, this fails.
    """

    types = {output.name: output.type for output in demo_draft().outputs}

    assert types == {
        "gdd_file": "csv",
        "total_degree_days": "float",
        "growing_season_days": "int",
        "first_growing_day": "str",
    }


def test_the_exemplar_source_is_readable_and_teaches_the_rules():
    source = demo_source_text()

    assert f"def {DEMO_FUNCTION}(" in source
    # The docstring is the teaching material, so keep the promise it makes.
    assert "examples/user_code/" in source


# ---------------------------------------------------------------------------
# Stage 2: synthesized sample data
# ---------------------------------------------------------------------------


def test_sample_data_is_written_with_the_columns_the_tool_expects(tmp_path):
    path = synthesize_sample_data(tmp_path / "data", days=30)

    with path.open() as stream:
        rows = list(csv.DictReader(stream))

    assert path.name.endswith(".csv")
    assert len(rows) == 30
    assert set(rows[0]) == {"date", "tas_degc"}
    assert rows[0]["date"] == "2001-01-01"


def test_sample_data_is_deterministic(tmp_path):
    """Reproducible numbers matter: the chat quotes them back to the user."""

    first = synthesize_sample_data(tmp_path / "a", days=20).read_text()
    second = synthesize_sample_data(tmp_path / "b", days=20).read_text()

    assert first == second


def test_sample_data_spans_the_freezing_and_growing_seasons(tmp_path):
    """The demo is only interesting if some days are above the base temperature."""

    path = synthesize_sample_data(tmp_path / "data", days=180)
    with path.open() as stream:
        values = [float(row["tas_degc"]) for row in csv.DictReader(stream)]

    assert min(values) < 10.0 < max(values)


# ---------------------------------------------------------------------------
# The workflow the run stage executes
# ---------------------------------------------------------------------------


def test_demo_workflow_declares_both_settings_directories(tmp_path):
    workflow = demo_workflow(tmp_path / "in.csv", tmp_path / "out", tmp_path / "data")

    # Settings requires both; a missing one fails validation, not execution.
    assert set(workflow["settings"]) == {"data_dir", "output_dir"}


def test_demo_workflow_is_one_step_calling_the_demo_tool(tmp_path):
    workflow = demo_workflow(tmp_path / "in.csv", tmp_path / "out", tmp_path / "data")
    (step,) = workflow["steps"]

    assert step["tool"] == DEMO_TOOL
    assert step["type"] == demo_draft().subagent
    assert step["params"]["input_file"] == str(tmp_path / "in.csv")
    # Only the file-valued output is declared; scalars come back inline.
    assert list(step["outputs"]) == ["gdd_file"]


def test_demo_workflow_snippet_is_valid_yaml_for_one_step():
    parsed = yaml.safe_load(demo_workflow_snippet())

    assert isinstance(parsed, list) and len(parsed) == 1
    assert parsed[0]["tool"] == DEMO_TOOL
    assert parsed[0]["id"] == "growing_degree_days"


# ---------------------------------------------------------------------------
# The guidance text -- this is what answers "where does my code go?"
# ---------------------------------------------------------------------------


def test_where_code_goes_distinguishes_user_written_from_generated():
    text = where_code_goes()

    assert "examples/user_code/" in text
    assert "extensions/registry.yaml" in text
    assert "tools/tool_catalog.yaml" in text
    assert "YOU WRITE THIS" in text
    assert "GENERATED" in text


def test_where_code_goes_embeds_the_live_snippet_and_a_reference():
    text = where_code_goes()

    assert f"tool: {DEMO_TOOL}" in text
    # Brace-heavy content: this used to blow up under str.format.
    assert "${growing_degree_days.outputs.gdd_file}" in text


def test_where_code_goes_states_the_restart_caveat():
    assert "new" in where_code_goes() and "session" in where_code_goes()


def test_demo_overview_promises_the_offboarding_step():
    overview = demo_overview()

    assert DEMO_SOURCE in overview
    assert "un-registers" in overview
    assert "repeatable" in overview


# ---------------------------------------------------------------------------
# Stage 3/5: start and end, with the registry primitives stubbed out
# ---------------------------------------------------------------------------


def _entry() -> RegistryEntry:
    draft = demo_draft()
    return RegistryEntry(
        tool_name=draft.tool_name,
        category=draft.category,
        subagent=draft.subagent,
        source_module=draft.source_module,
        source_function=draft.source_function,
        adapter_path="tools/analyzers/compute_growing_degree_days.py",
    )


@pytest.fixture
def stub_status(monkeypatch):
    """Drive start/end from a fabricated on-disk status."""

    def _apply(**fields):
        status = DemoStatus(**fields)
        monkeypatch.setattr(demo_mod, "demo_status", lambda *a, **k: status)
        return status

    return _apply


def test_start_demo_registers_and_reports_the_introspector_notes(
    monkeypatch, stub_status
):
    stub_status()
    calls = {}

    def _register(draft, **kwargs):
        calls.update(kwargs)
        calls["draft"] = draft
        return _entry(), "catalog regenerated"

    monkeypatch.setattr(demo_mod, "register_capability", _register)
    result = start_demo()

    assert isinstance(result, DemoStartResult)
    assert result.tool_name == DEMO_TOOL
    assert not result.already_active
    # Overwriting is how a half-finished previous demo gets repaired.
    assert calls["overwrite"] is True
    assert calls["regenerate"] is True
    assert result.notes  # the 'assumed csv' / 'defaulted to diagnostics' notes


def test_start_demo_is_a_no_op_when_the_demo_is_fully_active(
    monkeypatch, stub_status
):
    stub_status(registered=True, adapter_exists=True, in_catalog=True)
    monkeypatch.setattr(
        demo_mod,
        "register_capability",
        lambda *a, **k: pytest.fail("must not re-register an active demo"),
    )

    result = start_demo()

    assert result.already_active
    assert "already registered" in result.summary()


def test_start_demo_refuses_to_touch_a_real_capability_of_the_same_name(
    monkeypatch, stub_status
):
    """A user's own tool must never be hijacked by the demonstration."""

    stub_status(registered=False, foreign_capability=True)
    monkeypatch.setattr(
        demo_mod,
        "register_capability",
        lambda *a, **k: pytest.fail("must not write over a foreign capability"),
    )

    with pytest.raises(DemoError, match="NOT"):
        start_demo()


def test_run_demo_requires_the_capability_to_be_registered(stub_status):
    stub_status()

    with pytest.raises(DemoError, match="start_demo"):
        run_demo()


def test_end_demo_on_a_clean_repository_does_nothing(stub_status):
    stub_status()

    result = end_demo()

    assert isinstance(result, DemoEndResult)
    assert result.was_already_clean
    assert result.leftovers == []
    assert "already absent" in result.summary()


def test_end_demo_removes_the_registration_and_the_adapter(
    monkeypatch, stub_status, tmp_path
):
    statuses = iter(
        [
            DemoStatus(registered=True, adapter_exists=True, in_catalog=True,
                       adapter_path="tools/analyzers/x.py"),
            DemoStatus(),  # the post-removal re-check
        ]
    )
    monkeypatch.setattr(demo_mod, "demo_status", lambda *a, **k: next(statuses))

    removed = {}

    def _remove(tool_name, **kwargs):
        removed["tool_name"] = tool_name
        removed.update(kwargs)
        return _entry()

    monkeypatch.setattr(demo_mod, "remove_capability", _remove)
    monkeypatch.setattr(
        demo_mod, "adapter_path_for", lambda draft: tmp_path / "gone.py"
    )

    result = end_demo()

    assert removed["tool_name"] == DEMO_TOOL
    assert removed["delete_adapter"] is True
    assert removed["regenerate"] is True
    assert result.removed_capability and result.removed_adapter
    assert result.regenerated_catalog
    assert result.leftovers == []


def test_end_demo_deletes_an_orphaned_adapter(monkeypatch, tmp_path):
    """The process can die between writing the adapter and the registry entry."""

    orphan = tmp_path / "orphan_adapter.py"
    orphan.write_text("# generated\n")

    statuses = iter(
        [
            DemoStatus(registered=False, adapter_exists=True,
                       adapter_path=str(orphan)),
            DemoStatus(),
        ]
    )
    monkeypatch.setattr(demo_mod, "demo_status", lambda *a, **k: next(statuses))
    monkeypatch.setattr(demo_mod, "adapter_path_for", lambda draft: orphan)
    monkeypatch.setattr(
        demo_mod,
        "remove_capability",
        lambda *a, **k: pytest.fail("nothing is registered to remove"),
    )
    regenerated = []
    monkeypatch.setattr(
        demo_mod, "regenerate_catalog", lambda: regenerated.append(True) or ""
    )

    result = end_demo()

    assert not orphan.exists()
    assert result.removed_adapter and result.regenerated_catalog
    assert regenerated == [True]


def test_end_demo_reports_leftovers_rather_than_claiming_success(
    monkeypatch, tmp_path
):
    statuses = iter(
        [
            DemoStatus(registered=True, adapter_exists=True, in_catalog=True),
            # Removal did not take effect -- say so instead of lying.
            DemoStatus(registered=True, adapter_exists=True, in_catalog=True,
                       adapter_path="tools/analyzers/x.py"),
        ]
    )
    monkeypatch.setattr(demo_mod, "demo_status", lambda *a, **k: next(statuses))
    monkeypatch.setattr(demo_mod, "remove_capability", lambda *a, **k: _entry())
    monkeypatch.setattr(
        demo_mod, "adapter_path_for", lambda draft: tmp_path / "still_here.py"
    )

    result = end_demo()

    assert len(result.leftovers) == 3
    assert "WARNING" in result.summary()


def test_end_demo_refuses_to_delete_a_real_capability_of_the_same_name(
    stub_status,
):
    stub_status(foreign_capability=True)

    with pytest.raises(DemoError, match="NOT"):
        end_demo()
