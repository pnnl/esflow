"""Tests for the data-driven planner category registry.

The invariant that matters most: with an empty ``extensions/registry.yaml`` the
merged specs must be *byte-identical* to the five historical builtin categories,
so replacing the hand-written ``Literal[...]`` allow-lists with generated ones
cannot silently change planner behaviour.
"""

from pathlib import Path

import pytest
import yaml

from common.tool_categories import (
    BUILTIN_CATEGORIES,
    BUILTIN_CATEGORY_NAMES,
    CategorySpec,
    category_specs,
    extension_category_specs,
    load_registry,
    registered_capabilities,
)


def _write_registry(path: Path, **sections) -> Path:
    payload = {"version": 1, "capabilities": [], "subagents": []}
    payload.update(sections)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.fixture
def empty_registry(tmp_path) -> Path:
    return _write_registry(tmp_path / "registry.yaml")


def test_load_registry_tolerates_missing_and_empty_files(tmp_path):
    missing = tmp_path / "nope.yaml"
    assert load_registry(missing) == {
        "version": 1,
        "capabilities": [],
        "subagents": [],
    }

    blank = tmp_path / "blank.yaml"
    blank.write_text("")
    assert load_registry(blank)["capabilities"] == []

    junk = tmp_path / "junk.yaml"
    junk.write_text("- not\n- a mapping\n")
    assert load_registry(junk)["subagents"] == []


def test_empty_registry_reproduces_the_builtin_categories_exactly(empty_registry):
    specs = category_specs(empty_registry)

    assert list(specs) == list(BUILTIN_CATEGORY_NAMES)
    for builtin in BUILTIN_CATEGORIES:
        assert specs[builtin.name] == builtin
    assert extension_category_specs(empty_registry) == {}


def test_capability_widens_only_its_own_category(tmp_path):
    registry = _write_registry(
        tmp_path / "registry.yaml",
        capabilities=[
            {
                "tool_name": "compute_my_index",
                "category": "analyzers",
                "subagent": "diagnostics",
                "source_module": "examples.user_code.mine",
                "source_function": "compute_my_index",
                "adapter_path": "tools/analyzers/compute_my_index.py",
            }
        ],
    )

    specs = category_specs(registry)
    builtin = {c.name: c for c in BUILTIN_CATEGORIES}

    diagnostics = specs["diagnostics"]
    assert diagnostics.tools == builtin["diagnostics"].tools + ("compute_my_index",)
    # Everything else, including ordering, is untouched.
    for name, spec in specs.items():
        if name != "diagnostics":
            assert spec == builtin[name]


def test_capability_is_not_duplicated_when_already_allow_listed(tmp_path):
    already = BUILTIN_CATEGORIES[2].tools[0]
    registry = _write_registry(
        tmp_path / "registry.yaml",
        capabilities=[
            {
                "tool_name": already,
                "category": "analyzers",
                "subagent": "diagnostics",
                "source_module": "m",
                "source_function": "f",
                "adapter_path": "p",
            }
        ],
    )

    tools = category_specs(registry)["diagnostics"].tools
    assert tools.count(already) == 1


def test_capability_for_an_unknown_subagent_is_ignored(tmp_path):
    registry = _write_registry(
        tmp_path / "registry.yaml",
        capabilities=[
            {
                "tool_name": "orphan_tool",
                "category": "analyzers",
                "subagent": "no_such_subagent",
                "source_module": "m",
                "source_function": "f",
                "adapter_path": "p",
            }
        ],
    )

    specs = category_specs(registry)
    assert "no_such_subagent" not in specs
    assert all("orphan_tool" not in spec.tools for spec in specs.values())


def test_registry_subagent_becomes_a_non_builtin_category(tmp_path):
    registry = _write_registry(
        tmp_path / "registry.yaml",
        subagents=[{"name": "snow_hydrology", "description": "Snowpack analysis."}],
        capabilities=[
            {
                "tool_name": "compute_snow_season_metrics",
                "category": "analyzers",
                "subagent": "snow_hydrology",
                "source_module": "examples.user_code.snow_metrics",
                "source_function": "compute_snow_season_metrics",
                "adapter_path": "tools/analyzers/compute_snow_season_metrics.py",
            }
        ],
    )

    specs = category_specs(registry)
    snow = specs["snow_hydrology"]

    assert isinstance(snow, CategorySpec)
    assert snow.builtin is False
    # Conventional names are derived from the subagent name.
    assert snow.step_class == "SnowHydrologyStep"
    assert snow.call_tool_name == "call_snow_hydrology"
    assert snow.result_label == "snow hydrology"
    assert snow.tools == ("compute_snow_season_metrics",)
    # Builtins still come first, in their historical order.
    assert list(specs)[: len(BUILTIN_CATEGORY_NAMES)] == list(BUILTIN_CATEGORY_NAMES)
    assert list(extension_category_specs(registry)) == ["snow_hydrology"]


def test_registry_subagent_never_shadows_a_builtin(tmp_path):
    registry = _write_registry(
        tmp_path / "registry.yaml",
        subagents=[
            {"name": "diagnostics", "display_name": "Hijacked", "tools": []},
        ],
    )

    assert category_specs(registry)["diagnostics"] == {
        c.name: c for c in BUILTIN_CATEGORIES
    }["diagnostics"]


def test_registered_capabilities_returns_raw_entries(tmp_path):
    registry = _write_registry(
        tmp_path / "registry.yaml",
        capabilities=[{"tool_name": "a"}, {"tool_name": "b"}],
    )

    assert [entry["tool_name"] for entry in registered_capabilities(registry)] == [
        "a",
        "b",
    ]
