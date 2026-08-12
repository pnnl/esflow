from pathlib import Path

import pytest
import yaml

from common.workflow_validation import (
    _catalog_singletons,
    check_completeness,
    load_raw_catalog,
    load_tool_specs,
    validate_workflow,
)


@pytest.fixture
def catalog_path(tmp_path) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {
                        "name": "example",
                        "inputs": {
                            "required_value": {"required": True, "type": "str"},
                            "optional_value": {"required": False, "type": "str"},
                            "years": {"required": False, "type": "list[int]"},
                        },
                    }
                ]
            }
        )
    )
    return path


def test_load_tool_specs_returns_empty_for_missing_catalog(tmp_path):
    assert load_tool_specs(tmp_path / "missing.yaml") == {}


def test_load_raw_catalog_loads_once_per_resolved_path(catalog_path):
    first = load_raw_catalog(catalog_path)
    second = load_raw_catalog(catalog_path)

    assert first is second
    assert list(_catalog_singletons) == [catalog_path.resolve()]


def test_load_raw_catalog_uses_one_singleton_for_relative_and_resolved_paths(catalog_path, monkeypatch):
    monkeypatch.chdir(catalog_path.parent)

    relative = load_raw_catalog(Path(catalog_path.name))
    resolved = load_raw_catalog(catalog_path.resolve())

    assert relative is resolved
    assert list(_catalog_singletons) == [catalog_path.resolve()]


@pytest.mark.parametrize(
    ("workflow", "expected"),
    [
        ({"steps": []}, "Workflow has no steps"),
        ({"steps": [{"id": "one"}]}, "Step 'one' missing 'tool' field"),
        ({"steps": [{"id": "one", "tool": "missing"}]}, "Step 'one' uses unknown tool: missing"),
        ({"steps": [{"id": "one", "tool": "example", "params": {}}]}, "missing required input: required_value"),
        ({"steps": [{"id": "one", "tool": "example", "params": {"required_value": "x", "years": "2000"}}]}, "expected list[int]"),
    ],
)
def test_validate_workflow_reports_catalog_errors(catalog_path, workflow, expected):
    assert any(expected in error for error in validate_workflow(workflow, catalog_path))


def test_validate_workflow_reports_duplicate_ids_and_accepts_valid_config(catalog_path):
    duplicate = {
        "steps": [
            {"id": "same", "tool": "example", "params": {"required_value": "a"}},
            {"id": "same", "tool": "example", "params": {"required_value": "b", "years": "2000-2001"}},
        ]
    }
    assert validate_workflow(duplicate, catalog_path) == ["Duplicate step id: 'same'"]


def test_check_completeness_reports_invalid_values_and_unknown_outputs(catalog_path):
    workflow = {
        "steps": [
            {
                "id": "stats",
                "tool": "example",
                "params": {
                    "required_value": "",
                    "optional_value": "",
                    "null_value": None,
                    "placeholder": "TBD",
                    "missing": "${other.outputs.file}",
                    "data_dir": "${settings.data_dir}",
                },
            }
        ]
    }

    assert check_completeness(workflow, catalog_path) == [
        "stats.required_value is empty",
        "stats.null_value is null",
        "stats.placeholder is a placeholder",
        "stats.missing references unknown output '${other.outputs.file}'",
    ]


def test_check_completeness_accepts_known_output_references(catalog_path):
    workflow = {
        "steps": [
            {
                "id": "source",
                "tool": "example",
                "params": {"required_value": "source"},
                "outputs": {"result": "source.csv"},
            },
            {
                "id": "stats",
                "tool": "example",
                "params": {
                    "required_value": "${source.outputs.result}",
                    "optional_value": "",
                },
            },
        ]
    }

    assert check_completeness(workflow, catalog_path) == []
