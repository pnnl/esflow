from pathlib import Path

import pytest
import yaml

from common.workflow_validation import load_tool_specs, validate_workflow


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
