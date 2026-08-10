from pathlib import Path

import pytest
import yaml

from common.workflow_runner import (
    check_step_outputs,
    load_tool,
    load_workflow_file,
    resolve_references,
    run_workflow_definition,
    run_workflow_file,
    validate_workflow,
)


def test_resolve_references_preserves_native_type_for_full_references():
    context = {"settings": {"years": [2000, 2001]}, "step": {"outputs": {"file": "input.csv"}}}

    assert resolve_references("${settings.years}", context) == [2000, 2001]
    assert resolve_references("prefix_${step.outputs.file}", context) == "prefix_input.csv"
    assert resolve_references({"items": ["${step.outputs.file}", 4]}, context) == {
        "items": ["input.csv", 4]
    }
    with pytest.raises(ValueError, match="Cannot resolve reference"):
        resolve_references("${step.outputs.missing}", context)


def test_check_step_outputs_honors_declared_and_overridden_output_dirs(tmp_path):
    default_dir = tmp_path / "default"
    overridden_dir = tmp_path / "overridden"
    overridden_dir.mkdir()
    (overridden_dir / "result.csv").write_text("ok")
    context = {"settings": {"custom_output": str(overridden_dir)}}
    step = {
        "params": {"output_dir": "${settings.custom_output}"},
        "outputs": {"result": "result.csv"},
    }

    assert check_step_outputs(step, context, default_dir) == (
        True,
        {"result": str(overridden_dir / "result.csv")},
    )
    assert check_step_outputs({"params": {}, "outputs": {}}, context, default_dir) == (False, {})


def test_load_tool_uses_catalog_path_and_requires_run_function(tmp_path):
    tools_dir = tmp_path / "tools"
    category_dir = tools_dir / "custom"
    category_dir.mkdir(parents=True)
    (category_dir / "hello.py").write_text("def run(config):\n    return {'value': config['value']}\n")
    catalog = {"tools": [{"name": "hello", "path": "custom/hello.py"}]}

    assert load_tool("hello", catalog, tools_dir).run({"value": 3}) == {"value": 3}
    with pytest.raises(FileNotFoundError, match="Tool not found"):
        load_tool("missing", catalog, tools_dir)


def test_runner_renames_declared_outputs_and_reuses_them(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    tool_file = tools_dir / "stub_tool.py"
    tool_file.write_text(
        "from pathlib import Path\n"
        "def run(config):\n"
        "    actual = Path(config['output_dir']) / 'actual.csv'\n"
        "    actual.write_text('value\\n1\\n')\n"
        "    return {'result': str(actual)}\n"
    )
    catalog_path = tools_dir / "tool_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {"tools": [{"name": "stub", "path": "stub_tool.py", "inputs": {}, "outputs": {}}]}
        )
    )
    workflow = {
        "name": "stub workflow",
        "settings": {"output_dir": str(output_dir)},
        "steps": [{"id": "write", "tool": "stub", "params": {}, "outputs": {"result": "declared.csv"}}],
    }

    from common import workflow_runner

    monkeypatch.setattr(workflow_runner, "_REPO_ROOT", tmp_path)
    context = run_workflow_definition(workflow)
    declared_file = output_dir / "declared.csv"
    assert declared_file.read_text() == "value\n1\n"
    assert context["write"]["outputs"] == {"result": str(declared_file)}

    reused = run_workflow_definition(workflow, reuse=True)
    assert reused["write"]["result"] == {"reused": True}


def test_runner_returns_none_when_validation_fails(tmp_path, monkeypatch):
    from common import workflow_runner

    monkeypatch.setattr(workflow_runner, "_REPO_ROOT", tmp_path)
    assert run_workflow_definition({"name": "invalid", "steps": []}) is None


def test_runner_and_shared_validator_stay_behaviorally_aligned(tmp_path):
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump({"tools": []}))
    workflow = {"steps": [{"id": "unknown", "tool": "missing", "params": {}}]}

    from common.workflow_validation import validate_workflow as shared_validate_workflow

    assert validate_workflow(workflow, catalog_path) == shared_validate_workflow(workflow, catalog_path)


def test_runner_start_from_marks_prior_steps_skipped_and_rejects_unknown_step(tmp_path, monkeypatch):
    from common import workflow_runner

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "stub.py").write_text(
        "from pathlib import Path\n"
        "def run(config):\n"
        "    output = Path(config['output_dir']) / 'actual.txt'\n"
        "    output.write_text('ran')\n"
        "    return {'result': str(output)}\n"
    )
    (tools_dir / "tool_catalog.yaml").write_text(
        yaml.safe_dump({"tools": [{"name": "stub", "path": "stub.py", "inputs": {}, "outputs": {}}]})
    )
    workflow = {
        "settings": {"output_dir": str(tmp_path / "output")},
        "steps": [
            {"id": "first", "tool": "stub", "params": {}, "outputs": {"result": "first.txt"}},
            {"id": "second", "tool": "stub", "params": {}, "outputs": {"result": "second.txt"}},
        ],
    }
    monkeypatch.setattr(workflow_runner, "_REPO_ROOT", tmp_path)

    context = run_workflow_definition(workflow, start_from="second")
    assert context["first"]["result"] == {"skipped": True}
    assert (tmp_path / "output" / "second.txt").read_text() == "ran"
    assert run_workflow_definition(workflow, start_from="missing") is None


def test_runner_records_tool_errors_continues_and_can_reuse_created_outputs(tmp_path, monkeypatch):
    from common import workflow_runner

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "fail.py").write_text(
        "from pathlib import Path\n"
        "def run(config):\n"
        "    Path(config['output_result']).write_text('fallback')\n"
        "    raise RuntimeError('expected failure')\n"
    )
    (tools_dir / "succeed.py").write_text(
        "from pathlib import Path\n"
        "def run(config):\n"
        "    output = Path(config['output_dir']) / 'actual.txt'\n"
        "    output.write_text('success')\n"
        "    return {'result': str(output)}\n"
    )
    (tools_dir / "tool_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {"name": "fail", "path": "fail.py", "inputs": {}, "outputs": {}},
                    {"name": "succeed", "path": "succeed.py", "inputs": {}, "outputs": {}},
                ]
            }
        )
    )
    workflow = {
        "settings": {"output_dir": str(tmp_path / "output")},
        "steps": [
            {"id": "failed", "tool": "fail", "params": {}, "outputs": {"result": "fallback.txt"}},
            {"id": "succeeded", "tool": "succeed", "params": {}, "outputs": {"result": "success.txt"}},
        ],
    }
    monkeypatch.setattr(workflow_runner, "_REPO_ROOT", tmp_path)

    context = run_workflow_definition(workflow, reuse=True)
    assert context["failed"]["result"] == {"error": "expected failure"}
    assert context["failed"]["outputs"] == {"result": str(tmp_path / "output" / "fallback.txt")}
    assert context["succeeded"]["result"] == {"result": str(tmp_path / "output" / "actual.txt")}


def test_workflow_file_helpers_load_and_run_yaml_definition(tmp_path, monkeypatch):
    from common import workflow_runner

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "stub.py").write_text(
        "from pathlib import Path\n"
        "def run(config):\n"
        "    output = Path(config['output_dir']) / 'result.txt'\n"
        "    output.write_text('ok')\n"
        "    return {'result': str(output)}\n"
    )
    (tools_dir / "tool_catalog.yaml").write_text(
        yaml.safe_dump({"tools": [{"name": "stub", "path": "stub.py", "inputs": {}, "outputs": {}}]})
    )
    workflow = {
        "name": "from file",
        "settings": {"output_dir": str(tmp_path / "output")},
        "steps": [{"id": "step", "tool": "stub", "params": {}, "outputs": {"result": "result.txt"}}],
    }
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(yaml.safe_dump(workflow))
    monkeypatch.setattr(workflow_runner, "_REPO_ROOT", tmp_path)

    assert load_workflow_file(workflow_path) == (workflow, workflow_path)
    assert run_workflow_file(workflow_path)["step"]["outputs"] == {
        "result": str(tmp_path / "output" / "result.txt")
    }
    with pytest.raises(FileNotFoundError, match="Workflow not found"):
        load_workflow_file(tmp_path / "missing.yaml")
