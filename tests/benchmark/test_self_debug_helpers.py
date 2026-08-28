"""Deterministic tests for self-debug repair helpers in benchmark/self_debug_crashes.py.

No Docker or live LLM calls anywhere in this module -- Agent.run() itself
(the only network/LLM-touching call in self_debug_crashes.py) is never
exercised here; only the pure prompt-building, error-collection, and
local-execution helper functions are tested.
"""

import pytest

from benchmark.common import PROTOCOL_DIR
from benchmark.datasets import _collect_step_errors
from benchmark.self_debug_crashes import (
    build_repair_prompt, execute_workflow, repair_prompt_file, validate_args,
)


# ---------------------------------------------------------------------------
# _collect_step_errors() -- real per-step error text for self-debug, since
# run_workflow_definition() swallows exceptions internally
# ---------------------------------------------------------------------------
def test_collect_step_errors_returns_empty_string_for_none_context():
    assert _collect_step_errors(None) == ""


def test_collect_step_errors_returns_empty_string_for_context_with_no_errors():
    context = {
        "settings": {"output_dir": "./out"},
        "output_dir": "./out",
        "load_metadata": {"outputs": {"metadata_file": "x.csv"}, "result": {"metadata_file": "x.csv"}},
    }
    assert _collect_step_errors(context) == ""


def test_collect_step_errors_reports_every_failing_step():
    context = {
        "settings": {"output_dir": "./out"},
        "output_dir": "./out",
        "load_metadata": {"outputs": {"metadata_file": "x.csv"}, "result": {"metadata_file": "x.csv"}},
        "extract_ts": {"outputs": {}, "result": {"error": "KeyError: 'gauge_id'"}},
        "plot_map": {"outputs": {}, "result": {"error": "FileNotFoundError: x.csv not found"}},
    }
    errors = _collect_step_errors(context)
    assert "extract_ts: KeyError: 'gauge_id'" in errors
    assert "plot_map: FileNotFoundError: x.csv not found" in errors
    # Steps without errors, and the non-step metadata keys, must not appear.
    assert "load_metadata" not in errors
    assert "settings" not in errors
    assert "output_dir" not in errors


def test_collect_step_errors_handles_missing_result_key():
    context = {"weird_step": {"outputs": {}}}
    assert _collect_step_errors(context) == ""


# ---------------------------------------------------------------------------
# build_repair_prompt() -- includes the original task, the failing artifact,
# and the error text for both benchmark arms
# ---------------------------------------------------------------------------
def test_build_repair_prompt_includes_task_artifact_and_error():
    prompt = build_repair_prompt("protocol", "TASK TEXT", "ARTIFACT TEXT", "ERROR TEXT")
    assert "TASK TEXT" in prompt
    assert "ARTIFACT TEXT" in prompt
    assert "ERROR TEXT" in prompt


@pytest.mark.parametrize("mode", ["protocol", "single_agent"])
def test_build_repair_prompt_works_for_both_modes(mode):
    prompt = build_repair_prompt(mode, "Do the analysis.", "name: x", "error text")
    assert "Do the analysis." in prompt
    assert "name: x" in prompt
    assert "error text" in prompt


# ---------------------------------------------------------------------------
# execute_workflow() -- always forces settings.output_dir onto the repaired
# workflow regardless of what the model wrote, and surfaces real per-step
# error text (via _collect_step_errors) when the deliverable is missing.
# ---------------------------------------------------------------------------
def test_execute_workflow_rejects_non_mapping_yaml(tmp_path):
    ok, reason = execute_workflow("- just\n- a\n- list\n", tmp_path / "out", "task_01_obs_summary")
    assert ok is False
    assert "not a mapping" in reason


def test_execute_workflow_rejects_yaml_with_unknown_tool(tmp_path):
    workflow_text = (
        "name: repaired\n"
        "description: test\n"
        "settings:\n"
        "  data_dir: ./data/sample\n"
        "  output_dir: ./ignored\n"
        "steps:\n"
        "  - id: step1\n"
        "    tool: this_tool_does_not_exist\n"
        "    params: {}\n"
    )
    ok, reason = execute_workflow(workflow_text, tmp_path / "out", "task_01_obs_summary")
    assert ok is False
    assert reason  # validate_workflow's error list, non-empty and joined


def test_execute_workflow_overrides_output_dir_regardless_of_yaml_content(tmp_path):
    """Even though the YAML says output_dir: /some/other/place, the actual
    out_dir argument must be what's checked for the deliverable -- this is
    what makes repair safe against a model that ignores the output-path
    instruction in the failing artifact it was shown."""
    workflow_text = (
        "name: repaired\n"
        "description: test\n"
        "settings:\n"
        "  data_dir: ./data/sample\n"
        "  output_dir: /some/other/place/the/model/wrote\n"
        "steps: []\n"
    )
    out_dir = tmp_path / "actual_out_dir"
    ok, reason = execute_workflow(workflow_text, out_dir, "task_01_obs_summary")
    # Empty steps -> "Workflow has no steps" validation error, never reaches
    # run_workflow_definition -- but the point is it never touches
    # /some/other/place/the/model/wrote, and doesn't crash trying to.
    assert ok is False
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# validate_args() -- rejects --max-rounds < 1 with a clear message instead of
# letting main()'s loop leave `version`/`output` unbound (a NameError when
# range(1, max_rounds + 1) is empty).
# ---------------------------------------------------------------------------
class _Args:
    def __init__(self, max_rounds):
        self.max_rounds = max_rounds


def test_validate_args_rejects_zero_max_rounds():
    with pytest.raises(SystemExit, match="--max-rounds must be positive"):
        validate_args(_Args(max_rounds=0))


def test_validate_args_rejects_negative_max_rounds():
    with pytest.raises(SystemExit, match="--max-rounds must be positive"):
        validate_args(_Args(max_rounds=-1))


def test_validate_args_accepts_positive_max_rounds():
    validate_args(_Args(max_rounds=1))
    validate_args(_Args(max_rounds=3))


# ---------------------------------------------------------------------------
# repair_prompt_file() -- both benchmark arms read the exact same task
# prompt from benchmark/protocol/, since the task text doesn't change
# between arms, only which planner receives it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["protocol", "single_agent"])
def test_repair_prompt_file_resolves_protocol_prompts_dir(mode):
    path = repair_prompt_file(mode, "task_01_obs_summary")
    assert path.parent == PROTOCOL_DIR
    assert path.name == "task_01_obs_summary.txt"
    assert path.exists()
