"""Deterministic tests for self-debug repair helpers in benchmark/self_debug_crashes.py.

No Docker or live LLM calls anywhere in this module -- Agent.run() itself
(the only network/LLM-touching call in self_debug_crashes.py) is never
exercised here; only the pure prompt-building, error-collection, and
local-execution helper functions are tested.
"""

from pathlib import Path

import pytest

from benchmark.common import BASELINE_OUTPUT_INSTRUCTION, BASELINES_DIR, PROTOCOL_DIR
from benchmark.run_benchmark import _collect_step_errors
from benchmark.self_debug_crashes import (
    build_repair_prompt, execute_protocol, repair_prompt_file, validate_args,
)


# ---------------------------------------------------------------------------
# _collect_step_errors() -- Fix 3: real per-step error text for protocol
# self-debug, since run_workflow_definition() swallows exceptions internally
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
# build_repair_prompt() -- Fix 4: baseline repair prompts must carry the same
# output-directory instruction the original generation prompt did
# ---------------------------------------------------------------------------
def test_build_repair_prompt_includes_output_instruction_for_baseline():
    prompt = build_repair_prompt("baseline", "Do the analysis.", "print(1)", "Traceback...")
    assert BASELINE_OUTPUT_INSTRUCTION in prompt


def test_build_repair_prompt_omits_output_instruction_for_protocol():
    prompt = build_repair_prompt("protocol", "Do the analysis.", "name: x", "error text")
    assert BASELINE_OUTPUT_INSTRUCTION not in prompt


def test_build_repair_prompt_includes_task_artifact_and_error():
    prompt = build_repair_prompt("baseline", "TASK TEXT", "ARTIFACT TEXT", "ERROR TEXT")
    assert "TASK TEXT" in prompt
    assert "ARTIFACT TEXT" in prompt
    assert "ERROR TEXT" in prompt


# ---------------------------------------------------------------------------
# execute_protocol() -- always forces settings.output_dir onto the repaired
# workflow regardless of what the model wrote, and surfaces real per-step
# error text (via _collect_step_errors) when the deliverable is missing.
# ---------------------------------------------------------------------------
def test_execute_protocol_rejects_non_mapping_yaml(tmp_path):
    ok, reason = execute_protocol("- just\n- a\n- list\n", tmp_path / "out", "task_01_obs_summary")
    assert ok is False
    assert "not a mapping" in reason


def test_execute_protocol_rejects_yaml_with_unknown_tool(tmp_path):
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
    ok, reason = execute_protocol(workflow_text, tmp_path / "out", "task_01_obs_summary")
    assert ok is False
    assert reason  # validate_workflow's error list, non-empty and joined


def test_execute_protocol_overrides_output_dir_regardless_of_yaml_content(tmp_path):
    """Even though the YAML says output_dir: /some/other/place, the actual
    out_dir argument must be what's checked for the deliverable -- this is
    what makes protocol-mode repair safe against a model that ignores the
    output-path instruction (unlike baseline mode, which is why Fix 4 was
    needed for the baseline branch instead)."""
    workflow_text = (
        "name: repaired\n"
        "description: test\n"
        "settings:\n"
        "  data_dir: ./data/sample\n"
        "  output_dir: /some/other/place/the/model/wrote\n"
        "steps: []\n"
    )
    out_dir = tmp_path / "actual_out_dir"
    ok, reason = execute_protocol(workflow_text, out_dir, "task_01_obs_summary")
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
# repair_prompt_file() -- must resolve to benchmark/baselines/ (plural) for
# baseline mode, not benchmark/baseline/ (singular). A prior inline path
# construction hardcoded the singular directory, which doesn't exist and
# raises FileNotFoundError as soon as the repair loop tries to read it.
# ---------------------------------------------------------------------------
def test_repair_prompt_file_resolves_baseline_prompts_dir():
    path = repair_prompt_file("baseline", "task_01_obs_summary")
    assert path.parent == BASELINES_DIR
    assert path.name == "task_01_obs_summary.txt"
    assert path.exists()


def test_repair_prompt_file_resolves_protocol_prompts_dir():
    path = repair_prompt_file("protocol", "task_01_obs_summary")
    assert path.parent == PROTOCOL_DIR
    assert path.name == "task_01_obs_summary.txt"
    assert path.exists()
