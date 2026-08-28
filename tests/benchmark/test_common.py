"""Deterministic tests for benchmark/common.py.

No Docker or live LLM calls anywhere in this module.
"""

import pytest

from benchmark.common import (
    ERROR_RE,
    NETWORK_REQUIRED_TASKS,
    clean_artifact,
    output_dir,
    prompt_path,
    run_dir,
    slug,
)


# ---------------------------------------------------------------------------
# slug()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Claude Opus 4.8", "claude_opus_4_8"),
        ("GPT o4 Mini", "gpt_o4_mini"),
        ("gemini-2.5-flash", "gemini_2_5_flash"),
        ("  leading and trailing  ", "leading_and_trailing"),
    ],
)
def test_slug_normalizes_to_filesystem_safe_label(value, expected):
    assert slug(value) == expected


# ---------------------------------------------------------------------------
# path composition
# ---------------------------------------------------------------------------
def test_run_dir_and_output_dir_composition(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmark.common.RESULTS_DIR", tmp_path)
    rd = run_dir("protocol", "GPT 5.4", "task_01_obs_summary")
    assert rd == tmp_path / "gpt_5_4_protocol" / "task_01_obs_summary"

    od = output_dir("protocol", "GPT 5.4", "task_01_obs_summary", 3)
    assert od == rd / "run3_output"


def test_prompt_path_selects_protocol_or_baseline_dir():
    protocol = prompt_path("protocol", "task_01_obs_summary")
    baseline = prompt_path("baseline", "task_01_obs_summary")
    assert protocol.parent.name == "protocol"
    assert baseline.parent.name == "baselines"
    assert protocol.name == baseline.name == "task_01_obs_summary.txt"


# ---------------------------------------------------------------------------
# clean_artifact()
# ---------------------------------------------------------------------------
def test_clean_artifact_strips_fence_with_language_tag():
    text = "```python\nprint('hi')\n```"
    assert clean_artifact(text, "python") == "print('hi')"


def test_clean_artifact_strips_fence_without_language_tag():
    text = "```\nname: workflow\n```"
    assert clean_artifact(text, "yaml") == "name: workflow"


def test_clean_artifact_returns_text_unchanged_when_no_fence():
    text = "print('hi')"
    assert clean_artifact(text, "python") == text


# ---------------------------------------------------------------------------
# NETWORK_REQUIRED_TASKS
# ---------------------------------------------------------------------------
def test_network_required_tasks_are_exactly_the_ilamb_dependent_tasks():
    assert NETWORK_REQUIRED_TASKS == {"task_03_et_benchmark", "task_07_integrated_diagnostic"}


# ---------------------------------------------------------------------------
# ERROR_RE -- word-boundary anchored so a baseline run's benign stdout/stderr
# text doesn't get misclassified as a crash by run_baseline_in_sandbox().
# Mirrors v1's pattern; a prior port here dropped the \b anchors and matched
# any of these words as a bare substring, which this regression-tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "benign_text",
    [
        "0 errors found",
        "error_bar",
        "column error_bar computed successfully",
        "Errors: 0",
        "failed_count = 0  # no actual failures",
    ],
)
def test_error_re_does_not_match_benign_substrings(benign_text):
    assert ERROR_RE.search(benign_text) is None


@pytest.mark.parametrize(
    "genuine_error_text",
    [
        "Traceback (most recent call last):",
        "raised an Exception during processing",
        "Fatal error: cannot continue",
        "the script failed to converge",
        "encountered an error while reading the file",
    ],
)
def test_error_re_matches_genuine_error_signals(genuine_error_text):
    assert ERROR_RE.search(genuine_error_text) is not None
