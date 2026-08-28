"""Deterministic tests for benchmark/common.py.

No Docker or live LLM calls anywhere in this module.
"""

import pytest

from benchmark.common import (
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


def test_prompt_path_is_identical_for_both_benchmark_arms():
    """Both "protocol" and "single_agent" read the same task prompt --
    the task text doesn't change between arms, only which planner
    receives it."""
    protocol = prompt_path("protocol", "task_01_obs_summary")
    single_agent = prompt_path("single_agent", "task_01_obs_summary")
    assert protocol == single_agent
    assert protocol.parent.name == "protocol"
    assert protocol.name == "task_01_obs_summary.txt"


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
