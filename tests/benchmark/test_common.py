"""Deterministic tests for benchmark/common.py.

No Docker or live LLM calls anywhere in this module.
"""

import json
import os
import time

import pytest

from benchmark.common import (
    ERROR_RE,
    NETWORK_REQUIRED_TASKS,
    clean_artifact,
    latest_result,
    output_dir,
    prompt_path,
    read_json,
    run_dir,
    slug,
    timestamped_result_path,
    write_json,
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
# write_json / read_json round trip
# ---------------------------------------------------------------------------
def test_write_json_read_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "file.json"
    value = [{"a": 1, "b": [1, 2, 3]}]
    write_json(path, value)
    assert read_json(path) == value


# ---------------------------------------------------------------------------
# latest_result() -- regression tests for the two confirmed bugs:
#   1. prefix collision (e.g. "scores_selfdebug" matching
#      "scores_selfdebug_resolved_*.json")
#   2. lexicographic filename sort picking the wrong file regardless of
#      actual write order
# ---------------------------------------------------------------------------
def test_latest_result_does_not_collide_with_longer_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmark.common.RESULTS_DIR", tmp_path)
    write_json(tmp_path / "scores_selfdebug_resolved_20260101_090000.json", [{"note": "resolved"}])
    write_json(tmp_path / "scores_selfdebug_graded_20260101_100000.json", [{"note": "graded"}])
    write_json(tmp_path / "scores_selfdebug_final_20260101_110000.json", [{"note": "final"}])

    picked = latest_result("scores_selfdebug_graded")
    assert read_json(picked) == [{"note": "graded"}]


def test_latest_result_uses_mtime_not_filename_string(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmark.common.RESULTS_DIR", tmp_path)
    # Filename timestamp says "235959" is newest; force the file whose name
    # says "000000" to actually have the newest mtime on disk.
    older_by_name_but_newer_by_mtime = tmp_path / "scores_structural_20260101_000000.json"
    newer_by_name_but_older_by_mtime = tmp_path / "scores_structural_20260101_235959.json"
    write_json(newer_by_name_but_older_by_mtime, [{"note": "looks newest by filename"}])
    write_json(older_by_name_but_newer_by_mtime, [{"note": "actually newest by mtime"}])

    now = time.time()
    os.utime(newer_by_name_but_older_by_mtime, (now - 100, now - 100))
    os.utime(older_by_name_but_newer_by_mtime, (now, now))

    picked = latest_result("scores_structural")
    assert read_json(picked) == [{"note": "actually newest by mtime"}]


def test_latest_result_raises_when_no_files_match(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmark.common.RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        latest_result("scores_structural")


def test_timestamped_result_path_matches_latest_result_pattern(monkeypatch, tmp_path):
    """timestamped_result_path()'s own output must be found by latest_result()."""
    monkeypatch.setattr("benchmark.common.RESULTS_DIR", tmp_path)
    path = timestamped_result_path("scores_structural")
    write_json(path, [{"note": "ok"}])
    assert latest_result("scores_structural") == path


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
