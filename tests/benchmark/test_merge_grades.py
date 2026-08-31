"""Deterministic tests for the manual-review merge logic in
benchmark/merge_grades.py.

No Docker or live LLM calls anywhere in this module.
"""

from benchmark.merge_grades import resolve_final_grade


def test_manual_override_wins_over_auto_success():
    assert resolve_final_grade("success", "silent") == "silent"


def test_manual_override_wins_over_auto_crash():
    assert resolve_final_grade("crash", "obvious") == "obvious"


def test_auto_success_used_when_no_manual_label():
    assert resolve_final_grade("success", None) == "success"


def test_auto_crash_used_when_no_manual_label():
    assert resolve_final_grade("crash", None) == "crash"


def test_undetermined_falls_back_when_no_manual_label():
    assert resolve_final_grade("undetermined", None) == "undetermined"


def test_undetermined_uses_manual_label_when_present():
    assert resolve_final_grade("undetermined", "success") == "success"
