"""Deterministic tests for the manual-review merge logic in
benchmark/merge_grades.py and benchmark/merge_grades_selfdebug.py.

No Docker or live LLM calls anywhere in this module.
"""

from benchmark.merge_grades import resolve_final_grade as resolve_final_grade_structural
from benchmark.merge_grades_selfdebug import resolve_final_grade as resolve_final_grade_selfdebug


def test_manual_override_wins_over_auto_success():
    assert resolve_final_grade_structural("success", "silent") == "silent"


def test_manual_override_wins_over_auto_crash():
    assert resolve_final_grade_structural("crash", "obvious") == "obvious"


def test_auto_success_used_when_no_manual_label():
    assert resolve_final_grade_structural("success", None) == "success"


def test_auto_crash_used_when_no_manual_label():
    assert resolve_final_grade_structural("crash", None) == "crash"


def test_undetermined_falls_back_when_no_manual_label():
    assert resolve_final_grade_structural("undetermined", None) == "undetermined"


def test_undetermined_uses_manual_label_when_present():
    assert resolve_final_grade_structural("undetermined", "success") == "success"


def test_selfdebug_resolver_has_identical_precedence_rules():
    assert resolve_final_grade_selfdebug("success", "silent") == "silent"
    assert resolve_final_grade_selfdebug("crash", None) == "crash"
    assert resolve_final_grade_selfdebug("undetermined", None) == "undetermined"
    assert resolve_final_grade_selfdebug("undetermined", "obvious") == "obvious"
