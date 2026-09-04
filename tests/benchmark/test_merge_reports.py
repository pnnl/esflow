"""Deterministic tests for benchmark/merge_reports.py's dedup-by-name
merge logic.

Covers the last-report-wins semantics added to support patching a single
retried case back into an already-written chunk report (see
benchmark/RUNBOOK.md "Retrying a partial chunk failure") without disturbing
the original disjoint-chunks concatenation behavior (different models per
chunk => no shared case names => merge is unaffected).

No Docker or live LLM calls anywhere in this module.
"""

from pydantic_evals.reporting import EvaluationReport, ReportCase, ReportCaseFailure

from benchmark.merge_reports import merge_reports


def _case(name: str) -> ReportCase:
    return ReportCase(
        name=name,
        inputs=name,
        metadata=None,
        expected_output=None,
        output=None,
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=0.0,
        total_duration=0.0,
    )


def _failure(name: str, message: str = "boom") -> ReportCaseFailure:
    return ReportCaseFailure(
        name=name,
        inputs=name,
        metadata=None,
        expected_output=None,
        error_message=message,
        error_stacktrace="",
    )


def _report(*, cases=(), failures=(), name="test") -> EvaluationReport:
    return EvaluationReport(name=name, cases=list(cases), failures=list(failures))


# ---------------------------------------------------------------------------
# Disjoint names (the original chunked-merge use case) -- plain concatenation
# ---------------------------------------------------------------------------
def test_disjoint_names_are_simply_concatenated():
    report_a = _report(cases=[_case("Model A/task_01_obs_summary/run1")])
    report_b = _report(cases=[_case("Model B/task_01_obs_summary/run1")])

    merged = merge_reports([report_a, report_b], name="merged")

    assert [c.name for c in merged.cases] == [
        "Model A/task_01_obs_summary/run1",
        "Model B/task_01_obs_summary/run1",
    ]
    assert merged.failures == []


# ---------------------------------------------------------------------------
# Same name in cases in two reports -- last report wins
# ---------------------------------------------------------------------------
def test_duplicate_case_name_last_report_wins():
    name = "Gemini 3.7 Flash/task_06_water_balance/run1"
    old = _case(name)
    old.labels["stale"] = None  # sentinel to distinguish identity below
    new = _case(name)

    report_a = _report(cases=[old])
    report_b = _report(cases=[new])

    merged = merge_reports([report_a, report_b], name="merged")

    assert len(merged.cases) == 1
    assert merged.cases[0] is new


# ---------------------------------------------------------------------------
# A name that was a failure in an earlier report and a success in a later
# (patch) report is promoted to `cases`, not left in `failures`.
# ---------------------------------------------------------------------------
def test_failure_promoted_to_case_by_later_patch_report():
    name = "Gemini 3.7 Flash/task_06_water_balance/run1"
    original_chunk = _report(
        cases=[_case("Gemini 3.7 Flash/task_06_water_balance/run2")],
        failures=[_failure(name, "RetryError: exhausted transient retries")],
    )
    patch = _report(cases=[_case(name)])

    merged = merge_reports([original_chunk, patch], name="merged")

    assert {c.name for c in merged.cases} == {
        "Gemini 3.7 Flash/task_06_water_balance/run2",
        name,
    }
    assert merged.failures == []


# ---------------------------------------------------------------------------
# The reverse direction: a case that succeeded before but fails on retry
# ends up in `failures`, not silently kept as the old success.
# ---------------------------------------------------------------------------
def test_case_demoted_to_failure_by_later_patch_report():
    name = "Gemini 3.7 Flash/task_06_water_balance/run1"
    original_chunk = _report(cases=[_case(name)])
    patch = _report(failures=[_failure(name, "still flaky")])

    merged = merge_reports([original_chunk, patch], name="merged")

    assert merged.cases == []
    assert [f.name for f in merged.failures] == [name]


# ---------------------------------------------------------------------------
# Original first-seen position is preserved even when content is overwritten
# ---------------------------------------------------------------------------
def test_first_seen_order_preserved_across_overwrite():
    name_a = "Model A/task_01_obs_summary/run1"
    name_b = "Model B/task_01_obs_summary/run1"

    report_a = _report(cases=[_case(name_a), _case(name_b)])
    patch = _report(cases=[_case(name_a)])  # only patches the first case

    merged = merge_reports([report_a, patch], name="merged")

    assert [c.name for c in merged.cases] == [name_a, name_b]


# ---------------------------------------------------------------------------
# report_evaluator_failures still concatenate across all inputs
# ---------------------------------------------------------------------------
def test_report_evaluator_failures_concatenate():
    report_a = EvaluationReport(
        name="a", cases=[], failures=[], report_evaluator_failures=["err-a"]
    )
    report_b = EvaluationReport(
        name="b", cases=[], failures=[], report_evaluator_failures=["err-b"]
    )

    merged = merge_reports([report_a, report_b], name="merged")

    assert merged.report_evaluator_failures == ["err-a", "err-b"]
