"""Deterministic tests for benchmark/grading.py.

No Docker or live LLM calls anywhere in this module. Covers both the raw
deliverable/comparison helpers (formerly evals/structural_grading_helpers.py)
and the StructuralGrade pydantic_evals.Evaluator wrapper (formerly split
between benchmark/structural_grading.py and
evals/workflow_execution_numerical_tolerance_eval.py).
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pydantic_evals import Case, Dataset

from benchmark.grading import (
    GRADE_CRASH,
    GRADE_SUCCESS,
    GRADE_UNDETERMINED,
    MIN_CSV_BYTES,
    TOLERANCE_MATCH_REASON,
    StructuralGrade,
    csv_matches,
    find_key_csv,
    find_key_nc,
    grade,
    has_deliverable,
    nc_matches,
    protocol_matches_reference,
    try_float,
)


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["value"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset(path, variables):
    xr.Dataset({name: ("x", values) for name, values in variables.items()}).to_netcdf(path)


# ---------------------------------------------------------------------------
# try_float()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.25", 1.25), (3, 3.0), (None, None), ("not-a-number", None)],
)
def test_try_float(value, expected):
    assert try_float(value) == expected


# ---------------------------------------------------------------------------
# has_deliverable()
# ---------------------------------------------------------------------------
def test_has_deliverable_covers_missing_empty_csv_and_png_dirs(tmp_path):
    assert has_deliverable(tmp_path / "missing", "task_01_obs_summary") == (False, "no run_output directory")

    empty = tmp_path / "empty"
    empty.mkdir()
    assert has_deliverable(empty, "task_01_obs_summary") == (False, "empty run_output directory")

    (empty / "tiny.csv").write_text("x\n1\n")
    assert has_deliverable(empty, "task_01_obs_summary") == (False, "no non-trivial CSV deliverable")

    valid_csv = empty / "summary.csv"
    valid_csv.write_text("value\n" + "1\n" * MIN_CSV_BYTES)
    assert has_deliverable(empty, "task_01_obs_summary") == (True, "csv deliverable: summary.csv")

    png_dir = tmp_path / "png"
    png_dir.mkdir()
    (png_dir / "plot.png").write_bytes(b"not a real image")
    assert has_deliverable(png_dir, "task_02_seasonal_runoff") == (True, "1 png deliverable(s)")


def test_has_deliverable_rejects_unknown_kind(tmp_path, monkeypatch):
    import benchmark.grading as grading_module

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "result.txt").write_text("result")
    monkeypatch.setitem(grading_module.DELIVERABLE, "unknown", "other")

    with pytest.raises(ValueError, match="unknown deliverable kind"):
        has_deliverable(tmp_path, "unknown")


# ---------------------------------------------------------------------------
# find_key_csv() / find_key_nc()
# ---------------------------------------------------------------------------
def test_find_key_files_prefers_non_excluded_matches(tmp_path):
    for name in ["streamflow_stats.csv", "summary_stats.csv", "raw_runoff.nc", "runoff.nc"]:
        (tmp_path / name).touch()

    assert find_key_csv(tmp_path, "task_01_obs_summary").name == "summary_stats.csv"
    assert find_key_nc(tmp_path, "task_02_seasonal_runoff").name == "runoff.nc"
    assert find_key_csv(tmp_path / "missing", "task_01_obs_summary") is None
    assert find_key_nc(tmp_path, "task_01_obs_summary") is None


# ---------------------------------------------------------------------------
# csv_matches() / nc_matches()
# ---------------------------------------------------------------------------
def test_csv_matches_handles_order_tolerance_nan_and_failures(tmp_path):
    reference = tmp_path / "reference.csv"
    reordered = tmp_path / "reordered.csv"
    _write_csv(reference, [{"id": "a", "value": "1.0", "note": "ok"}, {"id": "b", "value": "nan", "note": "ok"}])
    _write_csv(reordered, [{"id": "b", "value": "nan", "note": "ok"}, {"id": "a", "value": "1.0000000000005", "note": "ok"}])

    assert csv_matches(reference, reordered) == (True, TOLERANCE_MATCH_REASON)

    changed = tmp_path / "changed.csv"
    _write_csv(changed, [{"id": "a", "value": "2.0", "note": "ok"}, {"id": "b", "value": "nan", "note": "changed"}])
    matched, reason = csv_matches(reference, changed)
    assert not matched
    assert "value:" in reason

    missing_column = tmp_path / "missing_column.csv"
    _write_csv(missing_column, [{"id": "a", "value": "1.0"}])
    assert csv_matches(reference, missing_column) == (False, "row count 2 vs 1")


def test_nc_matches_handles_matching_missing_variables_shape_and_value_differences(tmp_path):
    reference = tmp_path / "reference.nc"
    matching = tmp_path / "matching.nc"
    _write_dataset(reference, {"runoff": np.array([1.0, np.nan])})
    _write_dataset(matching, {"runoff": np.array([1.0 + 5e-13, np.nan])})
    assert nc_matches(reference, matching) == (True, TOLERANCE_MATCH_REASON)

    no_shared = tmp_path / "no_shared.nc"
    _write_dataset(no_shared, {"other": np.array([1.0, 2.0])})
    assert nc_matches(reference, no_shared) == (False, "no shared data variables")

    wrong_shape = tmp_path / "wrong_shape.nc"
    _write_dataset(wrong_shape, {"runoff": np.array([1.0, 2.0, 3.0])})
    assert nc_matches(reference, wrong_shape) == (False, "runoff: shape (2,) vs (3,)")

    changed = tmp_path / "changed.nc"
    _write_dataset(changed, {"runoff": np.array([2.0, np.nan])})
    assert nc_matches(reference, changed) == (False, "runoff: values differ beyond tolerance")


def test_protocol_matches_reference_requires_csv_and_optional_netcdf(tmp_path):
    reference_csv = tmp_path / "reference.csv"
    _write_csv(reference_csv, [{"id": "a", "value": "1.0"}])

    assert protocol_matches_reference(tmp_path, "task_01_obs_summary", None, None) == (False, "no reference csv")
    assert protocol_matches_reference(tmp_path, "task_01_obs_summary", reference_csv, None) == (False, "key csv not found in run")

    _write_csv(tmp_path / "summary_stats.csv", [{"id": "a", "value": "1.0"}])
    assert protocol_matches_reference(tmp_path, "task_01_obs_summary", reference_csv, None) == (True, f"csv: {TOLERANCE_MATCH_REASON}")


# ---------------------------------------------------------------------------
# grade() -- the (grade, reason) helper StructuralGrade wraps
# ---------------------------------------------------------------------------
def test_grade_is_crash_when_deliverable_missing(tmp_path):
    out_dir = tmp_path / "run1_output"
    out_dir.mkdir()
    auto_grade, _reason = grade(out_dir, "protocol", "task_01_obs_summary", tmp_path / "reference")
    assert auto_grade == GRADE_CRASH


def test_grade_is_success_when_protocol_matches_reference(tmp_path):
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    out_dir = tmp_path / "run" / "task_01_obs_summary" / "run1_output"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    auto_grade, _reason = grade(out_dir, "protocol", "task_01_obs_summary", tmp_path / "reference")
    assert auto_grade == GRADE_SUCCESS


def test_grade_is_undetermined_when_protocol_values_mismatch(tmp_path):
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    out_dir = tmp_path / "run" / "task_01_obs_summary" / "run1_output"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "999.0"}])

    auto_grade, _reason = grade(out_dir, "protocol", "task_01_obs_summary", tmp_path / "reference")
    assert auto_grade == GRADE_UNDETERMINED


def test_grade_never_auto_grades_baseline_as_success(tmp_path):
    """Baseline filenames are unpredictable free-form code output; Step 2
    (numeric reference comparison) only applies to protocol mode, even when
    a CSV with the exact reference values happens to be present."""
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    out_dir = tmp_path / "run" / "task_01_obs_summary" / "run1_output"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    auto_grade, _reason = grade(out_dir, "baseline", "task_01_obs_summary", tmp_path / "reference")
    assert auto_grade == GRADE_UNDETERMINED


def test_grade_baseline_crash_still_detected(tmp_path):
    out_dir = tmp_path / "run1_output"
    out_dir.mkdir()
    auto_grade, _reason = grade(out_dir, "baseline", "task_02_seasonal_runoff", tmp_path / "reference")
    assert auto_grade == GRADE_CRASH


# ---------------------------------------------------------------------------
# StructuralGrade -- the pydantic_evals.Evaluator wrapper used as the
# evaluator on every Case in benchmark/datasets.py
# ---------------------------------------------------------------------------
def test_structural_grade_scores_crash_success_and_undetermined(tmp_path):
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    crash_dir = tmp_path / "crash_output"
    crash_dir.mkdir()

    success_dir = tmp_path / "success_output"
    success_dir.mkdir()
    _write_csv(success_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    dataset = Dataset(
        name="t",
        cases=[
            Case(name="crash", inputs="crash", evaluators=[
                StructuralGrade(mode="protocol", task="task_01_obs_summary", reference_dir=tmp_path / "reference")
            ]),
            Case(name="success", inputs="success", evaluators=[
                StructuralGrade(mode="protocol", task="task_01_obs_summary", reference_dir=tmp_path / "reference")
            ]),
        ],
    )
    outputs = {"crash": crash_dir, "success": success_dir}
    report = dataset.evaluate_sync(lambda label: outputs[label], progress=False)
    scores = {case.name: case.scores["StructuralGrade"] for case in report.cases}
    assert scores["crash"].value == 0.0
    assert scores["crash"].reason.startswith("crash:")
    assert scores["success"].value == 1.0
    assert scores["success"].reason.startswith("success:")
