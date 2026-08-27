"""Deterministic tests for benchmark/structural_grading.py.

No Docker or live LLM calls anywhere in this module.
"""

from pathlib import Path

from benchmark.structural_grading import grade_row


def _write_csv(path: Path, rows):
    import csv

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_grade_row_is_crash_when_deliverable_missing(tmp_path):
    out_dir = tmp_path / "run1_output"
    out_dir.mkdir()
    row = {
        "mode": "protocol", "model": "GPT 5.4", "task": "task_01_obs_summary",
        "run": 1, "output_dir": str(out_dir),
    }
    graded = grade_row(row)
    assert graded["auto_grade"] == "crash"
    assert graded["mode"] == "protocol"
    assert graded["model"] == "GPT 5.4"
    assert graded["task"] == "task_01_obs_summary"
    assert graded["run"] == 1


def test_grade_row_is_success_when_protocol_matches_reference(monkeypatch, tmp_path):
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    out_dir = tmp_path / "run" / "task_01_obs_summary" / "run1_output"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    monkeypatch.setattr(
        "benchmark.structural_grading.REFERENCE_DIR", tmp_path / "reference"
    )
    row = {
        "mode": "protocol", "model": "GPT 5.4", "task": "task_01_obs_summary",
        "run": 1, "output_dir": str(out_dir),
    }
    graded = grade_row(row)
    assert graded["auto_grade"] == "success"


def test_grade_row_is_undetermined_when_protocol_values_mismatch(monkeypatch, tmp_path):
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    out_dir = tmp_path / "run" / "task_01_obs_summary" / "run1_output"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "999.0"}])

    monkeypatch.setattr(
        "benchmark.structural_grading.REFERENCE_DIR", tmp_path / "reference"
    )
    row = {
        "mode": "protocol", "model": "GPT 5.4", "task": "task_01_obs_summary",
        "run": 1, "output_dir": str(out_dir),
    }
    graded = grade_row(row)
    assert graded["auto_grade"] == "undetermined"


def test_grade_row_never_auto_grades_baseline_as_success(monkeypatch, tmp_path):
    """Baseline filenames are unpredictable free-form code output; Step 2
    (numeric reference comparison) only applies to protocol mode, even when
    a CSV with the exact reference values happens to be present."""
    ref_dir = tmp_path / "reference" / "task_01_obs_summary"
    ref_dir.mkdir(parents=True)
    _write_csv(ref_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    out_dir = tmp_path / "run" / "task_01_obs_summary" / "run1_output"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "summary_stats.csv", [{"gauge_id": "1", "mean": "1.5"}])

    monkeypatch.setattr(
        "benchmark.structural_grading.REFERENCE_DIR", tmp_path / "reference"
    )
    row = {
        "mode": "baseline", "model": "GPT 5.4", "task": "task_01_obs_summary",
        "run": 1, "output_dir": str(out_dir),
    }
    graded = grade_row(row)
    assert graded["auto_grade"] == "undetermined"


def test_grade_row_baseline_crash_still_detected(tmp_path):
    out_dir = tmp_path / "run1_output"
    out_dir.mkdir()
    row = {
        "mode": "baseline", "model": "Claude Haiku 4.5", "task": "task_02_seasonal_runoff",
        "run": 4, "output_dir": str(out_dir),
    }
    graded = grade_row(row)
    assert graded["auto_grade"] == "crash"
