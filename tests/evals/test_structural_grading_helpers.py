import csv

import numpy as np
import pytest
import xarray as xr

from evals.structural_grading_helpers import (
    MIN_CSV_BYTES,
    csv_matches,
    find_key_csv,
    find_key_nc,
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.25", 1.25), (3, 3.0), (None, None), ("not-a-number", None)],
)
def test_try_float(value, expected):
    assert try_float(value) == expected


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
    import evals.structural_grading_helpers as helpers

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "result.txt").write_text("result")
    monkeypatch.setitem(helpers.DELIVERABLE, "unknown", "other")

    with pytest.raises(ValueError, match="unknown deliverable kind"):
        has_deliverable(tmp_path, "unknown")


def test_find_key_files_prefers_non_excluded_matches(tmp_path):
    for name in ["streamflow_stats.csv", "summary_stats.csv", "raw_runoff.nc", "runoff.nc"]:
        (tmp_path / name).touch()

    assert find_key_csv(tmp_path, "task_01_obs_summary").name == "summary_stats.csv"
    assert find_key_nc(tmp_path, "task_02_seasonal_runoff").name == "runoff.nc"
    assert find_key_csv(tmp_path / "missing", "task_01_obs_summary") is None
    assert find_key_nc(tmp_path, "task_01_obs_summary") is None


def test_csv_matches_handles_order_tolerance_nan_and_failures(tmp_path):
    reference = tmp_path / "reference.csv"
    reordered = tmp_path / "reordered.csv"
    _write_csv(reference, [{"id": "a", "value": "1.0", "note": "ok"}, {"id": "b", "value": "nan", "note": "ok"}])
    _write_csv(reordered, [{"id": "b", "value": "nan", "note": "ok"}, {"id": "a", "value": "1.0000000000005", "note": "ok"}])

    assert csv_matches(reference, reordered) == (True, "match within 1%")

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
    assert nc_matches(reference, matching) == (True, "match within 1%")

    no_shared = tmp_path / "no_shared.nc"
    _write_dataset(no_shared, {"other": np.array([1.0, 2.0])})
    assert nc_matches(reference, no_shared) == (False, "no shared data variables")

    wrong_shape = tmp_path / "wrong_shape.nc"
    _write_dataset(wrong_shape, {"runoff": np.array([1.0, 2.0, 3.0])})
    assert nc_matches(reference, wrong_shape) == (False, "runoff: shape (2,) vs (3,)")

    changed = tmp_path / "changed.nc"
    _write_dataset(changed, {"runoff": np.array([2.0, np.nan])})
    assert nc_matches(reference, changed) == (False, "runoff: values differ beyond 1%")


def test_protocol_matches_reference_requires_csv_and_optional_netcdf(tmp_path):
    reference_csv = tmp_path / "reference.csv"
    _write_csv(reference_csv, [{"id": "a", "value": "1.0"}])

    assert protocol_matches_reference(tmp_path, "task_01_obs_summary", None, None) == (False, "no reference csv")
    assert protocol_matches_reference(tmp_path, "task_01_obs_summary", reference_csv, None) == (False, "key csv not found in run")

    _write_csv(tmp_path / "summary_stats.csv", [{"id": "a", "value": "1.0"}])
    assert protocol_matches_reference(tmp_path, "task_01_obs_summary", reference_csv, None) == (True, "csv: match within 1%")
