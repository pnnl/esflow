"""Deterministic structural grading, shared by every benchmark run mode.

Two grades can be assigned automatically; the rest need human review:

  Step 1 - Crash detection
      CRASH iff the required final deliverable is missing:
        task_01               -> a summary-statistics CSV
        task_02 .. task_07    -> at least one PNG figure

  Step 2 - Success detection (protocol mode only)
      SUCCESS iff the key data file is numerically identical to the
      reference within float64 precision (rtol=1e-12, atol=1e-15).

  Step 3 - Manual review
      UNDETERMINED runs are flagged for human inspection (see
      make_manual_queue.py / merge_grades.py).

``StructuralGrade`` is a ``pydantic_evals.Evaluator`` that combines Steps 1
and 2 into a single numeric score, used as the evaluator on every
``Case`` in ``benchmark/datasets.py`` (protocol, baseline, and self-debug
retries alike).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

# Numerical tolerance for the success check.
# Set to float64 precision following cprnc-style verification, but allowing
# the last ~1e-12 of reassociation noise: different LLM-generated workflows
# perform reductions in different orders and can disagree at one float64 ULP
# (~2e-16) without any scientific difference.
RTOL = 1e-12
ATOL = 1e-15

# Grade taxonomy scores used by StructuralGrade.
GRADE_CRASH = "crash"
GRADE_SUCCESS = "success"
GRADE_UNDETERMINED = "undetermined"
SCORE_BY_GRADE = {GRADE_CRASH: 0.0, GRADE_UNDETERMINED: 0.5, GRADE_SUCCESS: 1.0}

# Required final deliverable per task.
#   "csv" -> a non-trivial CSV must exist (task_01 only)
#   "png" -> at least one PNG must exist (task_02 .. task_07)
DELIVERABLE = {
    "task_01_obs_summary":           "csv",
    "task_02_seasonal_runoff":       "png",
    "task_03_et_benchmark":          "png",
    "task_04_streamflow_fdc":        "png",
    "task_05_basin_streamflow":      "png",
    "task_06_water_balance":         "png",
    "task_07_integrated_diagnostic": "png",
}

# Filename keywords used to locate the key data file in a protocol
# run for Step 2's numerical comparison. Protocol runs use validated
# tools so filenames are predictable; baseline runs are not auto-graded
# in Step 2.
KEY_CSV_PATTERNS = {
    "task_01_obs_summary":           ["summary", "stats"],
    "task_02_seasonal_runoff":       ["stats", "global"],
    "task_03_et_benchmark":          ["bias_stats", "et_bias"],
    "task_04_streamflow_fdc":        ["fdc_metrics"],
    "task_05_basin_streamflow":      ["metrics", "validation"],
    "task_06_water_balance":         ["basin_mean_et", "basin_et",
                                      "evapotranspiration"],
    "task_07_integrated_diagnostic": ["budget_summary", "basin_budget",
                                      "basin_water_budget"],
}
KEY_NC_PATTERNS = {
    "task_01_obs_summary":           [],
    "task_02_seasonal_runoff":       ["runoff"],
    "task_03_et_benchmark":          ["bias"],
    "task_04_streamflow_fdc":        [],
    "task_05_basin_streamflow":      [],
    "task_06_water_balance":         ["water_balance_residual",
                                      "field_residual",
                                      "residual_field",
                                      "residual_p_et_minus_q",
                                      "residual_p_minus_et_minus_q",
                                      "residual"],
    "task_07_integrated_diagnostic": [],
}
CSV_EXCLUDE = ["gauge", "matched", "timeseries", "discharge", "streamflow",
               "percentile", "zonal"]
NC_EXCLUDE = ["raw", "fetched", "ilamb", "modis_et_raw", "obs_raw"]

MIN_CSV_BYTES = 20


# ---------------------------------------------------------------------------
# Step 1 helpers — final deliverable detection
# ---------------------------------------------------------------------------
def has_deliverable(out_dir: Path, task: str):
    """Return (ok, reason). True iff the final deliverable exists."""
    if not out_dir.is_dir():
        return False, "no run_output directory"
    files = [p for p in out_dir.iterdir() if p.is_file()]
    if not files:
        return False, "empty run_output directory"

    kind = DELIVERABLE[task]
    if kind == "csv":
        csvs = [p for p in files if p.suffix.lower() == ".csv"
                and p.stat().st_size >= MIN_CSV_BYTES]
        if not csvs:
            return False, "no non-trivial CSV deliverable"
        return True, f"csv deliverable: {csvs[0].name}"
    if kind == "png":
        pngs = [p for p in files if p.suffix.lower() == ".png"]
        if not pngs:
            return False, "no PNG deliverable"
        return True, f"{len(pngs)} png deliverable(s)"
    raise ValueError(f"unknown deliverable kind: {kind}")


# ---------------------------------------------------------------------------
# Step 2 helpers — protocol-only numerical comparison
# ---------------------------------------------------------------------------
def find_key_csv(out_dir: Path, task: str):
    if not out_dir.is_dir():
        return None
    csvs = sorted(out_dir.glob("*.csv"))
    for pattern in KEY_CSV_PATTERNS[task]:
        matches = [f for f in csvs if pattern.lower() in f.name.lower()]
        if matches:
            filtered = [f for f in matches
                        if not any(ex in f.name.lower() for ex in CSV_EXCLUDE)]
            return filtered[0] if filtered else matches[0]
    return None


def find_key_nc(out_dir: Path, task: str):
    patterns = KEY_NC_PATTERNS.get(task, [])
    if not patterns or not out_dir.is_dir():
        return None
    ncs = sorted(out_dir.glob("*.nc"))
    for pattern in patterns:
        matches = [f for f in ncs if pattern.lower() in f.name.lower()]
        if matches:
            filtered = [f for f in matches
                        if not any(ex in f.name.lower() for ex in NC_EXCLUDE)]
            return filtered[0] if filtered else matches[0]
    return None


def try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def csv_matches(ref_path: Path, test_path: Path):
    """Compare two CSVs cell by cell with rtol=RTOL, atol=ATOL."""
    try:
        with open(ref_path) as f:
            ref_rows = list(csv.DictReader(f))
        with open(test_path) as f:
            test_rows = list(csv.DictReader(f))
    except Exception as e:
        return False, f"read error: {e}"

    if len(ref_rows) != len(test_rows):
        return False, f"row count {len(ref_rows)} vs {len(test_rows)}"
    if not ref_rows:
        return True, "both empty"
    if set(ref_rows[0].keys()) != set(test_rows[0].keys()):
        return False, "column mismatch"

    cols = list(ref_rows[0].keys())

    def key(r):
        return tuple(r.get(c, "") for c in cols)

    ref_rows = sorted(ref_rows, key=key)
    test_rows = sorted(test_rows, key=key)

    for rr, tr in zip(ref_rows, test_rows):
        for col in cols:
            rv, tv = rr.get(col, ""), tr.get(col, "")
            if rv == tv:
                continue
            rf, tf = try_float(rv), try_float(tv)
            if rf is None or tf is None:
                return False, f"string mismatch at {col}"
            if np.isnan(rf) and np.isnan(tf):
                continue
            if not np.isclose(rf, tf, rtol=RTOL, atol=ATOL, equal_nan=True):
                return False, f"{col}: {rv} vs {tv}"
    return True, "match within 1%"


def nc_matches(ref_path: Path, test_path: Path):
    try:
        ref_ds = xr.open_dataset(ref_path)
        test_ds = xr.open_dataset(test_path)
    except Exception as e:
        return False, f"nc read error: {e}"
    try:
        shared = set(ref_ds.data_vars) & set(test_ds.data_vars)
        if not shared:
            return False, "no shared data variables"
        for var in sorted(shared):
            rv = ref_ds[var].values
            tv = test_ds[var].values
            if rv.shape != tv.shape:
                return False, f"{var}: shape {rv.shape} vs {tv.shape}"
            try:
                rv_f = rv.astype(float)
                tv_f = tv.astype(float)
            except (TypeError, ValueError):
                if not np.array_equal(rv, tv):
                    return False, f"{var}: non-numeric differ"
                continue
            if not np.allclose(rv_f, tv_f, rtol=RTOL, atol=ATOL,
                               equal_nan=True):
                return False, f"{var}: values differ beyond 1%"
        return True, "match within 1%"
    finally:
        ref_ds.close()
        test_ds.close()


def protocol_matches_reference(out_dir: Path, task: str,
                               ref_csv: Path | None, ref_nc: Path | None):
    """Return (success, reason) for a protocol run.

    The protocol mode succeeds iff (a) there is a key CSV in the run that
    matches the reference key CSV within tolerance, AND (b) if the task
    has a key NC, the run's key NC also matches.
    """
    if ref_csv is None:
        return False, "no reference csv"
    test_csv = find_key_csv(out_dir, task)
    if test_csv is None:
        return False, "key csv not found in run"
    csv_ok, csv_reason = csv_matches(ref_csv, test_csv)
    if not csv_ok:
        return False, f"csv: {csv_reason}"

    if ref_nc is not None:
        test_nc = find_key_nc(out_dir, task)
        if test_nc is None:
            return False, "key nc not found in run"
        nc_ok, nc_reason = nc_matches(ref_nc, test_nc)
        if not nc_ok:
            return False, f"nc: {nc_reason}"
        return True, f"csv: {csv_reason}; nc: {nc_reason}"
    return True, f"csv: {csv_reason}"


def grade(out_dir: Path, mode: str, task: str, reference_dir: Path) -> tuple[str, str]:
    """Return (grade, reason) for a single run's output directory.

    grade is one of GRADE_CRASH / GRADE_SUCCESS / GRADE_UNDETERMINED.
    Baseline mode is never auto-graded success in Step 2 -- filenames are
    unpredictable free-form code output -- so it always falls to
    undetermined once a deliverable exists.
    """
    ok, reason = has_deliverable(out_dir, task)
    if not ok:
        return GRADE_CRASH, reason
    if mode == "protocol":
        ref_dir = reference_dir / task
        match, reason = protocol_matches_reference(
            out_dir, task, find_key_csv(ref_dir, task), find_key_nc(ref_dir, task)
        )
        if match:
            return GRADE_SUCCESS, reason
    return GRADE_UNDETERMINED, reason


@dataclass
class StructuralGrade(Evaluator[str, Path]):
    """Combined Steps 1 + 2 returning a numeric score.

    Scores:
      crash        -> 0.0
      success      -> 1.0
      undetermined -> 0.5  (needs manual review)
    """
    mode: str
    task: str
    reference_dir: Path

    def evaluate(self, ctx: EvaluatorContext[str, Path]) -> EvaluationReason:
        auto_grade, reason = grade(ctx.output, self.mode, self.task, self.reference_dir)
        return EvaluationReason(value=SCORE_BY_GRADE[auto_grade], reason=f"{auto_grade}: {reason}")
