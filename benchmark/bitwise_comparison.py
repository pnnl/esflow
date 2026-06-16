#!/usr/bin/env python3
"""
Bit-for-bit comparison of protocol benchmark outputs across all models and runs.
CSV comparison uses the standard library; NC comparison uses xarray/numpy.
"""

import json
import os
import hashlib
import glob
import csv
from collections import defaultdict
import numpy as np
import xarray as xr

RESULTS_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results"
MODELS = ["claude-opus-4-6", "claude-haiku-4-5-20251001", "gpt-5", "gemini-2.5-flash", "o4-mini", "phi-4"]
TASKS = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    "task_03_et_benchmark",
    "task_04_streamflow_fdc",
    "task_05_basin_streamflow",
    "task_06_water_balance",
    "task_07_integrated_diagnostic",
]
RUNS = [1, 2, 3, 4]

KEY_FILE_PATTERNS = {
    "task_01_obs_summary": ["summary"],
    "task_02_seasonal_runoff": ["stats", "global"],
    "task_03_et_benchmark": ["bias_stats", "et_bias"],
    "task_04_streamflow_fdc": ["fdc_metrics"],
    "task_05_basin_streamflow": ["metrics", "validation"],
    "task_06_water_balance": ["basin_mean_et", "basin_et", "evapotranspiration"],
    "task_07_integrated_diagnostic": ["budget_summary", "basin_budget", "basin_water_budget"],
}

# Most-downstream NC files before plotting (checked via MD5)
# Patterns are searched in order; first match wins.
# For tasks with a single downstream NC, use broad patterns.
KEY_NC_PATTERNS = {
    "task_01_obs_summary": [],
    "task_02_seasonal_runoff": ["runoff"],       # catches qrunoff_mean, runoff_mean, elm_mean_runoff, runoff_climatology, etc.
    "task_03_et_benchmark": ["bias"],            # catches et_bias_field, et_bias_model_minus_obs, bias_field, etc.
    "task_04_streamflow_fdc": [],
    "task_05_basin_streamflow": [],
    "task_06_water_balance": ["water_balance_residual", "field_residual",
                              "residual_field", "residual_p_et_minus_q",
                              "residual_p_minus_et_minus_q", "residual"],
    "task_07_integrated_diagnostic": [],
}

EXCLUDE_PATTERNS = ["gauge", "matched", "timeseries", "discharge", "streamflow", "percentile", "zonal"]

REF_MODEL = "claude-opus-4-6"
REF_RUN = 2


def find_key_csv(output_dir, task):
    if not os.path.isdir(output_dir):
        return None
    csvs = sorted(glob.glob(os.path.join(output_dir, "*.csv")))
    if not csvs:
        return None

    patterns = KEY_FILE_PATTERNS[task]
    for pattern in patterns:
        matches = [f for f in csvs if pattern.lower() in os.path.basename(f).lower()]
        if matches:
            filtered = [f for f in matches
                       if not any(ex in os.path.basename(f).lower() for ex in EXCLUDE_PATTERNS)]
            if filtered:
                return filtered[0]
            return matches[0]
    return None


def find_key_nc(output_dir, task):
    """Find the most-downstream NetCDF file for a task."""
    if not os.path.isdir(output_dir):
        return None
    patterns = KEY_NC_PATTERNS.get(task, [])
    if not patterns:
        return None
    ncs = sorted(glob.glob(os.path.join(output_dir, "*.nc")))
    if not ncs:
        return None
    # Exclude raw/fetched observation files
    nc_exclude = ["raw", "fetched", "ilamb", "modis_et_raw", "obs_raw"]
    for pattern in patterns:
        matches = [f for f in ncs if pattern.lower() in os.path.basename(f).lower()]
        if matches:
            filtered = [f for f in matches
                       if not any(ex in os.path.basename(f).lower() for ex in nc_exclude)]
            if filtered:
                return filtered[0]
            return matches[0]
    return None


def md5_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_nc_values(ref_path, test_path):
    """Compare data variables in two NC files. Returns (match, detail_string).

    'match' is True if all shared data variables are identical within
    floating-point tolerance (rtol=1e-12, well above machine epsilon).
    """
    try:
        ref_ds = xr.open_dataset(ref_path)
        test_ds = xr.open_dataset(test_path)
    except Exception as e:
        return False, f"NC read error: {e}"

    ref_vars = set(ref_ds.data_vars)
    test_vars = set(test_ds.data_vars)
    shared = ref_vars & test_vars

    if not shared:
        ref_ds.close(); test_ds.close()
        return False, f"no shared data vars: ref={sorted(ref_vars)} vs test={sorted(test_vars)}"

    diffs = []
    for var in sorted(shared):
        rv = ref_ds[var].values
        tv = test_ds[var].values
        if rv.shape != tv.shape:
            diffs.append(f"{var}: shape {rv.shape} vs {tv.shape}")
        else:
            try:
                rv_f = rv.astype(float)
                tv_f = tv.astype(float)
                # Allow machine-epsilon differences
                if not np.allclose(rv_f, tv_f, rtol=1e-12, atol=1e-15, equal_nan=True):
                    abs_diff = np.abs(rv_f - tv_f)
                    max_abs = np.nanmax(abs_diff)
                    scale = np.nanmax(np.abs(rv_f))
                    rel = max_abs / max(scale, 1e-30)
                    diffs.append(f"{var}: max_rel_diff={rel:.2e}")
            except (TypeError, ValueError):
                if not np.array_equal(rv, tv):
                    diffs.append(f"{var}: values differ (non-numeric)")

    missing_in_test = ref_vars - test_vars
    extra_in_test = test_vars - ref_vars

    ref_ds.close(); test_ds.close()

    detail_parts = []
    if missing_in_test:
        detail_parts.append(f"missing vars: {sorted(missing_in_test)}")
    if extra_in_test:
        detail_parts.append(f"extra vars: {sorted(extra_in_test)}")
    if diffs:
        detail_parts.append("; ".join(diffs))

    if not diffs and not missing_in_test:
        return True, "NC data values identical"
    else:
        return False, "; ".join(detail_parts) if detail_parts else "unknown diff"


def read_csv(path):
    """Read CSV and return (headers, rows) where rows is list of dicts."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def compare_csv_values(ref_path, test_path):
    """Compare two CSVs by value. Returns (match, detail_string)."""
    try:
        ref_headers, ref_rows = read_csv(ref_path)
        test_headers, test_rows = read_csv(test_path)
    except Exception as e:
        return False, f"read error: {e}"

    if len(ref_rows) != len(test_rows):
        return False, f"row count mismatch: ref={len(ref_rows)} vs test={len(test_rows)}"

    ref_cols = sorted(ref_headers)
    test_cols = sorted(test_headers)
    if ref_cols != test_cols:
        return False, f"column mismatch: ref={ref_cols} vs test={test_cols}"

    # Sort rows by all columns for order-independent comparison
    def sort_key(row):
        return tuple(row.get(c, '') for c in sorted(ref_headers))

    ref_sorted = sorted(ref_rows, key=sort_key)
    test_sorted = sorted(test_rows, key=sort_key)

    diffs = []
    for i, (rr, tr) in enumerate(zip(ref_sorted, test_sorted)):
        for col in ref_headers:
            rv = rr.get(col, '')
            tv = tr.get(col, '')
            if rv == tv:
                continue
            # Try numeric comparison
            rf = try_float(rv)
            tf = try_float(tv)
            if rf is not None and tf is not None:
                if rf != tf:
                    rel = abs(rf - tf) / max(abs(rf), 1e-30)
                    diffs.append(f"  {col}[row{i}]: ref={rv} vs test={tv} (rel_diff={rel:.2e})")
            else:
                diffs.append(f"  {col}[row{i}]: ref={rv} vs test={tv}")

    if not diffs:
        return True, "identical values (different filenames)"
    else:
        return False, f"{len(diffs)} value diffs:\n" + "\n".join(diffs[:8])


def main():
    with open(os.path.join(RESULTS_DIR, "scores.json")) as f:
        scores = json.load(f)

    grade_map = {}
    for s in scores:
        if s["mode"] == "protocol":
            grade_map[(s["model"], s["task"], s["run"])] = s.get("c_grade", "N/A")

    results = []

    for task in TASKS:
        print(f"\n{'='*80}")
        print(f"TASK: {task}")
        print(f"{'='*80}")

        ref_dir = os.path.join(RESULTS_DIR, f"{REF_MODEL}_protocol", task, f"run{REF_RUN}_output")
        ref_file = find_key_csv(ref_dir, task)
        if not ref_file:
            print(f"  WARNING: No reference file found in {ref_dir}")
            for model in MODELS:
                for run in RUNS:
                    grade = grade_map.get((model, task, run), "N/A")
                    results.append((task, model, run, grade, "N/A", "N/A", "no ref file", "", ""))
            continue

        ref_md5 = md5_file(ref_file)
        ref_basename = os.path.basename(ref_file)
        print(f"  Reference CSV: {REF_MODEL} run{REF_RUN} -> {ref_basename} (md5={ref_md5[:12]}...)")

        # Check for reference NC file
        ref_nc = find_key_nc(ref_dir, task)
        ref_nc_basename = os.path.basename(ref_nc) if ref_nc else None
        if ref_nc:
            print(f"  Reference NC:  {REF_MODEL} run{REF_RUN} -> {ref_nc_basename}")

        for model in MODELS:
            for run in RUNS:
                grade = grade_map.get((model, task, run), "N/A")

                if model == REF_MODEL and run == REF_RUN:
                    results.append((task, model, run, grade, "REF", "REF", "reference run", ref_basename, ref_basename, "REF"))
                    continue

                test_dir = os.path.join(RESULTS_DIR, f"{model}_protocol", task, f"run{run}_output")
                test_file = find_key_csv(test_dir, task)

                if not test_file:
                    results.append((task, model, run, grade, "N/A", "N/A", "no key CSV found", ref_basename, "", "N/A"))
                    continue

                test_basename = os.path.basename(test_file)
                test_md5 = md5_file(test_file)

                md5_match = "YES" if test_md5 == ref_md5 else "no"

                if md5_match == "YES":
                    val_match = "YES"
                    detail = "bit-for-bit identical (md5 match)"
                else:
                    match, detail = compare_csv_values(ref_file, test_file)
                    val_match = "YES" if match else "no"

                # Compare NC file data values if task has one
                nc_match = "N/A"
                if ref_nc:
                    test_nc = find_key_nc(test_dir, task)
                    if test_nc:
                        nc_identical, nc_detail = compare_nc_values(ref_nc, test_nc)
                        nc_match = "YES" if nc_identical else "no"
                        if nc_match == "no":
                            detail += f"; NC: {nc_detail}"
                    else:
                        nc_match = "missing"
                        detail += "; NC file missing"

                results.append((task, model, run, grade, md5_match, val_match, detail, ref_basename, test_basename, nc_match))

    # ========== SUMMARY TABLE ==========
    print("\n\n")
    print("=" * 140)
    print("SUMMARY: BIT-FOR-BIT COMPARISON OF PROTOCOL BENCHMARK OUTPUTS")
    print("=" * 140)
    print(f"Reference: {REF_MODEL} run{REF_RUN}")
    print(f"Comparison: First MD5 checksum, then value-level if MD5 differs")
    print("=" * 140)

    for task in TASKS:
        task_results = [r for r in results if r[0] == task]
        print(f"\n--- {task} ---")
        print(f"{'Model':<35s} {'Run':<5s} {'Grade':<10s} {'CSV-MD5':<9s} {'CSV-Val':<9s} {'NC-Val':<8s} {'Ref File':<30s} {'Test File':<30s}")
        print("-" * 150)
        for t, m, r, g, md5, val, d, rf, tf, nc in task_results:
            print(f"{m:<35s} run{r:<3d} {g:<10s} {md5:<9s} {val:<9s} {nc:<8s} {rf:<30s} {tf:<30s}")

    # ========== CONDENSED MATRIX ==========
    print("\n\n")
    print("=" * 120)
    print("CONDENSED REPRODUCIBILITY MATRIX")
    print("=" * 120)
    print(f"Reference: {REF_MODEL} run{REF_RUN}")
    print("Cell format: csv_md5 / csv_val / nc_val / total_runs")
    print()

    short_names = {
        "claude-opus-4-6": "opus",
        "claude-haiku-4-5-20251001": "haiku",
        "gpt-5": "gpt-5",
        "gemini-2.5-flash": "gemini",
        "o4-mini": "o4-mini",
        "phi-4": "phi-4",
    }

    header = f"{'Task':<30s}"
    for model in MODELS:
        header += f" {short_names[model]:<14s}"
    print(header)
    print("-" * 120)

    for task in TASKS:
        row = f"{task:<30s}"
        for model in MODELS:
            task_model = [(md5, val, nc) for t, m, r, g, md5, val, d, rf, tf, nc in results
                         if t == task and m == model]
            md5_count = sum(1 for md5, val, nc in task_model if md5 in ("YES", "REF"))
            val_count = sum(1 for md5, val, nc in task_model if val in ("YES", "REF"))
            nc_count = sum(1 for md5, val, nc in task_model if nc in ("YES", "REF"))
            nc_applicable = sum(1 for md5, val, nc in task_model if nc != "N/A")
            total = len(task_model)
            if nc_applicable > 0:
                cell = f"{md5_count}/{val_count}/{nc_count}/{total}"
            else:
                cell = f"{md5_count}/{val_count}/-/{total}"
            row += f" {cell:<14s}"
        print(row)

    print()
    print("Format: csv_md5 / csv_values / nc_val / total_runs  (- = no NC file for task)")

    # ========== NON-MATCHING DETAILS ==========
    print("\n\n")
    print("=" * 120)
    print("DETAILS FOR NON-MATCHING RUNS (values or NC differ from reference)")
    print("=" * 120)

    has_nonmatch = False
    for t, m, r, g, md5, val, d, rf, tf, nc in results:
        if val not in ("YES", "REF", "N/A") or nc == "no" or nc == "missing":
            has_nonmatch = True
            print(f"\n{t} | {m} run{r} (grade={g})")
            print(f"  ref={rf}, test={tf}, nc={nc}")
            print(f"  {d}")

    if not has_nonmatch:
        print("\nAll runs with key output files are identical to the reference!")

    # ========== GRAND SUMMARY ==========
    print("\n\n")
    print("=" * 120)
    print("GRAND SUMMARY")
    print("=" * 120)

    total_runs = sum(1 for r in results if r[4] not in ("N/A",))
    md5_matches = sum(1 for r in results if r[4] in ("YES", "REF"))
    val_matches = sum(1 for r in results if r[5] in ("YES", "REF"))
    no_file = sum(1 for r in results if r[4] == "N/A")
    nc_applicable = sum(1 for r in results if r[9] not in ("N/A",))
    nc_matches = sum(1 for r in results if r[9] in ("YES", "REF"))

    print(f"Total protocol runs with key CSV:    {total_runs}")
    print(f"  CSV MD5-identical to reference:     {md5_matches} ({100*md5_matches/max(total_runs,1):.1f}%)")
    print(f"  CSV value-identical to reference:   {val_matches} ({100*val_matches/max(total_runs,1):.1f}%)")
    print(f"  Runs with no key output CSV:        {no_file}")
    print(f"  Runs with CSV value differences:    {total_runs - val_matches}")
    print(f"NC comparison (tasks with downstream NC files):")
    print(f"  Runs with NC files to compare:      {nc_applicable}")
    print(f"  NC value-identical to reference:       {nc_matches} ({100*nc_matches/max(nc_applicable,1):.1f}%)")


if __name__ == "__main__":
    main()
