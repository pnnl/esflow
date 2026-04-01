#!/usr/bin/env python3
"""
Execute generated YAML workflows (or baseline Python scripts) and score them.

Scoring pipeline per run:
  V — validates: YAML parses + dry-run passes (from run_benchmark.py)
  X — executes: workflow completes without runtime error
  C — correct:  key outputs match reference (TODO: automated comparison)

Results are accumulated into a single scores.json file (append-safe).
Use --cleanup to remove bulky output directories after scoring.

Usage:
  # Execute all protocol runs for a task
  python benchmark/execute_runs.py --task task_01_obs_summary

  # Execute baseline scripts for a task
  python benchmark/execute_runs.py --task task_01_obs_summary --baseline

  # Clean up output dirs for a task (free disk space)
  python benchmark/execute_runs.py --task task_01_obs_summary --cleanup

  # Execute a specific model only
  python benchmark/execute_runs.py --task task_01_obs_summary --model claude-opus-4-6

  # Dry run — show what would be executed
  python benchmark/execute_runs.py --task task_01_obs_summary --dry-run

  # Show current scoreboard
  python benchmark/execute_runs.py --summary
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_FILE = RESULTS_DIR / "scores.json"
RUN_WORKFLOW = REPO_ROOT / "run_workflow.py"


# ---------------------------------------------------------------------------
# Load V scores from run_benchmark.py JSON files
# ---------------------------------------------------------------------------

def load_v_scores():
    """Load V (validates) scores from all benchmark_*.json files.

    Returns dict keyed by (model, task, run) -> {s0, s1, tokens, elapsed}.
    """
    v_scores = {}
    for jf in sorted(RESULTS_DIR.glob("benchmark_*.json")):
        with open(jf) as f:
            for entry in json.load(f):
                mode = entry.get("mode", "protocol")
                key = (mode, entry["model"], entry["task"], entry["run"])
                v_scores[key] = {
                    "v_parse": entry.get("s0", False),
                    "v_dryrun": entry.get("s1", False),
                    "v_pass": entry.get("s0", False) and entry.get("s1", False),
                    "v_msg": entry.get("s1_msg", entry.get("s0_msg", "")),
                    "gen_tokens": entry.get("tokens"),
                    "gen_elapsed": entry.get("elapsed"),
                    "mode": entry.get("mode", "protocol"),
                }
    return v_scores


# ---------------------------------------------------------------------------
# Find runs
# ---------------------------------------------------------------------------

def find_yaml_runs(task=None, model=None):
    """Find all run*.yaml files under *_protocol/ directories."""
    runs = []
    for yaml_file in sorted(RESULTS_DIR.rglob("run*.yaml")):
        parts = yaml_file.relative_to(RESULTS_DIR).parts
        if len(parts) != 3:
            continue
        model_dir, task_dir, filename = parts
        if not filename.endswith(".yaml"):
            continue
        if not model_dir.endswith("_protocol"):
            continue
        if task and task_dir != task:
            continue
        if model and model not in model_dir:
            continue

        run_num = int(filename.replace("run", "").replace(".yaml", ""))
        output_dir = yaml_file.parent / f"run{run_num}_output"

        runs.append({
            "model": model_dir.replace("_protocol", ""),
            "task": task_dir,
            "run": run_num,
            "yaml_path": yaml_file,
            "output_dir": output_dir,
        })
    return runs


def find_script_runs(task=None, model=None):
    """Find all run*.py files under *_baseline/ directories."""
    runs = []
    for py_file in sorted(RESULTS_DIR.rglob("run*.py")):
        parts = py_file.relative_to(RESULTS_DIR).parts
        if len(parts) != 3:
            continue
        model_dir, task_dir, filename = parts
        if not filename.endswith(".py"):
            continue
        if not model_dir.endswith("_baseline"):
            continue
        if task and task_dir != task:
            continue
        if model and model not in model_dir:
            continue

        run_num = int(filename.replace("run", "").replace(".py", ""))
        output_dir = py_file.parent / f"run{run_num}_output"

        runs.append({
            "model": model_dir.replace("_baseline", ""),
            "task": task_dir,
            "run": run_num,
            "script_path": py_file,
            "output_dir": output_dir,
        })
    return runs


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def execute_yaml(run_info, timeout=300):
    """Execute a YAML workflow via run_workflow.py. Return (x_pass, msg, elapsed)."""
    yaml_path = run_info["yaml_path"]
    output_dir = run_info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(RUN_WORKFLOW), str(yaml_path), "--no-freeze"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.time() - t0
        success = result.returncode == 0

        # Save logs alongside the YAML (these are small, keep them)
        log_dir = yaml_path.parent
        (log_dir / f"run{run_info['run']}_stdout.txt").write_text(result.stdout)
        (log_dir / f"run{run_info['run']}_stderr.txt").write_text(result.stderr)

        if success:
            return True, f"Completed in {elapsed:.1f}s", elapsed
        else:
            err_lines = result.stderr.strip().splitlines()
            if not err_lines:
                err_lines = result.stdout.strip().splitlines()
            err_msg = err_lines[-1] if err_lines else "unknown error"
            return False, f"Failed: {err_msg[:200]}", elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, f"Timed out ({timeout}s)", elapsed


def execute_script(run_info, timeout=300):
    """Execute a baseline Python script. Return (x_pass, msg, elapsed)."""
    script_path = run_info["script_path"]
    output_dir = run_info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.time() - t0
        success = result.returncode == 0

        log_dir = script_path.parent
        (log_dir / f"run{run_info['run']}_stdout.txt").write_text(result.stdout)
        (log_dir / f"run{run_info['run']}_stderr.txt").write_text(result.stderr)

        if success:
            return True, f"Completed in {elapsed:.1f}s", elapsed
        else:
            err_lines = result.stderr.strip().splitlines()
            err_msg = err_lines[-1] if err_lines else "unknown error"
            return False, f"Failed: {err_msg[:200]}", elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, f"Timed out ({timeout}s)", elapsed


# ---------------------------------------------------------------------------
# Scoreboard (persistent scores.json)
# ---------------------------------------------------------------------------

def load_scores():
    """Load existing scores.json or return empty list."""
    if SCORES_FILE.exists():
        with open(SCORES_FILE) as f:
            return json.load(f)
    return []


def save_scores(scores):
    """Write scores.json (atomic via temp file)."""
    tmp = SCORES_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(scores, f, indent=2)
    tmp.rename(SCORES_FILE)


def upsert_score(scores, entry):
    """Insert or update a score entry, keyed by (mode, model, task, run)."""
    key = (entry["mode"], entry["model"], entry["task"], entry["run"])
    for i, existing in enumerate(scores):
        existing_key = (existing["mode"], existing["model"],
                        existing["task"], existing["run"])
        if existing_key == key:
            scores[i] = entry
            return
    scores.append(entry)


def print_summary(scores):
    """Print a summary table of all scores."""
    if not scores:
        print("No scores recorded yet.")
        return

    # Group by mode
    for mode in ["protocol", "baseline"]:
        mode_scores = [s for s in scores if s["mode"] == mode]
        if not mode_scores:
            continue

        print(f"\n{'='*78}")
        print(f"  {mode.upper()} SCOREBOARD")
        print(f"{'='*78}")

        # Get unique tasks and models
        tasks = sorted(set(s["task"] for s in mode_scores))
        models = sorted(set(s["model"] for s in mode_scores))

        for task in tasks:
            print(f"\n  {task}")
            print(f"  {'Model':<30} {'Run':<5} {'V':<6} {'X':<6} {'C':<6} "
                  f"{'GenTime':<9} {'ExecTime':<9}")
            print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*6} "
                  f"{'-'*9} {'-'*9}")

            for model in models:
                task_model = [s for s in mode_scores
                              if s["task"] == task and s["model"] == model]
                for s in sorted(task_model, key=lambda x: x["run"]):
                    v = "pass" if s.get("v_pass") else "FAIL" if s.get("v_pass") is not None else "—"
                    x = "pass" if s.get("x_pass") else "FAIL" if s.get("x_pass") is not None else "—"
                    c = "pass" if s.get("c_pass") else "FAIL" if s.get("c_pass") is not None else "—"
                    gt = f"{s['gen_elapsed']:.1f}s" if s.get("gen_elapsed") else "—"
                    xt = f"{s['exec_elapsed']:.1f}s" if s.get("exec_elapsed") else "—"
                    print(f"  {model:<30} {s['run']:<5} {v:<6} {x:<6} {c:<6} "
                          f"{gt:<9} {xt:<9}")

        # Aggregate
        total = len(mode_scores)
        v_pass = sum(1 for s in mode_scores if s.get("v_pass"))
        x_pass = sum(1 for s in mode_scores if s.get("x_pass"))
        c_pass = sum(1 for s in mode_scores if s.get("c_pass"))
        x_tested = sum(1 for s in mode_scores if s.get("x_pass") is not None)
        c_tested = sum(1 for s in mode_scores if s.get("c_pass") is not None)
        print(f"\n  Totals: V={v_pass}/{total}  X={x_pass}/{x_tested or '?'}  "
              f"C={c_pass}/{c_tested or '?'}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_outputs(runs):
    """Remove run*_output/ directories to free disk space."""
    removed = 0
    freed = 0
    for run_info in runs:
        out_dir = run_info["output_dir"]
        if out_dir.exists():
            # Estimate size
            size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
            shutil.rmtree(out_dir)
            removed += 1
            freed += size
    if removed:
        print(f"\nCleanup: removed {removed} output directories "
              f"({freed / 1024 / 1024:.1f} MB freed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Execute benchmark runs and score V/X/C")
    parser.add_argument("--task", type=str,
                        help="Task name (e.g., task_01_obs_summary)")
    parser.add_argument("--model", type=str,
                        help="Model name filter (e.g., claude-opus-4-6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed")
    parser.add_argument("--baseline", action="store_true",
                        help="Execute baseline Python scripts instead of YAML")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per run in seconds (default: 300)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove output directories (no execution)")
    parser.add_argument("--summary", action="store_true",
                        help="Print scoreboard from scores.json and exit")
    args = parser.parse_args()

    # Summary mode
    if args.summary:
        print_summary(load_scores())
        return

    # Cleanup mode — just remove output dirs, no execution
    if args.cleanup:
        if args.baseline:
            runs = find_script_runs(task=args.task, model=args.model)
        else:
            runs = find_yaml_runs(task=args.task, model=args.model)
        cleanup_outputs(runs)
        return

    # Find runs
    if args.baseline:
        runs = find_script_runs(task=args.task, model=args.model)
        run_type = "baseline"
    else:
        runs = find_yaml_runs(task=args.task, model=args.model)
        run_type = "protocol"

    if not runs:
        print(f"No {run_type} runs found.")
        if args.task:
            print(f"  Task filter: {args.task}")
        if args.model:
            print(f"  Model filter: {args.model}")
        return

    print(f"\nExecuting {len(runs)} {run_type} runs")
    print(f"{'='*70}")

    # Load existing scores and V scores from benchmark JSONs
    scores = load_scores()
    v_scores = load_v_scores()

    for i, run_info in enumerate(runs):
        label = f"{run_info['model']} | {run_info['task']} | run{run_info['run']}"
        print(f"\n[{i+1}/{len(runs)}] {label}")

        if args.dry_run:
            path = run_info.get("yaml_path", run_info.get("script_path"))
            print(f"  Would execute: {path}")
            continue

        # Execute
        if args.baseline:
            x_pass, x_msg, x_elapsed = execute_script(run_info,
                                                       timeout=args.timeout)
        else:
            x_pass, x_msg, x_elapsed = execute_yaml(run_info,
                                                     timeout=args.timeout)

        icon = "X:pass" if x_pass else "X:FAIL"
        print(f"  {icon} — {x_msg}")

        # Look up V score
        v_key = (run_type, run_info["model"], run_info["task"], run_info["run"])
        v_info = v_scores.get(v_key, {})

        # Build score entry
        entry = {
            "mode": run_type,
            "model": run_info["model"],
            "task": run_info["task"],
            "run": run_info["run"],
            "v_pass": v_info.get("v_pass"),
            "v_msg": v_info.get("v_msg", ""),
            "x_pass": x_pass,
            "x_msg": x_msg,
            "c_pass": None,   # TODO: automated correctness check
            "c_msg": "",
            "gen_tokens": v_info.get("gen_tokens"),
            "gen_elapsed": v_info.get("gen_elapsed"),
            "exec_elapsed": round(x_elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        upsert_score(scores, entry)

    if args.dry_run:
        return

    # Save accumulated scores
    save_scores(scores)
    print(f"\nScores saved: {SCORES_FILE}")

    # Print summary for this batch
    task_filter = args.task
    batch = [s for s in scores
             if s["mode"] == run_type
             and (not task_filter or s["task"] == task_filter)]
    total = len(batch)
    v_pass = sum(1 for s in batch if s.get("v_pass"))
    x_pass = sum(1 for s in batch if s.get("x_pass"))
    print(f"\n{'='*70}")
    print(f"  {run_type.upper()}: V={v_pass}/{total}  X={x_pass}/{total}")
    print(f"{'='*70}")



if __name__ == "__main__":
    main()
