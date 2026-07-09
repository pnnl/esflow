#!/usr/bin/env python3
"""
Self-debug recovery for crashed benchmark runs (both protocol and baseline).

Reads scores_structural.json to identify crashes, then for each:
  1. Loads the existing artifact (runN.py or runN.yaml) and its error output
  2. Sends artifact + error to the SAME model that generated it
  3. The model returns a fixed artifact; we execute it
  4. Repeat up to --max-rounds times

No first-run regeneration — the original single-shot artifact is preserved.
Debug iterations are saved in runN_debug/{v0, v1, v2, ...}.

Usage:
    # Debug all crashes — both protocol and baseline (up to 3 rounds each):
    python benchmark/self_debug_crashes.py --max-rounds 3

    # Debug only baseline crashes:
    python benchmark/self_debug_crashes.py --max-rounds 3 --mode baseline

    # Debug only protocol crashes:
    python benchmark/self_debug_crashes.py --max-rounds 3 --mode protocol

    # Debug only a specific model:
    python benchmark/self_debug_crashes.py --max-rounds 3 --model claude-opus-4-6

    # Debug only a specific task:
    python benchmark/self_debug_crashes.py --max-rounds 3 --task task_03_et_benchmark

    # Dry run — show what would be debugged without calling APIs:
    python benchmark/self_debug_crashes.py --max-rounds 3 --dry-run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse infrastructure from run_benchmark
from run_benchmark import (
    REPO_ROOT,
    RESULTS_DIR,
    build_system_prompt,
    call_gemini_messages,
    call_llm_messages,
    get_api_key_for_model,
    is_gemini_model,
    is_local_model,
    strip_markdown_fences,
    score_python_s2,
    _rewrite_output_dir,
    LOCAL_BASE_URL,
    LOCAL_API_KEY,
    DEFAULT_BASE_URL,
    RUN_WORKFLOW,
)

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# Map short model names back to API model names
MODEL_NAME_MAP = {
    "claude-opus-4-6": "claude-opus-4-6-v1-project",
    "gpt-5": "gpt-5-birthright",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "o4-mini": "o4-mini-birthright",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001-v1-birthright",
    "phi-4": "phi-4",
}

BASELINE_DEBUG_PROMPT = """\
Your script crashed with the following error:

```
{traceback}
```

Fix the script and return the complete corrected Python script. \
Output ONLY the Python script, no explanation."""

PROTOCOL_DEBUG_PROMPT = """\
Your YAML workflow failed during execution with the following error:

```
{traceback}
```

Fix the workflow and return the complete corrected YAML workflow. \
Output ONLY the YAML workflow, no explanation."""


# ---------------------------------------------------------------------------
# Crash loading and artifact resolution
# ---------------------------------------------------------------------------

def load_crashes(scores_path, mode_filter=None, model_filter=None, task_filter=None):
    """Load crash entries from scores_structural.json.

    mode_filter: "baseline", "protocol", or None (both).
    """
    with open(scores_path) as f:
        data = json.load(f)

    crashes = []
    for r in data:
        grade = r.get("new_grade", r.get("auto_grade"))
        if grade != "crash":
            continue
        if mode_filter and r["mode"] != mode_filter:
            continue
        if model_filter and r["model"] != model_filter:
            continue
        if task_filter and r["task"] != task_filter:
            continue
        crashes.append(r)
    return crashes


def get_original_artifacts(crash):
    """Return paths to the original artifact and error output for a crash.

    Returns (artifact_path, stderr_path, stdout_path, task_dir, run_dir).
    """
    mode = crash["mode"]
    suffix = "_protocol" if mode == "protocol" else "_baseline"
    task_dir = RESULTS_DIR / f"{crash['model']}{suffix}" / crash["task"]
    run = crash["run"]

    if mode == "protocol":
        artifact = task_dir / f"run{run}.yaml"
    else:
        artifact = task_dir / f"run{run}.py"

    stderr = task_dir / f"run{run}_stderr.txt"
    stdout = task_dir / f"run{run}_stdout.txt"
    run_dir = task_dir / f"run{run}_output"
    return artifact, stderr, stdout, task_dir, run_dir


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def execute_yaml_workflow(yaml_path, timeout=300, python_exe=None):
    """Execute a YAML workflow via run_workflow.py.

    Returns (passed, short_msg, combined, stdout, stderr).
    combined merges stdout+stderr for error feedback; stdout/stderr are also
    returned separately so callers can log them to distinct files.
    """
    exe = python_exe or sys.executable
    try:
        result = subprocess.run(
            [exe, str(RUN_WORKFLOW), str(yaml_path), "--no-freeze"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        # For protocol, errors often appear in stdout (run_workflow.py logs there)
        combined = ""
        if stderr_text.strip():
            combined += stderr_text.strip() + "\n"
        if stdout_text.strip():
            combined += stdout_text.strip()

        if result.returncode == 0:
            # Check if stdout contains FAILED steps (run_workflow.py may exit 0
            # even when individual steps fail)
            if "FAILED" in stdout_text:
                short = "Workflow steps failed"
                return False, short, combined, stdout_text, stderr_text
            return True, "Execution succeeded", combined, stdout_text, stderr_text
        else:
            lines = combined.strip().splitlines()
            short = lines[-1][:200] if lines else "unknown error"
            return False, f"Runtime error: {short}", combined, stdout_text, stderr_text
    except subprocess.TimeoutExpired:
        msg = f"TimeoutError: exceeded {timeout}s"
        return False, f"Execution timed out ({timeout}s)", msg, "", msg


# ---------------------------------------------------------------------------
# Main debug loop
# ---------------------------------------------------------------------------

def debug_one_crash(crash, system_prompt, max_rounds, base_url, exec_timeout=120, python_exe=None, resume=True):
    """Run the self-debug loop for a single crashed run.

    Directory layout:
        task_dir/
        ├── run1.py (or .yaml)   # original (untouched)
        ├── run1_output/         # original output (untouched)
        ├── run1_stderr.txt      # original error (untouched)
        └── run1_debug/          # self-debug history
            ├── v0/              # copy of original (no API call)
            │   ├── script.py (or workflow.yaml)
            │   ├── stderr.txt
            │   └── output/
            ├── v1/              # first fix attempt (round 1)
            │   ├── script.py (or workflow.yaml)
            │   ├── raw.txt      # raw LLM response
            │   ├── stderr.txt
            │   └── output/
            ├── v2/              # second fix attempt (round 2)
            └── v3/              # third fix attempt (round 3)
    """
    mode = crash["mode"]
    model_short = crash["model"]
    api_model = MODEL_NAME_MAP.get(model_short, model_short)
    task = crash["task"]
    run = crash["run"]
    run_label = f"run{run}"
    is_protocol = mode == "protocol"

    artifact_path, stderr_path, stdout_path, task_dir, run_dir = get_original_artifacts(crash)

    if not artifact_path.exists():
        return {"status": "skip", "reason": f"No artifact: {artifact_path}"}

    # Create debug directory
    debug_dir = task_dir / f"{run_label}_debug"

    # Check if this crash was already debugged (resume mode)
    if resume and debug_dir.exists():
        final_ver = debug_dir / f"v{max_rounds}"
        if final_ver.exists():
            print(f"    Skipping — already has v{max_rounds} (use --no-resume to redo)")
            return {"status": "skipped_resume", "reason": "already completed"}
        # Also skip if an earlier round genuinely passed (no error markers)
        import re
        error_pattern = re.compile(
            r"\b(traceback|error|exception|failed|fatal)\b",
            re.IGNORECASE,
        )
        for r in range(1, max_rounds + 1):
            ver = debug_dir / f"v{r}"
            # Prefer combined.txt (new layout); fall back to stderr.txt (old layout
            # where stderr.txt contained combined stdout+stderr).
            log_file = ver / "combined.txt"
            if not log_file.exists():
                log_file = ver / "stderr.txt"
            if ver.exists() and log_file.exists():
                err = log_file.read_text().strip()
                has_error = bool(error_pattern.search(err))
                if not has_error and (not err or "Execution succeeded" in err):
                    print(f"    Skipping — v{r} already passed (use --no-resume to redo)")
                    return {"status": "skipped_resume", "reason": f"v{r} passed"}

    debug_dir.mkdir(parents=True, exist_ok=True)

    # Load original artifact
    original_artifact = artifact_path.read_text()

    # Load error output — combine stderr + stdout (some scripts catch
    # exceptions and print to stdout instead of letting them propagate)
    original_error = ""
    if stderr_path.exists():
        original_error += stderr_path.read_text().strip()
    if stdout_path.exists():
        stdout_content = stdout_path.read_text().strip()
        if stdout_content:
            if original_error:
                original_error += "\n"
            original_error += stdout_content

    if not original_error.strip():
        # Re-execute to capture error
        print(f"    No error output — re-executing original to capture error...")
        if is_protocol:
            _, _, original_error, _, _ = execute_yaml_workflow(artifact_path, timeout=exec_timeout, python_exe=python_exe)
        else:
            _, _, original_error, _, _ = score_python_s2(artifact_path, timeout=exec_timeout, python_exe=python_exe)

    if not original_error.strip():
        return {"status": "skip", "reason": "Empty error output — may not be a real crash"}

    # Save v0 (copy of original) into debug directory
    v0_dir = debug_dir / "v0"
    v0_dir.mkdir(exist_ok=True)
    artifact_name = "workflow.yaml" if is_protocol else "script.py"
    (v0_dir / artifact_name).write_text(original_artifact)
    (v0_dir / "stderr.txt").write_text(original_error)
    (v0_dir / "output").mkdir(exist_ok=True)

    # Build the task prompt (same as original generation)
    if is_protocol:
        prompt_dir = Path(__file__).resolve().parent / "protocol"
    else:
        prompt_dir = Path(__file__).resolve().parent / "baselines"
    task_file = prompt_dir / f"{task}.txt"
    if not task_file.exists():
        # Fallback: try the other directory
        alt_dir = Path(__file__).resolve().parent / ("protocol" if not is_protocol else "baselines")
        task_file = alt_dir / f"{task}.txt"
    if not task_file.exists():
        return {"status": "skip", "reason": f"Task file not found: {task}"}

    task_prompt = task_file.read_text().strip()
    if not is_protocol:
        task_prompt = f"{task_prompt}\n\nSave all output files to: {run_dir}"

    # Build initial conversation: system + task + original response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": original_artifact},
    ]

    feedback_template = PROTOCOL_DEBUG_PROMPT if is_protocol else BASELINE_DEBUG_PROMPT

    versions = [{"version": "v0", "dir": str(v0_dir), "passed": False,
                 "error": original_error.strip().splitlines()[-1] if original_error.strip() else "unknown"}]

    current_error = original_error

    for round_idx in range(max_rounds):
        # Truncate long error output to last 80 lines
        err_lines = current_error.strip().splitlines()
        if len(err_lines) > 80:
            truncated = "\n".join(["[...truncated...]"] + err_lines[-80:])
        else:
            truncated = current_error.strip()

        feedback = feedback_template.format(traceback=truncated)
        messages.append({"role": "user", "content": feedback})

        # Save the full conversation that will be sent to the model
        version = f"v{round_idx + 1}"  # v1, v2, v3
        ver_dir = debug_dir / version
        ver_dir.mkdir(exist_ok=True)
        with open(ver_dir / "messages.json", "w") as f:
            json.dump(messages, f, indent=2)

        print(f"    Round {round_idx + 1}/{max_rounds}: sending error to {api_model}...")

        # Call the model
        if is_gemini_model(api_model):
            gemini_contents = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            raw_text, elapsed, usage, error = call_gemini_messages(
                api_model, gemini_contents, system_prompt
            )
        elif is_local_model(api_model):
            raw_text, elapsed, usage, error = call_llm_messages(
                api_model, messages, LOCAL_BASE_URL, LOCAL_API_KEY
            )
        else:
            model_api_key = get_api_key_for_model(api_model)
            raw_text, elapsed, usage, error = call_llm_messages(
                api_model, messages, base_url, model_api_key
            )

        if error:
            print(f"    API error: {error}")
            return {"status": "api_error", "error": error, "versions": versions,
                    "rounds_used": round_idx + 1}

        tokens = usage.get("total_tokens") if usage else None
        print(f"    Response: {elapsed:.1f}s, {tokens or '?'} tokens")

        # ver_dir already created above (for messages.json)
        ver_output = ver_dir / "output"
        ver_output.mkdir(exist_ok=True)

        # Pre-populate output with .nc files from previous version (or original
        # run) so that ILAMB data fetcher cache hits and we skip re-downloads.
        prev_output = (debug_dir / f"v{round_idx}" / "output") if round_idx > 0 else run_dir
        if prev_output.exists():
            for nc_file in prev_output.glob("*.nc"):
                dest = ver_output / nc_file.name
                if not dest.exists():
                    shutil.copy2(nc_file, dest)

        # Save artifact and raw response
        fixed_artifact = strip_markdown_fences(raw_text)
        fixed_path = ver_dir / artifact_name
        fixed_path.write_text(fixed_artifact)
        (ver_dir / "raw.txt").write_text(raw_text)

        messages.append({"role": "assistant", "content": fixed_artifact})

        # Execute the fixed artifact
        if is_protocol:
            # Rewrite output_dir in YAML to point to this version's output
            patched_yaml = _rewrite_output_dir(fixed_artifact, str(ver_output))
            fixed_path.write_text(patched_yaml)
            passed, short_msg, full_error, stdout_text, stderr_text = execute_yaml_workflow(
                fixed_path, timeout=exec_timeout, python_exe=python_exe
            )
        else:
            # Rewrite output dir in Python script
            patched_script = fixed_artifact.replace(str(run_dir), str(ver_output))
            if patched_script != fixed_artifact:
                fixed_path.write_text(patched_script)
            passed, short_msg, full_error, stdout_text, stderr_text = score_python_s2(
                fixed_path, timeout=exec_timeout, python_exe=python_exe
            )

        # Save stdout and stderr separately, plus the combined feedback that
        # was fed back to the model in the next round.
        (ver_dir / "stdout.txt").write_text(stdout_text)
        (ver_dir / "stderr.txt").write_text(stderr_text)
        (ver_dir / "combined.txt").write_text(full_error)

        short_err = full_error.strip().splitlines()[-1] if full_error.strip() else ""

        versions.append({
            "version": version,
            "dir": str(ver_dir),
            "passed": passed,
            "error": None if passed else short_err,
        })

        print(f"    [{version}] {'PASS' if passed else 'FAIL'} — {short_msg}")

        if passed:
            return {"status": "fixed", "rounds_used": round_idx + 1,
                    "final_dir": str(ver_dir), "versions": versions}

        current_error = full_error

    return {"status": "exhausted", "rounds_used": max_rounds,
            "final_dir": str(versions[-1]["dir"]), "versions": versions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-debug recovery for crashed benchmark runs")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max debug rounds per crash (default: 3)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["protocol", "baseline"],
                        help="Only debug crashes from this mode (default: both)")
    parser.add_argument("--model", type=str, default=None,
                        help="Only debug crashes from this model (short name, e.g. claude-opus-4-6)")
    parser.add_argument("--task", type=str, default=None,
                        help="Only debug crashes from this task (e.g. task_03_et_benchmark)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="API base URL")
    parser.add_argument("--exec-timeout", type=int, default=300,
                        help="Execution timeout in seconds (default: 300)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Redo all crashes even if debug runs already exist")
    parser.add_argument("--python", type=str, default=None,
                        help="Python interpreter for executing scripts/workflows "
                             "(default: same as this process). Use this to point to "
                             "a conda/venv env that has ESM libraries installed, e.g. "
                             "--python /path/to/conda/envs/esm/bin/python")
    parser.add_argument("--dry-run", action="store_true",
                        help="List crashes that would be debugged, without calling APIs")
    args = parser.parse_args()

    scores_path = RESULTS_DIR / "scores_structural.json"
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found")
        sys.exit(1)

    crashes = load_crashes(scores_path, mode_filter=args.mode,
                           model_filter=args.model, task_filter=args.task)

    # Resolve Python interpreter for script execution
    python_exe = args.python or sys.executable
    if args.python and not Path(args.python).exists():
        print(f"ERROR: --python interpreter not found: {args.python}")
        sys.exit(1)

    mode_label = args.mode or "all"
    print(f"\nSelf-Debug Recovery for Crashed Runs ({mode_label})")
    print(f"  Crashes found:  {len(crashes)}")
    print(f"  Max rounds:     {args.max_rounds}")
    print(f"  Python exe:     {python_exe}")
    if args.mode:
        print(f"  Mode filter:    {args.mode}")
    if args.model:
        print(f"  Model filter:   {args.model}")
    if args.task:
        print(f"  Task filter:    {args.task}")

    if not crashes:
        print("\nNo crashes to debug.")
        return

    # Group by mode and model for display
    from collections import Counter
    by_mode = Counter(c["mode"] for c in crashes)
    print(f"\n  Per mode:")
    for m, n in sorted(by_mode.items()):
        print(f"    {m}: {n} crashes")

    by_model = Counter((c["mode"], c["model"]) for c in crashes)
    print(f"\n  Per model:")
    for (mode, model), n in sorted(by_model.items()):
        api_name = MODEL_NAME_MAP.get(model, model)
        print(f"    [{mode}] {model} ({api_name}): {n} crashes")

    if args.dry_run:
        print(f"\n  Per crash:")
        for c in crashes:
            artifact, stderr, stdout, _, _ = get_original_artifacts(c)
            ext = ".yaml" if c["mode"] == "protocol" else ".py"
            has_err = "yes" if stderr.exists() or (c["mode"] == "protocol" and stdout.exists()) else "NO"
            print(f"    [{c['mode']}] {c['model']} | {c['task']} | run{c['run']} | error: {has_err}")
        print("\n[Dry run — no API calls made]")
        return

    # Run debug loop — build system prompt per mode
    system_prompts = {}
    results = []
    for i, crash in enumerate(crashes):
        mode = crash["mode"]
        model = crash["model"]
        task = crash["task"]
        run = crash["run"]

        # Cache system prompt per mode
        if mode not in system_prompts:
            system_prompts[mode] = build_system_prompt(mode)

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(crashes)}] [{mode}] {model} | {task} | run{run}")
        print(f"{'='*70}")

        result = debug_one_crash(
            crash, system_prompts[mode], args.max_rounds,
            args.base_url, args.exec_timeout, python_exe=args.python,
            resume=not args.no_resume
        )
        result["mode"] = mode
        result["model"] = model
        result["task"] = task
        result["run"] = run
        results.append(result)

        print(f"  Result: {result['status']}")

    # Summary
    print(f"\n{'='*70}")
    print("SELF-DEBUG SUMMARY")
    print(f"{'='*70}")

    for mode in ["protocol", "baseline"]:
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        print(f"\n  {mode.upper()}:")
        statuses = Counter(r["status"] for r in mode_results)
        for s, n in sorted(statuses.items()):
            print(f"    {s}: {n}")
        print(f"    Total: {len(mode_results)}")

    fixed = [r for r in results if r["status"] == "fixed"]
    if fixed:
        print(f"\n  Fixed ({len(fixed)}):")
        for r in fixed:
            print(f"    [{r['mode']}] {r['model']} | {r['task']} | run{r['run']} "
                  f"— fixed in {r['rounds_used']} round(s)")

    exhausted = [r for r in results if r["status"] == "exhausted"]
    if exhausted:
        print(f"\n  Still crashing after {args.max_rounds} rounds ({len(exhausted)}):")
        for r in exhausted:
            print(f"    [{r['mode']}] {r['model']} | {r['task']} | run{r['run']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"self_debug_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
