#!/usr/bin/env python3
"""Run the v2 protocol and v1-style free-code benchmark conditions."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_ai import Agent

from agents.planner.oneshot_planner import plan_workflow_one_shot
from common.config import MODELS
from common.workflow import Settings
from common.workflow_runner import run_workflow_definition
from common.workflow_validation import validate_workflow
from benchmark.common import (
    BASELINE_OUTPUT_INSTRUCTION, BASELINES_DIR, ERROR_RE, NETWORK_REQUIRED_TASKS,
    PILOT_MODELS, ROOT, SANDBOX_IMAGE, TASKS, clean_artifact, output_dir,
    prompt_path, run_dir, timestamped_result_path, write_json,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("protocol", "baseline"), required=True)
    p.add_argument("--models", nargs="+", default=PILOT_MODELS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--timeout", type=int, default=300, help="baseline wall timeout")
    p.add_argument("--skip-execution", action="store_true")
    p.add_argument("--sandbox-image", default=SANDBOX_IMAGE)
    return p


def validate_args(args: argparse.Namespace) -> None:
    unknown_models = sorted(set(args.models) - set(MODELS))
    unknown_tasks = sorted(set(args.tasks) - set(TASKS))
    if unknown_models:
        raise SystemExit(f"unknown v2 model labels: {', '.join(unknown_models)}")
    if unknown_tasks:
        raise SystemExit(f"unknown benchmark tasks: {', '.join(unknown_tasks)}")
    if args.runs < 1:
        raise SystemExit("--runs must be positive")


def score_python_s0(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    return True, "Python compiles"


def score_python_s1(code: str) -> tuple[bool, str]:
    imports = "\n".join(
        line for line in code.splitlines()
        if line.lstrip().startswith(("import ", "from "))
    )
    if not imports:
        return True, "no imports"
    result = subprocess.run(
        [sys.executable, "-c", imports], text=True, capture_output=True, timeout=30
    )
    if result.returncode:
        return False, (result.stderr or result.stdout).strip()[-2000:]
    return True, "imports resolve"


def run_baseline_in_sandbox(
    script: Path, out_dir: Path, timeout: int, image: str, task: str
) -> tuple[bool, str, str, str]:
    """Run arbitrary generated code with no writable host input data.

    Network is disabled by default. task_03_et_benchmark and
    task_07_integrated_diagnostic are a deliberate exception -- they call
    fetch_ilamb_data, which needs to reach https://www.ilamb.org, and their
    plots may trigger a first-time cartopy Natural Earth shapefile download.
    See NETWORK_REQUIRED_TASKS and AGENTS.md for details.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "sample"
    command = ["docker", "run", "--rm", "--read-only"]
    if task not in NETWORK_REQUIRED_TASKS:
        command.append("--network=none")
    command += [
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        "--memory=2g", "--cpus=2", "--pids-limit=256",
        "-v", f"{data_dir.resolve()}:/workspace/data/sample:ro",
        "-v", f"{script.resolve()}:/workspace/script.py:ro",
        "-v", f"{out_dir.resolve()}:/workspace/output:rw",
        "-w", "/workspace", image, "python", "/workspace/script.py",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {timeout}s", exc.stdout or "", exc.stderr or ""
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode or ERROR_RE.search(combined):
        return False, f"exit={result.returncode}; {combined.strip()[-2000:]}", result.stdout, result.stderr
    return True, "execution passed", result.stdout, result.stderr


async def generate_baseline(model_name: str, task: str) -> str:
    system = (BASELINES_DIR / "system_codegen.txt").read_text(encoding="utf-8")
    task_text = prompt_path("baseline", task).read_text(encoding="utf-8")
    user = f"{task_text}\n\n{BASELINE_OUTPUT_INSTRUCTION}"
    agent = Agent(MODELS[model_name], output_type=str, instructions=system)
    result = await agent.run(user)
    return result.output


def _collect_step_errors(context: dict | None) -> str:
    """Return "step_id: error message" lines for every step that raised.

    run_workflow_definition() swallows per-step exceptions internally and
    keeps executing (it only returns None for a pre-execution validation
    failure), so a workflow can "complete" while one or more of its steps
    silently failed and the required deliverable never got produced. This
    is the real error signal for that case -- has_deliverable()'s generic
    reason string alone tells a repair model nothing about which step or
    tool actually failed.
    """
    if not context:
        return ""
    lines = []
    for step_id, step_state in context.items():
        if step_id in ("settings", "output_dir"):
            continue
        error = (step_state or {}).get("result", {}).get("error")
        if error:
            lines.append(f"{step_id}: {error}")
    return "\n".join(lines)


def run_protocol(model_name: str, task: str, run: int, skip_execution: bool) -> dict:
    artifact = run_dir("protocol", model_name, task) / f"run{run}.yaml"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir("protocol", model_name, task, run)
    prompt = prompt_path("protocol", task).read_text(encoding="utf-8")
    settings = Settings(data_dir="./data/sample", output_dir=str(out_dir))
    started = time.monotonic()
    row = {"mode": "protocol", "model": model_name, "task": task, "run": run,
           "output_file": str(artifact), "output_dir": str(out_dir)}
    try:
        workflow = asyncio.run(plan_workflow_one_shot(prompt, settings, MODELS[model_name]))
        workflow.write_to_file(artifact)
        row.update(s0=True, s0_msg="structured Workflow produced")
        errors = validate_workflow(workflow.to_yaml_dict())
        row.update(s1=not errors, s1_msg="; ".join(errors) if errors else "catalog validation passed")
        if not errors and not skip_execution:
            context = run_workflow_definition(workflow.to_yaml_dict(), verbose=False)
            row.update(s2=context is not None, s2_msg="execution passed" if context else "runner rejected workflow")
            step_errors = _collect_step_errors(context)
            if step_errors:
                artifact.with_suffix(".error.txt").write_text(step_errors, encoding="utf-8")
        else:
            row.update(s2=None, s2_msg="execution skipped" if skip_execution else "validation failed")
    except Exception as exc:
        artifact.with_suffix(".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        row.update(s0=False, s1=False, s2=False, s0_msg=f"{type(exc).__name__}: {exc}",
                   s1_msg="planning failed", s2_msg="planning failed")
    row["elapsed"] = round(time.monotonic() - started, 3)
    return row


def run_baseline(model_name: str, task: str, run: int, args: argparse.Namespace) -> dict:
    artifact = run_dir("baseline", model_name, task) / f"run{run}.py"
    raw = artifact.with_name(f"run{run}_raw.txt")
    logs = artifact.with_name(f"run{run}_execution.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir("baseline", model_name, task, run)
    started = time.monotonic()
    row = {"mode": "baseline", "model": model_name, "task": task, "run": run,
           "output_file": str(artifact), "output_dir": str(out_dir)}
    try:
        raw_text = asyncio.run(generate_baseline(model_name, task))
        raw.write_text(raw_text, encoding="utf-8")
        code = clean_artifact(raw_text, "python")
        artifact.write_text(code + "\n", encoding="utf-8")
        row["s0"], row["s0_msg"] = score_python_s0(code)
        row["s1"], row["s1_msg"] = score_python_s1(code) if row["s0"] else (False, "syntax failed")
        if row["s1"] and not args.skip_execution:
            ok, msg, stdout, stderr = run_baseline_in_sandbox(
                artifact, out_dir, args.timeout, args.sandbox_image, task
            )
            logs.write_text(f"STDOUT\n{stdout}\nSTDERR\n{stderr}", encoding="utf-8")
            row.update(s2=ok, s2_msg=msg)
        else:
            row.update(s2=None, s2_msg="execution skipped" if args.skip_execution else "import check failed")
    except Exception as exc:
        raw.write_text(traceback.format_exc(), encoding="utf-8")
        row.update(s0=False, s1=False, s2=False, s0_msg=f"{type(exc).__name__}: {exc}",
                   s1_msg="generation failed", s2_msg="generation failed")
    row["elapsed"] = round(time.monotonic() - started, 3)
    return row


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    rows = []
    for model_name in args.models:
        for task in args.tasks:
            for run in range(1, args.runs + 1):
                print(f"{args.mode}: {model_name} {task} run {run}", flush=True)
                rows.append(
                    run_protocol(model_name, task, run, args.skip_execution)
                    if args.mode == "protocol" else run_baseline(model_name, task, run, args)
                )
    path = timestamped_result_path("benchmark")
    write_json(path, rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
