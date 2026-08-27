#!/usr/bin/env python3
"""Retry crashed benchmark artifacts with traceback-driven repairs."""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from pydantic_ai import Agent

from benchmark.common import (
    BASELINE_OUTPUT_INSTRUCTION, BASELINES_DIR, ROOT, clean_artifact,
    latest_result, read_json, timestamped_result_path, write_json,
)
from benchmark.run_benchmark import _collect_step_errors, run_baseline_in_sandbox
from common.config import MODELS
from common.config import load_prompt
from common.workflow_runner import run_workflow_definition
from common.workflow_validation import validate_workflow
from evals.structural_grading_helpers import has_deliverable


def feedback(mode: str, error: str) -> str:
    artifact = "YAML workflow" if mode == "protocol" else "Python script"
    return (
        f"The {artifact} crashed. Return a complete corrected {artifact} only, without "
        f"Markdown fences. Preserve the requested scientific analysis.\n\n"
        f"Traceback/log (last 80 lines):\n{error[-12000:]}"
    )


def build_repair_prompt(mode: str, task_prompt: str, artifact: str, error: str) -> str:
    """Build the repair user message, pure/testable without an Agent call.

    For baseline mode this must carry the same output-directory instruction
    the original generation prompt did (BASELINE_OUTPUT_INSTRUCTION), or a
    repaired script that changes/drops the output path will be re-graded as
    still-crashed even when the analysis logic itself was fixed.
    """
    if mode == "baseline":
        task_prompt = f"{task_prompt}\n\n{BASELINE_OUTPUT_INSTRUCTION}"
    return f"Original task:\n{task_prompt}\n\nFailing artifact:\n{artifact}\n\n{feedback(mode, error)}"


async def repair(model_name: str, mode: str, task_prompt: str, artifact: str, error: str) -> str:
    if mode == "baseline":
        instructions = (BASELINES_DIR / "system_codegen.txt").read_text(encoding="utf-8")
        language = "python"
    else:
        instructions = load_prompt(
            "You repair ESMFlow YAML workflows. Output only complete valid YAML."
        )
        language = "yaml"
    agent = Agent(MODELS[model_name], output_type=str, instructions=instructions)
    result = await agent.run(build_repair_prompt(mode, task_prompt, artifact, error))
    return clean_artifact(result.output, language)


def execute_protocol(workflow_text: str, out_dir: Path, task: str) -> tuple[bool, str]:
    try:
        workflow = yaml.safe_load(workflow_text)
        if not isinstance(workflow, dict):
            return False, "YAML is not a mapping"
        workflow.setdefault("settings", {})["output_dir"] = str(out_dir)
        errors = validate_workflow(workflow)
        if errors:
            return False, "; ".join(errors)
        context = run_workflow_definition(workflow, verbose=False)
        ok, reason = has_deliverable(Path(context["output_dir"]) if context else out_dir, task)
        if not ok:
            step_errors = _collect_step_errors(context)
            if step_errors:
                reason = f"{reason}\n{step_errors}"
        return ok, reason
    except Exception:
        return False, traceback.format_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="scores_structural_*.json (default: latest)")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sandbox-image", default="esflow-v2-benchmark-baseline")
    args = parser.parse_args()
    scores = read_json(args.scores or latest_result("scores_structural"))
    results = []
    for score in scores:
        if score["auto_grade"] != "crash":
            continue
        mode, model, task, run = (score[key] for key in ("mode", "model", "task", "run"))
        source = Path(score["output_dir"]).parent / (f"run{run}.yaml" if mode == "protocol" else f"run{run}.py")
        if not source.exists():
            results.append(score | {"status": "skip", "reason": "source artifact missing"})
            continue
        prompt_file = ROOT / "benchmark" / mode / f"{task}.txt"
        current = source.read_text(encoding="utf-8")
        error_file = (
            source.with_suffix(".error.txt") if mode == "protocol"
            else source.with_name(f"run{run}_execution.txt")
        )
        error = error_file.read_text(encoding="utf-8") if error_file.exists() else score["reason"]
        status = "exhausted"
        for version in range(1, args.max_rounds + 1):
            debug_dir = source.parent / f"run{run}_debug" / f"v{version}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            current = asyncio.run(repair(model, mode, prompt_file.read_text(encoding="utf-8"), current, error))
            repaired = debug_dir / source.name
            repaired.write_text(current + "\n", encoding="utf-8")
            output = debug_dir / "output"
            if mode == "protocol":
                ok, error = execute_protocol(current, output, task)
            else:
                ok, error, stdout, stderr = run_baseline_in_sandbox(
                    repaired, output, args.timeout, args.sandbox_image, task
                )
                (debug_dir / "execution.txt").write_text(f"STDOUT\n{stdout}\nSTDERR\n{stderr}", encoding="utf-8")
            (debug_dir / "result.txt").write_text(error, encoding="utf-8")
            if ok:
                status = "fixed"
                break
        results.append(score | {"status": status, "rounds_used": version, "final_output_dir": str(output)})
    path = timestamped_result_path("self_debug")
    write_json(path, results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
