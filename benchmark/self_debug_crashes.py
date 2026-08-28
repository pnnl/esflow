#!/usr/bin/env python3
"""Retry crashed benchmark artifacts with traceback-driven repairs.

Reuses the same StructuralGrade evaluator each round (via a small per-round
Dataset) instead of re-implementing has_deliverable/protocol_matches_reference
inline -- one grading implementation for the main run and every self-debug
round alike.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import EvaluationReportAdapter

from benchmark.common import (
    BASELINE_OUTPUT_INSTRUCTION, BASELINES_DIR, REFERENCE_DIR, RESULTS_DIR,
    clean_artifact, prompt_path,
)
from benchmark.datasets import _collect_step_errors, parse_case_name, run_baseline_in_sandbox
from benchmark.grading import StructuralGrade, has_deliverable
from common.config import MODELS
from common.config import load_prompt
from common.workflow_runner import run_workflow_definition
from common.workflow_validation import validate_workflow


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


def validate_args(args: argparse.Namespace) -> None:
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be positive")


def repair_prompt_file(mode: str, task: str) -> Path:
    """Resolve the on-disk task prompt file for a crashed run's repair loop.

    Delegates to benchmark.common.prompt_path() -- the same helper
    run_benchmark.py uses for initial generation -- instead of constructing
    the path inline, since the on-disk directory for baseline prompts is
    benchmark/baselines/ (plural), not benchmark/baseline/.
    """
    return prompt_path(mode, task)


def _crashed_cases(report, mode: str) -> list[tuple[str, str, int, Path]]:
    """Return (model, task, run, output_dir) for every crashed case in report."""
    crashed = []
    for case in report.cases:
        score = case.scores.get("StructuralGrade")
        if score is None or not (score.reason or "").startswith("crash:"):
            continue
        model, task, run = parse_case_name(case.name)
        crashed.append((model, task, run, Path(case.output)))
    return crashed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="{mode}_report.json")
    parser.add_argument("--mode", required=True, choices=("protocol", "baseline"))
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sandbox-image", default="esflow-v2-benchmark-baseline")
    args = parser.parse_args()
    validate_args(args)

    report = EvaluationReportAdapter.validate_json(args.report.read_bytes())
    results = []
    for model, task, run, out_dir in _crashed_cases(report, args.mode):
        source_dir = out_dir.parent
        source = source_dir / (f"run{run}.yaml" if args.mode == "protocol" else f"run{run}.py")
        if not source.exists():
            results.append({
                "mode": args.mode, "model": model, "task": task, "run": run,
                "status": "skip", "reason": "source artifact missing",
            })
            continue
        prompt_file = repair_prompt_file(args.mode, task)
        current = source.read_text(encoding="utf-8")
        error_file = (
            source.with_suffix(".error.txt") if args.mode == "protocol"
            else source.with_name(f"run{run}_execution.txt")
        )
        error = error_file.read_text(encoding="utf-8") if error_file.exists() else "unknown error"
        status = "exhausted"
        output = out_dir
        for version in range(1, args.max_rounds + 1):
            debug_dir = source_dir / f"run{run}_debug" / f"v{version}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            current = asyncio.run(
                repair(model, args.mode, prompt_file.read_text(encoding="utf-8"), current, error)
            )
            repaired = debug_dir / source.name
            repaired.write_text(current + "\n", encoding="utf-8")
            output = debug_dir / "output"
            if args.mode == "protocol":
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

        # Re-grade the final attempt with the same StructuralGrade evaluator
        # used for the main run, instead of reimplementing crash/success
        # detection inline.
        case_id = f"{model}/{task}/run{run}"
        dataset = Dataset(
            name="self_debug",
            cases=[Case(
                name=case_id, inputs=case_id,
                evaluators=[StructuralGrade(mode=args.mode, task=task, reference_dir=REFERENCE_DIR)],
            )],
        )
        regraded = dataset.evaluate_sync(lambda _label, output=output: output, progress=False)
        score = regraded.cases[0].scores["StructuralGrade"]
        results.append({
            "mode": args.mode, "model": model, "task": task, "run": run,
            "status": status, "rounds_used": version, "final_output_dir": str(output),
            "selfdebug_auto_grade": (score.reason or "").split(":", 1)[0].strip(),
            "selfdebug_reason": score.reason,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"self_debug_{args.mode}.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
