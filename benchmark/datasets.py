"""pydantic_evals Dataset/Case construction for the v2 benchmark.

Each (model, task) pair becomes a Case; ``--runs N`` maps onto pydantic_evals'
``repeat=N`` on ``Dataset.evaluate``/``evaluate_sync``, and grading happens
inline via the shared ``StructuralGrade`` evaluator (``benchmark/grading.py``)
instead of a second pass over JSON files.
"""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset

from agents.planner.oneshot_planner import plan_workflow_one_shot
from common.config import MODELS
from common.workflow import Settings
from common.workflow_runner import run_workflow_definition
from common.workflow_validation import validate_workflow

from benchmark.common import (
    BASELINE_OUTPUT_INSTRUCTION, BASELINES_DIR, ERROR_RE, NETWORK_REQUIRED_TASKS,
    REFERENCE_DIR, ROOT, SANDBOX_IMAGE, clean_artifact, output_dir, prompt_path, run_dir,
)
from benchmark.grading import StructuralGrade


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


def case_name(model: str, task: str, run: int) -> str:
    return f"{model}/{task}/run{run}"


def parse_case_name(name: str) -> tuple[str, str, int]:
    model, task, run = name.split("/")
    return model, task, int(run.removeprefix("run"))


async def run_protocol_case(label: str, *, skip_execution: bool = False) -> Path:
    """Plan, validate, and execute a protocol run. Returns the output dir.

    Catalog validation failures and runner rejections are *expected* LLM
    output-quality failure modes -- not exceptional -- so this returns
    out_dir normally for them (with the reason written to .error.txt) and
    lets StructuralGrade's has_deliverable() check grade the run crash,
    since no deliverable was ever produced. This keeps every run in
    report.cases, where make_manual_queue.py/merge_grades.py/
    self_debug_crashes.py look, instead of report.failures, which they
    don't.

    Only genuinely unexpected exceptions (e.g. an Agent.run() call failing
    due to a network/API error, or a bug in the harness itself) still
    propagate, landing in pydantic_evals' report.failures -- those are
    infra/tooling failures, not model-quality signal, and shouldn't be
    counted as a model "crash" in the benchmark stats.
    """
    model_name, task, run = parse_case_name(label)
    artifact = run_dir("protocol", model_name, task) / f"run{run}.yaml"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir("protocol", model_name, task, run)
    prompt = prompt_path("protocol", task).read_text(encoding="utf-8")
    settings = Settings(data_dir="./data/sample", output_dir=str(out_dir))

    try:
        workflow = await plan_workflow_one_shot(prompt, settings, MODELS[model_name])
        workflow.write_to_file(artifact)
        errors = validate_workflow(workflow.to_yaml_dict())
        if errors:
            artifact.with_suffix(".error.txt").write_text(
                "catalog validation failed: " + "; ".join(errors), encoding="utf-8"
            )
            return out_dir
        if skip_execution:
            return out_dir
        context = run_workflow_definition(workflow.to_yaml_dict(), verbose=False)
        if context is None:
            artifact.with_suffix(".error.txt").write_text("runner rejected workflow", encoding="utf-8")
            return out_dir
        step_errors = _collect_step_errors(context)
        if step_errors:
            artifact.with_suffix(".error.txt").write_text(step_errors, encoding="utf-8")
        return out_dir
    except Exception:
        artifact.with_suffix(".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


async def run_baseline_case(
    label: str, *, timeout: int = 300,
    sandbox_image: str = SANDBOX_IMAGE, skip_execution: bool = False,
) -> Path:
    """Generate, syntax/import-check, and sandbox-execute a baseline run.

    Returns the output dir. Syntax and import-resolution failures are
    *expected* LLM output-quality failure modes -- not exceptional -- so
    this returns out_dir normally for them (with the reason written to
    run{N}_execution.txt, the same file self_debug_crashes.py already
    reads for baseline-mode error lookup) and lets StructuralGrade's
    has_deliverable() check grade the run crash, since no deliverable was
    ever produced. This keeps every run in report.cases, where
    make_manual_queue.py/merge_grades.py/self_debug_crashes.py look,
    instead of report.failures, which they don't.

    Only genuinely unexpected exceptions still propagate, landing in
    pydantic_evals' report.failures -- those are infra/tooling failures,
    not model-quality signal.
    """
    model_name, task, run = parse_case_name(label)
    artifact = run_dir("baseline", model_name, task) / f"run{run}.py"
    raw = artifact.with_name(f"run{run}_raw.txt")
    logs = artifact.with_name(f"run{run}_execution.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir("baseline", model_name, task, run)

    try:
        raw_text = await generate_baseline(model_name, task)
        raw.write_text(raw_text, encoding="utf-8")
        code = clean_artifact(raw_text, "python")
        artifact.write_text(code + "\n", encoding="utf-8")

        ok, msg = score_python_s0(code)
        if not ok:
            logs.write_text(f"syntax check failed: {msg}", encoding="utf-8")
            return out_dir
        ok, msg = score_python_s1(code)
        if not ok:
            logs.write_text(f"import check failed: {msg}", encoding="utf-8")
            return out_dir

        if skip_execution:
            return out_dir
        ok, msg, stdout, stderr = run_baseline_in_sandbox(
            artifact, out_dir, timeout, sandbox_image, task
        )
        logs.write_text(f"STDOUT\n{stdout}\nSTDERR\n{stderr}", encoding="utf-8")
        return out_dir
    except Exception:
        raw.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def make_dataset(mode: str, models: list[str], tasks: list[str], runs: int) -> Dataset[str, Path]:
    """Build a Dataset of (model, task, run) Cases graded by StructuralGrade."""
    cases: list[Case[str, Path, None]] = []
    for model in models:
        for task in tasks:
            for run in range(1, runs + 1):
                name = case_name(model, task, run)
                cases.append(Case(
                    name=name,
                    inputs=name,
                    evaluators=[StructuralGrade(mode=mode, task=task, reference_dir=REFERENCE_DIR)],
                ))
    return Dataset(name=f"{mode}_benchmark", cases=cases)
