"""Deterministic tests for run_protocol_case()/run_baseline_case() in
benchmark/datasets.py -- specifically the fix for a bug where "expected"
LLM output-quality failure modes (invalid workflow YAML failing catalog
validation, a runner rejecting the workflow, generated Python failing to
compile or resolve its imports) were raised as exceptions and landed in
pydantic_evals' report.failures, which make_manual_queue.py,
merge_grades.py, and self_debug_crashes.py never inspect -- silently
dropping those runs from every downstream grading/review/retry step.

These functions now return out_dir normally for those four failure modes
(writing the same diagnostic text an operator would have seen before) so
the run stays in report.cases and StructuralGrade's has_deliverable()
check grades it "crash" via the normal path. Only genuinely unexpected
exceptions (e.g. Agent.run()/plan_workflow_one_shot() itself raising)
should still propagate.

No Docker or live LLM calls anywhere in this module -- plan_workflow_one_shot()
and generate_baseline() are monkeypatched with plain async stand-ins.
"""

from pathlib import Path

import pytest

import benchmark.datasets as datasets
from common.workflow import Settings, Workflow


MODEL_NAME = "GPT 5.4"
TASK = "task_01_obs_summary"


def _fake_workflow(tmp_path: Path) -> Workflow:
    return Workflow(
        name="fake",
        description="fake workflow for tests",
        settings=Settings(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output")),
        steps=[],
    )


@pytest.fixture(autouse=True)
def _isolate_results_dir(monkeypatch, tmp_path):
    """Route every path helper at a tmp_path so no test touches real results/."""
    monkeypatch.setattr("benchmark.common.RESULTS_DIR", tmp_path)


# ---------------------------------------------------------------------------
# run_protocol_case() -- catalog validation failure
# ---------------------------------------------------------------------------
async def test_run_protocol_case_returns_normally_on_catalog_validation_failure(monkeypatch, tmp_path):
    async def fake_plan(prompt, settings, model_override=None):
        return _fake_workflow(tmp_path)

    monkeypatch.setattr(datasets, "plan_workflow_one_shot", fake_plan)
    monkeypatch.setattr(datasets, "validate_workflow", lambda workflow_dict: ["unknown tool: bogus_tool"])

    label = f"{MODEL_NAME}/{TASK}/run1"
    out_dir = await datasets.run_protocol_case(label, skip_execution=True)

    from benchmark.common import run_dir
    artifact = run_dir("protocol", MODEL_NAME, TASK) / "run1.yaml"
    error_text = artifact.with_suffix(".error.txt").read_text(encoding="utf-8")
    assert "catalog validation failed" in error_text
    assert "unknown tool: bogus_tool" in error_text
    assert out_dir == datasets.output_dir("protocol", MODEL_NAME, TASK, 1)


# ---------------------------------------------------------------------------
# run_protocol_case() -- runner rejection
# ---------------------------------------------------------------------------
async def test_run_protocol_case_returns_normally_when_runner_rejects_workflow(monkeypatch, tmp_path):
    async def fake_plan(prompt, settings, model_override=None):
        return _fake_workflow(tmp_path)

    monkeypatch.setattr(datasets, "plan_workflow_one_shot", fake_plan)
    monkeypatch.setattr(datasets, "validate_workflow", lambda workflow_dict: [])
    monkeypatch.setattr(datasets, "run_workflow_definition", lambda workflow_dict, verbose=False: None)

    label = f"{MODEL_NAME}/{TASK}/run2"
    out_dir = await datasets.run_protocol_case(label)

    from benchmark.common import run_dir
    artifact = run_dir("protocol", MODEL_NAME, TASK) / "run2.yaml"
    error_text = artifact.with_suffix(".error.txt").read_text(encoding="utf-8")
    assert "runner rejected workflow" in error_text
    assert out_dir == datasets.output_dir("protocol", MODEL_NAME, TASK, 2)


# ---------------------------------------------------------------------------
# run_protocol_case() -- unexpected exceptions still propagate
# ---------------------------------------------------------------------------
async def test_run_protocol_case_still_raises_on_unexpected_exception(monkeypatch):
    async def fake_plan(prompt, settings, model_override=None):
        raise ConnectionError("simulated network failure talking to the model gateway")

    monkeypatch.setattr(datasets, "plan_workflow_one_shot", fake_plan)

    label = f"{MODEL_NAME}/{TASK}/run3"
    with pytest.raises(ConnectionError, match="simulated network failure"):
        await datasets.run_protocol_case(label)


# ---------------------------------------------------------------------------
# run_baseline_case() -- syntax failure
# ---------------------------------------------------------------------------
async def test_run_baseline_case_returns_normally_on_syntax_failure(monkeypatch):
    async def fake_generate(model_name, task):
        return "def broken(:\n    pass\n"

    monkeypatch.setattr(datasets, "generate_baseline", fake_generate)

    label = f"{MODEL_NAME}/{TASK}/run1"
    out_dir = await datasets.run_baseline_case(label, skip_execution=True)

    from benchmark.common import run_dir
    artifact = run_dir("baseline", MODEL_NAME, TASK) / "run1.py"
    logs = artifact.with_name("run1_execution.txt")
    assert "syntax check failed" in logs.read_text(encoding="utf-8")
    assert out_dir == datasets.output_dir("baseline", MODEL_NAME, TASK, 1)


# ---------------------------------------------------------------------------
# run_baseline_case() -- unresolvable import
# ---------------------------------------------------------------------------
async def test_run_baseline_case_returns_normally_on_unresolvable_import(monkeypatch):
    async def fake_generate(model_name, task):
        return "import this_module_definitely_does_not_exist_xyz\n"

    monkeypatch.setattr(datasets, "generate_baseline", fake_generate)

    label = f"{MODEL_NAME}/{TASK}/run2"
    out_dir = await datasets.run_baseline_case(label, skip_execution=True)

    from benchmark.common import run_dir
    artifact = run_dir("baseline", MODEL_NAME, TASK) / "run2.py"
    logs = artifact.with_name("run2_execution.txt")
    log_text = logs.read_text(encoding="utf-8")
    assert "import check failed" in log_text
    assert out_dir == datasets.output_dir("baseline", MODEL_NAME, TASK, 2)


# ---------------------------------------------------------------------------
# run_baseline_case() -- unexpected exceptions still propagate
# ---------------------------------------------------------------------------
async def test_run_baseline_case_still_raises_on_unexpected_exception(monkeypatch):
    async def fake_generate(model_name, task):
        raise ConnectionError("simulated network failure talking to the model gateway")

    monkeypatch.setattr(datasets, "generate_baseline", fake_generate)

    label = f"{MODEL_NAME}/{TASK}/run3"
    with pytest.raises(ConnectionError, match="simulated network failure"):
        await datasets.run_baseline_case(label)


# ---------------------------------------------------------------------------
# run_baseline_case() -- sandbox timeout/exit-code reason must survive into
# run{N}_execution.txt even when stdout/stderr are empty, since
# self_debug_crashes.py reads exactly this file as the repair-model error
# signal (a prior version of run_baseline_case() dropped this "msg" value
# entirely, silently discarding e.g. "timeout after 300s").
# ---------------------------------------------------------------------------
async def test_run_baseline_case_preserves_sandbox_timeout_reason_with_empty_output(monkeypatch):
    async def fake_generate(model_name, task):
        return "print('hello')\n"

    def fake_sandbox(script, out_dir, timeout, image, task):
        return False, "timeout after 300s", "", ""

    monkeypatch.setattr(datasets, "generate_baseline", fake_generate)
    monkeypatch.setattr(datasets, "run_baseline_in_sandbox", fake_sandbox)

    label = f"{MODEL_NAME}/{TASK}/run4"
    out_dir = await datasets.run_baseline_case(label)

    from benchmark.common import run_dir
    artifact = run_dir("baseline", MODEL_NAME, TASK) / "run4.py"
    logs = artifact.with_name("run4_execution.txt")
    log_text = logs.read_text(encoding="utf-8")
    assert "timeout after 300s" in log_text
    assert out_dir == datasets.output_dir("baseline", MODEL_NAME, TASK, 4)
