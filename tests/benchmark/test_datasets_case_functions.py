"""Deterministic tests for run_protocol_case()/run_single_agent_case() in
benchmark/datasets.py -- specifically the fix for a bug where "expected"
LLM output-quality failure modes (invalid workflow YAML failing catalog
validation, a runner rejecting the workflow) were raised as exceptions and
landed in pydantic_evals' report.failures, which make_manual_queue.py,
merge_grades.py, and self_debug_crashes.py never inspect -- silently
dropping those runs from every downstream grading/review/retry step.

Both functions now return out_dir normally for those failure modes
(writing the same diagnostic text an operator would have seen before) so
the run stays in report.cases and StructuralGrade's has_deliverable()
check grades it "crash" via the normal path. Only genuinely unexpected
exceptions (e.g. Agent.run()/the planner call itself raising) should still
propagate.

No Docker or live LLM calls anywhere in this module -- the planner
functions are monkeypatched with plain async stand-ins.
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


@pytest.fixture(autouse=True)
def _patch_planners(monkeypatch, tmp_path):
    """Default both arms' planners to a canned Workflow; overridden per-test."""
    async def fake_plan(prompt, settings, model_override=None):
        return _fake_workflow(tmp_path)

    monkeypatch.setitem(datasets.PLANNERS, "protocol", fake_plan)
    monkeypatch.setitem(datasets.PLANNERS, "single_agent", fake_plan)


# ---------------------------------------------------------------------------
# run_protocol_case() / run_single_agent_case() -- catalog validation failure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,run_case", [
    ("protocol", datasets.run_protocol_case),
    ("single_agent", datasets.run_single_agent_case),
])
async def test_run_case_returns_normally_on_catalog_validation_failure(monkeypatch, mode, run_case):
    monkeypatch.setattr(datasets, "validate_workflow", lambda workflow_dict: ["unknown tool: bogus_tool"])

    label = f"{MODEL_NAME}/{TASK}/run1"
    out_dir = await run_case(label, skip_execution=True)

    from benchmark.common import run_dir
    artifact = run_dir(mode, MODEL_NAME, TASK) / "run1.yaml"
    error_text = artifact.with_suffix(".error.txt").read_text(encoding="utf-8")
    assert "catalog validation failed" in error_text
    assert "unknown tool: bogus_tool" in error_text
    assert out_dir == datasets.output_dir(mode, MODEL_NAME, TASK, 1)


# ---------------------------------------------------------------------------
# run_protocol_case() / run_single_agent_case() -- runner rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,run_case", [
    ("protocol", datasets.run_protocol_case),
    ("single_agent", datasets.run_single_agent_case),
])
async def test_run_case_returns_normally_when_runner_rejects_workflow(monkeypatch, mode, run_case):
    monkeypatch.setattr(datasets, "validate_workflow", lambda workflow_dict: [])
    monkeypatch.setattr(datasets, "run_workflow_definition", lambda workflow_dict, verbose=False: None)

    label = f"{MODEL_NAME}/{TASK}/run2"
    out_dir = await run_case(label)

    from benchmark.common import run_dir
    artifact = run_dir(mode, MODEL_NAME, TASK) / "run2.yaml"
    error_text = artifact.with_suffix(".error.txt").read_text(encoding="utf-8")
    assert "runner rejected workflow" in error_text
    assert out_dir == datasets.output_dir(mode, MODEL_NAME, TASK, 2)


# ---------------------------------------------------------------------------
# run_protocol_case() / run_single_agent_case() -- unexpected exceptions
# still propagate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,run_case", [
    ("protocol", datasets.run_protocol_case),
    ("single_agent", datasets.run_single_agent_case),
])
async def test_run_case_still_raises_on_unexpected_exception(monkeypatch, mode, run_case):
    async def fake_plan(prompt, settings, model_override=None):
        raise ConnectionError("simulated network failure talking to the model gateway")

    monkeypatch.setitem(datasets.PLANNERS, mode, fake_plan)

    label = f"{MODEL_NAME}/{TASK}/run3"
    with pytest.raises(ConnectionError, match="simulated network failure"):
        await run_case(label)


# ---------------------------------------------------------------------------
# make_dataset_from_cases() -- builds a Dataset from explicit case names
# instead of the full models x tasks x runs cross product, so a chunk's
# report.failures can be retried by name alone (see
# benchmark/RUNBOOK.md "Retrying a partial chunk failure").
# ---------------------------------------------------------------------------
def test_make_dataset_from_cases_builds_exactly_the_requested_cases():
    names = [
        "GPT 5.4/task_01_obs_summary/run2",
        "Claude Opus 4.8/task_05_basin_streamflow/run1",
    ]
    dataset = datasets.make_dataset_from_cases("protocol", names)

    assert [c.name for c in dataset.cases] == names
    assert [c.inputs for c in dataset.cases] == names


def test_make_dataset_from_cases_preserves_order():
    names = [
        "Claude Opus 4.8/task_02_seasonal_runoff/run3",
        "GPT 5.4/task_01_obs_summary/run1",
    ]
    dataset = datasets.make_dataset_from_cases("single_agent", names)

    assert [c.name for c in dataset.cases] == names


def test_make_dataset_from_cases_rejects_duplicate_case_name():
    """Dataset itself rejects duplicate case names (pydantic_evals.Dataset's
    own constructor check) -- make_dataset_from_cases doesn't need its own
    dedup logic, but a caller passing the same case twice should still get a
    clear error rather than a silently-dropped duplicate."""
    names = [
        "GPT 5.4/task_01_obs_summary/run1",
        "GPT 5.4/task_01_obs_summary/run1",
    ]
    with pytest.raises(ValueError, match="Duplicate case name"):
        datasets.make_dataset_from_cases("single_agent", names)


def test_make_dataset_from_cases_uses_the_evaluator_for_the_cases_task():
    from benchmark.grading import StructuralGrade

    names = ["GPT 5.4/task_03_et_benchmark/run4"]
    dataset = datasets.make_dataset_from_cases("protocol", names)

    [case] = dataset.cases
    [evaluator] = case.evaluators
    assert isinstance(evaluator, StructuralGrade)
    assert evaluator.task == "task_03_et_benchmark"
    assert evaluator.mode == "protocol"


def test_make_dataset_from_cases_rejects_malformed_case_name():
    with pytest.raises(ValueError):
        datasets.make_dataset_from_cases("protocol", ["not-a-valid-case-name"])
