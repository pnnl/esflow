"""Deterministic tests for benchmark/run_benchmark.py's validate_args() --
specifically the --cases flag added to retry an explicit set of
(model, task, run) cases (e.g. ones that landed in a prior chunk's
report.failures) without redoing an entire already-mostly-successful chunk.

No Docker or live LLM calls anywhere in this module.
"""

import pytest

from benchmark.common import PILOT_MODELS, TASKS
from benchmark.run_benchmark import parser, validate_args


def _parse(argv):
    return parser().parse_args(argv)


# ---------------------------------------------------------------------------
# --cases alone is accepted
# ---------------------------------------------------------------------------
def test_cases_alone_is_accepted():
    args = _parse([
        "--mode", "single_agent",
        "--cases", "Gemini 3.7 Flash/task_06_water_balance/run1",
    ])
    validate_args(args)  # must not raise


def test_multiple_cases_alone_is_accepted():
    args = _parse([
        "--mode", "protocol",
        "--cases",
        "GPT 5.6 Terra/task_04_streamflow_fdc/run2",
        "Claude Sonnet 5/task_05_basin_streamflow/run3",
    ])
    validate_args(args)  # must not raise


# ---------------------------------------------------------------------------
# --cases combined with an explicit --models/--tasks/--runs is a conflict
# ---------------------------------------------------------------------------
def test_cases_with_explicit_models_conflicts():
    args = _parse([
        "--mode", "single_agent",
        "--cases", "Gemini 3.7 Flash/task_06_water_balance/run1",
        "--models", "Gemini 3.7 Flash",
    ])
    with pytest.raises(SystemExit, match="--cases cannot be combined"):
        validate_args(args)


def test_cases_with_explicit_tasks_conflicts():
    args = _parse([
        "--mode", "single_agent",
        "--cases", "Gemini 3.7 Flash/task_06_water_balance/run1",
        "--tasks", "task_06_water_balance",
    ])
    with pytest.raises(SystemExit, match="--cases cannot be combined"):
        validate_args(args)


def test_cases_with_explicit_runs_conflicts():
    args = _parse([
        "--mode", "single_agent",
        "--cases", "Gemini 3.7 Flash/task_06_water_balance/run1",
        "--runs", "4",
    ])
    with pytest.raises(SystemExit, match="--cases cannot be combined"):
        validate_args(args)


# ---------------------------------------------------------------------------
# --cases with a malformed or unknown model/task name
# ---------------------------------------------------------------------------
def test_cases_with_malformed_name_rejected():
    args = _parse(["--mode", "single_agent", "--cases", "not-a-valid-case-name"])
    with pytest.raises(SystemExit, match="--cases must be exact"):
        validate_args(args)


def test_cases_with_unknown_model_rejected():
    args = _parse([
        "--mode", "single_agent",
        "--cases", "Totally Fake Model/task_01_obs_summary/run1",
    ])
    with pytest.raises(SystemExit, match="--cases must be exact"):
        validate_args(args)


def test_cases_with_unknown_task_rejected():
    args = _parse([
        "--mode", "single_agent",
        "--cases", "Gemini 3.7 Flash/task_99_not_real/run1",
    ])
    with pytest.raises(SystemExit, match="--cases must be exact"):
        validate_args(args)


# ---------------------------------------------------------------------------
# Existing --models/--tasks/--runs validation is unaffected when --cases is
# not passed
# ---------------------------------------------------------------------------
def test_default_args_without_cases_still_validate():
    args = _parse(["--mode", "protocol"])
    assert args.models == PILOT_MODELS
    assert args.tasks == TASKS
    validate_args(args)  # must not raise


def test_unknown_model_without_cases_still_rejected():
    args = _parse(["--mode", "protocol", "--models", "Totally Fake Model"])
    with pytest.raises(SystemExit, match="unknown v2 model labels"):
        validate_args(args)


def test_zero_runs_without_cases_still_rejected():
    args = _parse(["--mode", "protocol", "--runs", "0"])
    with pytest.raises(SystemExit, match="--runs must be positive"):
        validate_args(args)
