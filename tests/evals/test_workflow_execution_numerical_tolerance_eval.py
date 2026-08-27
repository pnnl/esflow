"""Deterministic import-time checks for evals/workflow_execution_numerical_tolerance_eval.py.

This module is a live-LLM eval harness (see AGENTS.md's "no live LLM calls
in tests/" policy), so run_task()/make_dataset() are never exercised here.
This only guards against a specific regression: the module previously
imported common.config.MODELS (a dict[str, Model]) and then immediately
shadowed it with its own module-level MODELS = [...] list of eval labels,
so MODELS[model] at the run_task() call site indexed a list with a string
key and raised TypeError as soon as an eval actually ran.
"""

from pydantic_ai.models import Model

from evals.workflow_execution_numerical_tolerance_eval import EVAL_MODELS, MODELS


def test_models_is_the_common_config_registry_not_a_list():
    assert isinstance(MODELS, dict)
    assert all(isinstance(value, Model) for value in MODELS.values())


def test_eval_models_are_valid_keys_in_the_models_registry():
    assert isinstance(EVAL_MODELS, list)
    for label in EVAL_MODELS:
        assert label in MODELS
