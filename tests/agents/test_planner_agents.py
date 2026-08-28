from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.test import TestModel

from agents.planner.multistep_planner import planner
from agents.planner.multistep_planner_executor import plan_with_multistep_planner
from agents.planner.oneshot_planner import oneshot_planner, plan_workflow_one_shot
from agents.planner.oneshot_planner_executor import plan_with_planner
from common.workflow import Settings, Workflow


def _fake_workflow(**overrides) -> Workflow:
    defaults = dict(
        name="model workflow", description="d",
        settings=Settings(data_dir="WRONG_DATA_DIR", output_dir="WRONG_OUTPUT_DIR"),
        steps=[],
    )
    defaults.update(overrides)
    return Workflow(**defaults)


async def test_oneshot_planner_returns_a_structured_workflow_with_test_model(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output"))

    with oneshot_planner.override(model=TestModel()):
        workflow = await plan_workflow_one_shot("Create a workflow", settings)

    assert isinstance(workflow, Workflow)
    assert isinstance(workflow.steps, list)


async def test_plan_workflow_one_shot_preserves_caller_settings_over_model_output(tmp_path):
    """The model's final structured-output turn constructs a fresh Workflow and
    is free to rewrite `settings` from the task text (e.g. a literal "save to
    ./output/foo" instruction in the prompt) instead of echoing the settings
    it was actually given. TestModel's default structured-output stub fills
    Settings.data_dir/output_dir with an arbitrary placeholder, exactly this
    failure mode -- this regression-tests that plan_workflow_one_shot forces
    the caller's settings back on before returning, regardless of what the
    model's own output claims."""
    settings = Settings(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output"))

    workflow = await plan_workflow_one_shot("Create a workflow", settings, TestModel())

    assert workflow.settings == settings


async def test_multistep_planner_runs_with_test_model(run_context):
    with planner.override(model=TestModel()):
        result = await planner.run("Help me plan", deps=run_context.deps)

    assert isinstance(result.output, (Workflow, str))


async def test_oneshot_executor_planning_wrapper_updates_workflow_state(run_context):
    with oneshot_planner.override(model=TestModel()):
        message = await plan_with_planner(run_context, "Create a workflow")

    assert message.startswith("Planner updated the workflow to ")
    assert isinstance(run_context.deps.workflow, Workflow)


async def test_plan_with_planner_preserves_session_settings_over_model_output(monkeypatch, run_context):
    """Same failure mode as plan_workflow_one_shot(), but for the interactive
    planner-executor's tool wrapper: ctx.deps.workflow gets wholesale-replaced
    by the model's fresh Workflow, so the session's actual settings must be
    reattached before the replacement, not left to whatever the model wrote.

    Monkeypatches oneshot_planner.run() directly (rather than .override(),
    which doesn't take effect for this agent's subagent-tool-calling path in
    this environment) to return a fabricated Workflow with different settings.
    """
    original_settings = run_context.deps.workflow.settings

    async def fake_run(*args, **kwargs):
        return AgentRunResult(output=_fake_workflow())

    monkeypatch.setattr(oneshot_planner, "run", fake_run)
    await plan_with_planner(run_context, "Create a workflow")

    assert run_context.deps.workflow.settings == original_settings


async def test_multistep_executor_planning_wrapper_returns_agent_response(run_context):
    with planner.override(model=TestModel()):
        message = await plan_with_multistep_planner(run_context, "Help me plan")

    assert isinstance(message, str)


async def test_plan_with_multistep_planner_preserves_session_settings_over_model_output(monkeypatch, run_context):
    """Same failure mode as the one-shot variants, for the multistep
    planner-executor's tool wrapper."""
    original_settings = run_context.deps.workflow.settings

    async def fake_run(*args, **kwargs):
        return AgentRunResult(output=_fake_workflow())

    monkeypatch.setattr(planner, "run", fake_run)
    await plan_with_multistep_planner(run_context, "Help me plan")

    assert run_context.deps.workflow.settings == original_settings
