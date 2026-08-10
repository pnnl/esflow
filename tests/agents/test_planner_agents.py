from pydantic_ai.models.test import TestModel

from agents.planner.multistep_planner import planner
from agents.planner.multistep_planner_executor import plan_with_multistep_planner
from agents.planner.oneshot_planner import oneshot_planner, plan_workflow_one_shot
from agents.planner.oneshot_planner_executor import plan_with_planner
from common.workflow import Settings, Workflow


async def test_oneshot_planner_returns_a_structured_workflow_with_test_model(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "output"))

    with oneshot_planner.override(model=TestModel()):
        workflow = await plan_workflow_one_shot("Create a workflow", settings)

    assert isinstance(workflow, Workflow)
    assert isinstance(workflow.steps, list)


async def test_multistep_planner_runs_with_test_model(run_context):
    with planner.override(model=TestModel()):
        result = await planner.run("Help me plan", deps=run_context.deps)

    assert isinstance(result.output, (Workflow, str))


async def test_oneshot_executor_planning_wrapper_updates_workflow_state(run_context):
    with oneshot_planner.override(model=TestModel()):
        message = await plan_with_planner(run_context, "Create a workflow")

    assert message.startswith("Planner updated the workflow to ")
    assert isinstance(run_context.deps.workflow, Workflow)


async def test_multistep_executor_planning_wrapper_returns_agent_response(run_context):
    with planner.override(model=TestModel()):
        message = await plan_with_multistep_planner(run_context, "Help me plan")

    assert isinstance(message, str)
