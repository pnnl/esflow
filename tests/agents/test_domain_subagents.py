import pytest
from pydantic_ai.models.test import TestModel

from agents.domain import data_discovery, diagnostics, extraction, visualization, water_cycle
from common.workflow import (
    BasinScaleWaterCycleSynthesisStep,
    DataDiscoveryStep,
    DiagnosticVisualizationStep,
    DiagnosticsAndSkillMetricsStep,
    SpatialTemporalExtractionStep,
)


@pytest.mark.parametrize(
    ("module", "function_name", "step_class"),
    [
        (data_discovery, "call_data_discovery", DataDiscoveryStep),
        (extraction, "call_extraction", SpatialTemporalExtractionStep),
        (diagnostics, "call_diagnostics", DiagnosticsAndSkillMetricsStep),
        (water_cycle, "call_water_cycle_synthesis", BasinScaleWaterCycleSynthesisStep),
        (visualization, "call_visualization", DiagnosticVisualizationStep),
    ],
)
async def test_domain_subagents_extend_the_shared_workflow(
    run_context, module, function_name, step_class
):
    function = getattr(module, function_name)

    with module.agent.override(model=TestModel()):
        message = await function(run_context, "Add one step")

    assert message.startswith("Added 1 ")
    assert len(run_context.deps.workflow.steps) == 1
    assert isinstance(run_context.deps.workflow.steps[0], step_class)
