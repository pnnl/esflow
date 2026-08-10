import yaml
import pytest
from pydantic import ValidationError

from common.workflow import (
    BasinScaleWaterCycleSynthesisStep,
    DataDiscoveryStep,
    DiagnosticVisualizationStep,
    DiagnosticsAndSkillMetricsStep,
    Settings,
    SpatialTemporalExtractionStep,
    Workflow,
)


@pytest.mark.parametrize(
    ("step_class", "valid_tool"),
    [
        (DataDiscoveryStep, "load_obs_metadata"),
        (SpatialTemporalExtractionStep, "extract_obs_timeseries"),
        (DiagnosticsAndSkillMetricsStep, "compute_summary_stats"),
        (BasinScaleWaterCycleSynthesisStep, "compute_basin_budget"),
        (DiagnosticVisualizationStep, "plot_timeseries"),
    ],
)
def test_step_categories_enforce_tool_allow_lists(step_class, valid_tool):
    assert step_class(id="step", tool=valid_tool).tool == valid_tool
    with pytest.raises(ValidationError):
        step_class(id="step", tool="not_allowed")


def test_workflow_serializes_and_writes_yaml(tmp_path):
    workflow = Workflow(
        name="Example",
        description="A test workflow",
        settings=Settings(data_dir="data", output_dir="output"),
        steps=[DataDiscoveryStep(id="metadata", tool="load_obs_metadata")],
    )

    assert workflow.to_yaml_dict()["settings"] == {"data_dir": "data", "output_dir": "output"}

    destination = tmp_path / "nested" / "workflow.yaml"
    workflow.write_to_file(destination)
    with destination.open() as stream:
        assert yaml.safe_load(stream) == workflow.to_yaml_dict()
