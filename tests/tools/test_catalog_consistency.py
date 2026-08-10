from pathlib import Path
from typing import get_args, get_type_hints

import yaml

from common.workflow import (
    BasinScaleWaterCycleSynthesisStep,
    DataDiscoveryStep,
    DiagnosticVisualizationStep,
    DiagnosticsAndSkillMetricsStep,
    SpatialTemporalExtractionStep,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_workflow_step_tools_are_registered_in_the_catalog():
    with (REPO_ROOT / "tools" / "tool_catalog.yaml").open() as stream:
        catalog_tools = {tool["name"] for tool in yaml.safe_load(stream)["tools"]}

    step_classes = [
        DataDiscoveryStep,
        SpatialTemporalExtractionStep,
        DiagnosticsAndSkillMetricsStep,
        BasinScaleWaterCycleSynthesisStep,
        DiagnosticVisualizationStep,
    ]
    workflow_tools = {
        tool
        for step_class in step_classes
        for tool in get_args(get_type_hints(step_class)["tool"])
    }

    assert workflow_tools <= catalog_tools
