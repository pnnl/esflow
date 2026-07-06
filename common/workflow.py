from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Settings configuration for a workflow."""

    case_name: Optional[str] = None
    data_dir: str
    output_dir: str


class Step(BaseModel):
    """A single step in a workflow."""

    id: str
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)


class DataDiscoveryStep(Step):
    """Step emitted by the ESM Data Discovery and Intake subagent."""

    tool: Literal["fetch_ilamb_data", "load_obs_metadata"]


class SpatialTemporalExtractionStep(Step):
    """Step emitted by the ESM Spatial-Temporal Extraction subagent."""

    tool: Literal[
        "match_to_grid",
        "extract_e3sm_timeseries",
        "extract_obs_timeseries",
        "extract_gridded_field",
        "extract_basin_mean",
    ]


class DiagnosticsAndSkillMetricsStep(Step):
    """Step emitted by the ESM Diagnostics and Skill Metrics subagent."""

    tool: Literal[
        "compute_climatology",
        "compute_summary_stats",
        "compute_metrics",
        "compute_fdc_metrics",
        "compute_spatial_bias",
        "compute_zonal_stats",
    ]


class BasinScaleWaterCycleSynthesisStep(Step):
    """Step emitted by the Basin-Scale Water Cycle Synthesis subagent."""

    tool: Literal[
        "compute_basin_budget",
        "extract_basin_mean",
        "compute_fdc_metrics",
        "compute_spatial_bias",
    ]


class DiagnosticVisualizationStep(Step):
    """Step emitted by the ESM Diagnostic Visualization subagent."""

    tool: Literal[
        "plot_map",
        "plot_gridded_map",
        "plot_timeseries",
        "plot_scatter",
        "plot_fdc",
        "plot_basin_timeseries",
        "plot_water_balance_basins",
        "plot_basin_budget_comparison",
        "plot_basin_radar",
        "plot_bias_comparison",
    ]


class Workflow(BaseModel):
    """A complete ESMFlow workflow definition."""

    name: str
    description: str
    settings: Settings
    steps: List[Step]

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Return a plain dict matching workflow YAML expectations."""

        return self.model_dump(exclude_none=True)

    def write_to_file(self, path: Union[str, Path]) -> None:
        """Write the workflow YAML dict to a file.

        Args:
            path: File path where YAML will be written.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)