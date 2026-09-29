from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import yaml
from pydantic import BaseModel, Field, create_model

from common.tool_categories import CategorySpec, category_specs


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


def _build_step_class(spec: CategorySpec) -> Type[Step]:
    """Create a ``Step`` subclass whose ``tool`` field is a Literal allow-list.

    The allow-list comes from ``common.tool_categories`` (builtin lists merged
    with ``extensions/registry.yaml``), so onboarding a tool into an existing
    category requires no source edit here.
    """

    if not spec.tools:
        # An empty Literal is invalid; a subagent with no tools yet keeps the
        # permissive `str` field it inherits from Step so the module still
        # imports. Validation still rejects unknown tools downstream.
        fields: Dict[str, Any] = {"tool": (str, ...)}
    else:
        fields = {"tool": (Literal[tuple(spec.tools)], ...)}  # type: ignore[valid-type]

    step_class = create_model(
        spec.step_class,
        __base__=Step,
        __module__=__name__,
        **fields,
    )
    step_class.__doc__ = f"Step emitted by the {spec.display_name} subagent."
    return step_class


CATEGORY_SPECS: Dict[str, CategorySpec] = category_specs()

STEP_CLASSES: Dict[str, Type[Step]] = {
    name: _build_step_class(spec) for name, spec in CATEGORY_SPECS.items()
}

# Bind every step class as a module-level attribute so both the five historical
# names and any user-onboarded ones are importable from `common.workflow`.
for _name, _spec in CATEGORY_SPECS.items():
    globals()[_spec.step_class] = STEP_CLASSES[_name]

# Explicit re-exports keep static analysers and existing imports happy.
DataDiscoveryStep: Type[Step] = STEP_CLASSES["data_discovery"]
SpatialTemporalExtractionStep: Type[Step] = STEP_CLASSES["extraction"]
DiagnosticsAndSkillMetricsStep: Type[Step] = STEP_CLASSES["diagnostics"]
BasinScaleWaterCycleSynthesisStep: Type[Step] = STEP_CLASSES["water_cycle"]
DiagnosticVisualizationStep: Type[Step] = STEP_CLASSES["visualization"]


def step_class_for(category: str) -> Type[Step]:
    """Return the Step subclass for a category name."""

    return STEP_CLASSES[category]


class SpatialBiasStep(Step):
    """Step emitted by the Spatial Bias Deep-Dive subagent.

    Focused on extracting gridded fields, computing bias and zonal statistics,
    and producing map-based visualisations for model-vs-observation comparisons.
    """

    tool: Literal[
        "extract_gridded_field",
        "compute_spatial_bias",
        "compute_zonal_stats",
        "plot_gridded_map",
        "plot_bias_comparison",
    ]


class ExtremesFlowStep(Step):
    """Step emitted by the Extreme Events / Flow Statistics subagent.

    Focused on extracting discharge time series, computing FDC and distributional
    metrics, and rendering FDC and time-series plots for drought/flood analysis.
    """

    tool: Literal[
        "extract_e3sm_timeseries",
        "extract_obs_timeseries",
        "match_to_grid",
        "compute_fdc_metrics",
        "compute_climatology",
        "compute_summary_stats",
        "plot_fdc",
        "plot_timeseries",
        "plot_basin_timeseries",
    ]


class ModelComparisonStep(Step):
    """Step emitted by the Cross-Case Model Comparison subagent.

    Focused on side-by-side comparison of two or more E3SM simulation cases
    using extraction, skill metrics, bias fields, and comparison plots.
    """

    tool: Literal[
        "extract_e3sm_timeseries",
        "extract_gridded_field",
        "extract_obs_timeseries",
        "match_to_grid",
        "compute_metrics",
        "compute_spatial_bias",
        "compute_summary_stats",
        "plot_bias_comparison",
        "plot_map",
        "plot_timeseries",
        "plot_scatter",
    ]


class ObsSurveyStep(Step):
    """Step emitted by the Observational Data Survey subagent.

    Focused on fetching remote observation datasets, loading gauge metadata,
    computing summary statistics, and producing overview scatter/map plots.
    """

    tool: Literal[
        "fetch_ilamb_data",
        "load_obs_metadata",
        "extract_obs_timeseries",
        "compute_summary_stats",
        "compute_climatology",
        "plot_scatter",
        "plot_map",
        "plot_gridded_map",
        "plot_timeseries",
    ]


class Workflow(BaseModel):
    """A complete ESMFlow workflow definition."""

    name: str
    description: str
    settings: Settings
    steps: List[Step]

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Return a plain dict matching workflow YAML expectations.

        ``exclude_none`` is intentionally False for the ``settings`` sub-model
        so that ``case_name: null`` is preserved in the serialised dict.  This
        ensures ``${settings.case_name}`` can always be resolved by the workflow
        runner — even when the user has not set ESFLOW_CASE_NAME in their .env
        (the runner will auto-detect it from E3SM filenames in that case).

        Other top-level None fields (e.g. optional step params) are still
        excluded to keep YAML output clean.
        """
        d = self.model_dump(exclude_none=False)
        d["steps"] = [step.model_dump(exclude_none=True) for step in self.steps]
        return d

    def write_to_file(self, path: Union[str, Path]) -> None:
        """Write the workflow YAML dict to a file.

        Args:
            path: File path where YAML will be written.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)