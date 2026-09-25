"""Single source of truth for planner categories (subagents) and their tools.

Historically the allow-list of tool names a subagent may emit lived as a
hand-written ``Literal[...]`` on a ``Step`` subclass in ``common/workflow.py``.
That made adding a capability a three-place source edit (tool module, generated
catalog, ``Literal``). This module turns the third place into *data*:

* ``BUILTIN_CATEGORIES`` holds the five shipped categories verbatim.
* ``extensions/registry.yaml`` holds capabilities and subagents added by the
  onboarding agent.

``category_specs()`` merges the two and is consumed by
``common/workflow.py`` (to build the ``Step`` subclasses dynamically) and by
``agents/domain/extensions.py`` (to build subagents for new categories).

Consequence: onboarding a tool into a *builtin* category needs no code edit at
all -- the ``Literal`` widens on the next process start.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "extensions" / "registry.yaml"


@dataclass(frozen=True)
class CategorySpec:
    """A planner category: one subagent, one Step subclass, one tool allow-list."""

    name: str
    step_class: str
    display_name: str
    result_label: str
    call_tool_name: str
    description: str
    tools: Tuple[str, ...] = ()
    builtin: bool = True


BUILTIN_CATEGORIES: Tuple[CategorySpec, ...] = (
    CategorySpec(
        name="data_discovery",
        step_class="DataDiscoveryStep",
        display_name="ESM Data Discovery and Intake",
        result_label="data discovery",
        call_tool_name="call_data_discovery",
        description=(
            "Delegate observational/reference dataset discovery and intake "
            "(ILAMB fetches, gauge metadata) to the data discovery subagent."
        ),
        tools=(
            "fetch_ilamb_data",
            "load_obs_metadata",
        ),
    ),
    CategorySpec(
        name="extraction",
        step_class="SpatialTemporalExtractionStep",
        display_name="ESM Spatial-Temporal Extraction",
        result_label="extraction",
        call_tool_name="call_extraction",
        description=(
            "Delegate spatial/temporal subsetting, grid matching and timeseries "
            "extraction to the extraction subagent."
        ),
        tools=(
            "match_to_grid",
            "extract_e3sm_timeseries",
            "extract_obs_timeseries",
            "extract_gridded_field",
            "extract_basin_mean",
        ),
    ),
    CategorySpec(
        name="diagnostics",
        step_class="DiagnosticsAndSkillMetricsStep",
        display_name="ESM Diagnostics and Skill Metrics",
        result_label="diagnostics",
        call_tool_name="call_diagnostics",
        description=(
            "Delegate climatologies, summary statistics and skill metrics to the "
            "diagnostics subagent."
        ),
        tools=(
            "compute_climatology",
            "compute_summary_stats",
            "compute_metrics",
            "compute_fdc_metrics",
            "compute_spatial_bias",
            "compute_zonal_stats",
        ),
    ),
    CategorySpec(
        name="water_cycle",
        step_class="BasinScaleWaterCycleSynthesisStep",
        display_name="Basin-Scale Water Cycle Synthesis",
        result_label="water cycle synthesis",
        call_tool_name="call_water_cycle_synthesis",
        description=(
            "Delegate basin-scale water budget closure and basin aggregation to "
            "the water cycle synthesis subagent."
        ),
        tools=(
            "compute_basin_budget",
            "extract_basin_mean",
            "compute_fdc_metrics",
            "compute_spatial_bias",
        ),
    ),
    CategorySpec(
        name="visualization",
        step_class="DiagnosticVisualizationStep",
        display_name="ESM Diagnostic Visualization",
        result_label="visualization",
        call_tool_name="call_visualization",
        description=(
            "Delegate figure generation (maps, timeseries, scatter, FDC, radar) "
            "to the visualization subagent."
        ),
        tools=(
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
        ),
    ),
)

BUILTIN_CATEGORY_NAMES: Tuple[str, ...] = tuple(c.name for c in BUILTIN_CATEGORIES)


def _empty_registry() -> Dict[str, Any]:
    return {"version": 1, "capabilities": [], "subagents": []}


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``extensions/registry.yaml``, tolerating absence or emptiness."""

    path = Path(registry_path) if registry_path else REGISTRY_PATH
    if not path.exists():
        return _empty_registry()
    with path.open() as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        return _empty_registry()
    registry = _empty_registry()
    registry["version"] = data.get("version", 1)
    registry["capabilities"] = list(data.get("capabilities") or [])
    registry["subagents"] = list(data.get("subagents") or [])
    return registry


def _pascal_case(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_") if part)


def _spec_from_registry_subagent(entry: Dict[str, Any]) -> CategorySpec:
    name = entry["name"]
    pascal = _pascal_case(name)
    display_name = entry.get("display_name") or f"{pascal} (user-onboarded)"
    return CategorySpec(
        name=name,
        step_class=entry.get("step_class") or f"{pascal}Step",
        display_name=display_name,
        result_label=entry.get("result_label") or name.replace("_", " "),
        call_tool_name=entry.get("call_tool_name") or f"call_{name}",
        description=entry.get("description")
        or f"Delegate {display_name} work to the {name} subagent.",
        tools=(),
        builtin=False,
    )


def category_specs(registry_path: Optional[Path] = None) -> Dict[str, CategorySpec]:
    """Merge builtin categories with registry subagents and capabilities.

    Returns an insertion-ordered mapping ``{category_name: CategorySpec}`` where
    builtin categories come first (in their historical order) followed by any
    user-onboarded subagents.
    """

    registry = load_registry(registry_path)

    specs: Dict[str, CategorySpec] = {c.name: c for c in BUILTIN_CATEGORIES}

    for entry in registry["subagents"]:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        if entry["name"] in specs:
            # A registry subagent never shadows a builtin category.
            continue
        spec = _spec_from_registry_subagent(entry)
        specs[spec.name] = spec

    # Widen the allow-lists with onboarded capabilities.
    extra_tools: Dict[str, List[str]] = {}
    for entry in registry["capabilities"]:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool_name")
        subagent = entry.get("subagent")
        if not tool_name or not subagent or subagent not in specs:
            continue
        bucket = extra_tools.setdefault(subagent, [])
        if tool_name not in bucket:
            bucket.append(tool_name)

    for name, tools in extra_tools.items():
        spec = specs[name]
        merged = list(spec.tools) + [t for t in tools if t not in spec.tools]
        specs[name] = replace(spec, tools=tuple(merged))

    return specs


def extension_category_specs(
    registry_path: Optional[Path] = None,
) -> Dict[str, CategorySpec]:
    """Only the non-builtin categories (those needing generated subagents)."""

    return {
        name: spec
        for name, spec in category_specs(registry_path).items()
        if not spec.builtin
    }


def registered_capabilities(registry_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Raw capability entries from the registry."""

    return load_registry(registry_path)["capabilities"]
