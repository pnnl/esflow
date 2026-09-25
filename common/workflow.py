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