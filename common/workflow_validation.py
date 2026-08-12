"""Workflow validation helpers shared by eval and runtime code."""

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_catalog_singletons: dict[Path, dict] = {}


def _read_catalog_yaml(resolved_path: Path) -> dict:
    """Load one static catalog file at most once per resolved path and process."""
    if resolved_path not in _catalog_singletons:
        with open(resolved_path) as f:
            _catalog_singletons[resolved_path] = yaml.safe_load(f) or {}
    return _catalog_singletons[resolved_path]


def load_raw_catalog(catalog_path: Path | None = None) -> dict:
    """Return parsed catalog YAML loaded once per resolved path and process.

    The generated catalog is static while the application runs. Regenerate it
    before starting a process, then restart to load a newer version.
    """
    if catalog_path is None:
        catalog_path = _REPO_ROOT / "tools" / "tool_catalog.yaml"

    if not catalog_path.exists():
        return {}

    return _read_catalog_yaml(catalog_path.resolve())


def load_tool_specs(catalog_path: Path | None = None) -> dict[str, dict]:
    """Return the tool catalog keyed by tool name, or {} if the catalog is missing."""
    catalog = load_raw_catalog(catalog_path)
    return {t["name"]: t for t in catalog.get("tools", [])}


def validate_workflow(workflow: dict, catalog_path: Path | None = None) -> list[str]:
    """Validate a workflow dict against the tool catalog and return errors."""
    errors: list[str] = []

    if catalog_path is None:
        catalog_path = _REPO_ROOT / "tools" / "tool_catalog.yaml"

    if not catalog_path.exists():
        errors.append(f"Tool catalog not found: {catalog_path}")
        return errors

    available_tools = load_tool_specs(catalog_path)

    steps = workflow.get("steps", [])
    if not steps:
        errors.append("Workflow has no steps")
        return errors

    step_ids = set()
    for i, step in enumerate(steps):
        step_id = step.get("id", f"step_{i}")

        if step_id in step_ids:
            errors.append(f"Duplicate step id: '{step_id}'")
        step_ids.add(step_id)

        tool_name = step.get("tool")
        if not tool_name:
            errors.append(f"Step '{step_id}' missing 'tool' field")
            continue

        if tool_name not in available_tools:
            errors.append(f"Step '{step_id}' uses unknown tool: {tool_name}")
            continue

        tool_spec = available_tools[tool_name]

        # Support both 'params' and 'config' keys
        params = step.get("params", step.get("config", {}))

        # Check required inputs
        for input_name, input_spec in tool_spec.get("inputs", {}).items():
            if input_spec.get("required", False) and input_name not in params:
                if "default" not in input_spec:
                    errors.append(
                        f"Step '{step_id}' missing required input: {input_name}"
                    )

        # Type-check params against catalog (catch common LLM errors)
        for param_name, param_value in params.items():
            if param_name in tool_spec.get("inputs", {}):
                expected_type = tool_spec["inputs"][param_name].get("type", "")
                if expected_type == "list[int]" and isinstance(param_value, str):
                    # LLMs often write years as string "2000" instead of [2000]
                    if not any(c in param_value for c in [",", "-", "[", "]"]):
                        try:
                            int(param_value)
                            errors.append(
                                f"Step '{step_id}' param '{param_name}': "
                                f"expected list[int], got string '{param_value}'. "
                                "Use [int] or int-int range."
                            )
                        except ValueError:
                            pass

    return errors
