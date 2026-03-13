"""
ESFlow v3 tool framework.

Provides:
- Param: typed parameter specification
- ToolSpec: tool metadata (name, description, inputs, outputs)
- @esflow_tool: decorator for registration and validation
"""

import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------

class Param:
    """Parameter specification for tool inputs.

    Supported types: str, int, float, bool, path, list[str], list[int]
    """

    def __init__(
        self,
        type: str,
        required: bool = False,
        default: Any = None,
        description: str = "",
    ):
        self.type = type
        self.required = required
        self.default = default
        self.description = description

    def coerce(self, value: Any) -> Any:
        """Convert value to the declared type."""
        if value is None:
            return self.default

        if self.type == 'str':
            return str(value) if value is not None else self.default
        elif self.type == 'int':
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return self.default
            return int(float(value))
        elif self.type == 'float':
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return self.default
            return float(value)
        elif self.type == 'bool':
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
        elif self.type == 'path':
            if value == '':
                return self.default
            return str(Path(value))
        elif self.type == 'list[str]':
            if isinstance(value, str):
                return [v.strip() for v in value.split(',')]
            return [str(v) for v in value] if value else []
        elif self.type == 'list[int]':
            if isinstance(value, str):
                # Handle "2000-2005" range syntax
                value = value.strip('[]')
                if '-' in value and not value.startswith('-'):
                    parts = value.split('-')
                    if len(parts) == 2:
                        try:
                            return list(range(int(parts[0]), int(parts[1]) + 1))
                        except ValueError:
                            pass
                return [int(float(v.strip())) for v in value.split(',')]
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            return [int(value)]
        return value

    def to_catalog_dict(self) -> Dict:
        """Convert to catalog YAML format."""
        d = {
            'type': self.type,
            'required': self.required,
            'description': self.description,
        }
        if self.default is not None:
            d['default'] = self.default
        return d


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

class ToolSpec:
    """Tool metadata — single source of truth.

    Outputs is a dict of {name: {type, description}}.
    Types: 'csv', 'png', 'netcdf', 'int', 'float', 'str', 'dict'
    """

    def __init__(
        self,
        name: str,
        description: str,
        inputs: Dict[str, Param],
        outputs: Dict[str, Dict[str, str]],
    ):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.outputs = outputs

    def parse_config(self, config: Dict) -> Dict:
        """Validate and coerce config. Rejects unknown params."""
        parsed = {}

        for param_name, param_spec in self.inputs.items():
            value = config.get(param_name)

            if value is None and param_spec.required:
                raise ValueError(
                    f"[{self.name}] Missing required parameter: {param_name}"
                )

            parsed[param_name] = param_spec.coerce(value)

        # Strict mode: reject unknown params
        known_keys = set(self.inputs.keys()) | {'output_dir'}
        for key in config:
            if key not in known_keys and not key.startswith('output_'):
                raise ValueError(
                    f"[{self.name}] Unknown parameter: '{key}'. "
                    f"Valid params: {sorted(self.inputs.keys())}"
                )

        return parsed

    def to_catalog_dict(self) -> Dict:
        """Generate catalog entry."""
        return {
            'name': self.name,
            'description': self.description,
            'inputs': {k: v.to_catalog_dict() for k, v in self.inputs.items()},
            'outputs': self.outputs,
        }


# ---------------------------------------------------------------------------
# Tool registry and decorator
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def esflow_tool(spec: ToolSpec):
    """Decorator for ESFlow tools.

    - Registers tool in TOOL_REGISTRY
    - Validates and coerces params via spec
    - Auto-creates output_dir
    - Strict: rejects unknown params
    """
    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY[spec.name] = spec

        @functools.wraps(func)
        def wrapper(config: Dict) -> Dict:
            parsed = spec.parse_config(config)

            # Ensure output_dir exists
            output_dir = config.get('output_dir', './output')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            parsed['output_dir'] = str(output_dir)

            result = func(parsed)

            # Verify output keys match spec
            for key, out_spec in spec.outputs.items():
                out_type = out_spec.get('type', '')
                if out_type in ('csv', 'png', 'netcdf') and key not in result:
                    raise ValueError(
                        f"[{spec.name}] Tool must return output key '{key}' "
                        f"(declared type: {out_type})"
                    )

            return result

        wrapper._spec = spec
        return wrapper

    return decorator
