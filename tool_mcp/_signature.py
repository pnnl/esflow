"""Build real, introspectable Python function signatures from ``ToolSpec``.

``tools/core/base.py``'s ``ToolSpec``/``Param`` pair is the single source of
truth for every esmflow tool's parameters (it's also what generates
``tools/tool_catalog.yaml``). Each tool's actual implementation is a plain
``run(config: dict) -> dict`` function, which gives FastMCP nothing to build a
JSON schema from on its own.

``build_tool_function`` bridges the two: it synthesizes a wrapper with a real
``inspect.Signature`` (one named, typed parameter per ``spec.inputs`` entry,
plus a trailing ``output_dir``) and a docstring with ``Args:``/``Returns:``
sections, so FastMCP produces the same quality of per-argument JSON schema
that a hand-written tool function would -- without duplicating any tool
logic. The wrapper's body just re-assembles ``config`` and calls the
original ``run``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

# Map ToolSpec/Param's type strings to real Python types for the synthesized
# signature. 'path' is just a string on the wire; esmflow_tool's Param.coerce
# does its own light validation/normalization once ``run()`` receives it.
PARAM_TYPE_MAP: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "path": str,
    "list[str]": list[str],
    "list[int]": list[int],
}


def _python_type_for(param) -> Any:
    """Resolve the annotation for a single ``Param``.

    Optional (non-required, defaultless) params get ``Optional[T]`` so a
    caller can omit them; required or defaulted params keep the bare type
    since a concrete default is always supplied.
    """
    base = PARAM_TYPE_MAP.get(param.type, str)
    if not param.required and param.default is None:
        return Optional[base]
    return base


def build_tool_function(name: str, spec, run_fn: Callable[[dict], dict]) -> Callable[..., dict]:
    """Synthesize a plain function exposing ``spec``'s inputs as real kwargs.

    The returned function's ``__signature__``, ``__annotations__``, and
    ``__doc__`` are all set explicitly so FastMCP's introspection (which
    reads these, not the wrapper's actual ``*args, **kwargs`` implementation)
    produces a schema matching ``spec`` exactly.
    """

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    # Required params first, then optional/defaulted ones, to keep the
    # signature valid (no required param after a defaulted one).
    ordered_names = sorted(spec.inputs, key=lambda n: not spec.inputs[n].required)

    for param_name in ordered_names:
        param = spec.inputs[param_name]
        annotation = _python_type_for(param)
        annotations[param_name] = annotation
        default = inspect.Parameter.empty if param.required else param.default
        parameters.append(
            inspect.Parameter(
                param_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )

    parameters.append(
        inspect.Parameter(
            "output_dir",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default="./output",
            annotation=str,
        )
    )
    annotations["output_dir"] = str
    annotations["return"] = dict

    def wrapper(**kwargs: Any) -> dict:
        config = dict(kwargs)
        return run_fn(config)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__signature__ = inspect.Signature(parameters, return_annotation=dict)
    wrapper.__annotations__ = annotations
    wrapper.__doc__ = _build_docstring(spec)
    return wrapper


def _build_docstring(spec) -> str:
    lines = [spec.description.strip(), "", "Args:"]
    for param_name in spec.inputs:
        param = spec.inputs[param_name]
        req = "required" if param.required else f"default: {param.default!r}"
        lines.append(f"    {param_name}: {param.description} ({req})")
    lines.append("    output_dir: Directory to write output files to (default: './output').")
    lines.append("")
    lines.append("Returns:")
    for out_name, out_spec in spec.outputs.items():
        out_type = out_spec.get("type", "") if isinstance(out_spec, dict) else out_spec
        out_desc = out_spec.get("description", "") if isinstance(out_spec, dict) else ""
        lines.append(f"    {out_name} ({out_type}): {out_desc}")
    return "\n".join(lines)
