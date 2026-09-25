"""Bridge between plain user Python functions and the ESMFlow tool contract.

Onboarded capabilities are *thin generated adapters* (``tools/<category>/<name>.py``)
that keep the user's own code untouched and unaware of ESMFlow. Each adapter:

1. calls the user function with only the arguments its signature accepts
   (:func:`call_user_function`), then
2. turns whatever the function returned into the declared on-disk artifacts and
   the ``{output_key: path}`` dict ``@esmflow_tool`` requires
   (:func:`materialize_result`).

Supported return values:

===========================  ==========================================
Returned object              Handling
===========================  ==========================================
``dict``                     Matched per declared output key
``pandas.DataFrame``/Series  Written as CSV
``xarray.Dataset``/DataArray Written as NetCDF
``matplotlib`` Figure        Saved as PNG
``str``/``Path``             Treated as an already-written file path
scalar (int/float/str)       Passed through as a value output
===========================  ==========================================
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# User code lives at the repo root (e.g. `examples.user_code.snow_metrics`), so
# the root must be importable regardless of which entrypoint loaded the adapter.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FILE_OUTPUT_TYPES = {"csv": ".csv", "png": ".png", "netcdf": ".nc"}


class AdapterError(RuntimeError):
    """Raised when a user function cannot be called or its result materialized."""


# ---------------------------------------------------------------------------
# Calling user code
# ---------------------------------------------------------------------------


def load_user_function(module_path: str, function_name: str):
    """Import ``module_path`` and return its ``function_name`` attribute."""

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AdapterError(
            f"Could not import user module '{module_path}': {exc}"
        ) from exc
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise AdapterError(
            f"User module '{module_path}' has no function '{function_name}'"
        ) from exc


def _accepts_kwargs(signature: inspect.Signature) -> bool:
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )


def call_user_function(
    module_path: str,
    function_name: str,
    spec,
    config: Dict[str, Any],
) -> Any:
    """Call the user function with the subset of ``config`` it can accept.

    ``config`` is the already-parsed/coerced dict handed over by
    ``@esmflow_tool``. Only names present in ``spec.inputs`` are forwarded, plus
    ``output_dir`` when the user function asks for it, so a user function is
    never surprised by ESMFlow plumbing keys.
    """

    func = load_user_function(module_path, function_name)
    signature = inspect.signature(func)
    takes_any = _accepts_kwargs(signature)

    kwargs: Dict[str, Any] = {}
    for name in spec.inputs:
        if name not in config:
            continue
        value = config[name]
        if value is None:
            # Let the user function's own default apply.
            continue
        if takes_any or name in signature.parameters:
            kwargs[name] = value

    if "output_dir" in signature.parameters and "output_dir" in config:
        kwargs["output_dir"] = config["output_dir"]

    missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if missing:
        raise AdapterError(
            f"[{spec.name}] user function '{module_path}.{function_name}' requires "
            f"argument(s) {missing} that are not declared as tool inputs"
        )

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    last_positional = max(
        (index for index, parameter in enumerate(positional) if parameter.name in kwargs),
        default=-1,
    )
    positional_args = []
    for index, parameter in enumerate(positional):
        if index > last_positional:
            break
        positional_args.append(kwargs.pop(parameter.name, parameter.default))
    return func(*positional_args, **kwargs)


# ---------------------------------------------------------------------------
# Materializing results
# ---------------------------------------------------------------------------


def _is_dataframe(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module.startswith("pandas") and type(value).__name__ in (
        "DataFrame",
        "Series",
    )


def _is_xarray(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module.startswith("xarray") and type(value).__name__ in (
        "Dataset",
        "DataArray",
    )


def _is_figure(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module.startswith("matplotlib") and type(value).__name__ == "Figure"


def _is_path_like(value: Any) -> bool:
    return isinstance(value, Path) or (
        isinstance(value, str) and Path(value).suffix != ""
    )


def _target_path(
    spec, output_key: str, out_type: str, config: Dict[str, Any]
) -> Path:
    """Where an artifact for ``output_key`` should be written.

    Honours the ``output_<key>`` filename the workflow runner injects, so the
    runner's post-hoc rename becomes a no-op.
    """

    requested = config.get(f"output_{output_key}")
    output_dir = Path(config.get("output_dir", "./output"))
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute() and candidate.parent == Path("."):
            candidate = output_dir / candidate.name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    suffix = FILE_OUTPUT_TYPES.get(out_type, "")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{spec.name}_{output_key}{suffix}"


def _write_artifact(value: Any, path: Path, out_type: str, spec, output_key: str) -> str:
    """Persist ``value`` at ``path`` according to the declared output type."""

    path.parent.mkdir(parents=True, exist_ok=True)

    if _is_path_like(value):
        source = Path(value)
        if not source.exists():
            raise AdapterError(
                f"[{spec.name}] output '{output_key}' looks like a path "
                f"('{value}') but no such file exists"
            )
        if source.resolve() != path.resolve():
            path.write_bytes(source.read_bytes())
        return str(path)

    if _is_dataframe(value):
        if out_type != "csv":
            raise AdapterError(
                f"[{spec.name}] output '{output_key}' is declared '{out_type}' but "
                "the user function returned a pandas object (expected 'csv')"
            )
        index = getattr(value, "index", None)
        write_index = bool(index is not None and index.name)
        value.to_csv(path, index=write_index)
        return str(path)

    if _is_xarray(value):
        if out_type != "netcdf":
            raise AdapterError(
                f"[{spec.name}] output '{output_key}' is declared '{out_type}' but "
                "the user function returned an xarray object (expected 'netcdf')"
            )
        value.to_netcdf(path)
        return str(path)

    if _is_figure(value):
        if out_type != "png":
            raise AdapterError(
                f"[{spec.name}] output '{output_key}' is declared '{out_type}' but "
                "the user function returned a matplotlib Figure (expected 'png')"
            )
        value.savefig(path, dpi=150, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(value)
        except Exception:  # pragma: no cover - headless safety net
            pass
        return str(path)

    raise AdapterError(
        f"[{spec.name}] cannot materialize output '{output_key}' of declared type "
        f"'{out_type}' from a {type(value).__name__}"
    )


def output_type_of(declared: Any) -> str:
    """Normalize a ``ToolSpec.outputs`` value to its type string.

    ``ToolSpec`` stores outputs as ``{key: 'csv'}``, while the generated catalog
    and some hand-written tools use the richer ``{key: {'type': 'csv', ...}}``
    form. Accept both so adapters work in either context.
    """

    if isinstance(declared, str):
        return declared
    if isinstance(declared, dict):
        return str(declared.get("type", ""))
    return ""


def materialize_result(spec, result: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a user function's return value into an ESMFlow result dict."""

    file_keys = [
        key
        for key, out in spec.outputs.items()
        if output_type_of(out) in FILE_OUTPUT_TYPES
    ]

    if isinstance(result, dict):
        per_key = result
    elif len(spec.outputs) == 1:
        per_key = {next(iter(spec.outputs)): result}
    elif len(file_keys) == 1 and result is not None:
        per_key = {file_keys[0]: result}
    else:
        raise AdapterError(
            f"[{spec.name}] user function must return a dict keyed by output name "
            f"({sorted(spec.outputs)}) because the tool declares "
            f"{len(spec.outputs)} outputs"
        )

    materialized: Dict[str, Any] = {}
    for key, out_spec in spec.outputs.items():
        out_type = output_type_of(out_spec)
        if key not in per_key:
            if out_type in FILE_OUTPUT_TYPES:
                raise AdapterError(
                    f"[{spec.name}] user function did not provide required output "
                    f"'{key}' (declared type '{out_type}')"
                )
            continue
        value = per_key[key]
        if out_type in FILE_OUTPUT_TYPES:
            path = _target_path(spec, key, out_type, config)
            materialized[key] = _write_artifact(value, path, out_type, spec, key)
        else:
            materialized[key] = value

    # Surface any extra scalars the user returned; harmless and useful for logs.
    for key, value in per_key.items():
        if key not in materialized and key not in spec.outputs:
            materialized.setdefault(key, value)

    return materialized


def run_user_capability(
    module_path: str,
    function_name: str,
    spec,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Convenience: call the user function and materialize its result."""

    result = call_user_function(module_path, function_name, spec, config)
    return materialize_result(spec, result, config)
