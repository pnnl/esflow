"""Programmatic workflow execution API for ESMFlow.

This module exposes the workflow runner without argparse or process-exit behavior,
so callers can import and execute workflows directly from Python code.
"""

import importlib.util
import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from common.workflow import Workflow
from common.workflow_validation import load_raw_catalog, validate_workflow


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
logger = logging.getLogger(__name__)


class ExecutionStepStatus(BaseModel):
    """Structured execution status for one workflow step."""

    step_id: str
    status: Literal["completed", "failed", "reused", "skipped"]
    error: str | None = None


def step_statuses_from_execution_context(
    workflow: Workflow, execution_context: dict
) -> list[ExecutionStepStatus]:
    """Convert workflow runner context into structured per-step statuses."""
    step_statuses: list[ExecutionStepStatus] = []
    for step in workflow.steps:
        step_state = execution_context.get(step.id, {})
        result = step_state.get("result", {})
        if result.get("error"):
            step_statuses.append(
                ExecutionStepStatus(
                    step_id=step.id,
                    status="failed",
                    error=result["error"],
                )
            )
        elif result.get("reused"):
            step_statuses.append(ExecutionStepStatus(step_id=step.id, status="reused"))
        elif result.get("skipped"):
            step_statuses.append(ExecutionStepStatus(step_id=step.id, status="skipped"))
        else:
            step_statuses.append(ExecutionStepStatus(step_id=step.id, status="completed"))
    return step_statuses


def _default_output_dir() -> Path:
    """Return the canonical output directory as an absolute path.

    Reads ``ESFLOW_OUTPUT_DIR`` via ``RuntimeConfig`` when available, so the
    server and the runner always agree on where outputs land.  Falls back to
    ``<repo>/outputs`` if the config cannot be loaded.
    """
    try:
        from common.config import runtime_config  # noqa: PLC0415
        return runtime_config.resolved_output_dir()
    except Exception:
        return (_REPO_ROOT / "output").resolve()


warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in intersection',
    category=RuntimeWarning,
)


def load_tool(tool_name: str, catalog: dict = None, tools_dir: Path = None):
    """Dynamically load a tool module by name."""
    if tools_dir is None:
        tools_dir = _REPO_ROOT / 'tools'

    tool_path = None
    if catalog:
        for tool in catalog.get('tools', []):
            if tool['name'] == tool_name:
                rel_path = tool.get('path', '')
                if rel_path:
                    tool_path = tools_dir / rel_path
                break

    if tool_path is None or not tool_path.exists():
        for category_dir in tools_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('_'):
                continue
            candidate = category_dir / f"{tool_name}.py"
            if candidate.exists():
                tool_path = candidate
                break

    if tool_path is None or not tool_path.exists():
        raise FileNotFoundError(f"Tool not found: {tool_name}")

    spec = importlib.util.spec_from_file_location(f"esmflow.tools.{tool_name}", tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'run'):
        raise AttributeError(f"Tool {tool_name} does not have a run() function")

    return module


def resolve_references(obj, context: dict):
    """Replace ${references} with values from context."""
    if isinstance(obj, str):
        pattern = r'\$\{([^}]+)\}'

        full_match = re.fullmatch(pattern, obj.strip())
        if full_match:
            path = full_match.group(1)
            return _resolve_path(path, context)

        def replacer(match):
            return str(_resolve_path(match.group(1), context))

        return re.sub(pattern, replacer, obj)

    if isinstance(obj, dict):
        return {k: resolve_references(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_references(v, context) for v in obj]
    return obj


def _resolve_path(path: str, context: dict):
    """Walk dotted path through nested dicts."""
    parts = path.split('.')
    value = context
    for part in parts:
        if isinstance(value, dict):
            if part not in value:
                raise ValueError(f"Cannot resolve reference: ${{{path}}}")
            value = value[part]
        else:
            raise ValueError(f"Cannot resolve reference: ${{{path}}}")
    return value


def check_step_outputs(step, context, output_dir):
    """Check if declared output files exist on disk."""
    step_params = step.get('params', step.get('config', {}))

    step_output_dir = output_dir
    try:
        resolved = resolve_references(step_params, context)
        if 'output_dir' in resolved:
            step_output_dir = Path(resolved['output_dir'])
    except (ValueError, KeyError):
        pass

    outputs = {}
    all_exist = True
    for yaml_key, filename in step.get('outputs', {}).items():
        try:
            filename = resolve_references(filename, context)
        except (ValueError, KeyError):
            pass

        expected_path = step_output_dir / filename
        outputs[yaml_key] = str(expected_path)
        if not expected_path.exists():
            all_exist = False

    if not step.get('outputs'):
        all_exist = False

    return all_exist, outputs


def load_workflow_file(workflow_path: str | Path) -> tuple[dict, Path]:
    """Load a workflow YAML file and return both content and normalized path."""
    resolved_path = Path(workflow_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Workflow not found: {resolved_path}")

    with open(resolved_path) as f:
        workflow = yaml.safe_load(f)

    return workflow, resolved_path


def run_workflow_definition(
    workflow: dict,
    workflow_path: str | Path = '<memory>',
    verbose: bool = False,
    start_from: str = None,
    reuse: bool = False,
    catalog_path: Path | None = None,
    tools_dir: Path | None = None,
):
    """Execute a workflow definition provided as a Python dict."""
    workflow_path = Path(workflow_path)
    if catalog_path is None:
        catalog_path = _REPO_ROOT / 'tools' / 'tool_catalog.yaml'
    if tools_dir is None:
        tools_dir = _REPO_ROOT / 'tools'
    workflow_name = workflow.get('name', workflow_path.stem)

    logger.info(
        "Running workflow %s from %s at %s",
        workflow_name,
        workflow_path,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    errors = validate_workflow(workflow, catalog_path)
    if errors:
        logger.error("Workflow validation failed:")
        for err in errors:
            logger.error("- %s", err)
        return None

    logger.info("Workflow validation passed")

    steps = workflow.get('steps', [])

    catalog = load_raw_catalog(catalog_path)

    settings = dict(workflow.get('settings', {}))

    # Fall back to runtime_config when settings are missing or an unresolved
    # template (e.g. the literal "${settings.output_dir}").
    _TEMPLATE_RE = re.compile(r'^\$\{.*\}$')

    def _is_template(v) -> bool:
        return isinstance(v, str) and bool(_TEMPLATE_RE.match(v.strip()))

    try:
        from common.config import runtime_config as _rc  # noqa: PLC0415
        default_output_dir = _rc.resolved_output_dir()
        default_data_dir   = _rc.resolved_data_dir()
    except Exception:
        _rc = None
        default_output_dir = _default_output_dir()
        default_data_dir   = None

    # Session settings are pinned by the planners, and validate_data checks the
    # same settings.data_dir, so preflight and execution see the same roots.
    if not settings.get('output_dir') or _is_template(settings['output_dir']):
        settings['output_dir'] = str(default_output_dir)
    if (not settings.get('data_dir') or _is_template(settings['data_dir'])) and default_data_dir:
        settings['data_dir'] = str(default_data_dir)
    canonical_data_dir = Path(settings['data_dir']).resolve() if settings.get('data_dir') else None

    try:
        canonical_obs_paths = _rc.obs_paths(canonical_data_dir)
    except Exception:
        canonical_obs_paths = {}

    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect case_name from E3SM filenames when the settings block does
    # not supply one.  This mirrors the same logic in validate_data / preflight
    # so the runner never fails on ${settings.case_name} references just
    # because ESFLOW_CASE_NAME was not set in the environment.
    if not settings.get('case_name') or _is_template(settings.get('case_name', '')):
        data_dir_for_detect = settings.get('data_dir', '')
        if data_dir_for_detect:
            try:
                from agents.data_preflight import _autodetect_e3sm_context  # noqa: PLC0415
                detected = _autodetect_e3sm_context(data_dir_for_detect)
                if detected['case_name']:
                    settings['case_name'] = detected['case_name']
                    logger.info("Auto-detected case_name: %s", detected['case_name'])
            except Exception as _exc:
                logger.warning("Could not auto-detect case_name: %s", _exc)

    # Re-run deterministic preflight using the *actual* workflow years before
    # execution. This is the final guard against a planner validating one time
    # range and then emitting steps for another (for example, validating the
    # bundled 1985--1989 streamflow records but requesting 2000--2004).
    requested_years = settings.get('years', [])
    if not requested_years:
        requested_years = []
        for step in steps:
            step_years = step.get('params', step.get('config', {})).get('years', [])
            if isinstance(step_years, (list, tuple)):
                requested_years.extend(step_years)
    try:
        requested_years = sorted({int(year) for year in requested_years})
    except (TypeError, ValueError):
        requested_years = []

    if canonical_data_dir:
        try:
            from agents.data_preflight import preflight_check  # noqa: PLC0415
            report = preflight_check(
                str(canonical_data_dir), str(catalog_path),
                case_name=settings.get('case_name', ''), years=requested_years,
            )
            requested_tools = {step.get('tool') for step in steps}
            blocked_requested = sorted(requested_tools & report.blocked)
            if blocked_requested:
                logger.error(
                    "PREFLIGHT BLOCKED EXECUTION: the workflow requests tools whose required "
                    "inputs are unavailable for the configured data/time range: %s",
                    ", ".join(blocked_requested),
                )
                for error in report.errors:
                    affected = error.get('affected_tools', [])
                    if isinstance(affected, str):
                        affected = [affected]
                    if requested_tools.intersection(affected):
                        logger.error("  - %s", error['detail'])
                return None
        except Exception as exc:
            logger.error("Execution preflight could not run: %s", exc)
            return None

    context = {
        'settings': settings,
        'output_dir': str(output_dir),
    }

    if reuse:
        logger.info("Reuse enabled; steps with existing output files will be skipped")

    total_steps = len(steps)
    start_idx = 0
    if start_from:
        step_ids = [s.get('id', f'step_{i}') for i, s in enumerate(steps)]
        if start_from in step_ids:
            start_idx = step_ids.index(start_from)
            logger.info("Starting from step %s (%s/%s)", start_from, start_idx + 1, total_steps)
        else:
            logger.error("Step %s not found. Available: %s", start_from, step_ids)
            return None

    logger.info("Executing %s of %s steps", total_steps - start_idx, total_steps)

    for i, step in enumerate(steps):
        step_id = step.get('id', f'step_{i}')
        tool_name = step.get('tool')

        if i < start_idx:
            _, outputs = check_step_outputs(step, context, output_dir)
            context[step_id] = {'outputs': outputs, 'result': {'skipped': True}}
            logger.info("[%s/%s] %s: skipped", i + 1, total_steps, step_id)
            continue

        if reuse:
            all_exist, existing = check_step_outputs(step, context, output_dir)
            if all_exist:
                context[step_id] = {'outputs': existing, 'result': {'reused': True}}
                logger.info("[%s/%s] %s: %s", i + 1, total_steps, step_id, tool_name)
                logger.info("Reusing existing files")
                continue

        logger.info("[%s/%s] %s: %s", i + 1, total_steps, step_id, tool_name)

        try:
            tool_module = load_tool(tool_name, catalog=catalog, tools_dir=tools_dir)
            raw_params = step.get('params', step.get('config', {}))
            params = resolve_references(raw_params, context)

            # Canonicalize observation paths after reference resolution.  This
            # prevents stale/LLM-hallucinated values like
            # ``data/sample/basins/basins.geojson`` from bypassing the exact
            # paths checked by preflight.
            if 'basins_file' in params and canonical_obs_paths:
                params['basins_file'] = str(canonical_obs_paths['basin_polygons'])
            if 'obs_dir' in params and canonical_obs_paths:
                params['obs_dir'] = str(canonical_obs_paths['streamflow_dir'])
            if 'gauge_metadata' in params and canonical_obs_paths:
                params['gauge_metadata'] = str(canonical_obs_paths['gauge_metadata'])
            if 'data_dir' in params and canonical_data_dir:
                params['data_dir'] = str(canonical_data_dir)

            step_output_dir = Path(params.get('output_dir', str(output_dir)))

            if 'outputs' in step:
                for key, filename in step['outputs'].items():
                    if isinstance(filename, str):
                        filename = resolve_references(filename, context)
                    _fn = Path(filename)
                    out_path = _fn.resolve() if _fn.is_absolute() else (step_output_dir / _fn).resolve()
                    try:
                        out_path.relative_to(step_output_dir.resolve())
                    except ValueError as e:
                        raise ValueError(f"Refusing to write outside output_dir: {filename}") from e
                    params[f'output_{key}'] = str(out_path)

            if verbose:
                logger.info("Params: %s", params)

            if 'output_dir' not in params:
                params['output_dir'] = str(output_dir)

            result = tool_module.run(params)

            outputs = {}
            declared = step.get('outputs', {})
            for yaml_key in declared:
                actual = result.get(yaml_key)

                if actual is None:
                    for suffix in ['', '_file']:
                        candidate = yaml_key + suffix
                        if candidate in result:
                            actual = result[candidate]
                            break

                if actual is not None:
                    declared_filename = declared[yaml_key]
                    if isinstance(declared_filename, str) and '${' in declared_filename:
                        declared_filename = resolve_references(declared_filename, context)
                    actual_file = Path(str(actual))
                    # ``declared_filename`` should be a bare filename (e.g.
                    # ``gauge_metadata_validated.csv``), but the LLM sometimes
                    # emits a full relative path.  If it already contains
                    # directory separators or is absolute, use it as-is to
                    # avoid double-prepending the parent directory.
                    _decl = Path(str(declared_filename))
                    if _decl.is_absolute():
                        declared_file = _decl
                    else:
                        declared_file = actual_file.parent / _decl.name

                    if actual_file.exists() and actual_file != declared_file:
                        declared_file.parent.mkdir(parents=True, exist_ok=True)
                        actual_file.rename(declared_file)
                        outputs[yaml_key] = str(declared_file)
                    else:
                        outputs[yaml_key] = str(actual)

            context[step_id] = {'outputs': outputs, 'result': result}
            logger.info("Done")

            if verbose and result:
                for key, value in result.items():
                    logger.info("%s: %s", key, value)

        except Exception as e:
            logger.exception("Step %s failed", step_id)

            if reuse:
                all_exist, fallback = check_step_outputs(step, context, output_dir)
                if all_exist:
                    context[step_id] = {'outputs': fallback, 'result': {'error': str(e)}}
                    logger.info("Falling back to existing files")
                    continue

            context[step_id] = {'outputs': {}, 'result': {'error': str(e)}}
            logger.info("Continuing with remaining steps")

    logger.info("Workflow complete; outputs: %s", output_dir)

    output_files = sorted(f for f in output_dir.glob('*') if f.is_file())
    if output_files:
        logger.info("Generated files:")
        for output_file in output_files:
            size = output_file.stat().st_size
            if size < 1024:
                size_text = f"{size} B"
            elif size < 1024 * 1024:
                size_text = f"{size/1024:.1f} KB"
            else:
                size_text = f"{size/1024/1024:.1f} MB"
            logger.info("%s (%s)", output_file.name, size_text)
    return context


def run_workflow_file(
    workflow_path: str | Path,
    verbose: bool = False,
    start_from: str = None,
    reuse: bool = False,
    catalog_path: Path | None = None,
    tools_dir: Path | None = None,
):
    """Load and execute a workflow YAML file."""
    workflow, resolved_path = load_workflow_file(workflow_path)
    return run_workflow_definition(
        workflow,
        workflow_path=resolved_path,
        verbose=verbose,
        start_from=start_from,
        reuse=reuse,
        catalog_path=catalog_path,
        tools_dir=tools_dir,
    )


run_workflow = run_workflow_file


__all__ = [
    'ExecutionStepStatus',
    'check_step_outputs',
    'load_tool',
    'load_workflow_file',
    'resolve_references',
    'run_workflow',
    'run_workflow_definition',
    'run_workflow_file',
    'step_statuses_from_execution_context',
    'validate_workflow',
]
