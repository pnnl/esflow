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

import yaml

from common.workflow_validation import validate_workflow


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
logger = logging.getLogger(__name__)


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
):
    """Execute a workflow definition provided as a Python dict."""
    workflow_path = Path(workflow_path)
    workflow_name = workflow.get('name', workflow_path.stem)

    logger.info(
        "Running workflow %s from %s at %s",
        workflow_name,
        workflow_path,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    errors = validate_workflow(workflow, _REPO_ROOT / 'tools' / 'tool_catalog.yaml')
    if errors:
        logger.error("Workflow validation failed:")
        for err in errors:
            logger.error("- %s", err)
        return None

    logger.info("Workflow validation passed")

    steps = workflow.get('steps', [])

    catalog_path = _REPO_ROOT / 'tools' / 'tool_catalog.yaml'
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    settings = workflow.get('settings', {})
    output_dir = Path(settings.get('output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)

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
            tool_module = load_tool(tool_name, catalog=catalog)
            raw_params = step.get('params', step.get('config', {}))
            params = resolve_references(raw_params, context)

            step_output_dir = Path(params.get('output_dir', str(output_dir)))

            if 'outputs' in step:
                for key, filename in step['outputs'].items():
                    if isinstance(filename, str):
                        filename = resolve_references(filename, context)
                    params[f'output_{key}'] = str(step_output_dir / filename)

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
                    declared_file = actual_file.parent / declared_filename

                    if actual_file.exists() and actual_file != declared_file:
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
):
    """Load and execute a workflow YAML file."""
    workflow, resolved_path = load_workflow_file(workflow_path)
    return run_workflow_definition(
        workflow,
        workflow_path=resolved_path,
        verbose=verbose,
        start_from=start_from,
        reuse=reuse,
    )


run_workflow = run_workflow_file


__all__ = [
    'check_step_outputs',
    'load_tool',
    'load_workflow_file',
    'resolve_references',
    'run_workflow',
    'run_workflow_definition',
    'run_workflow_file',
    'validate_workflow',
]
