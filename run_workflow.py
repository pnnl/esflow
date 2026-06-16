#!/usr/bin/env python3
"""
ESMFlow v3 Workflow Engine

Execute workflow YAML files by running tools in sequence,
passing outputs between steps via ${reference} resolution.

Usage:
    python run_workflow.py workflow.yaml [--dry-run] [--verbose] [--reuse]
"""

import argparse
import copy
import importlib.util
import re
import subprocess
import sys
import traceback as tb_module
import warnings
from datetime import datetime
from pathlib import Path

import yaml


class TeeWriter:
    """Write to both a stream and a log file simultaneously."""

    def __init__(self, original_stream, log_file):
        self.original = original_stream
        self.log_file = log_file

    def write(self, text):
        self.original.write(text)
        self.log_file.write(text)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return self.original.isatty()


warnings.filterwarnings('ignore', message='invalid value encountered in intersection',
                        category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

def load_tool(tool_name: str, catalog: dict = None, tools_dir: Path = None):
    """Dynamically load a tool module by name."""
    if tools_dir is None:
        tools_dir = Path(__file__).parent / 'tools'

    # Look up path from catalog
    tool_path = None
    if catalog:
        for tool in catalog.get('tools', []):
            if tool['name'] == tool_name:
                rel_path = tool.get('path', '')
                if rel_path:
                    tool_path = tools_dir / rel_path
                break

    # Fallback: search category subdirectories
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

    spec = importlib.util.spec_from_file_location(tool_name, tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'run'):
        raise AttributeError(f"Tool {tool_name} does not have a run() function")

    return module


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------

def resolve_references(obj, context: dict):
    """Replace ${references} with values from context.

    Supports:
        ${settings.key}
        ${step_id.outputs.key}
    """
    if isinstance(obj, str):
        pattern = r'\$\{([^}]+)\}'

        # Full-string reference: preserve native type
        full_match = re.fullmatch(pattern, obj.strip())
        if full_match:
            path = full_match.group(1)
            return _resolve_path(path, context)

        # Inline references: substitute as strings
        def replacer(match):
            return str(_resolve_path(match.group(1), context))
        return re.sub(pattern, replacer, obj)

    elif isinstance(obj, dict):
        return {k: resolve_references(v, context) for k, v in obj.items()}
    elif isinstance(obj, list):
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_workflow(workflow: dict, catalog_path: Path = None) -> list:
    """Validate workflow against tool catalog. Returns list of errors."""
    errors = []

    if catalog_path is None:
        catalog_path = Path(__file__).parent / 'tools' / 'tool_catalog.yaml'

    if not catalog_path.exists():
        errors.append(f"Tool catalog not found: {catalog_path}")
        return errors

    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    available_tools = {t['name']: t for t in catalog.get('tools', [])}

    steps = workflow.get('steps', [])
    if not steps:
        errors.append("Workflow has no steps")
        return errors

    step_ids = set()
    for i, step in enumerate(steps):
        step_id = step.get('id', f'step_{i}')

        if step_id in step_ids:
            errors.append(f"Duplicate step id: '{step_id}'")
        step_ids.add(step_id)

        tool_name = step.get('tool')
        if not tool_name:
            errors.append(f"Step '{step_id}' missing 'tool' field")
            continue

        if tool_name not in available_tools:
            errors.append(f"Step '{step_id}' uses unknown tool: {tool_name}")
            continue

        tool_spec = available_tools[tool_name]

        # Support both 'params' and 'config' keys
        params = step.get('params', step.get('config', {}))

        # Check required inputs
        for input_name, input_spec in tool_spec.get('inputs', {}).items():
            if input_spec.get('required', False) and input_name not in params:
                if 'default' not in input_spec:
                    errors.append(
                        f"Step '{step_id}' missing required input: {input_name}"
                    )

        # Type-check params against catalog (catch common LLM errors)
        for param_name, param_value in params.items():
            if param_name in tool_spec.get('inputs', {}):
                expected_type = tool_spec['inputs'][param_name].get('type', '')
                if expected_type == 'list[int]' and isinstance(param_value, str):
                    # LLMs often write years as string "2000" instead of [2000]
                    if not any(c in param_value for c in [',', '-', '[', ']']):
                        try:
                            int(param_value)
                            errors.append(
                                f"Step '{step_id}' param '{param_name}': "
                                f"expected list[int], got string '{param_value}'. "
                                f"Use [int] or int-int range."
                            )
                        except ValueError:
                            pass

    return errors


# ---------------------------------------------------------------------------
# Output file checking
# ---------------------------------------------------------------------------

def check_step_outputs(step, context, output_dir):
    """Check if declared output files exist on disk.

    Returns (all_exist, outputs_dict).
    """
    step_params = step.get('params', step.get('config', {}))

    # Try to resolve output_dir from params
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
        # Resolve any references in the filename
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


# ---------------------------------------------------------------------------
# Git info and frozen workflow
# ---------------------------------------------------------------------------

def get_git_info():
    """Get git commit hash and check for uncommitted changes."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        short_commit = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        return {
            'commit': commit,
            'short_commit': short_commit,
            'has_uncommitted': len(status) > 0,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def generate_frozen_workflow(workflow: dict, workflow_path: Path, output_dir: Path):
    """Generate a frozen workflow YAML with git provenance and timestamp."""
    git_info = get_git_info()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    header_lines = [
        f"# Frozen Workflow: {workflow.get('name', workflow_path.stem)}",
        f"# Generated: {timestamp}",
        f"# Original file: {workflow_path}",
    ]

    if git_info:
        header_lines.append(f"# Git commit: {git_info['commit']}")
        header_lines.append(f"#")
        header_lines.append(f"# To reproduce, checkout this commit:")
        header_lines.append(f"#   git checkout {git_info['short_commit']}")
        if git_info['has_uncommitted']:
            header_lines.append(f"#")
            header_lines.append(f"# WARNING: Generated with uncommitted changes!")
    else:
        header_lines.append(f"# Git commit: (not in a git repository)")

    header_lines.append(f"#")
    header_lines.append(f"# Run with: python run_workflow.py <this_file>")
    header_lines.append("")

    header = "\n".join(header_lines)

    frozen_workflow = copy.deepcopy(workflow)

    # Modify output_dir to include commit hash (avoid overwriting original)
    if 'settings' in frozen_workflow and 'output_dir' in frozen_workflow['settings']:
        original_output = frozen_workflow['settings']['output_dir']
        commit_suffix = git_info['short_commit'] if git_info else 'unknown'
        frozen_workflow['settings']['output_dir'] = f"{original_output}_reproduced_{commit_suffix}"

    frozen_content = header + yaml.dump(
        frozen_workflow, default_flow_style=False, sort_keys=False, allow_unicode=True
    )

    frozen_filename = f"{workflow_path.stem}_frozen.yaml"
    frozen_path = output_dir / frozen_filename

    with open(frozen_path, 'w') as f:
        f.write(frozen_content)

    return frozen_path


# ---------------------------------------------------------------------------
# Main workflow runner
# ---------------------------------------------------------------------------

def run_workflow(workflow_path: str, dry_run: bool = False, verbose: bool = False,
                 start_from: str = None, reuse: bool = False,
                 diagram: bool = False, freeze: bool = None):
    """Execute a workflow definition."""
    workflow_path = Path(workflow_path)

    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")

    with open(workflow_path) as f:
        workflow = yaml.safe_load(f)

    workflow_name = workflow.get('name', workflow_path.stem)

    print("=" * 70)
    print(f"WORKFLOW: {workflow_name}")
    print(f"File: {workflow_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Validate
    errors = validate_workflow(workflow)
    if errors:
        print("\nVALIDATION ERRORS:")
        for err in errors:
            print(f"   - {err}")
        return None

    print("\nWorkflow validation passed")

    steps = workflow.get('steps', [])

    if dry_run:
        print(f"\n[DRY RUN] Would execute {len(steps)} steps:")
        for i, step in enumerate(steps):
            print(f"   {i+1}. [{step.get('id')}] {step.get('tool')}")
        return {'dry_run': True}

    # Load catalog
    catalog_path = Path(__file__).parent / 'tools' / 'tool_catalog.yaml'
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    # Setup
    settings = workflow.get('settings', {})
    output_dir = Path(settings.get('output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = output_dir / 'workflow.log'
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeWriter(original_stdout, log_file)
    sys.stderr = TeeWriter(original_stderr, log_file)

    context = {
        'settings': settings,
        'output_dir': str(output_dir),
    }

    def _restore():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    # Decide whether to create a frozen workflow
    create_frozen = False
    if freeze is True:
        # --freeze: always create frozen workflow
        git_info = get_git_info()
        if git_info and git_info['has_uncommitted']:
            print("\n⚠️  Warning: You have uncommitted changes in the repository.")
            print("   The frozen workflow may not be fully reproducible.")
        create_frozen = True
    elif freeze is None:
        # Default: ask interactively
        try:
            freeze_response = input("\nCreate a frozen workflow file after completion? [y/N]: ").strip().lower()
            if freeze_response in ('y', 'yes'):
                git_info = get_git_info()
                if git_info and git_info['has_uncommitted']:
                    print("\n⚠️  Warning: You have uncommitted changes in the repository.")
                    print("   The frozen workflow may not be fully reproducible.")
                    try:
                        resp = input("Continue anyway? [y/N]: ").strip().lower()
                        if resp not in ('y', 'yes'):
                            print("\nAborting. Please commit your changes first, then rerun.")
                            return None
                        create_frozen = True
                    except EOFError:
                        print("\nAborting.")
                        return None
                else:
                    create_frozen = True
        except EOFError:
            pass
    # else: freeze is False (--no-freeze), skip entirely

    if reuse:
        print("\n--reuse: will skip steps with existing output files.")

    # Find start index
    total_steps = len(steps)
    start_idx = 0
    if start_from:
        step_ids = [s.get('id', f'step_{i}') for i, s in enumerate(steps)]
        if start_from in step_ids:
            start_idx = step_ids.index(start_from)
            print(f"\nStarting from step '{start_from}' ({start_idx + 1}/{total_steps})")
        else:
            print(f"\nStep '{start_from}' not found. Available: {step_ids}")
            _restore()
            return None

    print(f"\nExecuting {total_steps - start_idx} of {total_steps} steps...\n")

    # Log workflow YAML
    log_file.write("\n--- WORKFLOW YAML ---\n")
    yaml.dump(workflow, log_file, default_flow_style=False, sort_keys=False)
    log_file.write("--- END WORKFLOW YAML ---\n\n")
    log_file.flush()

    for i, step in enumerate(steps):
        step_id = step.get('id', f'step_{i}')
        tool_name = step.get('tool')

        # Skip steps before start_from
        if i < start_idx:
            _, outputs = check_step_outputs(step, context, output_dir)
            context[step_id] = {'outputs': outputs, 'result': {'skipped': True}}
            print(f"[{i+1}/{total_steps}] {step_id}: (skipped)")
            continue

        # Reuse existing outputs
        if reuse:
            all_exist, existing = check_step_outputs(step, context, output_dir)
            if all_exist:
                context[step_id] = {'outputs': existing, 'result': {'reused': True}}
                print(f"[{i+1}/{total_steps}] {step_id}: {tool_name}")
                print(f"    Reusing existing files")
                continue

        print(f"[{i+1}/{total_steps}] {step_id}: {tool_name}")

        try:
            tool_module = load_tool(tool_name, catalog=catalog)

            # Support both 'params' and 'config' keys
            raw_params = step.get('params', step.get('config', {}))
            params = resolve_references(raw_params, context)

            # Determine step output dir
            step_output_dir = Path(params.get('output_dir', str(output_dir)))

            # Inject output paths from 'outputs' section
            if 'outputs' in step:
                for key, filename in step['outputs'].items():
                    filename = resolve_references(filename, context) if isinstance(filename, str) else filename
                    params[f'output_{key}'] = str(step_output_dir / filename)

            if verbose:
                print(f"    Params: {params}")
            else:
                log_file.write(f"    Resolved params: {params}\n")
                log_file.flush()

            if 'output_dir' not in params:
                params['output_dir'] = str(output_dir)

            # Run
            result = tool_module.run(params)

            # Map declared outputs to result
            outputs = {}
            declared = step.get('outputs', {})
            for yaml_key in declared:
                actual = result.get(yaml_key)

                if actual is None:
                    # Try common suffixed keys
                    for suffix in ['', '_file']:
                        candidate = yaml_key + suffix
                        if candidate in result:
                            actual = result[candidate]
                            break

                if actual is not None:
                    # Rename file to declared name if needed
                    declared_filename = declared[yaml_key]
                    if isinstance(declared_filename, str):
                        declared_filename = resolve_references(declared_filename, context) if '${' in declared_filename else declared_filename
                    actual_file = Path(str(actual))
                    declared_file = actual_file.parent / declared_filename

                    if actual_file.exists() and actual_file != declared_file:
                        actual_file.rename(declared_file)
                        outputs[yaml_key] = str(declared_file)
                    else:
                        outputs[yaml_key] = str(actual)

            context[step_id] = {'outputs': outputs, 'result': result}
            print(f"    Done")

            if verbose and result:
                for k, v in result.items():
                    print(f"      {k}: {v}")

        except Exception as e:
            print(f"    FAILED: {e}")
            tb_str = tb_module.format_exc()
            if verbose:
                print(tb_str)
            else:
                log_file.write(tb_str)
                log_file.flush()

            # Fallback to existing files in reuse mode
            if reuse:
                all_exist, fallback = check_step_outputs(step, context, output_dir)
                if all_exist:
                    context[step_id] = {'outputs': fallback, 'result': {'error': str(e)}}
                    print(f"    Falling back to existing files")
                    continue

            context[step_id] = {'outputs': {}, 'result': {'error': str(e)}}
            print(f"    (Continuing with remaining steps...)")

    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nOutputs: {output_dir}")

    output_files = sorted(f for f in output_dir.glob('*') if f.is_file())
    if output_files:
        print("\nGenerated files:")
        for f in output_files:
            size = f.stat().st_size
            if size < 1024:
                s = f"{size} B"
            elif size < 1024 * 1024:
                s = f"{size/1024:.1f} KB"
            else:
                s = f"{size/1024/1024:.1f} MB"
            print(f"   {f.name} ({s})")

    print(f"\nLog: {log_path}")

    # Generate frozen workflow if requested
    if create_frozen:
        frozen_path = generate_frozen_workflow(workflow, workflow_path, output_dir)
        git_info = get_git_info()
        print(f"\n✓ Frozen workflow saved: {frozen_path}")
        if git_info:
            print(f"  To reproduce: git checkout {git_info['short_commit']} && python run_workflow.py {frozen_path}")

    # Generate workflow diagram
    if diagram:
        try:
            from visualize_workflow import visualize_workflow as viz
            diagram_path = output_dir / 'workflow_diagram.png'
            viz(workflow_path, diagram_path, show_settings=False)
        except Exception as e:
            print(f"\nDiagram generation failed: {e}")

    _restore()
    return context


def main():
    parser = argparse.ArgumentParser(
        description='Execute ESMFlow workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_workflow.py workflows/example.yaml
    python run_workflow.py workflows/example.yaml --dry-run
    python run_workflow.py workflows/example.yaml --reuse --verbose
    python run_workflow.py workflows/example.yaml --start-from compute_metrics
        """
    )

    parser.add_argument('workflow', help='Path to workflow YAML file')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Validate without executing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--start-from', '-s', metavar='STEP_ID',
                        help='Skip to this step (earlier steps assumed complete)')
    parser.add_argument('--reuse', '-r', action='store_true',
                        help='Reuse existing intermediate files')
    parser.add_argument('--diagram', '-d', action='store_true',
                        help='Generate workflow data-flow diagram (PNG)')

    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument('--freeze', action='store_true', default=None,
                              help='Create frozen workflow with git provenance (no prompt)')
    freeze_group.add_argument('--no-freeze', dest='freeze', action='store_false',
                              help='Skip frozen workflow creation (no prompt)')

    args = parser.parse_args()

    try:
        context = run_workflow(
            args.workflow,
            dry_run=args.dry_run,
            verbose=args.verbose,
            start_from=args.start_from,
            reuse=args.reuse,
            diagram=args.diagram,
            freeze=args.freeze,
        )
        sys.exit(0 if context is not None else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
