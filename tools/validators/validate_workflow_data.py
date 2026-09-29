"""
validate_workflow_data.py

ESFlow runtime validation tool. Executes as step 1 of any workflow.

Loads the tool catalog, calls the ValidationPlanningAgent to get (or
load from cache) a validation plan, executes all checks via primitives,
and writes three output files: validation_report.csv, validation_summary.txt,
feasible_tools.csv
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import yaml

# Add tools/ directory to path so core.base and validators are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from validators import primitives
from validators import planning_agent

SPEC = ToolSpec(
    name='validate_workflow_data',
    description=(
        'Validates user-provided data against the requirements derived from the ESFlow '
        'tool catalog. Inspects gauge metadata, streamflow CSVs, basin polygons, E3SM '
        'model output, and observation files. Produces a structured report of what data '
        'is present, what is missing or malformed, and which tools/analyses can be '
        'performed. Should be run as the first step in any workflow so the user is '
        'informed of data gaps before analysis begins.'
    ),
    inputs={
        'data_dir':     Param('path', required=True,
                              description='Root data directory to inspect'),
        'catalog_file': Param('path', required=True,
                              description='Path to tool_catalog.yaml — used to derive data '
                                          'requirements dynamically'),
        'case_name':    Param('str',  required=False,
                              description='E3SM case name (e.g. sample.v3.LR.historical). '
                                          'If provided, model output is checked.', default=''),
        'years':        Param('list[int]', required=False,
                              description='Years to check for model/obs coverage '
                                          '(e.g. [1985, 1986, 1987, 1988, 1989])', default=[]),
    },
    outputs={
        'report_file':         {'type': 'csv',
                                'description': 'Validation report CSV with columns check, '
                                               'status, detail, affected_tools'},
        'summary_file':        {'type': 'txt',
                                'description': 'Human-readable plain-text summary of '
                                               'validation results'},
        'feasible_tools_file': {'type': 'csv',
                                'description': 'CSV listing which tools can and cannot run '
                                               'given the available data'},
    },
)


def derive_feasibility(results: list, catalog: dict) -> list:
    """Determine which tools are feasible given the validation results."""
    tool_errors   = defaultdict(list)
    tool_warnings = defaultdict(list)

    for r in results:
        tools = r.get('affected_tools', [])
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(',') if t.strip() != '—']
        for t in tools:
            if r['status'] == 'ERROR':
                tool_errors[t].append(r['check_name'])
            elif r['status'] == 'WARN':
                tool_warnings[t].append(r['check_name'])

    rows = []
    for name, defn in catalog.items():
        if tool_errors[name]:
            rows.append({'tool': name,
                         'category': defn.get('category', ''),
                         'feasible': 'NO',
                         'reason': f"Blocked by: {', '.join(tool_errors[name])}"})
        elif tool_warnings[name]:
            rows.append({'tool': name,
                         'category': defn.get('category', ''),
                         'feasible': 'PARTIAL',
                         'reason': f"Warnings: {', '.join(tool_warnings[name])}"})
        else:
            rows.append({'tool': name,
                         'category': defn.get('category', ''),
                         'feasible': 'YES',
                         'reason': 'All data requirements met'})

    return sorted(rows,
                  key=lambda r: {'NO': 0, 'PARTIAL': 1, 'YES': 2}[r['feasible']])


def format_summary(results: list, feasibility: list,
                   data_dir: str, n_tools: int) -> str:
    """Format human-readable validation summary."""
    errors   = [r for r in results if r['status'] == 'ERROR']
    warnings = [r for r in results if r['status'] == 'WARN']
    ok       = [r for r in results if r['status'] == 'OK']
    blocked  = [f for f in feasibility if f['feasible'] == 'NO']
    partial  = [f for f in feasibility if f['feasible'] == 'PARTIAL']
    ready    = [f for f in feasibility if f['feasible'] == 'YES']
    W = 72

    lines = [
        '=' * W,
        'ESFlow Data Validation Report (Agent-Generated Plan)',
        f'Data directory : {data_dir}',
        f'Tools in catalog: {n_tools}  |  '
        f'Checks: {len(results)}  '
        f'(✓{len(ok)}  ⚠{len(warnings)}  ✗{len(errors)})',
        '=' * W, '',
    ]

    if errors:
        lines += ['── ERRORS ── must fix before affected tools can run ' +
                  '─' * (W - 52), '']
        for r in errors:
            lines += [f'  ✗  {r["check_name"]}',
                      f'     {r["detail"]}',
                      f'     Fix: {r.get("fix_hint", "")}',
                      f'     Affects: {r["affected_tools"]}', '']

    if warnings:
        lines += ['── WARNINGS ── tools may run with degraded results ' +
                  '─' * (W - 52), '']
        for r in warnings:
            lines += [f'  ⚠  {r["check_name"]}',
                      f'     {r["detail"]}',
                      f'     Fix: {r.get("fix_hint", "")}',
                      f'     Affects: {r["affected_tools"]}', '']

    lines += ['── TOOL FEASIBILITY ' + '─' * (W - 20), '',
              f'  {"Tool":<38} {"Status":<10} Reason',
              f'  {"-"*38} {"-"*10} {"-"*20}']
    for f in feasibility:
        icon = {'YES': '✓', 'PARTIAL': '⚠', 'NO': '✗'}[f['feasible']]
        lines.append(
            f'  {icon} {f["tool"]:<37} {f["feasible"]:<10} {f["reason"]}')

    lines += ['', '── SUMMARY ' + '─' * (W - 11), '',
              f'  ✓ Ready    : {len(ready)} tools',
              f'  ⚠ Partial  : {len(partial)} tools',
              f'  ✗ Blocked  : {len(blocked)} tools', '']

    if blocked:
        lines += ['  Resolve these ERRORs to unblock:']
        for f in blocked:
            lines.append(f'    ✗ {f["tool"]}')
        lines.append('')

    lines += ['=' * W]
    return '\n'.join(lines)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    """Execute the validation workflow."""
    data_dir     = config['data_dir']
    catalog_file = config['catalog_file']
    case_name    = config.get('case_name', '')
    years        = config.get('years', [])
    output_dir   = Path(config['output_dir'])

    # Load catalog
    with open(catalog_file) as f:
        cat = yaml.safe_load(f)
    catalog = {t['name']: t for t in cat.get('tools', [])}
    print(f"  Catalog: {len(catalog)} tools")

    # Get validation plan from agent (cached by catalog hash)
    context = {
        'data_dir':  data_dir,
        'case_name': case_name,
        'years':     years,
    }
    cache_dir = output_dir / '.validation_cache'
    plan = planning_agent.get_plan(catalog, context, cache_dir=cache_dir)
    print(f"  Plan: {len(plan)} checks to execute")

    # Execute all checks
    results = []
    for check in plan:
        result = primitives.execute(check)
        icon = {'OK': '✓', 'WARN': '⚠', 'ERROR': '✗'}.get(result['status'], '?')
        print(f"  {icon}  {result['check_name']}: {result['detail'][:80]}")
        results.append(result)

    # Derive feasibility
    feasibility = derive_feasibility(results, catalog)

    # Normalise affected_tools to string for CSV
    for r in results:
        if isinstance(r.get('affected_tools'), list):
            r['affected_tools'] = ', '.join(r['affected_tools'])

    # Write outputs
    pd.DataFrame(results).to_csv(
        output_dir / 'validation_report.csv', index=False)
    pd.DataFrame(feasibility).to_csv(
        output_dir / 'feasible_tools.csv', index=False)
    summary = format_summary(results, feasibility, data_dir, len(catalog))
    (output_dir / 'validation_summary.txt').write_text(summary)

    print()
    print(summary)

    return {
        'report_file':         str(output_dir / 'validation_report.csv'),
        'summary_file':        str(output_dir / 'validation_summary.txt'),
        'feasible_tools_file': str(output_dir / 'feasible_tools.csv'),
    }
