"""
data_preflight.py

Supervisor pre-check that runs BEFORE a workflow is composed.

Called by the workflow planning supervisor when a user provides a data
directory. Runs the validation planning agent and returns a structured
preflight report that the supervisor uses to:

  1. Warn the user about data issues before composing the workflow
  2. Only include tools that are feasible given the available data
  3. Suggest which gauge IDs to use in workflow params
  4. Surface actionable fix hints for any errors/warnings

Usage (from supervisor):
    from agents.data_preflight import preflight_check
    report = preflight_check(data_dir, catalog_file, case_name, years)
    # report.summary      → plain text for user
    # report.feasible     → set of tool names that can run
    # report.gauge_ids    → recommended gauge IDs
    # report.has_errors   → bool
    # report.has_warnings → bool
"""

import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, defaultdict

import yaml
import pandas as pd

# Add <repo>/tools to path for validators module
_tools_path = Path(__file__).parent.parent / 'tools'
if str(_tools_path) not in sys.path:
    sys.path.insert(0, str(_tools_path))

from validators import primitives
from validators import planning_agent

# Import runtime_config lazily inside functions where needed so that this module
# can still be imported without a fully-initialised config (e.g. in unit tests).


@dataclass
class PreflightReport:
    """Container for preflight check results."""
    summary:      str         = ''
    feasible:     set         = field(default_factory=set)
    partial:      set         = field(default_factory=set)
    blocked:      set         = field(default_factory=set)
    gauge_ids:    list        = field(default_factory=list)
    errors:       list        = field(default_factory=list)
    warnings:     list        = field(default_factory=list)
    has_errors:   bool        = False
    has_warnings: bool        = False
    raw_results:  list        = field(default_factory=list)


def preflight_check(
    data_dir:     str,
    catalog_file: str,
    case_name:    str  = '',
    years:        list = None,
    model:        str  = 'claude-sonnet-4-6-project',
    cache_dir:    Optional[Path] = None,
) -> PreflightReport:
    """
    Run the validation planning agent and execute all checks.
    Returns a PreflightReport for the supervisor to act on.
    """
    years = years or []

    # Load catalog
    with open(catalog_file) as f:
        cat = yaml.safe_load(f)
    catalog = {t['name']: t for t in cat.get('tools', [])}

    context = {
        'data_dir':  data_dir,
        'case_name': case_name,
        'years':     years,
    }

    _cache_dir = cache_dir or Path(data_dir).parent / '.validation_cache'
    # Feasibility must be derived from every catalog requirement. LLM-generated
    # and cached plans are useful for suggestions, but may omit a required
    # check and incorrectly label a workflow as feasible.
    planner = planning_agent.ValidationPlanningAgent(model=model, cache_dir=_cache_dir)
    plan = planner._fallback_plan(catalog, context)

    # Execute checks against canonical paths from the rule-based plan.
    results = [primitives.execute(c) for c in plan]

    # Separate errors and warnings
    errors   = [r for r in results if r['status'] == 'ERROR']
    warnings = [r for r in results if r['status'] == 'WARN']

    # Derive per-tool feasibility
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

    feasible = set()
    partial  = set()
    blocked  = set()
    for name in catalog:
        if tool_errors[name]:
            blocked.add(name)
        elif tool_warnings[name]:
            partial.add(name)
        else:
            feasible.add(name)

    # Discover usable gauge IDs from cross-checks
    gauge_ids = _discover_usable_gauge_ids(data_dir)

    # Format supervisor-friendly summary
    summary = _format_preflight_summary(
        errors, warnings, feasible, partial, blocked, gauge_ids, data_dir)

    return PreflightReport(
        summary      = summary,
        feasible     = feasible,
        partial      = partial,
        blocked      = blocked,
        gauge_ids    = gauge_ids,
        errors       = errors,
        warnings     = warnings,
        has_errors   = bool(errors),
        has_warnings = bool(warnings),
        raw_results  = results,
    )


_E3SM_FILENAME = re.compile(r'^(?P<case>.+)\.(?:elm|mosart|eam)\.h0\.(?P<year>\d{4})-\d{2}\.nc$')


def _autodetect_e3sm_context(data_dir: str) -> dict:
    """Infer case_name and available years from E3SM h0 filenames under ``<data_dir>/e3sm``.

    Uses the ``<case>.<component>.h0.YYYY-MM.nc`` layout checked by
    ``primitives.e3sm_files_exist``. Picks the most common case when several
    are present. Returns ``{'case_name': '', 'years': []}`` if nothing matches.
    """
    e3sm_dir = Path(data_dir) / 'e3sm'
    years_by_case: dict = defaultdict(set)
    counts: Counter = Counter()
    if e3sm_dir.is_dir():
        for subdir in ('lnd', 'rof', 'atm'):
            for f in (e3sm_dir / subdir).glob('*.h0.*.nc'):
                m = _E3SM_FILENAME.match(f.name)
                if m:
                    counts[m['case']] += 1
                    years_by_case[m['case']].add(int(m['year']))
    if not counts:
        return {'case_name': '', 'years': []}
    case_name = counts.most_common(1)[0][0]
    return {'case_name': case_name, 'years': sorted(years_by_case[case_name])}

def _discover_usable_gauge_ids(data_dir: str) -> list:
    """
    Find gauge IDs that exist in ALL THREE sources:
    gauge_metadata.csv, obs/streamflow/*.csv, basin_polygons.geojson

    Sub-directory names are read from ``RuntimeConfig`` so they can be
    overridden via environment variables (``ESFLOW_OBS_SUBDIR``,
    ``ESFLOW_GAUGE_METADATA_FILENAME``, ``ESFLOW_STREAMFLOW_SUBDIR``,
    ``ESFLOW_BASIN_POLYGONS_FILENAME``).

    Returns sorted list of usable IDs, or empty list if any source missing.
    """
    # Lazy import so tests that don't need a live config can still import this
    # module without triggering RuntimeConfig validation.
    try:
        from common.config import runtime_config  # noqa: PLC0415
        obs = runtime_config.obs_paths(Path(data_dir).resolve())
    except Exception:
        # Fallback to conventional layout when config is unavailable.
        d = Path(data_dir).resolve() / 'obs'
        obs = {
            'gauge_metadata': d / 'gauge_metadata.csv',
            'streamflow_dir': d / 'streamflow',
            'basin_polygons': d / 'basin_polygons.geojson',
        }

    try:
        meta_ids = set(
            pd.read_csv(obs['gauge_metadata'])['gauge_id'].astype(str)
        )
    except Exception:
        return []

    try:
        stream_ids = {p.stem for p in obs['streamflow_dir'].glob('*.csv')}
    except Exception:
        stream_ids = set()

    try:
        with open(obs['basin_polygons']) as f:
            gj = json.load(f)
        basin_ids = {str(feat['properties'].get('grdc_no', ''))
                     for feat in gj['features']}
    except Exception:
        basin_ids = set()

    usable = sorted(meta_ids & stream_ids & basin_ids)
    return usable


def _format_preflight_summary(errors, warnings, feasible, partial,
                               blocked, gauge_ids, data_dir) -> str:
    """Format supervisor-friendly preflight summary."""
    W = 70
    lines = [
        '─' * W,
        '⚡ ESFlow Pre-flight Data Check',
        f'   Data directory: {data_dir}',
        '─' * W,
    ]

    if not errors and not warnings:
        lines += ['', '  ✅ All data checks passed — workflow is ready to run.', '']
    else:
        if errors:
            lines += ['', '  🔴 ERRORS (must fix before workflow can run fully):']
            for e in errors:
                lines += [f'     ✗ {e["check_name"]}',
                          f'       {e["detail"]}',
                          f'       → {e.get("fix_hint", "")}']
            lines.append('')

        if warnings:
            lines += ['  🟡 WARNINGS (workflow will run with degraded results):']
            for w in warnings:
                lines += [f'     ⚠ {w["check_name"]}',
                          f'       {w["detail"]}',
                          f'       → {w.get("fix_hint", "")}']
            lines.append('')

    lines += [f'  Tools ready   : {len(feasible)}  |  '
              f'Partial: {len(partial)}  |  '
              f'Blocked: {len(blocked)}']

    if blocked:
        lines += ['', '  Blocked tools (fix errors above to enable):']
        for t in sorted(blocked):
            lines.append(f'    ✗ {t}')

    if gauge_ids:
        lines += ['',
                  '  📍 Recommended gauge IDs (present in metadata + '
                  'streamflow + basin polygons):',
                  f'     {", ".join(str(g) for g in gauge_ids)}']

    lines += ['', '─' * W]
    return '\n'.join(lines)
