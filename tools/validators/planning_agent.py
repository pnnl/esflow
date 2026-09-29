"""
planning_agent.py

LLM-powered validation planning agent for ESFlow.

Given the tool catalog and runtime context (data_dir, case_name, years),
the agent generates a structured validation plan — a list of primitive
check calls that collectively validate all data requirements for all tools.

The plan is cached by catalog hash so the LLM is only called when the
catalog changes. At runtime, the plan is executed by primitives.execute().
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import Optional

import yaml

from . import primitives


# Try to import PNNL Incubator config; fall back to standard OpenAI/Anthropic if not available
try:
    from common.config import runtime_config, INCUBATOR_BASE_URL
    _HAS_INCUBATOR = True
except ImportError:
    _HAS_INCUBATOR = False


def _default_cache_dir() -> Path:
    """Return the validation-plan cache directory.

    Resolution order:
    1. ``ESFLOW_VALIDATION_CACHE_DIR`` env var (or ``RuntimeConfig`` if available)
    2. ``ESFLOW_OUTPUT_DIR``/.validation_cache (or <repo>/outputs/.validation_cache)
    3. Hard fallback to /tmp/esflow_validation_cache
    """
    if _HAS_INCUBATOR:
        try:
            return runtime_config.resolved_validation_cache_dir()
        except Exception:
            pass
    # Env-var fallback without the full config object
    explicit = os.environ.get('ESFLOW_VALIDATION_CACHE_DIR')
    if explicit:
        return Path(explicit)
    output_dir = os.environ.get('ESFLOW_OUTPUT_DIR')
    if output_dir:
        return Path(output_dir) / '.validation_cache'
    return Path('/tmp/esflow_validation_cache')


def _obs_sub_paths(data_dir: str, context: dict) -> dict:
    """Return resolved observation sub-paths for *data_dir*.

    Looks up layout names from *context* first (keys: ``obs_subdir``,
    ``gauge_metadata_filename``, ``streamflow_subdir``,
    ``basin_polygons_filename``, ``ilamb_cache_subdir``), then falls back to
    ``RuntimeConfig`` env-vars, then to the ESFlow conventional layout.
    """
    obs_subdir              = context.get('obs_subdir')
    gauge_metadata_filename = context.get('gauge_metadata_filename')
    streamflow_subdir       = context.get('streamflow_subdir')
    basin_polygons_filename = context.get('basin_polygons_filename')
    ilamb_cache_subdir      = context.get('ilamb_cache_subdir')

    if _HAS_INCUBATOR:
        try:
            cfg = runtime_config
            obs_subdir              = obs_subdir              or cfg.ESFLOW_OBS_SUBDIR
            gauge_metadata_filename = gauge_metadata_filename or cfg.ESFLOW_GAUGE_METADATA_FILENAME
            streamflow_subdir       = streamflow_subdir       or cfg.ESFLOW_STREAMFLOW_SUBDIR
            basin_polygons_filename = basin_polygons_filename or cfg.ESFLOW_BASIN_POLYGONS_FILENAME
            ilamb_cache_subdir      = ilamb_cache_subdir      or cfg.ESFLOW_ILAMB_CACHE_SUBDIR
        except Exception:
            pass

    # Env-var fallback (no config object needed)
    obs_subdir              = obs_subdir              or os.environ.get('ESFLOW_OBS_SUBDIR',              'obs')
    gauge_metadata_filename = gauge_metadata_filename or os.environ.get('ESFLOW_GAUGE_METADATA_FILENAME', 'gauge_metadata.csv')
    streamflow_subdir       = streamflow_subdir       or os.environ.get('ESFLOW_STREAMFLOW_SUBDIR',       'streamflow')
    basin_polygons_filename = basin_polygons_filename or os.environ.get('ESFLOW_BASIN_POLYGONS_FILENAME', 'basin_polygons.geojson')
    ilamb_cache_subdir      = ilamb_cache_subdir      or os.environ.get('ESFLOW_ILAMB_CACHE_SUBDIR',     'ilamb_cache')

    base = f'{data_dir}/{obs_subdir}'
    return {
        'gauge_metadata':  f'{base}/{gauge_metadata_filename}',
        'streamflow_dir':  f'{base}/{streamflow_subdir}',
        'basin_polygons':  f'{base}/{basin_polygons_filename}',
        'ilamb_cache':     f'{base}/{ilamb_cache_subdir}',
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a data validation planning agent for ESFlow, an Earth System Model
analysis framework. Your job is to read the ESFlow tool catalog and produce
a comprehensive, structured validation plan that checks whether a user's data
directory contains all the data required to run each tool.

You have access to the following validation primitives — atomic checks that
can be executed against the filesystem:

{primitives}

RULES:
1. For EVERY required input across ALL tools in the catalog, produce at least
   one check using the most appropriate primitive.
2. Where inputs are files, always check file_exists first, then structural checks.
3. Where multiple tools share the same input type (e.g. gauge_metadata CSV),
   produce ONE shared check and list ALL affected tools.
4. Where inputs must be consistent across tools (e.g. gauge_id in metadata
   must match filenames in obs/streamflow/ AND grdc_no in basin GeoJSON),
   produce explicit cross-consistency checks.
5. Use severity ERROR for checks whose failure blocks a tool entirely.
   Use severity WARN for checks whose failure degrades but doesn't block.
6. PATH RULES — follow exactly:
   - data_dir root: {data_dir}
   - E3SM land (ELM) files: {data_dir}/e3sm/lnd/*.nc
   - E3SM river (MOSART) files: {data_dir}/e3sm/rof/*.nc
   - Observation files root: {data_dir}/obs/
   - Basin polygons GeoJSON: {data_dir}/obs/basin_polygons.geojson
     (the filename is ALWAYS basin_polygons.geojson — never basins.geojson)
   - Gauge metadata CSV: {data_dir}/obs/gauge_metadata.csv
   - Streamflow obs directory: {data_dir}/obs/streamflow/
   - ILAMB cached files: {data_dir}/obs/ilamb_cache/{{variable}}_{{dataset}}.nc
     e.g. {data_dir}/obs/ilamb_cache/pr_GPCPv2.3.nc
          {data_dir}/obs/ilamb_cache/evspsbl_GLEAMv3.3a.nc
          {data_dir}/obs/ilamb_cache/mrro_LORA.nc
     These files are DOWNLOADED at runtime — use severity WARN, not ERROR.
   - Case name: {case_name}
   - Years to check: {years}
7. For E3SM file checks, use the e3sm_files_exist primitive. Pass data_dir as
   the ROOT data directory ({data_dir}), NOT the component subdirectory.
   The primitive resolves e3sm/lnd/ or e3sm/rof/ internally from the component name.
   Example: {{"data_dir": "{data_dir}", "component": "elm", "case_name": "{case_name}", "years": {years}}}
8. Do NOT produce checks for tool_catalog.yaml or any config/code files —
   only check data files (NetCDF, CSV, GeoJSON) that tools read as inputs.
9. For optional inputs (required: false), skip checks entirely.
10. Do NOT invent paths — use ONLY the path patterns listed in rule 6.

Output a JSON array of check objects. Each object must have exactly these keys:
  - check_name:       string — unique snake_case identifier
  - primitive:        string — must be one of the registered primitive names
  - args:             object — keyword arguments for the primitive
  - failure_severity: string — ERROR or WARN  (the severity to report IF this check fails;
                      OK is reported when the check passes regardless of this value)
  - affected_tools:   array of strings — tool names from the catalog
  - fix_hint:         string — actionable advice if this check fails

Respond with ONLY the JSON array. No markdown, no explanation, no code blocks.
"""

USER_PROMPT = """
Here is the complete ESFlow tool catalog:

{catalog_yaml}

Generate the validation plan now.
"""


# ---------------------------------------------------------------------------
# Plan cache
# ---------------------------------------------------------------------------

    key_material = yaml.dump(
        {
            'catalog': catalog,
            'data_dir': context.get('data_dir') or '',
            'case_name': context.get('case_name') or '',
            'years': context.get('years') or [],
        },
        sort_keys=True,
    )
    return hashlib.md5(key_material.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, catalog: dict, context: dict) -> Path:
    return cache_dir / f'validation_plan_{_catalog_hash(catalog, context)}.json'


def load_cached_plan(cache_dir: Path, catalog: dict, context: dict) -> Optional[list]:
    p = _cache_path(cache_dir, catalog, context)
    if p.exists():
        print(f"  Validation plan cache hit: {p.name}")
        return json.loads(p.read_text())
    return None


def save_plan(cache_dir: Path, catalog: dict, context: dict, plan: list):
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, catalog, context)
    p.write_text(json.dumps(plan, indent=2))
    print(f"  Validation plan cached: {p.name}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ValidationPlanningAgent:
    """
    Calls an LLM to generate a validation plan from the tool catalog.
    Supports OpenAI-compatible APIs. Falls back to a rule-based plan
    if no API key is available.
    """

    def __init__(self, model: str = 'claude-sonnet-4-6-project', cache_dir: Optional[Path] = None):
        self.model     = model
        self.cache_dir = cache_dir or _default_cache_dir()

    def generate(self, catalog: dict, context: dict) -> list:
        """
        Generate (or load from cache) a validation plan for the given catalog.

        Args:
            catalog: dict of {tool_name: tool_def} from tool_catalog.yaml
            context: {data_dir, case_name, years}

        Returns:
            list of check dicts ready for primitives.execute()
        """
        # Check cache first — keyed on catalog + data_dir so different roots
        # never share (and corrupt) each other's plans.
        cached = load_cached_plan(self.cache_dir, catalog, context)
        if cached:
            # Re-resolve dynamic paths in cached plan
            return self._resolve_paths(cached, context)

        print(f"  Generating validation plan via LLM ({self.model})...")
        plan = self._call_llm(catalog, context)

        if plan:
            save_plan(self.cache_dir, catalog, context, plan)
            # Run path resolution on the freshly-generated plan too so that
            # any LLM-hallucinated relative paths or wrong filenames are fixed
            # before the plan is executed (not just when loaded from cache).
            return self._resolve_paths(plan, context)

        # Fallback if LLM unavailable
        print("  LLM unavailable — using rule-based fallback plan.")
        return self._fallback_plan(catalog, context)

    def _call_llm(self, catalog: dict, context: dict) -> Optional[list]:
        """Call the LLM API to generate the validation plan."""
        # Try PNNL Incubator first (if available)
        if _HAS_INCUBATOR:
            try:
                return self._call_incubator(catalog, context)
            except Exception as e:
                print(f"  PNNL Incubator call failed: {e}")

        # Fall back to standard OpenAI/Anthropic
        api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return None

        primitives_desc = '\n'.join(
            f'  - {p["name"]}: {p["description"]}'
            for p in primitives.list_primitives()
        )

        system = SYSTEM_PROMPT.format(
            primitives=primitives_desc,
            data_dir=context.get('data_dir', '{data_dir}'),
            case_name=context.get('case_name', ''),
            years=context.get('years', []),
        )
        user = USER_PROMPT.format(
            catalog_yaml=yaml.dump(catalog, sort_keys=False)
        )

        try:
            # Try OpenAI first
            if os.environ.get('OPENAI_API_KEY'):
                return self._call_openai(system, user)
            # Try Anthropic
            if os.environ.get('ANTHROPIC_API_KEY'):
                return self._call_anthropic(system, user)
        except Exception as e:
            print(f"  LLM call failed: {e}")
            return None

    def _call_incubator(self, catalog: dict, context: dict) -> Optional[list]:
        """Call PNNL AI Incubator (OpenAI-compatible Depot gateway)."""
        from openai import OpenAI

        primitives_desc = '\n'.join(
            f'  - {p["name"]}: {p["description"]}'
            for p in primitives.list_primitives()
        )

        system = SYSTEM_PROMPT.format(
            primitives=primitives_desc,
            data_dir=context.get('data_dir', '{data_dir}'),
            case_name=context.get('case_name', ''),
            years=context.get('years', []),
        )
        user = USER_PROMPT.format(
            catalog_yaml=yaml.dump(catalog, sort_keys=False)
        )

        client = OpenAI(
            api_key=runtime_config.AI_INCUBATOR_KEY,
            base_url=INCUBATOR_BASE_URL,
        )
        response = client.chat.completions.create(
            model=self.model,  # e.g., 'claude-sonnet-4-6-project'
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
            temperature=0,
            response_format={'type': 'json_object'},
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else parsed.get('checks', [])

    def _call_openai(self, system: str, user: str) -> Optional[list]:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
            temperature=0,
            response_format={'type': 'json_object'},
        )
        raw = response.choices[0].message.content
        # Model may return {"checks": [...]} or just [...]
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else parsed.get('checks', [])

    def _call_anthropic(self, system: str, user: str) -> Optional[list]:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model='claude-3-5-haiku-latest',
            max_tokens=4096,
            system=system,
            messages=[{'role': 'user', 'content': user}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else parsed.get('checks', [])

    def _resolve_paths(self, plan: list, context: dict) -> list:
        """
        Re-inject runtime values (data_dir, case_name, years) into a
        cached plan whose args may contain placeholder strings or relative paths.

        Handles:
        1. Placeholder strings like {data_dir} and {case_name}
        2. Relative paths like 'data/obs/...' which need to be made absolute
           (strips any leading components that overlap data_dir and rejoins)
        3. Filename normalisation: replaces any LLM-hallucinated basin-polygons
           filename (e.g. "basins.geojson") with the value configured in
           ESFLOW_BASIN_POLYGONS_FILENAME so stale cached plans never fail
           because the LLM used the wrong name.
        """
        data_dir  = context.get('data_dir') or ''
        case_name = context.get('case_name') or ''
        years     = context.get('years') or []
        data_dir_path = Path(data_dir).resolve() if data_dir else Path()

        # Canonical filenames from config (fall back to defaults if unavailable)
        try:
            from common.config import runtime_config as _rc  # noqa: PLC0415
            _basin_polygons_filename = _rc.ESFLOW_BASIN_POLYGONS_FILENAME
            _gauge_metadata_filename  = _rc.ESFLOW_GAUGE_METADATA_FILENAME
        except Exception:
            _basin_polygons_filename = 'basin_polygons.geojson'
            _gauge_metadata_filename  = 'gauge_metadata.csv'

        def _normalise_filename(path_str: str) -> str:
            """Fix well-known filenames the LLM may have got wrong."""
            p = Path(path_str)
            # Basin polygons: any *.geojson in the obs dir → configured filename
            if p.suffix == '.geojson' and p.name != _basin_polygons_filename:
                return str(p.parent / _basin_polygons_filename)
            return path_str

        def resolve_value(v):
            """Convert placeholder strings and relative paths to absolutes."""
            if isinstance(v, str):
                # First handle placeholder strings
                v = v.replace('{data_dir}', data_dir).replace('{case_name}', case_name)

                # Then check if it's a relative path that needs to be made absolute.
                # Paths contain '/' or end with file extensions.
                if ('/' in v or v.endswith(('.csv', '.nc', '.json', '.yaml', '.geojson'))) and v:
                    path = Path(v)

                    # Already absolute — check for double-prepend (path already
                    # starts with data_dir_path) and return as-is if so.
                    if path.is_absolute():
                        try:
                            path.relative_to(data_dir_path)
                            return _normalise_filename(str(path))
                        except ValueError:
                            return _normalise_filename(str(path))

                    # Relative path: strip any leading components that duplicate
                    # data_dir_path so we don't double-prepend.
                    #
                    # data_dir_path parts: e.g. [..., '<repo>', 'data', 'sample']
                    # path parts example:        ['data', 'sample', 'obs', 'gauge_metadata.csv']
                    #
                    # Walk data_dir_path from its last component backwards and
                    # strip matching leading parts from path.
                    data_parts = data_dir_path.parts  # absolute parts
                    path_parts = path.parts            # relative parts

                    # Find the longest suffix of data_parts that is a prefix of path_parts.
                    stripped = path_parts
                    for start in range(len(data_parts)):
                        suffix = data_parts[start:]
                        if path_parts[:len(suffix)] == suffix:
                            stripped = path_parts[len(suffix):]
                            break

                    if stripped:
                        resolved = str(data_dir_path / Path(*stripped))
                    else:
                        # path_parts were entirely consumed by data_dir — return data_dir itself
                        resolved = str(data_dir_path)
                    return _normalise_filename(resolved)
                return v
            elif isinstance(v, dict):
                return {k: resolve_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [resolve_value(item) for item in v]
            return v

        resolved = []
        for check in plan:
            c = check.copy()
            c['args'] = resolve_value(check.get('args', {}))

            # Inject years into e3sm_files_exist checks
            if c.get('primitive') == 'e3sm_files_exist' and years:
                c['args']['years'] = years
            resolved.append(c)
        return resolved

    def _fallback_plan(self, catalog: dict, context: dict) -> list:
        """Pure rule-based plan derived from catalog structure.

        Observation sub-directory names are resolved from *context* first, then
        from environment variables / ``RuntimeConfig``, so no paths are
        hard-coded here.
        """
        data_dir  = context.get('data_dir') or ''
        case_name = context.get('case_name') or ''
        years     = context.get('years') or []
        checks    = []
        seen      = set()

        # Resolve obs layout paths once — uses env/config overrides if set.
        obs = _obs_sub_paths(data_dir, context)
        gauge_meta_path   = obs['gauge_metadata']
        streamflow_path   = obs['streamflow_dir']
        basin_geojson_path = obs['basin_polygons']
        ilamb_cache_path  = obs['ilamb_cache']

        def add(check_name, primitive, args, failure_severity, tools, hint):
            key = (primitive, json.dumps(args, sort_keys=True))
            if key in seen:
                for c in checks:
                    if (c['primitive'], json.dumps(c['args'], sort_keys=True)) == key:
                        c['affected_tools'] = sorted(
                            set(c['affected_tools']) | set(tools))
                return
            seen.add(key)
            checks.append({
                'check_name':       check_name,
                'primitive':        primitive,
                'args':             args,
                'failure_severity': failure_severity,
                'affected_tools':   list(tools),
                'fix_hint':         hint,
            })

        for tool_name, tool_def in catalog.items():
            for inp_name, inp_def in tool_def.get('inputs', {}).items():
                if not inp_def.get('required', False):
                    continue

                desc = inp_def.get('description', '').lower()

                # --- gauge metadata CSV ---
                if 'gauge_metadata' in inp_name or \
                   ('gauge' in desc and 'metadata' in desc):
                    add('gauge_metadata_exists', 'file_exists',
                        {'path': gauge_meta_path}, 'ERROR', [tool_name],
                        f'Provide {gauge_meta_path} with columns: '
                        'gauge_id, lat, lon, area_km2, river_name')
                    add('gauge_metadata_columns', 'csv_has_columns',
                        {'path': gauge_meta_path,
                         'columns': ['gauge_id', 'lat', 'lon', 'area_km2']},
                        'ERROR', [tool_name],
                        f'Add missing columns to {gauge_meta_path}')
                    add('gauge_metadata_min_rows', 'csv_min_rows',
                        {'path': gauge_meta_path, 'min_rows': 3}, 'WARN', [tool_name],
                        'Provide real GRDC gauge data — sample data detected')

                # --- basin GeoJSON ---
                elif 'geojson' in desc or 'basin polygon' in desc or 'grdc_no' in desc:
                    add('basin_geojson_exists', 'file_exists',
                        {'path': basin_geojson_path}, 'ERROR', [tool_name],
                        f'Provide {basin_geojson_path} with grdc_no property')
                    add('basin_geojson_grdc_no', 'geojson_has_property',
                        {'path': basin_geojson_path, 'property_name': 'grdc_no'},
                        'ERROR', [tool_name],
                        'Each GeoJSON feature must have a grdc_no property')
                    add('basin_polygon_gauge_match',
                        'geojson_ids_match_csv_column',
                        {'geojson_path': basin_geojson_path,
                         'geojson_property': 'grdc_no',
                         'csv_path': gauge_meta_path,
                         'csv_column': 'gauge_id'},
                        'WARN', [tool_name],
                        'gauge_metadata gauge_id values must match grdc_no in basin polygons')

                # --- streamflow obs dir ---
                elif 'obs_dir' in inp_name or \
                     ('per-gauge csv' in desc and 'discharge' in desc):
                    add('streamflow_dir_exists', 'dir_exists',
                        {'path': streamflow_path}, 'ERROR', [tool_name],
                        f'Create {streamflow_path} directory with per-gauge CSVs')
                    add('streamflow_csvs_exist', 'dir_has_pattern',
                        {'path': streamflow_path, 'pattern': '*.csv'},
                        'ERROR', [tool_name],
                        'Add per-gauge CSV files named <gauge_id>.csv')
                    add('streamflow_gauge_id_match',
                        'csv_column_values_match_filenames',
                        {'path': gauge_meta_path,
                         'column': 'gauge_id',
                         'file_dir': streamflow_path,
                         'extension': '.csv'},
                        'WARN', [tool_name],
                        'gauge_metadata gauge_id must match streamflow filenames')
                    if years:
                        add('streamflow_year_coverage',
                            'streamflow_csvs_cover_years',
                            {'directory': streamflow_path, 'years': years},
                            'ERROR', [tool_name],
                            f'Provide streamflow CSVs in {streamflow_path} covering years {years}')

                # --- E3SM data_dir ---
                elif inp_name == 'data_dir' and 'e3sm' in desc and case_name:
                    for comp in ['elm', 'mosart']:
                        add(f'e3sm_{comp}_files', 'e3sm_files_exist',
                            {'data_dir': data_dir, 'case_name': case_name,
                             'component': comp, 'years': years},
                            'ERROR', [tool_name],
                            f'Ensure E3SM {comp} output files exist in {data_dir}/e3sm/')

                # --- ILAMB obs NetCDF ---
                elif 'ilamb' in desc or \
                     ('observation netcdf' in desc and inp_name == 'obs_file'):
                    add('ilamb_cache_dir', 'dir_exists',
                        {'path': ilamb_cache_path}, 'WARN', [tool_name],
                        'ILAMB files will be downloaded on first run')

        return checks


def get_plan(catalog: dict, context: dict,
             cache_dir: Optional[Path] = None,
             model: str = 'claude-sonnet-4-6-project') -> list:
    """Get a validation plan, using cache if available."""
    agent = ValidationPlanningAgent(model=model, cache_dir=cache_dir)
    return agent.generate(catalog, context)
