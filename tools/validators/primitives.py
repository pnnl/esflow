"""
primitives.py

Generic, atomic validation checks for ESFlow data files.
These never change — they are the stable I/O layer that the
validation planning agent maps catalog descriptions onto.

Each primitive takes explicit args and returns:
  {'passed': bool, 'detail': str}
"""

import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY = {}


def primitive(fn):
    """Register a primitive by function name."""
    REGISTRY[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------------------
# File / directory existence
# ---------------------------------------------------------------------------

@primitive
def file_exists(path: str) -> dict:
    """Check a file exists and is non-empty."""
    p = Path(path)
    if not p.exists():
        return {'passed': False, 'detail': f'File not found: {path}'}
    if p.stat().st_size == 0:
        return {'passed': False, 'detail': f'File exists but is empty: {path}'}
    return {'passed': True, 'detail': f'File exists ({p.stat().st_size:,} bytes): {path}'}


@primitive
def dir_exists(path: str) -> dict:
    """Check a directory exists."""
    p = Path(path)
    if not p.is_dir():
        return {'passed': False, 'detail': f'Directory not found: {path}'}
    return {'passed': True, 'detail': f'Directory exists: {path}'}


@primitive
def dir_has_pattern(path: str, pattern: str) -> dict:
    """Check a directory contains files matching a glob pattern."""
    p = Path(path)
    if not p.is_dir():
        return {'passed': False, 'detail': f'Directory not found: {path}'}
    # Strip any leading '*/' or absolute-path prefix that some callers add —
    # Path.glob on Python 3.12+ rejects non-relative patterns.
    clean_pattern = pattern.lstrip('/')
    if not clean_pattern:
        clean_pattern = '*'
    try:
        matches = list(p.glob(clean_pattern))
    except ValueError:
        # Last-resort fallback for truly pathological patterns
        import fnmatch
        matches = [f for f in p.iterdir() if fnmatch.fnmatch(f.name, clean_pattern)]
    if not matches:
        return {'passed': False,
                'detail': f'No files matching "{pattern}" in {path}'}
    return {'passed': True,
            'detail': f'Found {len(matches)} files matching "{pattern}" in {path}'}


# ---------------------------------------------------------------------------
# CSV checks
# ---------------------------------------------------------------------------

@primitive
def csv_has_columns(path: str, columns: list) -> dict:
    """Check a CSV has all required columns."""
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception as e:
        return {'passed': False, 'detail': f'Could not read CSV {path}: {e}'}
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return {'passed': False,
                'detail': f'Missing columns {missing} in {path}. '
                          f'Found: {list(df.columns)}'}
    return {'passed': True,
            'detail': f'All required columns {columns} present in {path}'}


@primitive
def csv_min_rows(path: str, min_rows: int) -> dict:
    """Check a CSV has at least min_rows data rows."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {'passed': False, 'detail': f'Could not read CSV {path}: {e}'}
    n = len(df)
    if n < min_rows:
        return {'passed': False,
                'detail': f'{path} has {n} rows, expected at least {min_rows}. '
                          f'This may be sample/placeholder data.'}
    return {'passed': True, 'detail': f'{path} has {n} rows (≥ {min_rows})'}


@primitive
def csv_column_values_match_filenames(path: str = None, column: str = None,
                                      file_dir: str = None, extension: str = '.csv',
                                      # LLM sometimes uses these alternative arg names:
                                      csv_path: str = None,
                                      directory: str = None,
                                      filename_pattern: str = None,
                                      dir_path: str = None) -> dict:
    """
    Check that values in a CSV column correspond to files in a directory.
    E.g. gauge_id column values should match <gauge_id>.csv files in obs/streamflow.
    Accepts alternative arg names used by LLM-generated plans:
      path / csv_path, file_dir / directory / dir_path, filename_pattern (ignored).
    """
    # Normalise arg aliases
    resolved_path = path or csv_path
    resolved_dir  = file_dir or directory or dir_path
    if not resolved_path:
        return {'passed': False, 'detail': 'csv_column_values_match_filenames: missing path/csv_path argument'}
    if not resolved_dir:
        return {'passed': False, 'detail': 'csv_column_values_match_filenames: missing file_dir/directory argument'}
    if not column:
        return {'passed': False, 'detail': 'csv_column_values_match_filenames: missing column argument'}

    # Infer extension from filename_pattern if provided, e.g. "{value}.csv" → ".csv"
    if filename_pattern and not extension:
        import re as _re
        m = _re.search(r'(\.\w+)$', filename_pattern)
        if m:
            extension = m.group(1)
    if not extension:
        extension = '.csv'

    try:
        df = pd.read_csv(resolved_path)
    except Exception as e:
        return {'passed': False, 'detail': f'Could not read {resolved_path}: {e}'}

    if column not in df.columns:
        return {'passed': False,
                'detail': f"Column '{column}' not found in {resolved_path}"}

    ids = df[column].astype(str).tolist()
    d   = Path(resolved_dir)
    if not d.is_dir():
        return {'passed': False, 'detail': f'Directory not found: {resolved_dir}'}

    available = {p.stem for p in d.glob(f'*{extension}')}
    missing   = [i for i in ids if i not in available]
    matched   = [i for i in ids if i in available]

    if missing:
        return {'passed': False,
                'detail': f'{len(missing)}/{len(ids)} IDs in column "{column}" of '
                          f'{Path(resolved_path).name} have no matching {extension} file in '
                          f'{resolved_dir}: {missing}. '
                          f'Available files (sample): {sorted(available)[:10]}'}
    return {'passed': True,
            'detail': f'All {len(matched)} IDs in "{column}" have matching '
                      f'{extension} files in {resolved_dir}'}


@primitive
def streamflow_csvs_cover_years(
    directory: str,
    years: list,
    date_column: str = "date",
    pattern: str = "*.csv",
) -> dict:
    """Check that per-gauge streamflow CSVs cover every requested year.

    Directory existence and filename matching alone are insufficient: a workflow
    for 2000--2004 cannot use otherwise valid files containing only 1985--1989.
    """
    path = Path(directory)
    if not path.is_dir():
        return {"passed": False, "detail": f"Directory not found: {directory}"}
    requested = {int(year) for year in years or []}
    if not requested:
        return {"passed": True, "detail": "No requested years specified for streamflow coverage"}
    files = sorted(path.glob(pattern))
    if not files:
        return {"passed": False, "detail": f"No files matching {pattern!r} in {directory}"}

    missing_by_file = []
    usable = 0
    for csv_file in files:
        try:
            dates = pd.to_datetime(pd.read_csv(csv_file, usecols=[date_column])[date_column])
            available = set(dates.dt.year.dropna().astype(int))
        except Exception as exc:
            missing_by_file.append(f"{csv_file.name} unreadable ({exc})")
            continue
        missing = sorted(requested - available)
        if missing:
            missing_by_file.append(f"{csv_file.name} missing {missing}")
        else:
            usable += 1

    if missing_by_file:
        return {
            "passed": False,
            "detail": (
                f"Streamflow CSVs do not cover requested years {sorted(requested)} in {directory}. "
                f"{'; '.join(missing_by_file[:5])}"
            ),
        }
    return {
        "passed": True,
        "detail": f"All {usable} streamflow CSVs cover requested years {sorted(requested)}",
    }


@primitive
def csv_ids_match_csv_ids(path_a: str, col_a: str,
                          path_b: str, col_b: str) -> dict:
    """Check that IDs in col_a of path_a overlap with col_b of path_b."""
    try:
        a = set(pd.read_csv(path_a)[col_a].astype(str))
        b = set(pd.read_csv(path_b)[col_b].astype(str))
    except Exception as e:
        return {'passed': False, 'detail': f'Could not compare IDs: {e}'}
    overlap = a & b
    only_a  = a - b
    if not overlap:
        return {'passed': False,
                'detail': f'No matching IDs between {Path(path_a).name} '
                          f'({col_a}) and {Path(path_b).name} ({col_b}). '
                          f'Sample from A: {sorted(a)[:5]}. '
                          f'Sample from B: {sorted(b)[:5]}.'}
    if only_a:
        return {'passed': False,
                'detail': f'{len(only_a)}/{len(a)} IDs in {Path(path_a).name}.{col_a} '
                          f'not found in {Path(path_b).name}.{col_b}: {sorted(only_a)}'}
    return {'passed': True,
            'detail': f'All {len(overlap)} IDs match between '
                      f'{Path(path_a).name} and {Path(path_b).name}'}


# ---------------------------------------------------------------------------
# GeoJSON checks
# ---------------------------------------------------------------------------

@primitive
def geojson_has_property(path: str, property_name: str = None,
                         property: str = None) -> dict:
    """Check all GeoJSON features have a given property.
    Accepts both ``property_name`` and ``property`` (LLM sometimes uses the latter).
    """
    prop = property_name or property
    if not prop:
        return {'passed': False, 'detail': 'geojson_has_property: missing property_name argument'}

    try:
        with open(path) as f:
            gj = json.load(f)
    except Exception as e:
        return {'passed': False, 'detail': f'Could not parse GeoJSON {path}: {e}'}

    features = gj.get('features', [])
    if not features:
        return {'passed': False, 'detail': f'GeoJSON has no features: {path}'}

    missing = [i for i, feat in enumerate(features)
               if prop not in feat.get('properties', {})]
    if missing:
        return {'passed': False,
                'detail': f'{len(missing)}/{len(features)} features missing '
                          f'property "{prop}" in {path}'}
    return {'passed': True,
            'detail': f'All {len(features)} features have property '
                      f'"{prop}" in {path}'}


@primitive
def geojson_ids_match_csv_column(geojson_path: str, geojson_property: str,
                                  csv_path: str, csv_column: str) -> dict:
    """Check GeoJSON feature property values overlap with a CSV column."""
    try:
        with open(geojson_path) as f:
            gj = json.load(f)
        geojson_ids = {str(feat['properties'].get(geojson_property, ''))
                       for feat in gj['features']}
        csv_ids = set(pd.read_csv(csv_path)[csv_column].astype(str))
    except Exception as e:
        return {'passed': False,
                'detail': f'Could not compare GeoJSON to CSV: {e}'}

    unmatched = csv_ids - geojson_ids
    if unmatched:
        return {'passed': False,
                'detail': f'{len(unmatched)}/{len(csv_ids)} CSV "{csv_column}" values '
                          f'not found in GeoJSON "{geojson_property}": {sorted(unmatched)}. '
                          f'Available GeoJSON IDs: {sorted(geojson_ids)}'}
    return {'passed': True,
            'detail': f'All {len(csv_ids)} CSV "{csv_column}" values match '
                      f'GeoJSON "{geojson_property}" values'}


# ---------------------------------------------------------------------------
# NetCDF checks
# ---------------------------------------------------------------------------

@primitive
def netcdf_has_variable(path: str, variable: str) -> dict:
    """Check a NetCDF file contains a given variable."""
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        if variable not in ds:
            return {'passed': False,
                    'detail': f'Variable "{variable}" not found in {path}. '
                              f'Available: {list(ds.data_vars)}'}
        return {'passed': True,
                'detail': f'Variable "{variable}" present in {path}'}
    except Exception as e:
        return {'passed': False,
                'detail': f'Could not open NetCDF {path}: {e}'}


@primitive
def netcdf_has_time_coverage(path: str, years: list) -> dict:
    """Check a NetCDF file covers the requested years."""
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        if 'time' not in ds:
            return {'passed': False,
                    'detail': f'No time dimension in {path}'}
        times = pd.to_datetime(ds['time'].values)
        avail_years = set(times.year)
        missing = [y for y in years if y not in avail_years]
        if missing:
            return {'passed': False,
                    'detail': f'Years {missing} not covered in {path}. '
                              f'Available: {sorted(avail_years)}'}
        return {'passed': True,
                'detail': f'All years {years} covered in {path}'}
    except Exception as e:
        return {'passed': False,
                'detail': f'Could not check time coverage in {path}: {e}'}


# ---------------------------------------------------------------------------
# E3SM-specific checks
# ---------------------------------------------------------------------------

@primitive
def e3sm_files_exist(data_dir: str, case_name: str,
                     component: str, years: list) -> dict:
    """Check E3SM output files exist for a given component and year range.

    ``data_dir`` should be the **root** data directory (e.g. ``/data/sample``).
    The primitive resolves the correct component sub-directory internally:
      elm    → e3sm/lnd/
      mosart → e3sm/rof/
      eam    → e3sm/atm/

    If ``data_dir`` happens to already point at the component directory (a
    common LLM mistake where it passes ``data/sample/e3sm/lnd``), the
    primitive detects the overlap and avoids double-prepending.
    """
    subdir_map = {'elm': 'lnd', 'mosart': 'rof', 'eam': 'atm'}
    suffix_map = {'elm': 'elm.h0', 'mosart': 'mosart.h0', 'eam': 'eam.h0'}

    subdir  = subdir_map.get(component, component)
    suffix  = suffix_map.get(component, f'{component}.h0')

    data_path = Path(data_dir)

    # Detect if the path already ends with e3sm/<subdir> or just <subdir>
    # to avoid double-prepending (e.g. data/sample/e3sm/lnd/e3sm/lnd/).
    parts = data_path.parts
    if len(parts) >= 2 and parts[-2] == 'e3sm' and parts[-1] == subdir:
        # Already points at the component directory
        comp_dir = data_path
    elif len(parts) >= 1 and parts[-1] == subdir:
        # Points at subdir directly without the e3sm parent
        comp_dir = data_path
    else:
        comp_dir = data_path / 'e3sm' / subdir

    if not comp_dir.is_dir():
        return {'passed': False,
                'detail': f'E3SM {component} directory not found: {comp_dir}'}

    pattern = f'{case_name}.{suffix}.*.nc'
    files   = list(comp_dir.glob(pattern))
    if not files:
        return {'passed': False,
                'detail': f'No files matching {pattern} in {comp_dir}'}

    if years:
        avail = set()
        for f in files:
            m = re.search(r'\.(\d{4})-\d{2}\.nc$', f.name)
            if m:
                avail.add(int(m.group(1)))
        missing = [y for y in years if y not in avail]
        if missing:
            return {'passed': False,
                    'detail': f'Years {missing} missing from {component} output. '
                              f'Available: {sorted(avail)}'}

    return {'passed': True,
            'detail': f'Found {len(files)} {component} files '
                      f'(case: {case_name})'}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _normalise_args(prim_name: str, args: dict) -> dict:
    """Normalise LLM-generated arg names to match primitive signatures.

    The LLM sometimes uses slightly different kwarg names than the primitive
    expects (e.g. ``property`` vs ``property_name``, ``start_year``/``end_year``
    vs ``years``, absolute paths in ``pattern``, etc.).  Rather than adding
    aliases to every primitive, we fix them here in one place.
    """
    a = dict(args)  # shallow copy — don't mutate the original

    if prim_name == 'dir_has_pattern':
        # LLM sometimes passes an absolute path as pattern (e.g.
        # "/data/sample/*.nc") instead of just "*.nc".  Keep only the
        # last path component so Path.glob() receives a relative pattern.
        pat = a.get('pattern', '')
        if pat and ('/' in pat):
            from pathlib import PurePosixPath
            a['pattern'] = PurePosixPath(pat).name or pat

    elif prim_name == 'geojson_ids_match_csv_column':
        # LLM uses 'property_name' or 'geojson_property_name' instead of 'geojson_property'
        if 'geojson_property' not in a:
            a['geojson_property'] = (
                a.pop('property_name', None)
                or a.pop('geojson_property_name', None)
                or a.pop('property', None)
            )
        # LLM uses 'column' instead of 'csv_column'
        if 'csv_column' not in a and 'column' in a:
            a['csv_column'] = a.pop('column')

    elif prim_name == 'netcdf_has_time_coverage':
        # LLM sometimes passes start_year + end_year instead of years list
        if 'years' not in a and ('start_year' in a or 'end_year' in a):
            start = a.pop('start_year', None)
            end   = a.pop('end_year', None)
            if start is not None and end is not None:
                a['years'] = list(range(int(start), int(end) + 1))
            elif start is not None:
                a['years'] = [int(start)]
            else:
                a['years'] = []

    return a


def execute(check: dict) -> dict:
    """
    Execute a single validation check dict produced by the planning agent.

    Expected check format:
      {
        'check_name':       str,
        'primitive':        str,       # must be in REGISTRY
        'args':             dict,      # kwargs passed to primitive
        'failure_severity': str,       # ERROR | WARN — severity reported when the
                                       # check FAILS; 'OK' is always reported on pass.
                                       # Legacy plans may use 'severity' instead.
        'affected_tools':   list[str],
        'fix_hint':         str,
      }

    Returns the check dict augmented with 'status' and 'detail'.
      status='OK'    → check passed
      status='ERROR' → check failed and failure_severity is ERROR
      status='WARN'  → check failed and failure_severity is WARN
    """
    prim_name = check.get('primitive')
    fn        = REGISTRY.get(prim_name)
    result    = check.copy()

    # Support both new ('failure_severity') and legacy ('severity') field names so
    # that cached plans generated before the rename continue to work correctly.
    failure_sev = (
        check.get('failure_severity')
        or check.get('severity')
        or 'ERROR'
    )

    if fn is None:
        result['status'] = 'WARN'
        result['detail'] = (f'Unknown primitive "{prim_name}" — '
                            f'skipping check "{check.get("check_name")}"')
        return result

    try:
        normalised_args = _normalise_args(prim_name, check.get('args', {}))
        outcome = fn(**normalised_args)
        result['status'] = 'OK' if outcome['passed'] else failure_sev
        result['detail'] = outcome['detail']
    except Exception as e:
        result['status'] = failure_sev
        result['detail'] = f'Primitive "{prim_name}" raised: {e}'

    return result


def list_primitives() -> list:
    """Return names and docstrings of all registered primitives."""
    return [{'name': name, 'description': fn.__doc__.strip()}
            for name, fn in REGISTRY.items()]
