"""
ESM-specific utilities for E3SM model data.

Provides:
- cftime calendar conversion
- E3SM file discovery (archive structure)
- Dataset opening with chunking
- Mesh type detection
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


def cftime_to_datetime(time_values) -> pd.Index:
    """Convert xarray time values to pandas Index.

    Handles ESM's non-standard calendars (noleap, 365_day, 360_day).
    """
    try:
        return pd.to_datetime(time_values)
    except (TypeError, ValueError):
        pass

    try:
        return pd.to_datetime([str(t) for t in time_values])
    except (TypeError, ValueError):
        pass

    try:
        return pd.Index([f"{t.year:04d}-{t.month:02d}-01" for t in time_values])
    except AttributeError:
        pass

    return pd.RangeIndex(len(time_values))


def detect_mesh_type(ds: xr.Dataset) -> str:
    """Detect mesh type from dataset dimensions."""
    dims = set(ds.dims)

    if ('lat' in dims and 'lon' in dims) or ('nj' in dims and 'ni' in dims):
        return 'latlon'
    if 'ncol' in dims or 'nCells' in dims or 'gridcell' in dims:
        return 'unstructured'

    spatial_dims = [d for d in dims if d not in ['time', 'hist_interval', 'nbnd']]
    if len(spatial_dims) == 2:
        return 'latlon'
    elif len(spatial_dims) == 1:
        return 'unstructured'
    return 'unknown'


def find_e3sm_files(
    base_dir: Union[str, Path],
    case_name: str,
    component: str,
    years: List[int],
    file_pattern: str = None,
    frequency: str = 'monthly',
) -> List[Path]:
    """Find E3SM output files with flexible path searching.

    Args:
        frequency: 'monthly' (default, matches h0/monthly files) or
                   'daily' (matches h1/daily files).

    Searches in order:
    1. {base_dir}/{case_name}/archive/{comp}/hist/
    2. {base_dir}/{case_name}/run/
    3. {base_dir}/{case_name}/{comp}/hist/
    4. {base_dir}/{case_name}/
    5. {base_dir}/
    """
    base_dir = Path(base_dir)

    comp_map = {
        'mosart': 'mosart', 'rof': 'mosart',
        'elm': 'lnd', 'lnd': 'lnd',
        'eam': 'atm', 'atm': 'atm',
        'mpaso': 'ocn', 'ocn': 'ocn',
    }
    comp_name = comp_map.get(component.lower(), component.lower())

    archive_map = {
        'mosart': 'rof', 'rof': 'rof',
        'elm': 'lnd', 'lnd': 'lnd',
        'eam': 'atm', 'atm': 'atm',
    }
    archive_dir = archive_map.get(component.lower(), component.lower())

    search_paths = [
        base_dir / case_name / 'archive' / archive_dir / 'hist',
        base_dir / case_name / 'run',
        base_dir / case_name / archive_dir / 'hist',
        base_dir / case_name,
        base_dir / f'{comp_name}_output',
        base_dir / comp_name,
        base_dir / archive_dir,
        base_dir,
    ]

    freq = frequency.lower().strip()
    is_daily = freq in ('daily', 'day', 'd', 'h1')

    if file_pattern is None:
        orig_comp = component.lower()
        if is_daily:
            patterns = [
                f"{case_name}.{comp_name}.h1.*.nc",
                f"{case_name}.{orig_comp}.h1.*.nc",
                f"{case_name}.{comp_name}.daily.*.nc",
                f"{case_name}.{orig_comp}.daily.*.nc",
                f"*.{comp_name}.h1.*.nc",
                f"*.{orig_comp}.h1.*.nc",
                f"*.{comp_name}.daily.*.nc",
                f"*.{orig_comp}.daily.*.nc",
            ]
        else:
            patterns = [
                f"{case_name}.{comp_name}.h0.*.nc",
                f"{case_name}.{orig_comp}.h0.*.nc",
                f"{case_name}.{comp_name}.h?.*.nc",
                f"{case_name}.{orig_comp}.h?.*.nc",
                f"{case_name}.{comp_name}.monthly.*.nc",
                f"{case_name}.{orig_comp}.monthly.*.nc",
                f"*.{comp_name}.h0.*.nc",
                f"*.{orig_comp}.h0.*.nc",
                f"*.{comp_name}.monthly.*.nc",
                f"*.{orig_comp}.monthly.*.nc",
                f"*.{comp_name}.*.nc",
                f"*.{orig_comp}.*.nc",
            ]
    else:
        patterns = [file_pattern]

    found_files = []
    for search_path in search_paths:
        if not search_path.exists():
            continue
        for pattern in patterns:
            matches = list(search_path.glob(pattern))
            if matches:
                found_files.extend(matches)
                break
        if found_files:
            break

    if not found_files:
        for pattern in patterns:
            matches = list(base_dir.rglob(pattern))
            if matches:
                found_files.extend(matches)
                break

    if not found_files:
        searched = [str(p) for p in search_paths if p.exists()]
        raise FileNotFoundError(
            f"No {component} files found for case '{case_name}'. "
            f"Searched: {searched or [str(p) for p in search_paths[:3]]}"
        )

    if years:
        year_files = []
        for f in found_files:
            for year in years:
                year_str = f"{year:04d}"
                if (f"-{year_str}-" in f.name or
                    f".{year_str}-" in f.name or
                    f".{year_str}." in f.name or
                    f.name.endswith(f".{year_str}.nc")):
                    year_files.append(f)
                    break
        if year_files:
            found_files = year_files

    return sorted(set(found_files))


def open_e3sm_dataset(
    files: Union[str, Path, List[Path]],
    variables: List[str] = None,
    chunks: Dict = None
) -> xr.Dataset:
    """Open E3SM dataset with appropriate settings."""
    if isinstance(files, (str, Path)):
        files = [Path(files)]

    # Use dask chunking if available, otherwise load eagerly
    try:
        import dask
        if chunks is None:
            chunks = {'time': 12}
    except ImportError:
        chunks = None

    coords_to_keep = ['lat', 'lon', 'time', 'areatotal2', 'area', 'landfrac']

    if variables:
        vars_to_load = list(set(variables + coords_to_keep))
    else:
        vars_to_load = None

    # open_mfdataset requires dask in newer xarray versions;
    # fall back to manual concatenation with open_dataset if unavailable
    try:
        open_kwargs = dict(
            combine='by_coords',
            data_vars='minimal',
            coords='minimal',
            compat='override',
        )
        if chunks is not None:
            open_kwargs['chunks'] = chunks
        ds = xr.open_mfdataset(files, **open_kwargs)
    except ImportError:
        # dask not available — open and concatenate manually
        if len(files) == 1:
            ds = xr.open_dataset(files[0])
        else:
            datasets = [xr.open_dataset(f) for f in files]
            ds = xr.concat(datasets, dim='time')

    if vars_to_load:
        available = [v for v in vars_to_load if v in ds.data_vars or v in ds.coords]
        ds = ds[available]

    ds.attrs['mesh_type'] = detect_mesh_type(ds)
    return ds
