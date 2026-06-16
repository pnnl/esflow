"""
Data I/O utilities for ESM model output.

Handles loading MOSART and ELM data with support for:
- Multiple file patterns and directory structures
- Both lat-lon and unstructured meshes
- Time concatenation across years
- Variable subsetting
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

from .e3sm import find_e3sm_files, detect_mesh_type


def load_mosart_data(
    case_name: str,
    base_dir: str,
    years: List[int],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """Load MOSART river routing model output."""
    files = find_e3sm_files(base_dir, case_name, 'mosart', years)

    coords_to_keep = ['lat', 'lon', 'time', 'areatotal2', 'area']

    if variables:
        vars_to_load = list(set(variables + coords_to_keep))
        ds = xr.open_mfdataset(
            files, combine='by_coords', chunks={'time': 12},
            data_vars='minimal', coords='minimal', compat='override',
        )
        available_vars = [v for v in vars_to_load if v in ds.data_vars or v in ds.coords]
        ds = ds[available_vars]
    else:
        ds = xr.open_mfdataset(
            files, combine='by_coords', chunks={'time': 12},
            data_vars='minimal', coords='minimal', compat='override',
        )

    ds.attrs['mesh_type'] = detect_mesh_type(ds)
    ds.attrs['case_name'] = case_name
    ds.attrs['component'] = 'mosart'
    return ds


def load_elm_data(
    case_name: str,
    base_dir: str,
    years: List[int],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """Load ELM land model output."""
    files = find_e3sm_files(base_dir, case_name, 'elm', years)

    coords_to_keep = ['lat', 'lon', 'time', 'area', 'landfrac']

    if variables:
        vars_to_load = list(set(variables + coords_to_keep))
        ds = xr.open_mfdataset(
            files, combine='by_coords', chunks={'time': 12},
            data_vars='minimal', coords='minimal', compat='override',
        )
        available_vars = [v for v in vars_to_load if v in ds.data_vars or v in ds.coords]
        ds = ds[available_vars]
    else:
        ds = xr.open_mfdataset(
            files, combine='by_coords', chunks={'time': 12},
            data_vars='minimal', coords='minimal', compat='override',
        )

    ds.attrs['mesh_type'] = detect_mesh_type(ds)
    ds.attrs['case_name'] = case_name
    ds.attrs['component'] = 'elm'
    return ds


def save_dataset(ds: xr.Dataset, output_path: str, **kwargs) -> str:
    """Save dataset to NetCDF with compression."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4}

    ds.to_netcdf(output_path, encoding=encoding, **kwargs)
    return str(output_path)
