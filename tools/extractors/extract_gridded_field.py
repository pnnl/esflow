"""
Extract a 2D gridded field from E3SM or observation NetCDF files.

Computes time-mean (or sum of multiple variables) and outputs a simple
lat-lon NetCDF file suitable for spatial comparison and plotting.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.e3sm import find_e3sm_files, open_e3sm_dataset, cftime_to_datetime

logger = logging.getLogger(__name__)


SPEC = ToolSpec(
    name='extract_gridded_field',
    description=(
        'Extract a 2D gridded field from E3SM output or observation NetCDF files. '
        'Computes the time-mean over the specified years. '
        'For composite variables, provide multiple names separated by "+" '
        '(e.g., "QVEGE+QVEGT+QSOIL" to sum three ET components). '
        'Input can be E3SM model output (provide data_dir, case_name, component) '
        'or a single observation NetCDF file (provide obs_file). '
        'Output is a simple lat-lon NetCDF with the time-averaged field.'
    ),
    inputs={
        'data_dir': Param('path', required=False, default='',
                          description='Base directory for E3SM output (use for model data)'),
        'case_name': Param('str', required=False, default='',
                           description='E3SM case name (use for model data)'),
        'component': Param('str', required=False, default='',
                           description='Model component: elm, mosart, eam (use for model data)'),
        'obs_file': Param('path', required=False, default='',
                          description='Path to observation NetCDF file (use for obs data)'),
        'variable': Param('str', required=True,
                          description='Variable name(s) to extract. Use "+" to sum multiple '
                                      '(e.g., "RAIN+SNOW" or "QVEGE+QVEGT+QSOIL")'),
        'years': Param('list[int]', required=False, default=None,
                       description='Years to include (e.g., [2001]). If omitted, uses all available time steps.'),
    },
    outputs={
        'field_file': {'type': 'netcdf', 'description': 'Time-averaged 2D field as lat-lon NetCDF'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    data_dir = config['data_dir']
    case_name = config['case_name']
    component = config['component']
    obs_file = config['obs_file']
    variable = config['variable']
    years = config['years']
    output_dir = Path(config['output_dir'])

    var_names = [v.strip() for v in variable.split('+')]
    is_composite = len(var_names) > 1

    # Open dataset — either E3SM model or observation file
    if obs_file:
        logger.info("Opening observation file: %s", obs_file)
        ds = xr.open_dataset(obs_file)
        source = 'obs'
    elif data_dir and case_name:
        logger.info("Opening E3SM %s data: %s", component, case_name)
        years_list = years if years else []
        files = find_e3sm_files(data_dir, case_name, component, years_list)
        logger.info("Files found: %s", len(files))
        ds = open_e3sm_dataset(files, variables=var_names)
        source = 'model'
    else:
        raise ValueError(
            "Provide either (data_dir + case_name + component) for model data "
            "or obs_file for observation data."
        )

    # Extract and sum variables
    logger.info("Variables: %s", var_names)
    field = None
    for vname in var_names:
        if vname not in ds:
            available = [v for v in ds.data_vars if not v.startswith('_')]
            raise ValueError(
                f"Variable '{vname}' not found. Available: {sorted(available)[:20]}"
            )
        v = ds[vname].astype('float64')
        if hasattr(v, 'compute'):
            v = v.compute()
        field = v if field is None else field + v

    # Compute time mean and record time range
    time_range = ''
    if 'time' in field.dims:
        logger.info("Time steps: %s", len(field.time))
        try:
            tvals = field.time.values
            # Works for both datetime64 and cftime objects
            y0 = int(tvals[0].year) if hasattr(tvals[0], 'year') else None
            y1 = int(tvals[-1].year) if hasattr(tvals[-1], 'year') else None
            if y0 is not None and y1 is not None:
                time_range = f"{y0}-{y1}" if y0 != y1 else str(y0)
        except Exception:
            pass
        field_mean = field.mean(dim='time')
    else:
        field_mean = field

    # Build clean output dataset
    out_name = variable.replace('+', '_plus_')
    out_ds = xr.Dataset({out_name: field_mean})

    # Ensure lat/lon coordinates are preserved
    for coord in ['lat', 'lon', 'landfrac', 'area']:
        if coord in ds.coords or coord in ds.data_vars:
            if coord not in out_ds:
                out_ds[coord] = ds[coord]

    out_ds.attrs['source'] = source
    out_ds.attrs['variables_summed'] = variable
    if years:
        out_ds.attrs['years'] = str(years)
    if time_range:
        out_ds.attrs['time_range'] = time_range

    # Save
    safe_name = variable.replace('+', '_plus_').lower()
    out_path = output_dir / f'{safe_name}_mean.nc'
    out_ds.to_netcdf(out_path)

    logger.info("Output shape: %s", dict(field_mean.sizes))
    logger.info("Saved: %s", out_path)

    ds.close()

    return {'field_file': str(out_path)}
