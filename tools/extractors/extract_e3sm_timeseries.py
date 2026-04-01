"""
Extract time series from E3SM model output at matched gauge locations.

Reads matched_gauges CSV for grid indices, opens model files for
the specified years, and extracts a time series at each location.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param
from core.e3sm import find_e3sm_files, open_e3sm_dataset, cftime_to_datetime

SPEC = ToolSpec(
    name='extract_e3sm_timeseries',
    description='Extract time series from E3SM output at matched gauge locations. '
                'Produces a CSV with time index and one column per gauge_id.',
    inputs={
        'data_dir': Param('path', required=True,
                          description='Base directory containing E3SM case output'),
        'case_name': Param('str', required=True,
                           description='E3SM case name'),
        'component': Param('str', required=True,
                           description='Model component (mosart, elm, eam)'),
        'variable': Param('str', required=True,
                          description='Variable to extract (e.g., RIVER_DISCHARGE_OVER_LAND_LIQ)'),
        'locations_file': Param('path', required=True,
                                description='Matched gauges CSV with lat_idx, lon_idx columns'),
        'years': Param('list[int]', required=True,
                       description='Years to extract (e.g., [2000, 2001] or 2000-2005)'),
        'frequency': Param('str', required=False, default='monthly',
                           description='File frequency: monthly (default) or daily'),
    },
    outputs={
        'timeseries_file': {'type': 'csv', 'description': 'Time series CSV (time index, gauge_id columns)'},
        'n_locations': {'type': 'int', 'description': 'Number of locations extracted'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    data_dir = config['data_dir']
    case_name = config['case_name']
    component = config['component']
    variable = config['variable']
    locations_file = config['locations_file']
    years = config['years']
    output_dir = Path(config['output_dir'])

    # Load locations
    locs = pd.read_csv(locations_file)
    locs['gauge_id'] = locs['gauge_id'].astype(str)
    print(f"  Locations: {len(locs)}")
    print(f"  Years: {years[0]}-{years[-1]}")
    print(f"  Variable: {variable}")

    # Find and open files
    frequency = config.get('frequency', 'monthly')
    files = find_e3sm_files(data_dir, case_name, component, years,
                            frequency=frequency)
    print(f"  Files found: {len(files)}")

    ds = open_e3sm_dataset(files, variables=[variable])
    mesh_type = ds.attrs.get('mesh_type', 'latlon')

    # Convert time
    time_index = cftime_to_datetime(ds['time'].values)

    # Extract at each location
    data = {}
    for _, row in locs.iterrows():
        gauge_id = row['gauge_id']
        lat_idx = int(row['lat_idx'])
        lon_idx = int(row['lon_idx'])

        if mesh_type == 'unstructured':
            # 1D indexing
            ts = ds[variable].isel(ncol=lat_idx).values
        else:
            # 2D indexing
            if ds[variable].ndim == 3:  # (time, lat, lon)
                ts = ds[variable].isel(lat=lat_idx, lon=lon_idx).values
            elif ds[variable].ndim == 2:  # (time, gridcell)
                ts = ds[variable].isel(gridcell=lat_idx).values
            else:
                ts = ds[variable].values

        # Compute if dask
        if hasattr(ts, 'compute'):
            ts = ts.compute()

        data[gauge_id] = ts.flatten()

    ds.close()

    # Build DataFrame
    df = pd.DataFrame(data, index=time_index)
    df.index.name = 'time'

    # Save
    out_path = output_dir / 'sim_timeseries.csv'
    df.to_csv(out_path)

    print(f"  Extracted {len(df.columns)} locations, {len(df)} timesteps")

    return {
        'timeseries_file': str(out_path),
        'n_locations': len(df.columns),
    }
