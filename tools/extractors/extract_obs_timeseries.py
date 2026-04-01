"""
Extract observation time series from per-gauge CSV files.

Reads per-gauge CSVs (date, discharge_m3s) from an observation directory,
filters to specified years, and produces a combined time series CSV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param

SPEC = ToolSpec(
    name='extract_obs_timeseries',
    description='Extract observation time series from per-gauge CSV files. '
                'Each gauge CSV has columns (date, discharge_m3s). '
                'Produces combined CSV with time index and gauge_id columns.',
    inputs={
        'obs_dir': Param('path', required=True,
                         description='Directory containing per-gauge CSV files'),
        'gauge_metadata': Param('path', required=True,
                                description='Gauge metadata CSV (for gauge_id list)'),
        'years': Param('list[int]', required=True,
                       description='Years to extract (e.g., [2000, 2001] or 2000-2005)'),
        'gauge_ids': Param('str', required=False, default='',
                           description='Comma-separated gauge IDs to extract (if omitted, all gauges in metadata)'),
    },
    outputs={
        'timeseries_file': {'type': 'csv', 'description': 'Time series CSV (time index, gauge_id columns)'},
        'n_gauges': {'type': 'int', 'description': 'Number of gauges with data'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    obs_dir = Path(config['obs_dir'])
    gauge_metadata = config['gauge_metadata']
    years = config['years']
    output_dir = Path(config['output_dir'])

    # Load metadata for gauge list
    meta = pd.read_csv(gauge_metadata)
    meta['gauge_id'] = meta['gauge_id'].astype(str)

    gauge_ids_filter = config.get('gauge_ids', '')
    if gauge_ids_filter:
        gauge_ids = [g.strip() for g in gauge_ids_filter.split(',')]
        print(f"  Filtering to {len(gauge_ids)} specified gauges")
    else:
        gauge_ids = meta['gauge_id'].tolist()

    print(f"  Gauges in metadata: {len(gauge_ids)}")
    print(f"  Years: {years[0]}-{years[-1]}")
    print(f"  Obs directory: {obs_dir}")

    # Read per-gauge CSVs
    all_series = {}
    n_found = 0

    for gauge_id in gauge_ids:
        csv_path = obs_dir / f"{gauge_id}.csv"
        if not csv_path.exists():
            continue

        n_found += 1
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.set_index('date')

        # Filter to requested years
        df = df[df.index.year.isin(years)]

        if len(df) > 0:
            all_series[gauge_id] = df['discharge_m3s']

    print(f"  Found CSVs: {n_found} of {len(gauge_ids)}")
    print(f"  With data in year range: {len(all_series)}")

    if not all_series:
        raise ValueError(
            f"No observation data found for years {years[0]}-{years[-1]} "
            f"in {obs_dir}"
        )

    # Combine into single DataFrame
    combined = pd.DataFrame(all_series)
    combined.index.name = 'time'
    combined = combined.sort_index()

    # Save
    out_path = output_dir / 'obs_timeseries.csv'
    combined.to_csv(out_path)

    print(f"  Combined: {len(combined)} timesteps, {len(combined.columns)} gauges")

    return {
        'timeseries_file': str(out_path),
        'n_gauges': len(combined.columns),
    }
