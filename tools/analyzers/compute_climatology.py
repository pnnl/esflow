"""
Compute monthly climatology from a time series CSV.

Groups by month (1-12) and computes mean for each gauge column.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param

SPEC = ToolSpec(
    name='compute_climatology',
    description='Compute monthly climatology (mean by month 1-12) from a time series CSV. '
                'Input: time series with time index and gauge_id columns. '
                'Output: climatology with month index and gauge_id columns.',
    inputs={
        'timeseries_file': Param('path', required=True,
                                 description='Time series CSV (time index, gauge_id columns)'),
    },
    outputs={
        'climatology_file': {'type': 'csv', 'description': 'Monthly climatology CSV (month 1-12 index, gauge_id columns)'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    timeseries_file = config['timeseries_file']
    output_dir = Path(config['output_dir'])

    df = pd.read_csv(timeseries_file, index_col=0, parse_dates=True)
    print(f"  Input: {len(df)} timesteps, {len(df.columns)} columns")

    # Group by month, compute mean
    months = pd.to_datetime(df.index).month
    clim = df.groupby(months).mean()
    clim.index.name = 'month'

    print(f"  Climatology: {len(clim)} months, {len(clim.columns)} columns")

    out_path = output_dir / 'climatology.csv'
    clim.to_csv(out_path)

    return {
        'climatology_file': str(out_path),
    }
