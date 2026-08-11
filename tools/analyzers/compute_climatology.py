"""
Compute monthly climatology from a time series CSV.

Groups by month (1-12) and computes mean for each gauge column.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param

logger = logging.getLogger(__name__)

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
    logger.info("Input: %s timesteps, %s columns", len(df), len(df.columns))

    # Group by month, compute mean
    months = pd.to_datetime(df.index).month
    clim = df.groupby(months).mean()
    clim.index.name = 'month'

    logger.info("Climatology: %s months, %s columns", len(clim), len(clim.columns))

    out_path = output_dir / 'climatology.csv'
    clim.to_csv(out_path)

    return {
        'climatology_file': str(out_path),
    }
