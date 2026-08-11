"""
Compute summary statistics from a time series CSV.

Computes mean, std, min, max for each gauge/column and optionally
ranks by a statistic. Outputs a CSV with one row per gauge.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param

logger = logging.getLogger(__name__)


SPEC = ToolSpec(
    name='compute_summary_stats',
    description=(
        'Compute summary statistics (mean, std, min, max) for each column in a '
        'time series CSV. Optionally rank and filter to top N entries by a statistic. '
        'Input: time series CSV with time index and value columns (e.g., gauge_id columns). '
        'Output: CSV with columns: column_name, mean, std, min, max. '
        'If gauge metadata is provided, river_name and area_km2 are included.'
    ),
    inputs={
        'timeseries_file': Param('path', required=True,
                                 description='Time series CSV (time index, gauge_id columns)'),
        'gauge_metadata': Param('path', required=False, default='',
                                description='Optional gauge metadata CSV for adding river_name and area_km2'),
        'top_n': Param('int', required=False, default=0,
                       description='If > 0, return only the top N entries ranked by mean (descending)'),
        'rank_by': Param('str', required=False, default='mean',
                         description='Statistic to rank by: mean, std, min, max'),
    },
    outputs={
        'stats_file': {'type': 'csv', 'description': 'Summary statistics CSV'},
        'n_columns': {'type': 'int', 'description': 'Number of columns analyzed'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    ts_file = config['timeseries_file']
    metadata_file = config['gauge_metadata']
    top_n = config['top_n']
    rank_by = config['rank_by']
    output_dir = Path(config['output_dir'])

    df = pd.read_csv(ts_file, index_col=0, parse_dates=True)
    logger.info("Time series: %s timesteps, %s columns", len(df), len(df.columns))

    results = []
    for col in df.columns:
        vals = df[col].dropna().values.astype(float)
        results.append({
            'column_name': col,
            'mean': float(np.mean(vals)) if len(vals) > 0 else np.nan,
            'std': float(np.std(vals)) if len(vals) > 0 else np.nan,
            'min': float(np.min(vals)) if len(vals) > 0 else np.nan,
            'max': float(np.max(vals)) if len(vals) > 0 else np.nan,
            'n_valid': len(vals),
        })

    stats_df = pd.DataFrame(results)

    # Add metadata if provided
    if metadata_file:
        try:
            meta = pd.read_csv(metadata_file)
            meta['gauge_id'] = meta['gauge_id'].astype(str)
            stats_df = stats_df.merge(
                meta[['gauge_id', 'river_name', 'area_km2']],
                left_on='column_name', right_on='gauge_id', how='left'
            ).drop(columns='gauge_id', errors='ignore')
        except Exception as e:
            logger.warning("Could not merge metadata: %s", e)

    # Rank and filter
    if rank_by not in stats_df.columns:
        rank_by = 'mean'

    stats_df = stats_df.sort_values(rank_by, ascending=False)

    if top_n > 0:
        stats_df = stats_df.head(top_n)
        logger.info("Top %s by %s:", top_n, rank_by)
    else:
        logger.info("All %s columns (sorted by %s):", len(stats_df), rank_by)

    for _, row in stats_df.head(5).iterrows():
        name = row.get('river_name', row['column_name'])
        logger.info("%s: mean=%.2f (%s)", row['column_name'], row['mean'], name)

    out_path = output_dir / 'summary_stats.csv'
    stats_df.to_csv(out_path, index=False)

    return {
        'stats_file': str(out_path),
        'n_columns': len(stats_df),
    }
