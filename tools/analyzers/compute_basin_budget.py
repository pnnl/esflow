"""
Compute per-basin water budget diagnostics.

Combines basin-averaged model and observation fields with streamflow FDC
metrics to produce a unified per-basin summary table. For each basin:
  - ET relative bias  (model_ET - obs_ET) / obs_ET
  - Runoff relative bias  (model_Q - obs_Q) / obs_Q
  - Water balance residual  (P - ET - Q) / P
  - Streamflow volume bias  (from FDC metrics)
  - Streamflow Wasserstein  (from FDC metrics)

The summary table is suitable for radar-chart visualisation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param

SPEC = ToolSpec(
    name='compute_basin_budget',
    description='Combine basin-averaged model/obs fields and streamflow FDC '
                'metrics into a per-basin water budget summary table with '
                'relative biases for ET, runoff, water balance, and streamflow.',
    inputs={
        'model_precip_file': Param('path', required=True,
                                   description='Basin-mean model P CSV (gauge_id, basin_mean)'),
        'model_et_file': Param('path', required=True,
                               description='Basin-mean model ET CSV'),
        'model_runoff_file': Param('path', required=True,
                                   description='Basin-mean model QRUNOFF CSV'),
        'obs_precip_file': Param('path', required=False, default='',
                                 description='Basin-mean obs precipitation CSV (optional)'),
        'obs_et_file': Param('path', required=True,
                             description='Basin-mean obs ET CSV'),
        'obs_runoff_file': Param('path', required=True,
                                 description='Basin-mean obs runoff CSV'),
        'fdc_metrics_file': Param('path', required=True,
                                  description='FDC metrics CSV with gauge_id, volume_bias, wasserstein'),
        'locations_file': Param('path', required=True,
                                description='Gauge metadata CSV with gauge_id, river_name'),
    },
    outputs={
        'budget_file': {'type': 'csv',
                        'description': 'Per-basin water budget summary CSV'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    output_dir = Path(config['output_dir'])

    # Load all basin-mean CSVs
    mp = pd.read_csv(config['model_precip_file'])
    me = pd.read_csv(config['model_et_file'])
    mq = pd.read_csv(config['model_runoff_file'])
    oe = pd.read_csv(config['obs_et_file'])
    oq = pd.read_csv(config['obs_runoff_file'])
    fdc = pd.read_csv(config['fdc_metrics_file'])
    locs = pd.read_csv(config['locations_file'])

    obs_precip_file = config.get('obs_precip_file', '')
    op = pd.read_csv(obs_precip_file) if obs_precip_file else None

    # Standardise gauge_id to string
    all_dfs = [mp, me, mq, oe, oq, fdc, locs]
    if op is not None:
        all_dfs.append(op)
    for df in all_dfs:
        df['gauge_id'] = df['gauge_id'].astype(str)

    # Merge all on gauge_id
    budget = mp.rename(columns={'basin_mean': 'model_P'})
    budget = budget.merge(me.rename(columns={'basin_mean': 'model_ET'}), on='gauge_id', how='outer')
    budget = budget.merge(mq.rename(columns={'basin_mean': 'model_Q'}), on='gauge_id', how='outer')
    if op is not None:
        budget = budget.merge(op.rename(columns={'basin_mean': 'obs_P'}), on='gauge_id', how='outer')
        # Auto-convert obs P units: if obs_P >> model_P, likely mm/day vs mm/s
        if budget['obs_P'].median() > budget['model_P'].median() * 1000:
            budget['obs_P'] = budget['obs_P'] / 86400.0
            print("  Auto-converted obs_P from mm/day to mm/s (÷86400)")
    budget = budget.merge(oe.rename(columns={'basin_mean': 'obs_ET'}), on='gauge_id', how='outer')
    budget = budget.merge(oq.rename(columns={'basin_mean': 'obs_Q'}), on='gauge_id', how='outer')

    # Merge FDC metrics (volume_bias, wasserstein)
    fdc_cols = ['gauge_id']
    if 'volume_bias' in fdc.columns:
        fdc_cols.append('volume_bias')
    if 'wasserstein' in fdc.columns:
        fdc_cols.append('wasserstein')
    budget = budget.merge(fdc[fdc_cols], on='gauge_id', how='left')

    # Merge river name
    if 'river_name' in locs.columns:
        budget = budget.merge(locs[['gauge_id', 'river_name']], on='gauge_id', how='left')

    # Compute derived metrics
    if 'obs_P' in budget.columns:
        budget['precip_rel_bias'] = np.where(
            budget['obs_P'] != 0,
            (budget['model_P'] - budget['obs_P']) / np.abs(budget['obs_P']),
            np.nan
        )
    budget['et_rel_bias'] = np.where(
        budget['obs_ET'] != 0,
        (budget['model_ET'] - budget['obs_ET']) / np.abs(budget['obs_ET']),
        np.nan
    )
    budget['runoff_rel_bias'] = np.where(
        budget['obs_Q'] != 0,
        (budget['model_Q'] - budget['obs_Q']) / np.abs(budget['obs_Q']),
        np.nan
    )
    budget['water_balance'] = np.where(
        budget['model_P'] != 0,
        (budget['model_P'] - budget['model_ET'] - budget['model_Q']) / budget['model_P'],
        np.nan
    )

    # Rename volume_bias to streamflow_bias for clarity
    if 'volume_bias' in budget.columns:
        budget = budget.rename(columns={'volume_bias': 'streamflow_bias'})

    # Print summary
    print(f"  Basins: {len(budget)}")
    for col in ['precip_rel_bias', 'et_rel_bias', 'runoff_rel_bias', 'water_balance', 'streamflow_bias', 'wasserstein']:
        if col in budget.columns:
            vals = budget[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: mean={vals.mean():.4f}, range=[{vals.min():.4f}, {vals.max():.4f}]")

    out_path = output_dir / 'basin_budget.csv'
    budget.to_csv(out_path, index=False)

    return {
        'budget_file': str(out_path),
    }
