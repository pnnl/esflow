"""
Compute flow duration curve (FDC) metrics between simulated and observed discharge.

For climate model evaluation, point-by-point metrics (NSE) are inappropriate
because climate models do not reproduce specific weather events. Instead,
distributional comparisons via FDCs assess whether the model reproduces the
statistical character of streamflow.

Metrics per gauge:
  - volume_bias: relative bias in mean discharge (sim-obs)/obs
  - wasserstein: Earth Mover's Distance (Wasserstein-1) between normalised FDCs
  - q10_ratio:  ratio of sim/obs high-flow quantile (exceedance prob 0.1)
  - q50_ratio:  ratio of sim/obs median flow
  - q90_ratio:  ratio of sim/obs low-flow quantile (exceedance prob 0.9)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param


def _fdc(q: np.ndarray):
    """Return sorted descending flow values and exceedance probabilities."""
    q_clean = q[~np.isnan(q)]
    q_sorted = np.sort(q_clean)[::-1]
    n = len(q_sorted)
    exceed_prob = np.arange(1, n + 1) / (n + 1)
    return exceed_prob, q_sorted


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 (Earth Mover's Distance) between two 1-D samples.

    Normalises both distributions by dividing by observed mean so
    the metric is scale-free and comparable across basins.
    """
    a_clean = a[~np.isnan(a)]
    b_clean = b[~np.isnan(b)]
    if len(a_clean) < 10 or len(b_clean) < 10:
        return np.nan
    obs_mean = np.mean(b_clean)
    if obs_mean == 0:
        return np.nan
    a_norm = np.sort(a_clean / obs_mean)
    b_norm = np.sort(b_clean / obs_mean)
    # Interpolate both onto common quantile grid
    n = 200
    probs = np.linspace(0, 1, n)
    a_q = np.quantile(a_norm, probs)
    b_q = np.quantile(b_norm, probs)
    return float(np.mean(np.abs(a_q - b_q)))


SPEC = ToolSpec(
    name='compute_fdc_metrics',
    description='Compute flow duration curve (FDC) distributional metrics '
                'between simulated and observed discharge. Appropriate for '
                'climate model evaluation where timing is not expected to match. '
                'Outputs per-gauge metrics and FDC percentile data.',
    inputs={
        'sim_file': Param('path', required=True,
                          description='Simulated time series CSV (time index, gauge_id columns)'),
        'obs_file': Param('path', required=True,
                          description='Observed time series CSV (time index, gauge_id columns)'),
    },
    outputs={
        'metrics_file': {'type': 'csv',
                         'description': 'Per-gauge metrics: gauge_id, volume_bias, wasserstein, q10_ratio, q50_ratio, q90_ratio'},
        'fdc_file': {'type': 'csv',
                     'description': 'FDC percentile data for all gauges (exceedance_prob, sim_q, obs_q per gauge)'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    sim_file = config['sim_file']
    obs_file = config['obs_file']
    output_dir = Path(config['output_dir'])

    sim = pd.read_csv(sim_file, index_col=0, parse_dates=True)
    obs = pd.read_csv(obs_file, index_col=0, parse_dates=True)

    sim.columns = [str(c) for c in sim.columns]
    obs.columns = [str(c) for c in obs.columns]

    common_cols = [c for c in sim.columns if c in obs.columns]
    print(f"  Sim columns: {len(sim.columns)}")
    print(f"  Obs columns: {len(obs.columns)}")
    print(f"  Matched: {len(common_cols)}")

    if not common_cols:
        raise ValueError(
            f"No matching columns. Sim: {list(sim.columns)[:5]}, Obs: {list(obs.columns)[:5]}"
        )

    # Percentile grid for FDC output
    n_pct = 101
    exceed_probs = np.linspace(0, 1, n_pct)

    metrics_rows = []
    fdc_rows = []

    for col in common_cols:
        sim_vals = sim[col].values.astype(float)
        obs_vals = obs[col].values.astype(float)

        sim_clean = sim_vals[~np.isnan(sim_vals)]
        obs_clean = obs_vals[~np.isnan(obs_vals)]

        if len(sim_clean) < 10 or len(obs_clean) < 10:
            continue

        # Volume bias
        obs_mean = np.mean(obs_clean)
        sim_mean = np.mean(sim_clean)
        vol_bias = (sim_mean - obs_mean) / obs_mean if obs_mean != 0 else np.nan

        # Wasserstein
        emd = _wasserstein_1d(sim_vals, obs_vals)

        # Quantile ratios (Q10 = high flow, Q90 = low flow)
        sim_q10 = np.quantile(sim_clean, 0.90)  # top 10% exceedance
        obs_q10 = np.quantile(obs_clean, 0.90)
        sim_q50 = np.quantile(sim_clean, 0.50)
        obs_q50 = np.quantile(obs_clean, 0.50)
        sim_q90 = np.quantile(sim_clean, 0.10)  # bottom 10% exceedance
        obs_q90 = np.quantile(obs_clean, 0.10)

        q10_ratio = sim_q10 / obs_q10 if obs_q10 > 0 else np.nan
        q50_ratio = sim_q50 / obs_q50 if obs_q50 > 0 else np.nan
        q90_ratio = sim_q90 / obs_q90 if obs_q90 > 0 else np.nan

        metrics_rows.append({
            'gauge_id': col,
            'volume_bias': round(vol_bias, 4),
            'wasserstein': round(emd, 4),
            'q10_ratio': round(q10_ratio, 4),
            'q50_ratio': round(q50_ratio, 4),
            'q90_ratio': round(q90_ratio, 4),
            'n_sim': len(sim_clean),
            'n_obs': len(obs_clean),
        })

        # FDC percentiles for this gauge
        sim_fdc = np.quantile(sim_clean, 1 - exceed_probs)  # descending
        obs_fdc = np.quantile(obs_clean, 1 - exceed_probs)
        for j in range(n_pct):
            fdc_rows.append({
                'gauge_id': col,
                'exceedance_prob': round(exceed_probs[j], 3),
                'sim_q': round(sim_fdc[j], 2),
                'obs_q': round(obs_fdc[j], 2),
            })

    metrics_df = pd.DataFrame(metrics_rows)
    fdc_df = pd.DataFrame(fdc_rows)

    # Print summary
    for m in ['volume_bias', 'wasserstein', 'q50_ratio']:
        vals = metrics_df[m].dropna()
        if len(vals) > 0:
            print(f"  {m}: median={vals.median():.3f}, mean={vals.mean():.3f}")

    print(f"  Gauges analyzed: {len(metrics_df)}")

    metrics_path = output_dir / 'fdc_metrics.csv'
    fdc_path = output_dir / 'fdc_data.csv'
    metrics_df.to_csv(metrics_path, index=False)
    fdc_df.to_csv(fdc_path, index=False)

    return {
        'metrics_file': str(metrics_path),
        'fdc_file': str(fdc_path),
    }
