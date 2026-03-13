"""
Compute validation metrics between simulated and observed time series.

Matches columns by name (gauge_id). Computes NSE, KGE, PBIAS, RMSE,
correlation for each gauge pair.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (-inf to 1, 1 is perfect)."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    if mask.sum() < 2:
        return np.nan
    obs, sim = obs[mask], sim[mask]
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def calc_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency (-inf to 1, 1 is perfect)."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    if mask.sum() < 2:
        return np.nan
    obs, sim = obs[mask], sim[mask]
    if np.std(obs) == 0 or np.std(sim) == 0:
        r = 0
    else:
        r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    beta = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    if np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def calc_pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent Bias (%, 0 is perfect)."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    if mask.sum() < 1:
        return np.nan
    obs, sim = obs[mask], sim[mask]
    if np.sum(obs) == 0:
        return np.nan
    return 100 * np.sum(sim - obs) / np.sum(obs)


def calc_rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Root Mean Square Error."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    if mask.sum() < 1:
        return np.nan
    obs, sim = obs[mask], sim[mask]
    return np.sqrt(np.mean((obs - sim) ** 2))


def calc_correlation(obs: np.ndarray, sim: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    if mask.sum() < 2:
        return np.nan
    obs, sim = obs[mask], sim[mask]
    if np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    return np.corrcoef(obs, sim)[0, 1]


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

SPEC = ToolSpec(
    name='compute_metrics',
    description='Compute validation metrics (NSE, KGE, PBIAS, RMSE, correlation) '
                'between simulated and observed time series. '
                'Matches sim/obs columns by gauge_id name.',
    inputs={
        'sim_file': Param('path', required=True,
                          description='Simulated time series CSV (time index, gauge_id columns)'),
        'obs_file': Param('path', required=True,
                          description='Observed time series CSV (time index, gauge_id columns)'),
    },
    outputs={
        'metrics_file': {'type': 'csv', 'description': 'Metrics CSV: gauge_id, nse, kge, pbias, rmse, correlation, n_valid'},
        'n_gauges': {'type': 'int', 'description': 'Number of gauge pairs analyzed'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    sim_file = config['sim_file']
    obs_file = config['obs_file']
    output_dir = Path(config['output_dir'])

    # Load data
    sim = pd.read_csv(sim_file, index_col=0, parse_dates=True)
    obs = pd.read_csv(obs_file, index_col=0, parse_dates=True)

    sim.columns = [str(c) for c in sim.columns]
    obs.columns = [str(c) for c in obs.columns]

    # Match by column name
    common_cols = [c for c in sim.columns if c in obs.columns]
    print(f"  Sim columns: {len(sim.columns)}")
    print(f"  Obs columns: {len(obs.columns)}")
    print(f"  Matched: {len(common_cols)}")

    if not common_cols:
        raise ValueError(
            f"No matching column names between sim and obs. "
            f"Sim: {list(sim.columns)[:5]}, Obs: {list(obs.columns)[:5]}"
        )

    # Find overlapping dates
    common_dates = sim.index.intersection(obs.index)
    print(f"  Overlapping dates: {len(common_dates)}")

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between sim and obs.")

    # Compute metrics
    metric_funcs = {
        'nse': calc_nse,
        'kge': calc_kge,
        'pbias': calc_pbias,
        'rmse': calc_rmse,
        'correlation': calc_correlation,
    }

    results = []
    for col in common_cols:
        sim_vals = sim.loc[common_dates, col].values.astype(float)
        obs_vals = obs.loc[common_dates, col].values.astype(float)

        row = {'gauge_id': col}
        for name, func in metric_funcs.items():
            row[name] = func(obs_vals, sim_vals)

        valid = ~(np.isnan(sim_vals) | np.isnan(obs_vals))
        row['n_valid'] = int(valid.sum())

        results.append(row)

    metrics_df = pd.DataFrame(results)

    # Print summary
    for metric in ['nse', 'kge', 'pbias', 'rmse']:
        vals = metrics_df[metric].dropna()
        if len(vals) > 0:
            print(f"  {metric}: median={vals.median():.3f}, mean={vals.mean():.3f}")

    # Save
    out_path = output_dir / 'metrics.csv'
    metrics_df.to_csv(out_path, index=False)

    return {
        'metrics_file': str(out_path),
        'n_gauges': len(metrics_df),
    }
