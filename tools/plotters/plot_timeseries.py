"""
Plot time series comparing simulated vs observed data.

Produces a multi-panel figure with one subplot per gauge.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.styling import apply_style, get_colors, COLOR_PALETTES

SPEC = ToolSpec(
    name='plot_timeseries',
    description='Plot time series comparing simulated vs observed data. '
                'Produces a multi-panel PNG with one subplot per gauge.',
    inputs={
        'sim_file': Param('path', required=True,
                          description='Simulated time series CSV (time index, gauge_id columns)'),
        'obs_file': Param('path', required=True,
                          description='Observed time series CSV (time index, gauge_id columns)'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': 'Time series comparison plot'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    sim_file = config['sim_file']
    obs_file = config['obs_file']
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    colors = get_colors('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    # Load data
    sim = pd.read_csv(sim_file, index_col=0, parse_dates=True)
    obs = pd.read_csv(obs_file, index_col=0, parse_dates=True)
    sim.columns = [str(c) for c in sim.columns]
    obs.columns = [str(c) for c in obs.columns]

    # Find common columns
    common = [c for c in sim.columns if c in obs.columns]
    if not common:
        raise ValueError("No matching columns between sim and obs files.")

    # Limit to 9 panels max
    plot_cols = common[:9]
    n = len(plot_cols)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharex=True)
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(plot_cols):
        ax = axes[i]

        s_sim = sim[col].dropna()
        s_obs = obs[col].dropna()

        if len(s_sim) > 0:
            ax.plot(s_sim.index, s_sim.values, color=colors[0],
                    linewidth=0.8, alpha=0.9, label='Simulated')
        if len(s_obs) > 0:
            ax.plot(s_obs.index, s_obs.values, color=colors[1],
                    linewidth=0.8, alpha=0.9, label='Observed')

        ax.set_title(col, fontsize=10)
        ax.set_ylabel('Discharge (m3/s)', fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Time Series Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / 'timeseries.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close()

    print(f"  Plotted {n} gauges")

    return {
        'plot_file': str(out_path),
    }
