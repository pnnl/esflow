"""
Plot scatter comparison between two time series files.

Computes mean values per gauge from both files and creates a scatter plot.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param
from core.styling import apply_style, get_colors, COLOR_PALETTES

SPEC = ToolSpec(
    name='plot_scatter',
    description='Scatter plot comparing mean values from two time series files (e.g., sim vs obs). '
                'Each point represents one gauge.',
    inputs={
        'x_file': Param('path', required=True,
                         description='X-axis time series CSV (e.g., observed)'),
        'y_file': Param('path', required=True,
                         description='Y-axis time series CSV (e.g., simulated)'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': 'Scatter plot PNG'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    x_file = config['x_file']
    y_file = config['y_file']
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    colors = get_colors('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    # Load
    x_df = pd.read_csv(x_file, index_col=0, parse_dates=True)
    y_df = pd.read_csv(y_file, index_col=0, parse_dates=True)
    x_df.columns = [str(c) for c in x_df.columns]
    y_df.columns = [str(c) for c in y_df.columns]

    # Match columns
    common = [c for c in x_df.columns if c in y_df.columns]
    if not common:
        raise ValueError("No matching columns between x and y files.")

    # Compute means
    x_means = x_df[common].mean()
    y_means = y_df[common].mean()

    # Remove NaN
    valid = ~(x_means.isna() | y_means.isna())
    x_vals = x_means[valid].values
    y_vals = y_means[valid].values

    print(f"  Matched gauges: {len(common)}, valid: {valid.sum()}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x_vals, y_vals, c=colors[0], s=40, alpha=0.7, edgecolors='white',
               linewidth=0.5)

    # 1:1 line
    all_vals = np.concatenate([x_vals, y_vals])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            'k--', alpha=0.5, linewidth=1)

    # Correlation
    r = np.corrcoef(x_vals, y_vals)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x_vals)}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Observed Mean Discharge (m3/s)', fontsize=11)
    ax.set_ylabel('Simulated Mean Discharge (m3/s)', fontsize=11)
    ax.set_title('Simulated vs Observed', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = output_dir / 'scatter.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close()

    return {
        'plot_file': str(out_path),
    }
