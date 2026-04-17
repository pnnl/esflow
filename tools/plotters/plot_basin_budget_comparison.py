"""
Plot per-basin model vs observation water budget comparison.

Produces a multi-panel figure with one panel per basin, each showing grouped
bar charts of P, ET, and Q for both model and observations.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.styling import apply_style, COLOR_PALETTES


SPEC = ToolSpec(
    name='plot_basin_budget_comparison',
    description='Plot per-basin grouped bar charts comparing model vs '
                'observation water budget components (P, ET, Q). '
                'Input is the basin budget CSV from compute_basin_budget.',
    inputs={
        'budget_file': Param('path', required=True,
                             description='Basin budget CSV from compute_basin_budget '
                                         '(must contain model_P, model_ET, model_Q, '
                                         'obs_P, obs_ET, obs_Q columns)'),
    },
    outputs={
        'plot_file': {'type': 'png',
                      'description': 'Model vs obs budget comparison figure'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    budget_file = config['budget_file']
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    budget = pd.read_csv(budget_file)
    budget['gauge_id'] = budget['gauge_id'].astype(str)

    n_basins = len(budget)
    print(f"  Basins: {n_basins}")

    # Layout
    n_cols = min(3, n_basins)
    n_rows = int(np.ceil(n_basins / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor(palette['background'])

    if n_basins == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    components = ['P', 'ET', 'Q']
    model_cols = ['model_P', 'model_ET', 'model_Q']
    obs_cols = ['obs_P', 'obs_ET', 'obs_Q']

    x = np.arange(len(components))
    bar_width = 0.35

    model_color = '#3498DB'
    obs_color = '#E74C3C'

    for i, (_, row) in enumerate(budget.iterrows()):
        ax = axes[i]
        ax.set_facecolor(palette['background'])

        model_vals = [row.get(c, np.nan) for c in model_cols]
        obs_vals = [row.get(c, np.nan) for c in obs_cols]

        bars_m = ax.bar(x - bar_width / 2, model_vals, bar_width,
                        label='Model', color=model_color, edgecolor='white',
                        linewidth=0.5)
        bars_o = ax.bar(x + bar_width / 2, obs_vals, bar_width,
                        label='Obs', color=obs_color, edgecolor='white',
                        linewidth=0.5)

        # Value labels
        for bar, val in zip(list(bars_m) + list(bars_o), model_vals + obs_vals):
            if np.isnan(val):
                continue
            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.2e}', ha='center', va=va, fontsize=6.5,
                    fontweight='bold')

        # Basin name
        river = row.get('river_name', '')
        gid = row['gauge_id']
        title = str(river) if river and str(river) != 'nan' else f'Basin {gid}'
        ax.set_title(title, fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(components, fontsize=10)
        ax.set_ylabel('mm/s', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)

        if i == 0:
            ax.legend(fontsize=9, loc='upper right')

    # Hide unused axes
    for j in range(n_basins, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Water Budget: Model vs Observations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / 'basin_budget_comparison.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {'plot_file': str(out_path)}
