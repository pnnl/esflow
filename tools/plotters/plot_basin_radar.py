"""
Plot radar charts of per-basin water cycle diagnostics.

Produces a multi-panel figure with one radar chart per basin, showing
normalised error metrics across water cycle components (ET bias, runoff
bias, streamflow bias, water balance residual, Wasserstein distance).
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
from core.styling import apply_style, COLOR_PALETTES

SPEC = ToolSpec(
    name='plot_basin_radar',
    description='Plot radar charts of per-basin water cycle diagnostics. '
                'Each basin gets a radar panel showing normalised error '
                'metrics for ET, runoff, streamflow, and water balance.',
    inputs={
        'budget_file': Param('path', required=True,
                             description='Basin budget CSV from compute_basin_budget'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': 'Radar chart figure'},
    },
)


# Metrics to display on radar, with display labels
# Each tuple: (column_name, display_label, transform)
# transform: 'abs' = take absolute value (lower=better),
#             'raw' = use raw value (lower=better)
RADAR_METRICS = [
    ('precip_rel_bias', 'Precip\nBias', 'abs'),
    ('et_rel_bias', 'ET Bias', 'abs'),
    ('runoff_rel_bias', 'Runoff\nBias', 'abs'),
    ('streamflow_bias', 'Streamflow\nBias', 'abs'),
    ('water_balance', 'Water Bal.\nResidual', 'abs'),
    ('wasserstein', 'FDC\nWasserstein', 'raw'),
]


def _radar_panel(ax, values, labels, title, color, bg_color):
    """Draw a single radar chart on the given axes."""
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    values_plot = values.tolist() + values[:1].tolist()

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw gridlines
    ax.set_rlabel_position(30)
    ax.set_facecolor(bg_color)

    # Plot
    ax.plot(angles, values_plot, 'o-', linewidth=2, color=color, markersize=5)
    ax.fill(angles, values_plot, alpha=0.25, color=color)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    # Radial limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=7, color='gray')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    budget_file = config['budget_file']
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    budget = pd.read_csv(budget_file)
    budget['gauge_id'] = budget['gauge_id'].astype(str)

    n_basins = len(budget)
    print(f"  Basins: {n_basins}")

    # Prepare metric columns
    available_metrics = [(col, label, xform) for col, label, xform in RADAR_METRICS
                         if col in budget.columns]

    if len(available_metrics) < 3:
        raise ValueError(f"Need at least 3 metrics for radar, found {len(available_metrics)}")

    labels = [m[1] for m in available_metrics]
    n_metrics = len(available_metrics)

    # Transform values: take abs for bias metrics, clip to [0, 1]
    radar_data = np.zeros((n_basins, n_metrics))
    for j, (col, _, xform) in enumerate(available_metrics):
        vals = budget[col].values.astype(float)
        if xform == 'abs':
            vals = np.abs(vals)
        # Clip to [0, 1] for radar display
        # For wasserstein, normalise by max across basins
        if col == 'wasserstein':
            max_val = np.nanmax(vals)
            if max_val > 0:
                vals = vals / max_val
        radar_data[:, j] = np.clip(vals, 0, 1.0)

    # Replace NaN with 0 for plotting
    radar_data = np.nan_to_num(radar_data, nan=0.0)

    # Layout
    n_cols = min(3, n_basins)
    n_rows = int(np.ceil(n_basins / n_cols))

    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows + 0.8))
    fig.patch.set_facecolor(palette['background'])

    colors = ['#2166ac', '#b2182b', '#1b7837', '#762a83', '#e66101']

    for i, (_, row) in enumerate(budget.iterrows()):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, polar=True)

        river = row.get('river_name', '')
        gid = row['gauge_id']
        title = str(river) if river and str(river) != 'nan' else f'Basin {gid}'

        color = colors[i % len(colors)]
        _radar_panel(ax, radar_data[i], labels, title, color, palette['background'])

    fig.suptitle('Basin Water Cycle Diagnostics (lower = better)',
                 fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = output_dir / 'basin_radar.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {
        'plot_file': str(out_path),
    }
