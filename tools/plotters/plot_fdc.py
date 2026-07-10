"""
Plot flow duration curve (FDC) comparisons.

Produces a multi-panel figure: a map of a distributional metric (e.g. Wasserstein
distance) at gauge locations, plus FDC comparison panels for selected gauges
spanning the metric range (best, median, worst).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.styling import apply_style, COLOR_PALETTES

SPEC = ToolSpec(
    name='plot_fdc',
    description='Plot flow duration curve comparisons between simulated and '
                'observed discharge. Produces a map of a distributional metric '
                'and FDC panels for representative gauges.',
    inputs={
        'metrics_file': Param('path', required=True,
                              description='FDC metrics CSV with gauge_id and metric columns'),
        'fdc_file': Param('path', required=True,
                          description='FDC percentile data CSV with gauge_id, exceedance_prob, sim_q, obs_q'),
        'locations_file': Param('path', required=True,
                                description='Gauge metadata CSV with gauge_id, lat, lon'),
        'metric': Param('str', required=False,
                         description='Metric column to map (default: wasserstein)'),
        'n_panels': Param('int', required=False,
                          description='Number of FDC comparison panels (default: 6, ignored if gauge_ids given)'),
        'gauge_ids': Param('str', required=False,
                           description='Comma-separated gauge IDs for FDC panels (overrides auto-selection)'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': 'Combined FDC comparison figure'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    metrics_file = config['metrics_file']
    fdc_file = config['fdc_file']
    locations_file = config['locations_file']
    metric = config.get('metric', 'wasserstein') or 'wasserstein'
    n_panels_raw = config.get('n_panels', 6)
    n_panels = int(n_panels_raw) if n_panels_raw is not None else 6
    gauge_ids_str = config.get('gauge_ids', '') or ''
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    # Load data
    metrics = pd.read_csv(metrics_file)
    metrics['gauge_id'] = metrics['gauge_id'].astype(str)
    fdc = pd.read_csv(fdc_file)
    fdc['gauge_id'] = fdc['gauge_id'].astype(str)
    locs = pd.read_csv(locations_file)
    locs['gauge_id'] = locs['gauge_id'].astype(str)

    # Merge metrics with locations
    merged = locs.merge(metrics[['gauge_id', metric]], on='gauge_id', how='inner')
    merged = merged.dropna(subset=[metric])

    print(f"  Metric: {metric}")
    print(f"  Gauges with data: {len(merged)}")

    if len(merged) == 0:
        raise ValueError(f"No valid data for metric '{metric}'")

    # Select gauges for FDC panels
    if gauge_ids_str:
        # Fixed selection by gauge ID
        requested_ids = [g.strip() for g in gauge_ids_str.split(',')]
        selected = merged[merged['gauge_id'].isin(requested_ids)]
        # Preserve requested order
        selected = selected.set_index('gauge_id').loc[
            [g for g in requested_ids if g in selected['gauge_id'].values]
        ].reset_index()
        n_panels = len(selected)
        print(f"  Fixed gauge selection: {n_panels} of {len(requested_ids)} requested found")
    else:
        # Auto-select: best, worst, and evenly spaced by metric
        sorted_gauges = merged.sort_values(metric)
        n_available = len(sorted_gauges)
        n_panels = min(n_panels, n_available)
        if n_panels <= 2:
            idx = [0, n_available - 1][:n_panels]
        else:
            idx = np.linspace(0, n_available - 1, n_panels, dtype=int)
        selected = sorted_gauges.iloc[idx]

    # Layout: map on top, FDC panels below
    n_cols = min(n_panels, 3)
    n_rows_fdc = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(14, 6 + 3.5 * n_rows_fdc))
    gs = fig.add_gridspec(1 + n_rows_fdc, n_cols, height_ratios=[2.0] + [1] * n_rows_fdc,
                          hspace=0.35, wspace=0.3)

    # --- Map panel ---
    proj = ccrs.PlateCarree()
    ax_map = fig.add_subplot(gs[0, :], projection=proj)
    ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=1)
    ax_map.add_feature(cfeature.OCEAN, facecolor='#d4e8f0', zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555', zorder=2)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#999999', zorder=2)

    # Colormap: lower wasserstein = better
    vmin = merged[metric].quantile(0.05)
    vmax = merged[metric].quantile(0.95)
    cmap = 'YlOrRd'

    sc = ax_map.scatter(
        merged['lon'], merged['lat'],
        c=merged[metric], cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=50, edgecolors='black', linewidths=0.5,
        zorder=5, transform=proj,
    )

    # Highlight selected gauges
    ax_map.scatter(
        selected['lon'].values, selected['lat'].values,
        facecolors='none', edgecolors='blue', linewidths=2,
        s=120, zorder=6, transform=proj,
    )

    cbar = plt.colorbar(sc, ax=ax_map, shrink=0.6, pad=0.02)
    metric_label = metric.replace('_', ' ').title()
    cbar.set_label(metric_label, fontsize=11)
    ax_map.set_global()
    ax_map.set_title(f'{metric_label} at Gauge Locations', fontsize=13, fontweight='bold')
    ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, color='gray')

    # Stats annotation
    vals = merged[metric].dropna()
    stats = f"Median: {vals.median():.3f}\nMean: {vals.mean():.3f}\nn = {len(vals)}"
    ax_map.text(0.02, 0.02, stats, transform=ax_map.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=10)

    # --- FDC panels ---
    for i, (_, gauge_row) in enumerate(selected.iterrows()):
        row_idx = 1 + i // n_cols
        col_idx = i % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])

        gid = str(gauge_row['gauge_id'])
        g_fdc = fdc[fdc['gauge_id'] == gid].sort_values('exceedance_prob')

        if len(g_fdc) == 0:
            ax.text(0.5, 0.5, 'No FDC data', transform=ax.transAxes, ha='center')
            continue

        ax.semilogy(g_fdc['exceedance_prob'] * 100, g_fdc['obs_q'],
                     color='#2166ac', linewidth=2, label='Observed')
        ax.semilogy(g_fdc['exceedance_prob'] * 100, g_fdc['sim_q'],
                     color='#b2182b', linewidth=2, linestyle='--', label='Simulated')

        ax.fill_between(g_fdc['exceedance_prob'] * 100,
                         g_fdc['obs_q'], g_fdc['sim_q'],
                         alpha=0.15, color='gray')

        # Get river name if available
        river = gauge_row.get('river_name', '')
        title = f"ID {gid}"
        if river and str(river) != 'nan':
            title = f"{river} ({gid})"
        metric_val = gauge_row[metric]
        ax.set_title(f"{title}\n{metric_label}={metric_val:.3f}", fontsize=9)
        ax.set_xlabel('Exceedance Probability (%)', fontsize=9)
        ax.set_ylabel('Discharge (m³/s)', fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, 100)

    out_path = output_dir / 'fdc_comparison.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close()

    return {
        'plot_file': str(out_path),
    }
