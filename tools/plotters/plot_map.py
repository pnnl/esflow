"""
Plot metrics on a map at gauge locations.

Displays a specific metric (nse, kge, etc.) as colored markers on a map.
"""

import logging
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

logger = logging.getLogger(__name__)

SPEC = ToolSpec(
    name='plot_map',
    description='Plot a validation metric on a map at gauge locations. '
                'Shows colored markers for each gauge.',
    inputs={
        'metrics_file': Param('path', required=True,
                              description='Metrics CSV with gauge_id and metric columns'),
        'locations_file': Param('path', required=True,
                                description='Gauge metadata or matched CSV with gauge_id, lat, lon'),
        'metric': Param('str', required=True,
                        description='Metric column to plot (e.g., nse, kge, pbias)'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': 'Map plot PNG'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    metrics_file = config['metrics_file']
    locations_file = config['locations_file']
    metric = config['metric']
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    # Load data
    metrics = pd.read_csv(metrics_file)
    metrics['gauge_id'] = metrics['gauge_id'].astype(str)

    locs = pd.read_csv(locations_file)
    locs['gauge_id'] = locs['gauge_id'].astype(str)

    # Merge
    merged = locs.merge(metrics[['gauge_id', metric]], on='gauge_id', how='inner')
    merged = merged.dropna(subset=[metric])

    logger.info("Metric: %s", metric)
    logger.info("Gauges with data: %s", len(merged))

    if len(merged) == 0:
        raise ValueError(f"No valid data for metric '{metric}'")

    # Set up colormap and normalization per metric
    if metric in ('nse', 'kge'):
        vmin, vmax = -1, 1
        cmap = 'RdYlGn'
    elif metric == 'pbias':
        vmax = max(abs(merged[metric].quantile(0.05)), abs(merged[metric].quantile(0.95)))
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmin = merged[metric].quantile(0.05)
        vmax = merged[metric].quantile(0.95)
        cmap = 'YlOrRd'

    # Plot with cartopy for coastlines and land mask
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': proj})

    # Add land, ocean, and coastlines
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='#d4e8f0', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#999999', zorder=2)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor='#88bbdd', zorder=2)

    sc = ax.scatter(
        merged['lon'], merged['lat'],
        c=merged[metric], cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=50, edgecolors='black', linewidths=0.5,
        zorder=5, transform=proj,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(metric.upper(), fontsize=11)

    ax.set_global()
    ax.set_title(f'{metric.upper()} at Gauge Locations', fontsize=13, fontweight='bold')
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, color='gray')

    # Annotate stats
    vals = merged[metric].dropna()
    stats = f"Median: {vals.median():.2f}\nMean: {vals.mean():.2f}\nn = {len(vals)}"
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=10)

    plt.tight_layout()

    out_path = output_dir / f'map_{metric}.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close()

    return {
        'plot_file': str(out_path),
    }
