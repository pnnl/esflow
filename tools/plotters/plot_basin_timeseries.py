"""
Plot drainage basin maps with discharge time series.

For each requested gauge, produces a two-panel figure:
  top  — basin polygon with gauge marker on a cartopy map
  bottom — simulated vs observed daily discharge time series
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.styling import apply_style, COLOR_PALETTES

SPEC = ToolSpec(
    name='plot_basin_timeseries',
    description='Plot drainage basin maps with discharge time series for '
                'selected gauges. Each gauge produces a figure with the basin '
                'polygon on a map and a sim-vs-obs time series panel.',
    inputs={
        'basins_file': Param('path', required=True,
                             description='GeoJSON file with basin polygons (property: grdc_no)'),
        'sim_file': Param('path', required=True,
                          description='Simulated discharge CSV (time index, gauge_id columns)'),
        'obs_file': Param('path', required=True,
                          description='Observed discharge CSV (time index, gauge_id columns)'),
        'locations_file': Param('path', required=True,
                                description='Gauge metadata CSV with gauge_id, lat, lon, river_name'),
        'gauge_ids': Param('str', required=True,
                           description='Comma-separated GRDC gauge IDs to plot'),
        'metrics_file': Param('path', required=False, default='',
                              description='Optional metrics CSV (from compute_metrics) to annotate plots with RMSE, NSE, etc.'),
    },
    outputs={
        'plot_files': {'type': 'list', 'description': 'List of generated PNG paths'},
    },
)


def _load_basins(basins_file):
    """Load basin polygons from GeoJSON."""
    with open(basins_file) as f:
        return json.load(f)


def _get_basin(basins, grdc_no):
    """Get basin feature for a specific gauge."""
    for feat in basins['features']:
        if feat['properties']['grdc_no'] == grdc_no:
            return feat
    return None


def _get_bounds(basin):
    """Get bounding box (minx, miny, maxx, maxy) of basin polygon."""
    coords = basin['geometry']['coordinates']
    all_x, all_y = [], []
    for ring in coords:
        if isinstance(ring[0][0], list):
            for subring in ring:
                all_x.extend([p[0] for p in subring])
                all_y.extend([p[1] for p in subring])
        else:
            all_x.extend([p[0] for p in ring])
            all_y.extend([p[1] for p in ring])
    return min(all_x), min(all_y), max(all_x), max(all_y)


def _plot_basin_map(ax, basin, gauge_lon, gauge_lat):
    """Render basin polygon with gauge marker on cartopy axes."""
    minx, miny, maxx, maxy = _get_bounds(basin)
    pad_x = (maxx - minx) * 0.15
    pad_y = (maxy - miny) * 0.15
    ax.set_extent([minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
                  crs=ccrs.PlateCarree())

    # Background
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#666666',
                   linestyle='--', zorder=3)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='#4a90d9', zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor='#a6cee3', edgecolor='#4a90d9',
                   linewidth=0.5, zorder=2)

    # Basin polygon
    transform = ccrs.PlateCarree()
    coords = basin['geometry']['coordinates']
    for ring in coords:
        if isinstance(ring[0][0], list):
            for subring in ring:
                xs = [p[0] for p in subring]
                ys = [p[1] for p in subring]
                ax.fill(xs, ys, alpha=0.4, color='#6baed6', edgecolor='#2171b5',
                        linewidth=2, transform=transform, zorder=5)
        else:
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            ax.fill(xs, ys, alpha=0.4, color='#6baed6', edgecolor='#2171b5',
                    linewidth=2, transform=transform, zorder=5)

    # Gauge marker
    ax.scatter([gauge_lon], [gauge_lat], c='#e31a1c', s=150, marker='^',
              edgecolors='white', linewidth=2, zorder=10, transform=transform)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    basins_file = config['basins_file']
    sim_file = config['sim_file']
    obs_file = config['obs_file']
    locations_file = config['locations_file']
    gauge_ids_str = config['gauge_ids']
    output_dir = Path(config['output_dir'])

    apply_style('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    # Parse gauge IDs
    gauge_ids = [g.strip() for g in gauge_ids_str.split(',')]

    metrics_file = config.get('metrics_file', '')

    # Load data
    basins = _load_basins(basins_file)
    sim = pd.read_csv(sim_file, index_col=0, parse_dates=True)
    obs = pd.read_csv(obs_file, index_col=0, parse_dates=True)
    locs = pd.read_csv(locations_file)

    # Load metrics if provided
    metrics_df = None
    if metrics_file:
        metrics_df = pd.read_csv(metrics_file)
        metrics_df['gauge_id'] = metrics_df['gauge_id'].astype(str)
        print(f"  Metrics loaded: {len(metrics_df)} gauges")

    sim.columns = [str(c) for c in sim.columns]
    obs.columns = [str(c) for c in obs.columns]
    locs['gauge_id'] = locs['gauge_id'].astype(str)

    print(f"  Basins loaded: {len(basins['features'])}")
    print(f"  Gauge IDs requested: {gauge_ids}")

    plot_files = []

    for gid in gauge_ids:
        grdc_no = int(gid)
        basin = _get_basin(basins, grdc_no)
        if basin is None:
            print(f"  Warning: No basin polygon for {gid}, skipping")
            continue

        # Get gauge metadata
        row = locs[locs['gauge_id'] == gid]
        if len(row) == 0:
            print(f"  Warning: No metadata for {gid}, skipping")
            continue
        row = row.iloc[0]
        lat, lon = float(row['lat']), float(row['lon'])
        river = row.get('river_name', '')
        area = basin['properties'].get('area_km2', np.nan)

        # Build title
        title = str(river) if river and str(river) != 'nan' else f'Gauge {gid}'
        title += f'\nGRDC #{gid}'
        if not np.isnan(area):
            title += f' | Area: {area:,.0f} km²'

        # Get time series
        has_sim = gid in sim.columns
        has_obs = gid in obs.columns

        if not has_sim and not has_obs:
            print(f"  Warning: No time series data for {gid}, skipping")
            continue

        # Create figure
        fig = plt.figure(figsize=(10, 12))
        fig.patch.set_facecolor(palette['background'])

        ax_map = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax_ts = fig.add_subplot(2, 1, 2)

        # Basin map
        _plot_basin_map(ax_map, basin, lon, lat)
        ax_map.set_title(title, fontsize=12, fontweight='bold')

        # Time series
        if has_obs:
            obs_ts = obs[gid].dropna()
            ax_ts.plot(obs_ts.index, obs_ts.values, color='#2166ac',
                      linewidth=0.8, alpha=0.8, label='Observed')
        if has_sim:
            sim_ts = sim[gid].dropna()
            ax_ts.plot(sim_ts.index, sim_ts.values, color='#b2182b',
                      linewidth=0.8, alpha=0.8, label='Simulated')

        ax_ts.set_xlabel('Time', fontsize=10)
        ax_ts.set_ylabel('Discharge (m³/s)', fontsize=10)
        ax_ts.legend(fontsize=9, loc='upper right')
        ax_ts.xaxis.set_major_locator(mdates.YearLocator())
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_facecolor(palette['background'])

        # Summary stats + metrics annotation
        if has_sim and has_obs:
            common_idx = sim_ts.index.intersection(obs_ts.index)
            if len(common_idx) > 0:
                sim_mean = sim[gid].loc[common_idx].mean()
                obs_mean = obs[gid].loc[common_idx].mean()
                lines = [
                    f'Obs mean: {obs_mean:,.0f} m³/s',
                    f'Sim mean: {sim_mean:,.0f} m³/s',
                ]
                # Add metrics from CSV if available
                if metrics_df is not None:
                    mrow = metrics_df[metrics_df['gauge_id'] == gid]
                    if len(mrow) > 0:
                        mrow = mrow.iloc[0]
                        for metric in ['rmse', 'nse', 'kge', 'pbias']:
                            if metric in mrow and not np.isnan(mrow[metric]):
                                if metric == 'rmse':
                                    lines.append(f'RMSE: {mrow[metric]:,.0f} m³/s')
                                elif metric == 'pbias':
                                    lines.append(f'PBIAS: {mrow[metric]:.1f}%')
                                else:
                                    lines.append(f'{metric.upper()}: {mrow[metric]:.3f}')
                stats = '\n'.join(lines)
                ax_ts.text(0.02, 0.95, stats, transform=ax_ts.transAxes,
                          fontsize=9, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        out_path = output_dir / f'basin_{gid}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                   facecolor=palette['background'])
        plt.close(fig)
        plot_files.append(str(out_path))
        print(f"  Saved: {out_path}")

    return {
        'plot_files': plot_files,
    }
