"""
Plot water balance components at global and basin scales.

Produces a composite figure with a global residual map on top, and per-basin
bar charts of P, ET, Q, and residual below.  Basin outlines are overlaid on the
global map.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.styling import apply_style, COLOR_PALETTES


SPEC = ToolSpec(
    name='plot_water_balance_basins',
    description='Plot a composite water balance figure: global residual map '
                'with basin outlines (top) and per-basin bar charts of P, ET, '
                'Q, and residual (bottom panels).',
    inputs={
        'residual_file': Param('path', required=True,
                               description='NetCDF of water balance residual (P-ET-Q)'),
        'basin_precip_file': Param('path', required=True,
                                   description='CSV of per-basin mean precipitation'),
        'basin_et_file': Param('path', required=True,
                               description='CSV of per-basin mean ET'),
        'basin_runoff_file': Param('path', required=True,
                                   description='CSV of per-basin mean runoff'),
        'basin_residual_file': Param('path', required=True,
                                     description='CSV of per-basin mean residual'),
        'basins_file': Param('path', required=True,
                             description='GeoJSON file with basin polygons'),
        'gauge_ids': Param('str', required=True,
                           description='Comma-separated GRDC gauge IDs'),
        'locations_file': Param('path', required=False, default='',
                                description='Gauge metadata CSV with river_name column'),
    },
    outputs={
        'plot_file': {'type': 'png',
                      'description': 'Composite water balance figure'},
    },
)


# Basin name lookup from gauge metadata
BASIN_NAMES = {
    '3629000': 'Amazon',
    '4121801': 'Missouri',
    '4115200': 'Columbia',
    '6742900': 'Danube',
    '2969100': 'Mekong',
    '1159100': 'Orange',
}


def _get_basin_name(gid, locations_df=None):
    """Get human-readable basin name."""
    gid_str = str(gid)
    if locations_df is not None:
        row = locations_df[locations_df['gauge_id'].astype(str) == gid_str]
        if len(row) > 0 and 'river_name' in row.columns:
            name = str(row.iloc[0]['river_name'])
            if name and name != 'nan':
                return name
    return BASIN_NAMES.get(gid_str, f'Basin {gid_str}')


def _draw_polygon(ax, feature, transform, color='black', lw=1.5):
    """Draw a GeoJSON polygon on a cartopy axes."""
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    geom = feature['geometry']
    coords = geom['coordinates']
    geom_type = geom['type']

    rings = []
    if geom_type == 'Polygon':
        rings.append(coords[0])
    elif geom_type == 'MultiPolygon':
        for poly in coords:
            rings.append(poly[0])

    patches = []
    for ring in rings:
        xy = np.array(ring)
        patches.append(MplPolygon(xy, closed=True))

    pc = PatchCollection(patches, facecolor='none', edgecolor=color,
                         linewidth=lw, transform=transform, zorder=5)
    ax.add_collection(pc)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    residual_file = config['residual_file']
    basin_precip_file = config['basin_precip_file']
    basin_et_file = config['basin_et_file']
    basin_runoff_file = config['basin_runoff_file']
    basin_residual_file = config['basin_residual_file']
    basins_file = config['basins_file']
    gauge_ids_str = config['gauge_ids']
    locations_file = config.get('locations_file', '')
    output_dir = Path(config['output_dir'])

    gauge_ids = [g.strip() for g in gauge_ids_str.split(',')]
    n_basins = len(gauge_ids)

    apply_style('cream_ink')
    palette = COLOR_PALETTES['cream_ink']

    # Load locations for river names
    locations_df = None
    if locations_file:
        locations_df = pd.read_csv(locations_file)

    # Load basin CSVs
    df_p = pd.read_csv(basin_precip_file)
    df_et = pd.read_csv(basin_et_file)
    df_q = pd.read_csv(basin_runoff_file)
    df_res = pd.read_csv(basin_residual_file)

    for df in [df_p, df_et, df_q, df_res]:
        df['gauge_id'] = df['gauge_id'].astype(str)

    # Load residual field
    ds = xr.open_dataset(residual_file)
    skip = {'lat', 'lon', 'latitude', 'longitude', 'time', 'time_bnds'}
    data_vars = [v for v in ds.data_vars if v.lower() not in skip]
    vname = data_vars[0]
    field = ds[vname]
    if 'time' in field.dims:
        field = field.mean(dim='time')
    lat = ds['lat'].values
    lon = ds['lon'].values
    data = field.values
    ds.close()

    # Load basins GeoJSON
    with open(basins_file) as f:
        basins = json.load(f)

    # Build basin lookup
    basin_features = {}
    for feat in basins['features']:
        gid_key = str(feat['properties']['grdc_no'])
        basin_features[gid_key] = feat

    # ── Figure layout: global map on top, basin bar charts below ─────
    n_cols = min(3, n_basins)
    n_rows_bars = int(np.ceil(n_basins / n_cols))

    fig_height = 7 + 3.2 * n_rows_bars
    fig = plt.figure(figsize=(14, fig_height))
    fig.patch.set_facecolor(palette['background'])

    # Gridspec: top row for map (height ratio ~6), bottom rows for bars
    gs = fig.add_gridspec(1 + n_rows_bars, n_cols,
                          height_ratios=[6] + [3] * n_rows_bars,
                          hspace=0.35, wspace=0.35)

    # ── Top panel: global residual map ───────────────────────────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.PlateCarree()
        ax_map = fig.add_subplot(gs[0, :], projection=proj)

        vmax = max(abs(np.nanpercentile(data[np.isfinite(data)], 2)),
                   abs(np.nanpercentile(data[np.isfinite(data)], 98)))
        vmin = -vmax

        im = ax_map.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                               cmap='RdBu_r', vmin=vmin, vmax=vmax,
                               shading='auto')
        ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
        ax_map.add_feature(cfeature.OCEAN, facecolor='#d4e8f0', zorder=0)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555', zorder=2)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#999999', zorder=2)
        ax_map.set_global()
        ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, color='gray')

        # Draw basin outlines
        basin_colors = ['#2166ac', '#b2182b', '#1b7837', '#762a83',
                        '#e66101', '#d6604d']
        for i, gid in enumerate(gauge_ids):
            if gid in basin_features:
                color = basin_colors[i % len(basin_colors)]
                _draw_polygon(ax_map, basin_features[gid],
                              ccrs.PlateCarree(), color=color, lw=2.0)

        cb = plt.colorbar(im, ax=ax_map, shrink=0.6, pad=0.05,
                          orientation='horizontal')
        cb.set_label('Water Balance Residual P−ET−Q (mm/s)', fontsize=10)

        ax_map.set_title('Global Water Balance Residual with Basin Outlines',
                         fontsize=13, fontweight='bold')

    except ImportError:
        ax_map = fig.add_subplot(gs[0, :])
        vmax = max(abs(np.nanpercentile(data[np.isfinite(data)], 2)),
                   abs(np.nanpercentile(data[np.isfinite(data)], 98)))
        im = ax_map.pcolormesh(lon, lat, data, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax_map, shrink=0.6)
        ax_map.set_title('Global Water Balance Residual', fontsize=13,
                         fontweight='bold')

    # ── Bottom panels: per-basin bar charts ──────────────────────────
    bar_colors = {
        'P': '#3498DB',
        'ET': '#E74C3C',
        'Q': '#27AE60',
        'Residual': '#9B59B6',
    }

    for i, gid in enumerate(gauge_ids):
        row_idx = 1 + i // n_cols
        col_idx = i % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.set_facecolor(palette['background'])

        # Get basin means
        p_val = df_p.loc[df_p['gauge_id'] == gid, 'basin_mean'].values
        et_val = df_et.loc[df_et['gauge_id'] == gid, 'basin_mean'].values
        q_val = df_q.loc[df_q['gauge_id'] == gid, 'basin_mean'].values
        res_val = df_res.loc[df_res['gauge_id'] == gid, 'basin_mean'].values

        p_val = float(p_val[0]) if len(p_val) > 0 else 0
        et_val = float(et_val[0]) if len(et_val) > 0 else 0
        q_val = float(q_val[0]) if len(q_val) > 0 else 0
        res_val = float(res_val[0]) if len(res_val) > 0 else 0

        components = ['P', 'ET', 'Q', 'Residual']
        values = [p_val, et_val, q_val, res_val]
        colors = [bar_colors[c] for c in components]

        bars = ax.bar(components, values, color=colors, edgecolor='white',
                      linewidth=0.5, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.2e}', ha='center', va=va, fontsize=7,
                    fontweight='bold')

        basin_name = _get_basin_name(gid, locations_df)
        ax.set_title(basin_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('mm/s', fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=8)

    fig.suptitle('Water Balance: Global Residual & Basin Components',
                 fontsize=15, fontweight='bold', y=1.01)

    out_path = output_dir / 'water_balance_basins.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=palette['background'])
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {'plot_file': str(out_path)}
