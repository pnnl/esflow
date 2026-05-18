"""
Plot a 3-panel bias comparison: Observation, Simulation, and Bias.

Produces a publication-ready figure with three vertically stacked map panels
(Obs, Sim, Bias) sharing a consistent color range for Obs/Sim and a diverging
colormap for the bias panel.  Optionally overlays summary statistics on the
bias panel.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param


UNIT_LOOKUP = {
    'QRUNOFF': 'mm/s',
    'RAIN': 'mm/s',
    'SNOW': 'mm/s',
    'QVEGE_plus_QVEGT_plus_QSOIL': 'mm/s',
    'pr': 'mm/day',
    'et': 'kg/m²/s',
    'evspsbl': 'kg/m²/s',
    'mrro': 'kg/m²/s',
    'bias': 'mm/s',
}


SPEC = ToolSpec(
    name='plot_bias_comparison',
    description=(
        'Plot a 3-panel stacked map comparison: Observation (top), Simulation '
        '(middle), and Bias (bottom).  Obs and Sim panels share a common '
        'sequential colorbar; the Bias panel uses a diverging colorbar.  '
        'Optionally overlays bias statistics (mean bias, RMSE, spatial r) as '
        'an inset on the bias panel. Input: three NetCDF files (obs field, '
        'model field, bias field) from extract_gridded_field and '
        'compute_spatial_bias.'
    ),
    inputs={
        'obs_file': Param('path', required=True,
                          description='Observation field NetCDF (from extract_gridded_field)'),
        'sim_file': Param('path', required=True,
                          description='Simulation field NetCDF (from extract_gridded_field)'),
        'bias_file': Param('path', required=True,
                           description='Bias field NetCDF (from compute_spatial_bias)'),
        'obs_label': Param('str', required=False, default='Observation',
                           description='Label for the obs panel title'),
        'sim_label': Param('str', required=False, default='Simulation',
                           description='Label for the sim panel title'),
        'bias_label': Param('str', required=False, default='Bias (Sim − Obs)',
                            description='Label for the bias panel title'),
        'time_range': Param('str', required=False, default='',
                            description='Time range string to append to obs/sim titles, e.g. "1985-1989"'),
        'units': Param('str', required=False, default='',
                       description='Units for colorbars. Auto-detected if omitted.'),
        'stats_file': Param('path', required=False, default='',
                            description='Bias stats CSV to overlay on the bias panel'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': '3-panel comparison PNG'},
    },
)


def _load_field(path):
    """Load a 2-D lat-lon field from NetCDF, averaging over time if present.

    Returns (lat, lon, data, vname, units_str, time_range_str, ds).
    """
    import pandas as pd

    ds = xr.open_dataset(path)
    skip = {'landfrac', 'area', 'landmask', 'pftmask', 'lat', 'lon'}
    data_vars = [v for v in ds.data_vars if v not in skip]
    if not data_vars:
        raise ValueError(f"No data variables in {path}")
    vname = data_vars[0]
    field = ds[vname]

    # Extract time range: from global attrs first, then from time dim
    time_range = ds.attrs.get('time_range', '')
    if not time_range and 'years' in ds.attrs:
        try:
            import ast
            yrs = ast.literal_eval(ds.attrs['years'])
            y0, y1 = min(yrs), max(yrs)
            time_range = f"{y0}–{y1}" if y0 != y1 else str(y0)
        except Exception:
            pass
    if not time_range and 'time' in field.dims and len(field.time) > 0:
        try:
            tvals = field.time.values
            y0 = int(tvals[0].year) if hasattr(tvals[0], 'year') else None
            y1 = int(tvals[-1].year) if hasattr(tvals[-1], 'year') else None
            if y0 is not None and y1 is not None:
                time_range = f"{y0}–{y1}" if y0 != y1 else str(y0)
        except Exception:
            pass

    if 'time' in field.dims:
        field = field.mean(dim='time')

    return ds['lat'].values, ds['lon'].values, field.values, vname, field.attrs.get('units', ''), time_range, ds


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    obs_file = config['obs_file']
    sim_file = config['sim_file']
    bias_file = config['bias_file']
    obs_label = config.get('obs_label', 'Observation')
    sim_label = config.get('sim_label', 'Simulation')
    bias_label = config.get('bias_label', 'Bias (Sim − Obs)')
    time_range = config.get('time_range', '')
    units = config.get('units', '')
    stats_file = config.get('stats_file', '')
    output_dir = Path(config['output_dir'])

    # Load fields
    lat_o, lon_o, data_o, vname_o, units_o, trange_o, ds_o = _load_field(obs_file)
    lat_s, lon_s, data_s, vname_s, units_s, trange_s, ds_s = _load_field(sim_file)
    lat_b, lon_b, data_b, vname_b, units_b, _,        ds_b = _load_field(bias_file)

    ds_o.close(); ds_s.close(); ds_b.close()

    # Resolve units
    if not units:
        units = units_o or units_s or UNIT_LOOKUP.get(vname_o, '') or UNIT_LOOKUP.get(vname_s, '')

    print(f"  Obs: {vname_o} {data_o.shape}, Sim: {vname_s} {data_s.shape}, Bias: {vname_b} {data_b.shape}")
    if units:
        print(f"  Units: {units}")

    # Build titles with time ranges from data (override with time_range if set)
    obs_tr = time_range if time_range else trange_o
    sim_tr = time_range if time_range else trange_s
    obs_title = f"{obs_label} ({obs_tr})" if obs_tr else obs_label
    sim_title = f"{sim_label} ({sim_tr})" if sim_tr else sim_label
    bias_title = bias_label

    # Common color range for Obs and Sim
    all_finite = np.concatenate([
        data_o[np.isfinite(data_o)],
        data_s[np.isfinite(data_s)],
    ])
    vmin_field = np.nanpercentile(all_finite, 2)
    vmax_field = np.nanpercentile(all_finite, 98)

    # Diverging range for Bias
    finite_bias = data_b[np.isfinite(data_b)]
    vmax_bias = max(abs(np.nanpercentile(finite_bias, 2)),
                    abs(np.nanpercentile(finite_bias, 98)))
    vmin_bias = -vmax_bias

    cb_label = f'{units}' if units else ''

    # ── Build figure: 3 rows, each map + colorbar on right ───────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    fig = plt.figure(figsize=(12, 14))
    # 3 rows for maps, each with a narrow colorbar column on the right
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.025],
                          hspace=0.15, wspace=0.03)

    panels = [
        (0, lat_o, lon_o, data_o, obs_title,  'viridis', vmin_field, vmax_field),
        (1, lat_s, lon_s, data_s, sim_title,   'viridis', vmin_field, vmax_field),
        (2, lat_b, lon_b, data_b, bias_title,  'PuOr_r',  vmin_bias,  vmax_bias),
    ]

    axes = []
    images = []

    for row, lat, lon, data, label, cmap, vmin, vmax in panels:
        if has_cartopy:
            proj = ccrs.PlateCarree()
            ax = fig.add_subplot(gs[row, 0], projection=proj)
            im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#555555')
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#999999')
            ax.set_global()
            ax.gridlines(linewidth=0.2, alpha=0.4, color='gray')
        else:
            ax = fig.add_subplot(gs[row, 0])
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='auto')

        ax.set_title(label, fontsize=13, fontweight='bold')
        axes.append(ax)
        images.append(im)

    # Colorbars on the right — Obs and Sim share the same one
    cax_field = fig.add_subplot(gs[0, 1])
    cb1 = fig.colorbar(images[0], cax=cax_field)
    cb1.set_label(cb_label, fontsize=10)
    cb1.ax.tick_params(labelsize=9)

    # Sim colorbar — same scale, just mirror the ticks
    cax_sim = fig.add_subplot(gs[1, 1])
    cb1b = fig.colorbar(images[1], cax=cax_sim)
    cb1b.set_label(cb_label, fontsize=10)
    cb1b.ax.tick_params(labelsize=9)

    # Bias colorbar
    cax_bias = fig.add_subplot(gs[2, 1])
    cb2 = fig.colorbar(images[2], cax=cax_bias)
    cb2.set_label(f'Bias ({cb_label})' if cb_label else 'Bias', fontsize=10)
    cb2.ax.tick_params(labelsize=9)

    # Panel labels (a), (b), (c)
    for idx, ax in enumerate(axes):
        ax.text(0.01, 1.02, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom')

    # Stats inset on bias panel
    if stats_file:
        try:
            import pandas as pd
            stats = pd.read_csv(stats_file)
            lines = []
            if 'mean_bias' in stats.columns:
                row = stats.iloc[0]
                mb = row.get('mean_bias', None)
                rmse = row.get('rmse', None)
                corr = row.get('spatial_correlation', None)
                if mb is not None:
                    lines.append(f"Mean Bias: {float(mb):.2e}")
                if rmse is not None:
                    lines.append(f"RMSE:      {float(rmse):.2e}")
                if corr is not None:
                    lines.append(f"Spatial r: {float(corr):.3f}")
            if lines:
                stats_text = '\n'.join(lines)
                axes[2].text(
                    0.02, 0.05, stats_text, transform=axes[2].transAxes,
                    fontsize=9, fontfamily='monospace', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              alpha=0.9, edgecolor='#555555', linewidth=0.8))
                print(f"  Stats overlay: {len(lines)} lines")
        except Exception as e:
            print(f"  Warning: Could not overlay stats: {e}")

    # Save
    out_path = output_dir / 'bias_comparison.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved: {out_path}")
    return {'plot_file': str(out_path)}
