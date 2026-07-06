"""
Plot a 2D gridded field on a map.

Reads a lat-lon NetCDF file and produces a global map with coastlines
and a colorbar. Suitable for model fields, observation fields, or bias maps.
Optionally overlays summary statistics from a CSV file.
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


# Auto-detect units from variable name
UNIT_LOOKUP = {
    'QRUNOFF': 'mm/s',
    'RAIN': 'mm/s',
    'SNOW': 'mm/s',
    'RAIN_plus_SNOW': 'mm/s',
    'QVEGE': 'mm/s',
    'QVEGT': 'mm/s',
    'QSOIL': 'mm/s',
    'QVEGE_plus_QVEGT_plus_QSOIL': 'mm/s',
    'pr': 'mm/day',
    'et': 'kg/m²/s',
    'evspsbl': 'kg/m²/s',
    'mrro': 'kg/m²/s',
    'bias': 'mm/s',
}


SPEC = ToolSpec(
    name='plot_gridded_map',
    description=(
        'Plot a 2D gridded field on a geographic map with coastlines. '
        'Input is a lat-lon NetCDF file (from extract_gridded_field or compute_spatial_bias). '
        'Automatically selects an appropriate colormap: diverging (RdBu_r) for bias fields, '
        'sequential for positive-only fields. Units are auto-detected from the variable name '
        'or can be specified explicitly. Optionally overlays summary statistics text. '
        'Output is a PNG map.'
    ),
    inputs={
        'field_file': Param('path', required=True,
                            description='Input NetCDF file with a 2D lat-lon field'),
        'variable': Param('str', required=False, default='',
                          description='Variable name to plot. If omitted, uses the first data variable.'),
        'title': Param('str', required=False, default='',
                       description='Plot title. If omitted, uses the variable name.'),
        'units': Param('str', required=False, default='',
                       description='Units for the colorbar label. If omitted, auto-detected from variable name.'),
        'stats_file': Param('path', required=False, default='',
                            description='Optional zonal stats CSV to overlay on the map as text annotation.'),
    },
    outputs={
        'plot_file': {'type': 'png', 'description': 'Map plot PNG'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    field_file = config['field_file']
    variable = config['variable']
    title = config['title']
    units = config.get('units', '')
    stats_file = config.get('stats_file', '')
    output_dir = Path(config['output_dir'])

    ds = xr.open_dataset(field_file)

    # Determine variable
    skip_vars = {'landfrac', 'area', 'landmask', 'pftmask', 'lat', 'lon'}
    if variable:
        vname = variable
    else:
        data_vars = [v for v in ds.data_vars if v not in skip_vars]
        if not data_vars:
            raise ValueError(f"No data variables found in {field_file}")
        vname = data_vars[0]
        print(f"  Auto-selected variable: {vname}")

    field = ds[vname]

    # Handle time dimension
    if 'time' in field.dims:
        field = field.mean(dim='time')

    lat = ds['lat'].values
    lon = ds['lon'].values
    data = field.values

    print(f"  Variable: {vname}, shape: {data.shape}")

    # Determine units
    if not units:
        # Try NetCDF attributes first
        units = field.attrs.get('units', '')
    if not units:
        # Fall back to lookup table
        units = UNIT_LOOKUP.get(vname, '')
    if units:
        print(f"  Units: {units}")

    # Determine colormap — diverging if data spans negative and positive
    finite_data = data[np.isfinite(data)]
    has_neg = np.nanmin(finite_data) < 0
    has_pos = np.nanmax(finite_data) > 0
    is_bias = 'bias' in vname.lower()

    if is_bias:
        cmap = 'PuOr_r'
        vmax = max(abs(np.nanpercentile(finite_data, 2)),
                   abs(np.nanpercentile(finite_data, 98)))
        vmin = -vmax
    else:
        cmap = 'viridis'
        vmin = np.nanpercentile(finite_data, 2)
        vmax = np.nanpercentile(finite_data, 98)

    # Colorbar label
    cb_label = f'{vname} ({units})' if units else vname

    # Plot
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if is_bias:
            # Bias maps: PlateCarree with detailed basemap
            proj = ccrs.PlateCarree()
            fig, ax = plt.subplots(figsize=(14, 6),
                                   subplot_kw={'projection': proj})
            im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#d4e8f0', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555', zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#999999', zorder=2)
            ax.set_global()
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, color='gray')
        else:
            # Non-bias maps: Robinson projection
            fig, ax = plt.subplots(figsize=(12, 6),
                                   subplot_kw={'projection': ccrs.Robinson()})
            im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.set_global()
    except ImportError:
        # Fallback without cartopy
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cb.set_label(cb_label, fontsize=10)

    plot_title = title if title else vname
    ax.set_title(plot_title, fontsize=13, fontweight='bold')

    # Overlay statistics inset box if provided
    if stats_file:
        try:
            import pandas as pd
            stats = pd.read_csv(stats_file)
            lines = []

            # Detect format: bias stats (mean_bias, rmse, spatial_correlation)
            # vs zonal stats (region, weighted_mean)
            if 'mean_bias' in stats.columns:
                row = stats.iloc[0]
                mean_bias = row.get('mean_bias', None)
                rmse = row.get('rmse', None)
                corr = row.get('spatial_correlation', None)
                if mean_bias is not None:
                    lines.append(f"Mean Bias:  {float(mean_bias):.2e} {units}")
                if rmse is not None:
                    lines.append(f"RMSE:       {float(rmse):.2e} {units}")
                if corr is not None:
                    lines.append(f"Spatial r:  {float(corr):.3f}")
            else:
                for _, row in stats.iterrows():
                    region = row.get('region', 'global')
                    wmean = row.get('weighted_mean', row.get('mean', None))
                    if wmean is not None:
                        lines.append(f"{region}: {wmean:.2e} {units}")

            if lines:
                stats_text = '\n'.join(lines)
                fig.text(0.02, 0.02, stats_text, fontsize=9,
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='white', alpha=0.9,
                                   edgecolor='#555555', linewidth=0.8),
                         verticalalignment='bottom')
                print(f"  Stats overlay: {len(lines)} lines")
        except Exception as e:
            print(f"  Warning: Could not overlay stats: {e}")

    # Save
    safe_name = vname.lower().replace('+', '_')
    out_path = output_dir / f'{safe_name}_map.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {out_path}")

    ds.close()

    return {'plot_file': str(out_path)}
