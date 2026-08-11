"""
Compute spatial bias between two gridded fields.

Computes difference (field_a minus field_b), global mean bias,
spatial RMSE, and spatial correlation. Regrids to coarser grid
if resolutions differ.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param

import pandas as pd

logger = logging.getLogger(__name__)


SPEC = ToolSpec(
    name='compute_spatial_bias',
    description=(
        'Compute spatial bias between two gridded fields (A minus B). '
        'Inputs are two NetCDF files from extract_gridded_field. '
        'If grids differ, regrids the finer to the coarser resolution using nearest-neighbor. '
        'Outputs: bias field as NetCDF, and a CSV with global statistics '
        '(mean_bias, rmse, spatial_correlation). '
        'Use field_a for model and field_b for observations to get (model minus obs).'
    ),
    inputs={
        'field_a': Param('path', required=True,
                         description='NetCDF file for field A (e.g., model output)'),
        'field_b': Param('path', required=True,
                         description='NetCDF file for field B (e.g., observations)'),
    },
    outputs={
        'bias_file': {'type': 'netcdf', 'description': 'Bias field (A minus B) as lat-lon NetCDF'},
        'stats_file': {'type': 'csv', 'description': 'CSV with mean_bias, rmse, spatial_correlation'},
    },
)


def _regrid_nearest(source, target_lat, target_lon):
    """Regrid source DataArray to target lat-lon grid using nearest neighbor."""
    return source.interp(lat=target_lat, lon=target_lon, method='nearest')


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    field_a_path = config['field_a']
    field_b_path = config['field_b']
    output_dir = Path(config['output_dir'])

    ds_a = xr.open_dataset(field_a_path)
    ds_b = xr.open_dataset(field_b_path)

    # Find data variable in each dataset
    skip_vars = {'landfrac', 'area', 'landmask', 'pftmask', 'lat', 'lon'}

    def _get_field(ds, path):
        data_vars = [v for v in ds.data_vars if v not in skip_vars]
        if not data_vars:
            raise ValueError(f"No data variables found in {path}")
        return ds[data_vars[0]]

    a = _get_field(ds_a, field_a_path)
    b = _get_field(ds_b, field_b_path)

    logger.info("Field A: %s %s", a.name, dict(a.sizes))
    logger.info("Field B: %s %s", b.name, dict(b.sizes))

    # Handle time dimension if present
    if 'time' in a.dims:
        a = a.mean(dim='time')
    if 'time' in b.dims:
        b = b.mean(dim='time')

    # Regrid if needed — regrid finer to coarser
    a_nlat = len(ds_a.lat)
    b_nlat = len(ds_b.lat)

    if a_nlat != b_nlat or len(ds_a.lon) != len(ds_b.lon):
        if a_nlat > b_nlat:
            logger.info("Regridding A (%s lat) to B grid (%s lat)", a_nlat, b_nlat)
            a = _regrid_nearest(a, ds_b.lat, ds_b.lon)
        else:
            logger.info("Regridding B (%s lat) to A grid (%s lat)", b_nlat, a_nlat)
            b = _regrid_nearest(b, ds_a.lat, ds_a.lon)

    # Compute bias
    bias = a.values.astype(float) - b.values.astype(float)
    mask = np.isfinite(bias)

    if not mask.any():
        raise ValueError("No valid grid cells after computing bias (all NaN)")

    # Statistics
    mean_bias = float(np.nanmean(bias[mask]))
    rmse = float(np.sqrt(np.nanmean(bias[mask] ** 2)))

    a_vals = a.values.astype(float)
    b_vals = b.values.astype(float)
    valid = np.isfinite(a_vals) & np.isfinite(b_vals)
    if valid.sum() > 2:
        corr = float(np.corrcoef(a_vals[valid].flatten(), b_vals[valid].flatten())[0, 1])
    else:
        corr = np.nan

    logger.info("Mean bias: %.6f", mean_bias)
    logger.info("RMSE: %.6f", rmse)
    logger.info("Spatial correlation: %.4f", corr)

    # Save bias field
    lat_coord = ds_b.lat if a_nlat > b_nlat else ds_a.lat
    lon_coord = ds_b.lon if a_nlat > b_nlat else ds_a.lon

    bias_da = xr.DataArray(
        bias,
        dims=['lat', 'lon'],
        coords={'lat': lat_coord, 'lon': lon_coord},
        name='bias',
        attrs={'long_name': f'Bias ({a.name} minus {b.name})', 'units': a.attrs.get('units', '')},
    )
    bias_ds = bias_da.to_dataset()
    bias_path = output_dir / 'spatial_bias.nc'
    bias_ds.to_netcdf(bias_path)

    # Save stats
    stats = pd.DataFrame([{
        'metric': 'spatial_comparison',
        'mean_bias': round(mean_bias, 6),
        'rmse': round(rmse, 6),
        'spatial_correlation': round(corr, 4),
    }])
    stats_path = output_dir / 'spatial_stats.csv'
    stats.to_csv(stats_path, index=False)

    ds_a.close()
    ds_b.close()

    return {
        'bias_file': str(bias_path),
        'stats_file': str(stats_path),
    }
