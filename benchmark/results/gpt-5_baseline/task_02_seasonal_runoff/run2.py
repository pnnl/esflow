#!/usr/bin/env python3
import os
import glob
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def find_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.variables and ds[c].dims == (c,) or len(ds[c].dims) in (1,2):
            return c
    # Try case-insensitive
    lower_map = {k.lower(): k for k in ds.variables}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find any of coordinate names {candidates} in dataset.")

def open_elm_files(data_dir, case_name, years):
    files = []
    for y in years:
        pattern = os.path.join(
            data_dir,
            f"{case_name}.elm.h0.{y:04d}-*.nc"
        )
        files_year = sorted(glob.glob(pattern))
        files.extend(files_year)
    if not files:
        raise FileNotFoundError(f"No ELM files found for years {years} with pattern {pattern}")
    # Restrict to necessary variables to lighten memory
    # We'll include likely helpful variables if they exist
    def preprocess(ds):
        keep_vars = []
        for v in ['QRUNOFF', 'area', 'AREA', 'landfrac', 'LANDFRAC', 'landmask', 'land_mask']:
            if v in ds.variables:
                keep_vars.append(v)
        # Always keep coords
        for c in ['lat', 'LAT', 'latitude', 'lon', 'LON', 'longitude', 'time']:
            if c in ds.variables and c not in keep_vars:
                keep_vars.append(c)
        # Drop others
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        try:
            ds = ds.drop_vars(drop_vars)
        except Exception:
            pass
        return ds

    ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True, preprocess=preprocess)
    return ds

def convert_qrunoff_to_mmday(da):
    # ELM QRUNOFF is in mm/s per prompt
    # Convert to mm/day
    return da * 86400.0

def infer_area_weights(ds, lat_name, lon_name):
    # Try to get 'area' variable if present
    area_var = None
    area_name = None
    for a in ['area', 'AREA']:
        if a in ds.variables:
            area_var = ds[a]
            area_name = a
            break
    weights = None
    weight_source = None

    if area_var is not None:
        # Convert to m2 if necessary
        units = (area_var.attrs.get('units') or '').lower()
        area = area_var
        try:
            if units in ['m2', 'm^2', 'square meters', 'square metre', 'meter^2']:
                pass
            elif units in ['km2', 'km^2', 'square kilometers', 'square kilometres']:
                area = area * 1e6
            elif 'ster' in units or units in ['sr', 'steradian', 'steradians']:
                R = 6371000.0
                area = area * (R**2)
            elif units == '' or units is None:
                # Try to infer by magnitude: if max <= 0.1 assume steradians
                try:
                    m = float(area.max())
                    if m <= 0.1:
                        R = 6371000.0
                        area = area * (R**2)
                except Exception:
                    pass
            weight_source = f"{area_name}"
            weights = area
        except Exception:
            weights = None

    if weights is None:
        # Fallback to cos(lat) weighting
        lat = ds[lat_name]
        # Broadcast/calc cos(lat) in radians
        lat_r = np.deg2rad(lat)
        try:
            coslat = xr.apply_ufunc(np.cos, lat_r)
        except Exception:
            coslat = np.cos(lat_r)
        # If lat is 1D, expand to 2D weights by broadcasting across lon
        if coslat.ndim == 1 and lon_name in ds.dims:
            coslat = coslat.broadcast_like(ds[lon_name] if ds[lon_name].ndim == 2 else ds[lat_name])
        # We need 2D weights matching (lat, lon) dims if possible
        # Create a DataArray with dims matching grid
        if lat_name in ds.dims and lon_name in ds.dims:
            # Build 2D grid weights
            if coslat.ndim == 1:
                # Expand to 2D by outer with ones over lon
                lon = ds[lon_name]
                ones = xr.ones_like(lon) if lon.ndim == 2 else xr.DataArray(np.ones(lon.shape), dims=lon.dims, coords=lon.coords)
                weights = coslat * ones
            else:
                weights = coslat
        else:
            # As last resort, just use coslat even if shapes mismatch
            weights = coslat
        weight_source = "cos(lat)"
    # Apply land fraction if present
    landfrac_var = None
    for lf in ['landfrac', 'LANDFRAC', 'landmask', 'land_mask']:
        if lf in ds.variables:
            landfrac_var = ds[lf]
            break
    if landfrac_var is not None:
        try:
            # Ensure broadcasting
            weights = weights * xr.where(np.isfinite(landfrac_var), landfrac_var, 0)
            if weight_source:
                weight_source = f"{weight_source}*{lf}"
        except Exception:
            pass

    return weights, weight_source

def normalize_longitudes(lon):
    """Convert longitudes to [-180, 180) and return sorted indices."""
    lon_vals = lon.values
    lon_adj = lon_vals.copy()
    lon_adj = ((lon_adj + 180) % 360) - 180
    # Get sort order
    sort_idx = np.argsort(lon_adj)
    return lon_adj, sort_idx

def shift_to_pm180(da, lon_name='lon'):
    lon = da[lon_name]
    if lon.ndim != 1:
        # For 2D lon, try to adjust values to [-180,180] without reordering
        lon_adj = ((lon + 180) % 360) - 180
        da = da.assign_coords({lon_name: lon_adj})
        return da
    lon_vals = lon.values
    if np.nanmax(lon_vals) <= 180 and np.nanmin(lon_vals) >= -180:
        return da  # already in -180..180
    lon_adj, sort_idx = normalize_longitudes(lon)
    da2 = da.assign_coords({lon_name: lon_adj})
    da2 = da2.sortby(lon_name)
    return da2

def area_weighted_mean(field, weights, lat_name, lon_name):
    # Align weights to field
    w = weights
    # Mask by finite field
    w_masked = xr.where(np.isfinite(field), w, 0)
    denom = w_masked.sum(dim=[d for d in w_masked.dims if d in field.dims])
    num = (field * w_masked).sum(dim=[d for d in w_masked.dims if d in field.dims])
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = num / denom
    return float(mean.values)

def main():
    # Paths and settings
    data_base = "./data/sample/e3sm/lnd"
    case_name = "sample.v3.LR.historical"
    years = list(range(1985, 1990))
    period_start = "1985-01-01"
    period_end = "1989-12-31"

    # Output directory from the task specification (absolute path)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_02_seasonal_runoff/run2_output"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # Open and select data
    try:
        ds = open_elm_files(data_base, case_name, years)
    except Exception as e:
        print(f"Failed to open ELM files: {e}", file=sys.stderr)
        sys.exit(2)

    # Identify coordinate names
    try:
        lat_name = find_coord_name(ds, ['lat', 'LAT', 'latitude'])
        lon_name = find_coord_name(ds, ['lon', 'LON', 'longitude'])
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)

    if 'QRUNOFF' not in ds.variables:
        print("QRUNOFF not found in dataset.", file=sys.stderr)
        sys.exit(4)

    # Convert units and compute period mean
    try:
        da = ds['QRUNOFF']
        # Slice time range if time exists
        if 'time' in da.dims and 'time' in ds.coords:
            da = da.sel(time=slice(period_start, period_end))
        da_mmday = convert_qrunoff_to_mmday(da)
        da_mean = da_mmday.mean(dim='time') if 'time' in da_mmday.dims else da_mmday
    except Exception as e:
        print(f"Error processing QRUNOFF data: {e}", file=sys.stderr)
        sys.exit(5)

    # Compute area weights and global mean
    try:
        weights, weight_source = infer_area_weights(ds, lat_name, lon_name)
        # Regrid weights to mean field dims if needed
        # If weights lack time dimension it's fine
        if weights is None:
            print("Warning: Could not infer area weights; defaulting to uniform weights.", file=sys.stderr)
            # Uniform weights (equal-area) -> unweighted mean
            weights = xr.ones_like(da_mean)
            weight_source = "uniform"
        # Ensure weights broadcastable to field
        for d in da_mean.dims:
            if d not in weights.dims:
                # Try to expand along missing dimension
                weights = weights.expand_dims({d: da_mean.sizes[d]})
                # Align coords
                weights = weights.assign_coords({d: da_mean[d]})
        global_mean_mmday = area_weighted_mean(da_mean, weights, lat_name, lon_name)
    except Exception as e:
        print(f"Error computing area-weighted mean: {e}", file=sys.stderr)
        sys.exit(6)

    # Prepare outputs
    # Save mean field as NetCDF
    mean_ds = xr.Dataset()
    mean_var_name = 'QRUNOFF_mean'
    mean_da = da_mean.rename(mean_var_name)
    mean_da.attrs['units'] = 'mm/day'
    mean_da.attrs['long_name'] = 'Climatological mean total runoff (QRUNOFF)'
    mean_da.attrs['period'] = f"{period_start} to {period_end}"
    mean_da.attrs['source'] = f"ELM {case_name}"
    mean_da.attrs['weighting'] = weight_source if weight_source else ''
    mean_ds[mean_var_name] = mean_da

    nc_path = os.path.join(output_dir, f"{case_name}.QRUNOFF_mean_1985_1989.nc")
    try:
        encoding = {mean_var_name: {"zlib": True, "complevel": 4}}
        mean_ds.to_netcdf(nc_path, encoding=encoding)
        print(f"Saved mean field to {nc_path}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}", file=sys.stderr)

    # Save global stats as CSV
    try:
        # Compute additional stats
        field = da_mean.where(np.isfinite(da_mean))
        global_min = float(field.min().values)
        global_max = float(field.max().values)
        stats = {
            "variable": ["QRUNOFF"],
            "units": ["mm/day"],
            "case_name": [case_name],
            "period_start": [period_start],
            "period_end": [period_end],
            "weighting": [weight_source if weight_source else ''],
            "global_mean_mmday": [global_mean_mmday],
            "global_min_mmday": [global_min],
            "global_max_mmday": [global_max],
        }
        df_stats = pd.DataFrame(stats)
        csv_path = os.path.join(output_dir, f"{case_name}.QRUNOFF_global_stats_1985_1989.csv")
        df_stats.to_csv(csv_path, index=False)
        print(f"Saved global stats to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV stats: {e}", file=sys.stderr)

    # Plot map with global stats overlay
    try:
        # Shift longitudes to [-180,180] for cartopy
        da_plot = shift_to_pm180(da_mean, lon_name=lon_name)
        lat = da_plot[lat_name]
        lon = da_plot[lon_name]

        # Determine plotting vmax based on robust statistics
        data_vals = da_plot.values
        finite_vals = data_vals[np.isfinite(data_vals)]
        if finite_vals.size == 0:
            vmax = 10.0
        else:
            vmax = np.nanpercentile(finite_vals, 99)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 10.0

        fig = plt.figure(figsize=(12, 6), dpi=150)
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.3)

        # Handle 1D or 2D lon/lat
        transform = ccrs.PlateCarree()
        if lat.ndim == 1 and lon.ndim == 1:
            mesh = ax.pcolormesh(lon, lat, da_plot.squeeze(), cmap='viridis', vmin=0, vmax=float(vmax), transform=transform)
        else:
            mesh = ax.pcolormesh(lon, lat, da_plot.squeeze(), cmap='viridis', vmin=0, vmax=float(vmax), transform=transform)

        cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=50)
        cbar.set_label('Runoff (mm/day)')

        title = f"ELM QRUNOFF climatology (1985-1989)\n{case_name}"
        plt.title(title, fontsize=12)

        # Overlay global statistics text box
        textstr = (
            f"Area-weighted global mean: {global_mean_mmday:.3f} mm/day\n"
            f"Min: {global_min:.3f}  Max: {global_max:.3f} mm/day\n"
            f"Weighting: {weight_source if weight_source else 'none'}"
        )
        plt.gcf().text(0.02, 0.02, textstr, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        png_path = os.path.join(output_dir, f"{case_name}.QRUNOFF_mean_map_1985_1989.png")
        plt.savefig(png_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved map to {png_path}")
    except Exception as e:
        print(f"Error creating or saving plot: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Silence warnings to keep output clean
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()