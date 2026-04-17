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
    # Try exact and case-insensitive matches in coords or variables
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
        c_low = c.lower()
        c_up = c.upper()
        c_cap = c.capitalize()
        for cc in [c_low, c_up, c_cap]:
            if cc in ds.coords or cc in ds.variables:
                return cc
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
        raise FileNotFoundError(f"No ELM files found for years {years} in {data_dir}")
    ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True)
    return ds

def convert_qrunoff_to_mmday(da):
    # ELM QRUNOFF is in mm/s per prompt; Convert to mm/day
    return da * 86400.0

def infer_area_weights(ds, field, lat_name, lon_name):
    # Determine spatial dims from lat/lon coordinate variables
    lat_da = ds[lat_name] if lat_name in ds else field[lat_name]
    lon_da = ds[lon_name] if lon_name in ds else field[lon_name]
    spatial_dims = tuple(d for d in field.dims if (d in lat_da.dims) or (d in lon_da.dims))

    # Try area variable
    area = None
    area_name = None
    for a in ['area', 'AREA']:
        if a in ds.variables:
            area = ds[a]
            area_name = a
            break

    weight_source = None
    if area is not None:
        # Convert to m2 if we can infer units; otherwise assume area is OK
        units = (area.attrs.get('units') or '').lower()
        if units in ['km2', 'km^2', 'square kilometers', 'square kilometres']:
            area = area * 1e6
        elif 'ster' in units or units in ['sr', 'steradian', 'steradians']:
            R = 6371000.0
            area = area * (R**2)
        elif units in ['', None]:
            # heuristic: if typical magnitudes are very small, assume steradian
            try:
                m = float(area.max())
                if m <= 0.1:
                    R = 6371000.0
                    area = area * (R**2)
            except Exception:
                pass
        # Broadcast area to field
        try:
            area_b, field_b = xr.broadcast(area, field)
            weights = area_b.where(np.isfinite(field_b))
        except Exception:
            # Fallback to aligning by indexes
            weights = area
        weight_source = area_name
    else:
        # Fallback: cosine latitude weighting
        # If lat is 2D, take cos(lat) directly; if 1D, create outer product lat*ones(lon)
        if lat_da.ndim == 2:
            weights = xr.apply_ufunc(np.cos, np.deg2rad(lat_da))
        else:
            wlat = xr.apply_ufunc(np.cos, np.deg2rad(lat_da))
            if lon_da.ndim == 2:
                ones_lon = xr.ones_like(lon_da)
            else:
                ones_lon = xr.ones_like(lon_da)
            # This multiplication broadcasts to union of dims (outer product if necessary)
            weights = wlat * ones_lon
        # Broadcast to field shape
        try:
            weights, _ = xr.broadcast(weights, field)
        except Exception:
            pass
        weight_source = "cos(lat)"

    # Apply land fraction if present
    landfrac_var = None
    landfrac_name = None
    for lf in ['landfrac', 'LANDFRAC', 'landmask', 'land_mask']:
        if lf in ds.variables:
            landfrac_var = ds[lf]
            landfrac_name = lf
            break
    if landfrac_var is not None:
        try:
            lf_b, w_b = xr.broadcast(landfrac_var, weights)
            weights = w_b * xr.where(np.isfinite(lf_b), lf_b, 0)
            weight_source = f"{weight_source}*{landfrac_name}" if weight_source else landfrac_name
        except Exception:
            pass

    # Ensure weights only have spatial dims to avoid leftover dims
    # Reduce any non-spatial dims by taking first index (assume constant)
    extra_dims = [d for d in weights.dims if d not in spatial_dims]
    for d in extra_dims:
        try:
            weights = weights.isel({d: 0})
        except Exception:
            # Try mean over that dim if cannot isel
            try:
                weights = weights.mean(dim=d)
            except Exception:
                pass

    return weights, spatial_dims, weight_source

def shift_to_pm180(da, lon_name='lon'):
    lon = da[lon_name]
    # If lon is 2D, just wrap values to [-180,180] without sorting
    if lon.ndim != 1:
        lon_adj = ((lon + 180) % 360) - 180
        return da.assign_coords({lon_name: lon_adj})
    lon_vals = lon.values
    if np.nanmax(lon_vals) <= 180 and np.nanmin(lon_vals) >= -180:
        return da  # already in -180..180
    lon_adj = ((lon_vals + 180) % 360) - 180
    da2 = da.assign_coords({lon_name: lon_adj})
    da2 = da2.sortby(lon_name)
    return da2

def area_weighted_global_mean(field, weights, spatial_dims):
    # Mask weights where field is not finite
    w_masked = weights.where(np.isfinite(field), 0)
    num = (field * w_masked).sum(dim=spatial_dims, skipna=True)
    denom = w_masked.sum(dim=spatial_dims, skipna=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = num / denom
    # Convert to scalar
    try:
        return float(mean.values.item())
    except Exception:
        return float(np.array(mean.values).ravel()[0])

def main():
    # Paths and settings
    data_base = "./data/sample/e3sm/lnd"
    case_name = "sample.v3.LR.historical"
    years = list(range(1985, 1990))
    period_start = "1985-01-01"
    period_end = "1989-12-31"

    # Output directory from the task specification (absolute path)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_02_seasonal_runoff/run2_debug/v1/output"
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
        lat_name = find_coord_name(ds, ['lat', 'latitude', 'LAT'])
        lon_name = find_coord_name(ds, ['lon', 'longitude', 'LON'])
    except KeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)

    if 'QRUNOFF' not in ds.variables:
        print("QRUNOFF not found in dataset.", file=sys.stderr)
        sys.exit(4)

    # Convert units and compute period mean
    try:
        da = ds['QRUNOFF']
        # Restrict to period
        if 'time' in da.dims:
            da = da.sel(time=slice(period_start, period_end))
            if da.sizes.get('time', 0) == 0:
                raise ValueError("No time steps found in the requested period.")
            da_mmday = convert_qrunoff_to_mmday(da)
            da_mean = da_mmday.mean(dim='time')
        else:
            da_mmday = convert_qrunoff_to_mmday(da)
            da_mean = da_mmday
    except Exception as e:
        print(f"Error processing QRUNOFF data: {e}", file=sys.stderr)
        sys.exit(5)

    # Compute area weights and global mean
    try:
        weights, spatial_dims, weight_source = infer_area_weights(ds, da_mean, lat_name, lon_name)
        if weights is None or len(spatial_dims) == 0:
            print("Warning: Could not infer area weights; defaulting to uniform weights.", file=sys.stderr)
            weights = xr.ones_like(da_mean)
            # Determine spatial dims as all dims of field
            spatial_dims = tuple(da_mean.dims)
            weight_source = "uniform"
        global_mean_mmday = area_weighted_global_mean(da_mean, weights, spatial_dims)
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

        transform = ccrs.PlateCarree()
        if lat.ndim == 1 and lon.ndim == 1:
            mesh = ax.pcolormesh(lon, lat, da_plot.squeeze(), cmap='viridis', vmin=0, vmax=float(vmax), transform=transform, shading='auto')
        else:
            mesh = ax.pcolormesh(lon, lat, da_plot.squeeze(), cmap='viridis', vmin=0, vmax=float(vmax), transform=transform, shading='auto')

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
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()