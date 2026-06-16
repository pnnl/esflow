#!/usr/bin/env python3
import os
import sys
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)

def find_variable(ds, names):
    for n in names:
        if n in ds.variables:
            return n
        if n in ds.data_vars:
            return n
    return None

def detect_lat_lon(ds):
    # Try common lat/lon names
    lat_candidates = ['lat', 'latitude', 'LAT', 'nav_lat']
    lon_candidates = ['lon', 'longitude', 'LON', 'nav_lon']
    lat_name = None
    lon_name = None
    for n in lat_candidates:
        if n in ds.variables or n in ds.coords:
            lat_name = n
            break
    for n in lon_candidates:
        if n in ds.variables or n in ds.coords:
            lon_name = n
            break
    return lat_name, lon_name

def to_2d_lat_lon(ds, lat_name, lon_name, spatial_dims):
    # Return 2D arrays (xarray.DataArray) of lat and lon aligned to spatial dims
    lat = ds[lat_name]
    lon = ds[lon_name]

    # Normalize longitude to [-180, 180] for plotting convenience but not necessary for area
    # Keep original lon for area calc
    # Generate 2D by broadcasting if needed
    # If lat and lon are 1D and match spatial dims separately
    if lat.ndim == 1 and lon.ndim == 1:
        # Identify which dims they align with
        # Assume lat aligns with one of spatial dims and lon with the other
        # xarray.broadcast will handle to 2D
        lat2d, lon2d = xr.broadcast(lat, lon)
        return lat2d.transpose(*spatial_dims), lon2d.transpose(*spatial_dims)
    elif lat.ndim == 2 and lon.ndim == 2:
        # Assume already matches spatial dims or can be transposed
        # Try to align ordering to spatial_dims
        lat2d = lat
        lon2d = lon
        # If dims not equal, try to transpose if contains dims
        if tuple(lat2d.dims) != tuple(spatial_dims):
            try:
                lat2d = lat2d.transpose(*spatial_dims)
            except Exception:
                pass
        if tuple(lon2d.dims) != tuple(spatial_dims):
            try:
                lon2d = lon2d.transpose(*spatial_dims)
            except Exception:
                pass
        return lat2d, lon2d
    else:
        # Mixed dimensionality; try to broadcast
        lat2d, lon2d = xr.broadcast(lat, lon)
        return lat2d, lon2d

def compute_cell_area_m2(ds, spatial_dims, lat_name, lon_name):
    # Compute grid cell area in m^2
    R = 6371000.0  # Earth radius in meters
    # Prefer 'area' variable if available
    area_var_name = find_variable(ds, ['area', 'AREA', 'cell_area'])
    area = None
    if area_var_name is not None:
        try:
            area = ds[area_var_name]
            # Align to spatial dims
            # Drop time dim if present
            if 'time' in area.dims:
                area = area.isel(time=0, drop=True)
            # Transpose to spatial dims
            if tuple(area.dims) != tuple(spatial_dims):
                try:
                    area = area.transpose(*spatial_dims)
                except Exception:
                    pass
            # Convert steradians to m^2 if needed
            try:
                units = area.attrs.get('units', '').lower()
            except Exception:
                units = ''
            area_vals = area.values
            if units in ['sr', 'steradian', 'steradians']:
                area = area * (R ** 2)
            else:
                # Heuristic: if values are <= 4*pi, likely steradians
                if np.nanmax(area_vals) <= 4.1 * np.pi:
                    area = area * (R ** 2)
            # Ensure numpy float
            area = area.astype('float64')
        except Exception as e:
            print(f"Warning: could not use existing area variable '{area_var_name}': {e}")
            area = None

    if area is None:
        # Compute from lat/lon
        lat2d, lon2d = to_2d_lat_lon(ds, lat_name, lon_name, spatial_dims)
        # If both are 1D initially, we can build edges easily
        # Check if original lat/lon are 1D aligned with each spatial dim
        lat = ds[lat_name]
        lon = ds[lon_name]
        try:
            if lat.ndim == 1 and lon.ndim == 1:
                # Compute edges
                lat_rad = np.deg2rad(lat.values)
                lon_rad = np.deg2rad(lon.values)

                # Edges arrays
                lat_edges = np.zeros(lat_rad.size + 1, dtype='float64')
                lon_edges = np.zeros(lon_rad.size + 1, dtype='float64')

                lat_edges[1:-1] = 0.5 * (lat_rad[1:] + lat_rad[:-1])
                lon_edges[1:-1] = 0.5 * (lon_rad[1:] + lon_rad[:-1])
                # Extrapolate boundaries
                lat_edges[0] = lat_rad[0] - (lat_rad[1] - lat_rad[0]) / 2.0
                lat_edges[-1] = lat_rad[-1] + (lat_rad[-1] - lat_rad[-2]) / 2.0
                # Limit to [-pi/2, pi/2]
                lat_edges[0] = max(lat_edges[0], -0.5 * np.pi)
                lat_edges[-1] = min(lat_edges[-1], 0.5 * np.pi)

                lon_wrap = 2.0 * np.pi
                lon_step0 = lon_rad[1] - lon_rad[0] if lon_rad.size > 1 else 2*np.pi/360.0
                lon_edges[0] = lon_rad[0] - lon_step0 / 2.0
                lon_edges[-1] = lon_rad[-1] + lon_step0 / 2.0

                dlat = np.abs(np.diff(lat_edges))  # size nlat
                dlon = np.abs(np.diff(lon_edges))  # size nlon

                # Broadcast to 2D
                dlat2d, dlon2d = np.meshgrid(dlat, dlon, indexing='ij')
                latc2d, _ = np.meshgrid(lat_rad, lon_rad, indexing='ij')

                area_vals = (R ** 2) * dlat2d * dlon2d * np.cos(latc2d)
                area = xr.DataArray(area_vals, dims=spatial_dims)
            else:
                # Fallback: approximate using 2D lat, estimate dlat, dlon from gradients
                latc = lat2d.values
                lonc = lon2d.values
                # Compute spacings by differences along each dimension
                # This is a rough approximation
                # Get indices mapping
                dim_y, dim_x = spatial_dims
                # Compute dlat along y by taking diff and pad
                dlat_y = np.zeros_like(latc)
                dlon_x = np.zeros_like(lonc)
                # Using np.gradient to estimate spacing (in radians)
                grad_lat_y = np.gradient(np.deg2rad(latc), axis=0)
                grad_lon_x = np.gradient(np.deg2rad(lonc), axis=1)
                dlat_y = np.abs(grad_lat_y)
                dlon_x = np.abs(grad_lon_x)
                # Use cos(lat) weighting
                area_vals = (R ** 2) * dlat_y * dlon_x * np.cos(np.deg2rad(latc))
                area = xr.DataArray(area_vals, dims=spatial_dims)
        except Exception as e:
            raise RuntimeError(f"Failed to compute grid cell areas from lat/lon: {e}")

    return area

def weighted_mean_and_std(field, weights):
    # Both are DataArray aligned on same dims
    w = weights.where(np.isfinite(field))
    f = field.where(np.isfinite(field))
    wsum = (w).sum()
    if np.asarray(wsum.values).size == 0 or np.isnan(wsum.values):
        return np.nan, np.nan
    mean = (f * w).sum() / wsum
    var = ((f - mean) ** 2 * w).sum() / wsum
    std = np.sqrt(var)
    return mean.item() if hasattr(mean, 'item') else float(mean.values), \
           std.item() if hasattr(std, 'item') else float(std.values)

def main():
    # Configuration
    case_name = "sample.v3.LR.historical"
    input_dir = "./data/sample/e3sm/lnd"
    # Output directory as specified by the user
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_02_seasonal_runoff/run1_output"
    ensure_dir(output_dir)

    start_year = 1985
    end_year = 1989
    var_name_candidates = ['QRUNOFF', 'qrunoff', 'Runoff', 'RUNOFF']

    # Gather files
    files = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            fname = f"{case_name}.elm.h0.{y:04d}-{m:02d}.nc"
            fpath = os.path.join(input_dir, fname)
            if os.path.exists(fpath):
                files.append(fpath)
    if len(files) == 0:
        print("No input files found for the requested period.", file=sys.stderr)
        sys.exit(1)

    # Open dataset
    try:
        ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True, use_cftime=True)
    except Exception as e:
        print(f"Error opening dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Find variable
    qrunoff_name = find_variable(ds, var_name_candidates)
    if qrunoff_name is None:
        print(f"QRUNOFF variable not found. Tried names: {var_name_candidates}", file=sys.stderr)
        sys.exit(1)

    qrunoff = ds[qrunoff_name]
    # Subset time
    try:
        qsel = qrunoff.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    except Exception as e:
        print(f"Error subsetting time: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute time-mean climatology
    try:
        qmean = qsel.mean(dim='time', skipna=True)
    except Exception as e:
        print(f"Error computing time-mean: {e}", file=sys.stderr)
        sys.exit(1)

    # Identify spatial dims
    spatial_dims = [d for d in qmean.dims if d != 'time']
    if len(spatial_dims) != 2:
        print(f"Unexpected number of spatial dims for QRUNOFF: {spatial_dims}", file=sys.stderr)
        # Try to squeeze singleton dims
        for d in list(spatial_dims):
            if qmean.sizes.get(d, 1) == 1:
                qmean = qmean.squeeze(d)
                spatial_dims = [dd for dd in qmean.dims if dd != 'time']
        if len(spatial_dims) != 2:
            sys.exit(1)

    # Detect lat/lon names
    lat_name, lon_name = detect_lat_lon(ds)
    if lat_name is None or lon_name is None:
        print("Latitude/longitude coordinates not found in dataset.", file=sys.stderr)
        sys.exit(1)

    # Compute grid cell areas
    try:
        area = compute_cell_area_m2(ds, spatial_dims, lat_name, lon_name)
    except Exception as e:
        print(f"Error computing grid cell areas: {e}", file=sys.stderr)
        sys.exit(1)

    # Land fraction weighting if available
    landfrac_name = find_variable(ds, ['landfrac', 'LANDFRAC'])
    if landfrac_name:
        try:
            landfrac = ds[landfrac_name]
            if 'time' in landfrac.dims:
                landfrac = landfrac.isel(time=0, drop=True)
            if tuple(landfrac.dims) != tuple(spatial_dims):
                landfrac = landfrac.transpose(*spatial_dims)
            area = area * landfrac
        except Exception as e:
            print(f"Warning: could not apply land fraction '{landfrac_name}': {e}")

    # Mask weights where qmean is nan
    weights = area.where(np.isfinite(qmean))

    # Compute area-weighted global mean and std
    global_mean_mms, global_std_mms = weighted_mean_and_std(qmean, weights)

    # Convert to mm/day for reporting
    sec_per_day = 86400.0
    global_mean_mmd = global_mean_mms * sec_per_day if np.isfinite(global_mean_mms) else np.nan
    global_std_mmd = global_std_mms * sec_per_day if np.isfinite(global_std_mms) else np.nan

    # Prepare 2D lat/lon for plotting
    lat2d, lon2d = to_2d_lat_lon(ds, lat_name, lon_name, spatial_dims)

    # Plot map
    plot_path = os.path.join(output_dir, f"runoff_climatology_{start_year}_{end_year}_map.png")
    try:
        plt.figure(figsize=(12, 6), dpi=150)
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3)

        # Convert data to mm/day for plotting
        qplot = (qmean * sec_per_day).where(np.isfinite(qmean))
        # Ensure data aligns with lat2d/lon2d dims
        qplot = qplot.transpose(*spatial_dims)

        # Plot with pcolormesh; need numeric arrays
        pcm = ax.pcolormesh(lon2d.values, lat2d.values, qplot.values,
                            transform=ccrs.PlateCarree(),
                            cmap='viridis',
                            shading='auto')
        cb = plt.colorbar(pcm, orientation='horizontal', pad=0.05, shrink=0.9)
        cb.set_label('Runoff climatology (QRUNOFF) [mm/day]')

        title = f"ELM QRUNOFF climatology {start_year}-{end_year}"
        plt.title(title, fontsize=13)

        # Overlay statistics
        txt = (f"Global area-weighted mean: {global_mean_mmd:.3f} mm/day\n"
               f"Global area-weighted std:  {global_std_mmd:.3f} mm/day\n"
               f"Total land area weighted (10^6 km^2): {float((weights.sum() / 1e12).values):.2f}")
        plt.annotate(txt, xy=(0.02, 0.02), xycoords='axes fraction',
                     fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved map to {plot_path}")
    except Exception as e:
        print(f"Error generating/saving plot: {e}", file=sys.stderr)

    # Save NetCDF of mean field
    nc_path = os.path.join(output_dir, f"runoff_climatology_{start_year}_{end_year}.nc")
    try:
        ds_out = xr.Dataset()
        ds_out['QRUNOFF_mean'] = qmean
        ds_out['QRUNOFF_mean'].attrs['long_name'] = 'Climatological mean total runoff'
        ds_out['QRUNOFF_mean'].attrs['units'] = 'mm s-1'
        # Also store mm/day convenience variable
        ds_out['QRUNOFF_mean_mm_day'] = qmean * sec_per_day
        ds_out['QRUNOFF_mean_mm_day'].attrs['long_name'] = 'Climatological mean total runoff'
        ds_out['QRUNOFF_mean_mm_day'].attrs['units'] = 'mm day-1'
        # Attach coordinates
        ds_out[lat_name] = ds[lat_name]
        ds_out[lon_name] = ds[lon_name]
        # Global attributes
        ds_out.attrs['source_case'] = case_name
        ds_out.attrs['period_start'] = str(start_year)
        ds_out.attrs['period_end'] = str(end_year)
        ds_out.to_netcdf(nc_path)
        print(f"Saved NetCDF to {nc_path}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}", file=sys.stderr)

    # Save CSV with global stats
    csv_path = os.path.join(output_dir, f"runoff_climatology_{start_year}_{end_year}_global_stats.csv")
    try:
        data = {
            'period_start': [start_year],
            'period_end': [end_year],
            'global_mean_mm_per_s': [global_mean_mms],
            'global_mean_mm_per_day': [global_mean_mmd],
            'global_std_mm_per_s': [global_std_mms],
            'global_std_mm_per_day': [global_std_mmd],
            'total_weight_m2': [float(weights.sum().values)]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)

    # Also save the mean field as a CSV summary (optional)
    try:
        # Compute global median for extra info
        global_median_mmd = float(np.nanmedian((qmean * sec_per_day).values))
        extra_csv_path = os.path.join(output_dir, f"runoff_climatology_{start_year}_{end_year}_summary.csv")
        pd.DataFrame({
            'metric': ['global_mean_mm_per_day', 'global_std_mm_per_day', 'global_median_mm_per_day'],
            'value': [global_mean_mmd, global_std_mmd, global_median_mmd]
        }).to_csv(extra_csv_path, index=False)
        print(f"Saved summary CSV to {extra_csv_path}")
    except Exception as e:
        print(f"Warning: could not save summary CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Suppress some numpy warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()