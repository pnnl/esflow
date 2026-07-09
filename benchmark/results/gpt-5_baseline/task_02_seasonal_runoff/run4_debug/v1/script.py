#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def find_coord_name(obj, key):
    """Find coordinate name for 'lat' or 'lon' in a Dataset or DataArray."""
    names = []
    if hasattr(obj, "coords"):
        names += list(obj.coords)
    if hasattr(obj, "variables"):
        names += list(obj.variables)
    if hasattr(obj, "dims"):
        names += list(obj.dims)
    # Prefer exact matches first
    for name in names:
        if name.lower() == key:
            return name
    for name in names:
        if key in name.lower():
            return name
    return None

def find_area_var(ds, target_da):
    """Try to find an area variable matching target_da's spatial dims."""
    candidate_names = [name for name in ds.variables if 'area' in name.lower()]
    tdims = set(target_da.dims)
    for name in candidate_names:
        var = ds[name]
        # Check that var dims are subset of target dims and not empty
        if set(var.dims).issubset(tdims) and var.size > 0:
            return var
    # Try common alternatives
    for name in ds.variables:
        if name.lower() in ["cell_area", "areacella", "gridcellarea", "area"]:
            var = ds[name]
            if set(var.dims).issubset(tdims) and var.size > 0:
                return var
    return None

def adjust_longitudes(da, lon_name):
    """Shift longitudes to [-180, 180); sort only if 1D lon."""
    lon = da[lon_name]
    lon_new = (((lon + 180) % 360) - 180)
    da2 = da.assign_coords({lon_name: lon_new})
    # Sort by longitude only if lon is 1D
    if lon.ndim == 1:
        try:
            da2 = da2.sortby(lon_name)
        except Exception:
            pass
    return da2

def compute_coslat_weights(da, lat_name):
    """Compute cosine latitude weights broadcast to da."""
    lat = da[lat_name]
    lat_rad = xr.apply_ufunc(np.deg2rad, lat)
    w = xr.apply_ufunc(np.cos, lat_rad)
    # Broadcast to data array shape by simple addition with zeros-like
    return xr.zeros_like(da) + w

def area_weighted_mean(da, weights, spatial_dims):
    """Compute weighted mean over spatial dims of da."""
    w = weights.where(da.notnull())
    num = (da * w).sum(dim=spatial_dims, skipna=True)
    den = w.sum(dim=spatial_dims, skipna=True)
    return num / den

def main():
    # Configuration
    base_dir = "./data/sample/e3sm"
    case_name = "sample.v3.LR.historical"
    lnd_dir = os.path.join(base_dir, "lnd")
    start_year = 1985
    end_year = 1989
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_02_seasonal_runoff/run4_debug/v1/output"
    os.makedirs(output_dir, exist_ok=True)

    # Collect monthly files
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 12 + 1):
            path = os.path.join(lnd_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
            if os.path.exists(path):
                files.append(path)
            else:
                print(f"Warning: Missing file {path}")
    if len(files) == 0:
        print("No input files found; exiting.")
        return

    # Open dataset
    try:
        ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    # Ensure variable exists
    var_name = "QRUNOFF"
    if var_name not in ds:
        print(f"Variable {var_name} not found in dataset variables: {list(ds.data_vars)}")
        try:
            ds.close()
        except Exception:
            pass
        return

    # Extract and time-slice
    try:
        da = ds[var_name].sel(time=slice(start_str, end_str))
    except Exception as e:
        print(f"Error selecting time slice {start_str} to {end_str}: {e}")
        try:
            ds.close()
        except Exception:
            pass
        return

    # Count months available
    n_months = int(da.sizes.get("time", 0))
    if n_months == 0:
        print("No data in the specified time range; exiting.")
        try:
            ds.close()
        except Exception:
            pass
        return

    # Compute climatological mean (time mean)
    qrunoff_mean = da.mean(dim="time", skipna=True)

    # Detect coordinates
    lat_name = find_coord_name(qrunoff_mean, "lat")
    lon_name = find_coord_name(qrunoff_mean, "lon")
    if (lat_name is None) or (lon_name is None):
        print(f"Could not detect latitude/longitude coordinates. Found lat: {lat_name}, lon: {lon_name}")
        try:
            ds.close()
        except Exception:
            pass
        return

    # Adjust longitude range for consistent plotting
    qrunoff_mean = adjust_longitudes(qrunoff_mean, lon_name)

    # Convert units from mm/s to mm/day (per problem statement)
    qrunoff_mean_mmd = qrunoff_mean * 86400.0
    qrunoff_mean_mmd = qrunoff_mean_mmd.assign_attrs({
        "units": "mm/day",
        "long_name": "Climatological mean total runoff (1985-1989)"
    })

    # Determine area weights
    area_var = find_area_var(ds, qrunoff_mean)
    if area_var is not None:
        try:
            area = area_var.reindex_like(qrunoff_mean, method=None)
        except Exception:
            area = area_var.broadcast_like(qrunoff_mean)
        weights = area
        weights_name = area_var.name
    else:
        # Fallback to cosine latitude
        weights = compute_coslat_weights(qrunoff_mean, lat_name)
        weights_name = f"cos({lat_name})"

    # Spatial dimensions (all except time if present)
    spatial_dims = [d for d in qrunoff_mean_mmd.dims if d != "time"]

    # Masked weights
    w_masked = weights.where(qrunoff_mean_mmd.notnull())
    # Compute global mean
    global_mean = area_weighted_mean(qrunoff_mean_mmd, w_masked, spatial_dims)
    try:
        global_mean_value = float(global_mean.values)
    except Exception:
        global_mean_value = np.nan

    # Save climatological field to NetCDF
    nc_out = os.path.join(output_dir, "ELM_QRUNOFF_climatology_1985-1989_mm_per_day.nc")
    try:
        comp = dict(zlib=True, complevel=4)
        encoding = {qrunoff_mean_mmd.name: comp}
        qrunoff_mean_mmd.to_netcdf(nc_out, encoding=encoding)
        print(f"Saved climatology NetCDF to {nc_out}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

    # Save global statistics to CSV
    stats = {
        "time_start": start_str,
        "time_end": end_str,
        "n_months": n_months,
        "area_weighting": weights_name,
        "global_mean_mm_day": global_mean_value
    }
    csv_out = os.path.join(output_dir, "global_runoff_statistics_1985-1989.csv")
    try:
        pd.DataFrame([stats]).to_csv(csv_out, index=False)
        print(f"Saved statistics CSV to {csv_out}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Plot map with statistics
    try:
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.5)

        lon = qrunoff_mean_mmd[lon_name]
        lat = qrunoff_mean_mmd[lat_name]
        data = qrunoff_mean_mmd

        # Build 2D lon/lat arrays for pcolormesh
        if (lon.ndim == 1) and (lat.ndim == 1):
            lon2d, lat2d = np.meshgrid(lon.values, lat.values)
        else:
            lon2d, lat2d = lon.values, lat.values

        valid = data.values[np.isfinite(data.values)]
        if valid.size > 0:
            vmin = max(0.0, float(np.nanpercentile(valid, 2)))
            vmax = float(np.nanpercentile(valid, 98))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        pcm = ax.pcolormesh(lon2d, lat2d, data.values, transform=ccrs.PlateCarree(),
                            cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(pcm, orientation='horizontal', pad=0.05, aspect=40)
        cb.set_label("Runoff (mm/day)")

        title = "ELM QRUNOFF climatological mean (1985-1989)"
        plt.title(title, fontsize=12)

        # Overlay stats text
        stats_text = (
            f"Period: {start_year}-01 to {end_year}-12  (months: {n_months})\n"
            f"Global area-weighted mean: {global_mean_value:.3f} mm/day"
        )
        plt.gcf().text(0.01, 0.02, stats_text, fontsize=10, va='bottom', ha='left',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        fig_out = os.path.join(output_dir, "ELM_QRUNOFF_climatology_1985-1989_map.png")
        plt.tight_layout()
        plt.savefig(fig_out, dpi=150)
        plt.close()
        print(f"Saved map figure to {fig_out}")
    except Exception as e:
        print(f"Error during plotting: {e}")

    # Close dataset
    try:
        ds.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()