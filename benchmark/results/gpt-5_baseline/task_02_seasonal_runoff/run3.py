import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def find_files(data_dir, case_name, start_year=1985, end_year=1989):
    pattern = os.path.join(data_dir, f"{case_name}.elm.h0.*.nc")
    files = sorted(glob.glob(pattern))
    sel_files = []
    for f in files:
        try:
            base = os.path.basename(f)
            parts = base.split(".")
            # Expect ...elm.h0.YYYY-MM.nc
            date_part = parts[-2]  # YYYY-MM
            year = int(date_part.split("-")[0])
            if start_year <= year <= end_year:
                sel_files.append(f)
        except Exception:
            continue
    return sel_files

def get_lat_lon(ds, var_da):
    lat = None
    lon = None
    # Prefer coordinates directly attached to the DataArray
    if 'lat' in var_da.coords:
        lat = var_da.coords['lat']
    elif 'lat' in ds:
        lat = ds['lat']
    if 'lon' in var_da.coords:
        lon = var_da.coords['lon']
    elif 'lon' in ds:
        lon = ds['lon']
    return lat, lon

def find_area_variable(ds):
    # Try common names
    for cand in ['area', 'AREA', 'areacella', 'areacell', 'cell_area', 'gridcell_area', 'areaCell']:
        if cand in ds.variables:
            return ds[cand]
    # Some datasets store area as coordinate/aux variable
    for cand in ['area', 'AREA', 'areacella', 'areacell', 'cell_area', 'gridcell_area', 'areaCell']:
        if cand in ds.coords:
            return ds[cand]
    return None

def compute_cell_areas_from_1d(lat, lon):
    # lat, lon are 1D in degrees
    R = 6371000.0
    lat_vals = np.asarray(lat.values, dtype=float)
    lon_vals = np.asarray(lon.values, dtype=float)

    # Compute edges for lat
    lat_edges = np.zeros(lat_vals.size + 1, dtype=float)
    lat_edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
    lat_edges[0] = lat_vals[0] - 0.5 * (lat_vals[1] - lat_vals[0])
    lat_edges[-1] = lat_vals[-1] + 0.5 * (lat_vals[-1] - lat_vals[-2]) if lat_vals.size > 1 else lat_vals[-1] + 0.5
    # Clip to valid bounds
    lat_edges = np.clip(lat_edges, -90.0, 90.0)

    # Compute edges for lon (assume periodic)
    lon_edges = np.zeros(lon_vals.size + 1, dtype=float)
    lon_edges[1:-1] = 0.5 * (lon_vals[:-1] + lon_vals[1:])
    dlon = (lon_vals[1] - lon_vals[0]) if lon_vals.size > 1 else 1.0
    lon_edges[0] = lon_vals[0] - 0.5 * dlon
    lon_edges[-1] = lon_vals[-1] + 0.5 * dlon

    # Convert to radians
    lat_edges_rad = np.deg2rad(lat_edges)
    lon_edges_rad = np.deg2rad(lon_edges)

    # Area per lat band (integral of sin(phi) dphi)
    dphi = np.diff(lat_edges_rad)  # length nlat
    sin_phi = np.sin(lat_edges_rad)
    band_factor = (sin_phi[1:] - sin_phi[:-1])  # length nlat

    # Area per lon segment
    dlambda = np.diff(lon_edges_rad)  # length nlon

    # 2D area = R^2 * band_factor[:,None] * dlambda[None,:]
    area = (R ** 2) * np.outer(band_factor, dlambda)
    return area  # shape (nlat, nlon)

def build_weights(ds, var_da):
    # Determine spatial dims (exclude 'time' if present)
    sp_dims = [d for d in var_da.dims if d != 'time']

    # Try to find a proper area variable
    area_da = find_area_variable(ds)
    if area_da is not None:
        try:
            area_b = area_da.broadcast_like(var_da.isel(time=0, drop=True) if 'time' in var_da.dims else var_da)
            weights = xr.DataArray(area_b, dims=sp_dims, coords={d: var_da.coords[d] for d in sp_dims})
            return weights
        except Exception:
            pass

    # Fallback to latitude-based weights
    lat, lon = get_lat_lon(ds, var_da)
    if lat is None or lon is None:
        # As a last resort, use ones
        weights = xr.ones_like(var_da.isel(time=0, drop=True) if 'time' in var_da.dims else var_da)
        return weights

    try:
        # If lat and lon are 1D and match dims
        lat_is_1d = (lat.ndim == 1) and (lat.dims[0] in sp_dims)
        lon_is_1d = (lon.ndim == 1) and (lon.dims[0] in sp_dims)
        if lat_is_1d and lon_is_1d and set(sp_dims) == set([lat.dims[0], lon.dims[0]]):
            area = compute_cell_areas_from_1d(lat, lon)
            # Build DataArray with correct dim order
            # Determine order
            if len(sp_dims) == 2:
                dim_yx = sp_dims
            else:
                dim_yx = [lat.dims[0], lon.dims[0]]
            # Create DataArray
            area_da2 = xr.DataArray(area, dims=(lat.dims[0], lon.dims[0]),
                                    coords={lat.dims[0]: lat, lon.dims[0]: lon})
            # Broadcast to var_da
            area_b = area_da2.broadcast_like(var_da.isel(time=0, drop=True) if 'time' in var_da.dims else var_da)
            return area_b
        else:
            # Use cosine latitude weighting
            lat_b = lat
            if lat.ndim == 0:
                lat_b = xr.zeros_like(var_da.isel(time=0, drop=True)) + float(lat.values)
            elif lat.ndim == 1 and lat.dims[0] in sp_dims and len(sp_dims) == 2:
                other_dim = [d for d in sp_dims if d != lat.dims[0]][0]
                lat_b = lat.broadcast_like(var_da.isel({other_dim: 0}, time=0, drop=True) if 'time' in var_da.dims else var_da.isel({other_dim: 0}, drop=True))
                lat_b = lat_b.broadcast_like(var_da.isel(time=0, drop=True) if 'time' in var_da.dims else var_da)
            elif lat.ndim == 2:
                lat_b = lat
                if set(lat.dims) != set(sp_dims):
                    lat_b = lat_b.transpose(*sp_dims)
            else:
                lat_b = xr.zeros_like(var_da.isel(time=0, drop=True)) + xr.DataArray(lat).values
            weights = np.cos(np.deg2rad(lat_b))
            weights = xr.DataArray(weights, dims=sp_dims, coords={d: var_da.coords[d] for d in sp_dims})
            # Ensure non-negative
            weights = xr.where(weights > 0, weights, 0)
            return weights
    except Exception:
        # Fallback to ones if anything fails
        weights = xr.ones_like(var_da.isel(time=0, drop=True) if 'time' in var_da.dims else var_da)
        return weights

def main():
    # Paths and settings
    data_dir = "./data/sample/e3sm/lnd"
    case_name = "sample.v3.LR.historical"
    start_year = 1985
    end_year = 1989

    # Output directory (user-specified)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_02_seasonal_runoff/run3_output"
    os.makedirs(output_dir, exist_ok=True)

    # Find files
    files = find_files(data_dir, case_name, start_year, end_year)
    if len(files) == 0:
        print("No ELM files found for the specified period. Exiting.")
        return

    # Open dataset
    try:
        ds = xr.open_mfdataset(files, combine='by_coords', decode_times=True, use_cftime=True)
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    varname = 'QRUNOFF'
    if varname not in ds.variables:
        print(f"Variable {varname} not found in dataset. Available variables: {list(ds.variables)}")
        try:
            ds.close()
        except Exception:
            pass
        return

    # Select time range
    try:
        ds_sel = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    except Exception as e:
        print(f"Error selecting time range: {e}")
        ds_sel = ds  # fallback to all

    da = ds_sel[varname]

    # Compute climatological mean over the period
    try:
        da_mean = da.mean(dim='time', skipna=True)
    except Exception as e:
        print(f"Error computing mean over time: {e}")
        try:
            ds.close()
        except Exception:
            pass
        return

    # Convert mm/s to mm/day for interpretation
    da_mean_mmd = da_mean * 86400.0
    da_mean_mmd.attrs['units'] = 'mm/day'
    da_mean_mmd.attrs['description'] = f'Climatological mean runoff ({start_year}-{end_year})'

    # Build area weights and compute global area-weighted mean
    weights = build_weights(ds_sel, da_mean_mmd)
    # Mask weights where data is NaN
    valid_mask = xr.where(xr.ufuncs.isfinite(da_mean_mmd), 1.0, 0.0)
    weights_masked = weights * valid_mask
    # Sum of weights
    wsum = weights_masked.sum(dim=[d for d in da_mean_mmd.dims], skipna=True)
    # Weighted mean
    num = (da_mean_mmd * weights_masked).sum(dim=[d for d in da_mean_mmd.dims], skipna=True)
    try:
        global_mean = (num / wsum).item()
    except Exception:
        global_mean = float(num.values / wsum.values)

    # Compute other statistics (unweighted, across space)
    try:
        global_min = da_mean_mmd.min(skipna=True).item()
        global_max = da_mean_mmd.max(skipna=True).item()
        global_median = da_mean_mmd.median(skipna=True).item()
        global_std = da_mean_mmd.std(skipna=True).item()
    except Exception:
        global_min = float(np.nanmin(da_mean_mmd.values))
        global_max = float(np.nanmax(da_mean_mmd.values))
        global_median = float(np.nanmedian(da_mean_mmd.values))
        global_std = float(np.nanstd(da_mean_mmd.values))

    # Save NetCDF of climatological mean
    nc_out = os.path.join(output_dir, f"ELM_QRUNOFF_clim_mean_{start_year}-{end_year}.nc")
    try:
        ds_out = xr.Dataset(
            {"QRUNOFF_clim_mean": da_mean_mmd},
            attrs={
                "source": "E3SM ELM monthly output",
                "case_name": case_name,
                "period": f"{start_year}-{end_year}",
                "notes": "QRUNOFF climatological mean converted to mm/day"
            }
        )
        ds_out.to_netcdf(nc_out)
        print(f"Saved climatological mean NetCDF to: {nc_out}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

    # Save global statistics to CSV
    stats_csv = os.path.join(output_dir, f"ELM_QRUNOFF_global_stats_{start_year}-{end_year}.csv")
    try:
        stats_dict = {
            "statistic": ["global_mean", "global_median", "global_std", "global_min", "global_max"],
            "value_mm_per_day": [global_mean, global_median, global_std, global_min, global_max]
        }
        pd.DataFrame(stats_dict).to_csv(stats_csv, index=False)
        print(f"Saved global statistics CSV to: {stats_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Plot map of mean runoff field with global stats overlaid
    fig_out = os.path.join(output_dir, f"ELM_QRUNOFF_clim_mean_{start_year}-{end_year}.png")
    try:
        # Identify plotting approach
        lat, lon = get_lat_lon(ds_sel, da_mean_mmd)
        # Create figure
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='gray')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3)

        plotted = False
        if lat is not None and lon is not None:
            if lat.ndim == 1 and lon.ndim == 1 and (lat.dims[0] in da_mean_mmd.dims) and (lon.dims[0] in da_mean_mmd.dims):
                # 1D lat/lon
                lon2d, lat2d = np.meshgrid(lon.values, lat.values)
                im = ax.pcolormesh(lon2d, lat2d, da_mean_mmd.transpose(lat.dims[0], lon.dims[0]).values,
                                   transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
                plotted = True
            elif lat.ndim == 2 and lon.ndim == 2:
                # 2D lat/lon
                # Align dims
                da_plot = da_mean_mmd
                if set(lat.dims) == set(da_plot.dims):
                    da_plot = da_plot.transpose(*lat.dims)
                im = ax.pcolormesh(lon.values, lat.values, da_plot.values,
                                   transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
                plotted = True
            elif lat.ndim == 1 and 'ncol' in da_mean_mmd.dims and lat.dims[0] == 'ncol':
                # Unstructured
                im = ax.scatter(lon.values, lat.values, c=da_mean_mmd.values, s=1,
                                transform=ccrs.PlateCarree(), cmap='viridis')
                plotted = True

        if not plotted:
            # Fallback: assume 2D regular grid with coords named same as dims
            # Try to guess dims
            sp_dims = [d for d in da_mean_mmd.dims]
            if len(sp_dims) == 2:
                ydim, xdim = sp_dims
                # Build fake lon/lat from indices if not present
                yvals = np.arange(da_mean_mmd.sizes[ydim])
                xvals = np.arange(da_mean_mmd.sizes[xdim])
                lon2d, lat2d = np.meshgrid(xvals, yvals)
                im = ax.pcolormesh(lon2d, lat2d, da_mean_mmd.values,
                                   transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
            else:
                # Scatter with zeros
                vals = da_mean_mmd.values.flatten()
                n = vals.size
                lons = np.linspace(-180, 180, n, endpoint=False)
                lats = np.linspace(-90, 90, n, endpoint=True)
                im = ax.scatter(lons, lats, c=vals, s=1, cmap='viridis', transform=ccrs.PlateCarree())

        cb = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.8)
        cb.set_label('Runoff (mm/day)')

        title = f"ELM QRUNOFF Climatological Mean ({start_year}-{end_year})"
        plt.title(title, fontsize=12)

        # Overlay text box with global stats
        stats_text = (f"Global mean: {global_mean:.3f} mm/day\n"
                      f"Median: {global_median:.3f} mm/day\n"
                      f"Std: {global_std:.3f} mm/day\n"
                      f"Min: {global_min:.3f} mm/day\n"
                      f"Max: {global_max:.3f} mm/day")
        plt.gcf().text(0.02, 0.02, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        plt.savefig(fig_out, dpi=200)
        plt.close()
        print(f"Saved figure to: {fig_out}")
    except Exception as e:
        print(f"Error creating/saving plot: {e}")

    # Close dataset
    try:
        ds.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()