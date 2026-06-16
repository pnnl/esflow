#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from urllib.request import urlretrieve

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)

def download_file(url, filepath):
    try:
        if not os.path.exists(filepath):
            print(f"Downloading {url} to {filepath}")
            urlretrieve(url, filepath)
        else:
            print(f"File already exists: {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False

def standardize_longitude(da, lon_name='lon'):
    """Convert longitudes to [-180,180) and sort."""
    if lon_name not in da.coords:
        # Try to find lon coordinate among data variables or coordinates
        raise ValueError(f"Longitude coordinate '{lon_name}' not found in dataset/dataarray.")
    lon = da[lon_name].values
    # Handle 1D longitude coordinate
    if np.ndim(lon) == 1:
        lon2 = lon.copy()
        if lon2.max() > 180:
            lon2 = ((lon2 + 180) % 360) - 180
        da = da.assign_coords({lon_name: lon2})
        try:
            da = da.sortby(lon_name)
        except Exception:
            # In case sortby fails due to non-monotonic coords
            order = np.argsort(da[lon_name].values)
            da = da.isel({lon_name: order})
        return da
    else:
        # If 2D lon, convert values but cannot sort easily; we will handle in interpolation step
        lon2 = lon.copy()
        mask = lon2 > 180
        lon2[mask] = lon2[mask] - 360.0
        da = da.assign_coords({lon_name: (da[lon_name].dims, lon2)})
        return da

def convert_et_units_to_mm_per_day(da, units_attr=None):
    """Convert ET units to mm/day. Supports mm/s, kg m-2 s-1, mm/day, kg m-2 day-1."""
    units = units_attr if units_attr is not None else da.attrs.get('units', '').lower()
    # Handle some common forms
    if units is None:
        units = ''
    units = units.replace(' ', '')
    out = da.copy()
    if ('s-1' in units) or ('/s' in units):
        out = out * 86400.0
        out.attrs['units'] = 'mm/day'
    elif ('mm/day' in units) or ('mm/d' in units) or ('mmd-1' in units) or ('mmperday' in units):
        out.attrs['units'] = 'mm/day'
    elif ('kgm-2s-1' in units) or ('kg/m2/s' in units) or ('kgm-2/s' in units):
        out = out * 86400.0
        out.attrs['units'] = 'mm/day'
    elif ('kgm-2day-1' in units) or ('kg/m2/day' in units) or ('kgm-2/d' in units):
        out.attrs['units'] = 'mm/day'
    else:
        # Unknown units; assume current values are mm/day
        out.attrs['units'] = 'mm/day'
    return out

def weighted_time_mean(da):
    """Compute time mean weighted by days in month if possible."""
    if 'time' not in da.dims:
        return da
    try:
        days = xr.DataArray(da['time'].dt.days_in_month, dims=('time',))
        w = days / days.sum()
        # expand weights for broadcasting
        while w.ndim < da.ndim:
            w = w.broadcast_like(da).isel({dim: 0 for dim in da.dims if dim != 'time'})  # ensure alignment
        # Alternatively, use explicit weighted
        wt = days
        res = (da * wt).sum(dim='time') / wt.sum(dim='time')
        return res
    except Exception:
        # Fallback to simple mean
        return da.mean(dim='time')

def meshgrid_from_latlon(lat, lon):
    if lat.ndim == 1 and lon.ndim == 1:
        lons2d, lats2d = np.meshgrid(lon.values, lat.values)
        return lats2d, lons2d
    return lat.values, lon.values

def regrid_to_obs_grid(model_da, obs_lat, obs_lon, method='linear'):
    """Regrid model_et (lat,lon) to obs grid defined by obs_lat (1D) and obs_lon (1D)."""
    # Ensure longitudes standardized
    model_da = standardize_longitude(model_da, lon_name='lon')
    # Ensure model lat/lon names
    if 'lat' not in model_da.dims or 'lon' not in model_da.dims:
        raise ValueError("Model DataArray must have 'lat' and 'lon' dimensions.")
    mlat = model_da['lat']
    mlon = model_da['lon']
    # If 1D lat/lon, try xarray interp
    if mlat.ndim == 1 and mlon.ndim == 1:
        try:
            regridded = model_da.interp(lat=obs_lat, lon=obs_lon, method=method)
            return regridded
        except Exception as e:
            print(f"xarray interp failed, falling back to griddata: {e}", file=sys.stderr)
    # Use griddata
    MLAT, MLON = meshgrid_from_latlon(mlat, mlon)
    target_lon2d, target_lat2d = np.meshgrid(obs_lon.values, obs_lat.values)
    pts = np.column_stack((MLON.ravel(), MLAT.ravel()))
    vals = model_da.values.ravel()
    mask = np.isfinite(vals) & np.isfinite(pts[:,0]) & np.isfinite(pts[:,1])
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough valid points in model data for interpolation.")
    vals_interp = griddata(pts[mask], vals[mask], (target_lon2d, target_lat2d), method=method)
    # Fill remaining NaNs with nearest
    if np.isnan(vals_interp).any():
        try:
            vals_nearest = griddata(pts[mask], vals[mask], (target_lon2d, target_lat2d), method='nearest')
            fillmask = np.isnan(vals_interp)
            vals_interp[fillmask] = vals_nearest[fillmask]
        except Exception as e:
            print(f"Nearest fill failed: {e}", file=sys.stderr)
    out = xr.DataArray(vals_interp, coords={'lat': obs_lat.values, 'lon': obs_lon.values}, dims=('lat','lon'))
    out.attrs.update(model_da.attrs)
    return out

def compute_area_weights(lat_1d, lon_1d):
    """Compute cosine latitude weights normalized to mean 1; returned shape (lat,lon)."""
    lat_rad = np.deg2rad(lat_1d.values if isinstance(lat_1d, xr.DataArray) else lat_1d)
    w = np.cos(lat_rad)
    w2d = w[:, None] * np.ones((len(w), len(lon_1d)))
    # Normalize weights so that mean weight over valid area ~1
    w2d = w2d / np.nanmean(w2d)
    return w2d

def main():
    # Paths and settings
    case_name = "sample.v3.LR.historical"
    elm_dir = "./data/sample/e3sm/lnd"
    years = list(range(1985, 1990))
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_03_et_benchmark/run1_output"
    ensure_dir(output_dir)

    # 1) Fetch MODIS ET dataset
    ilamb_var = "evspsbl"
    ilamb_dataset = "MODIS"
    ilamb_filename = "et_0.5x0.5.nc"
    ilamb_url = f"https://www.ilamb.org/ILAMB-Data/DATA/{ilamb_var}/{ilamb_dataset}/{ilamb_filename}"
    obs_file = os.path.join(output_dir, "MODIS_et_0.5x0.5.nc")
    ok = download_file(ilamb_url, obs_file)
    if not ok:
        print("Failed to download observation data; exiting.", file=sys.stderr)
        return

    # 2) Load MODIS ET and compute time mean (mm/day)
    try:
        ds_obs = xr.open_dataset(obs_file)
        if 'et' not in ds_obs:
            raise KeyError("Variable 'et' not found in MODIS dataset.")
        # Standardize lons
        if 'lon' in ds_obs.coords:
            ds_obs = standardize_longitude(ds_obs, lon_name='lon')
        else:
            raise KeyError("Observation dataset missing 'lon' coordinate.")
        obs_et = ds_obs['et']
        obs_et = convert_et_units_to_mm_per_day(obs_et)
        if 'time' in obs_et.dims:
            obs_et_mean = weighted_time_mean(obs_et)
        else:
            obs_et_mean = obs_et
        # Ensure lat/lon ascending for plotting
        if 'lat' in obs_et_mean.coords:
            try:
                obs_et_mean = obs_et_mean.sortby('lat')
            except Exception:
                pass
        obs_lat = obs_et_mean['lat']
        obs_lon = obs_et_mean['lon']
    except Exception as e:
        print(f"Error processing MODIS ET: {e}", file=sys.stderr)
        return

    # 3) Load ELM ET for 1985-1989 and compute time-mean climatology (mm/day)
    try:
        file_list = []
        for y in years:
            pattern = os.path.join(elm_dir, f"{case_name}.elm.h0.{y}-*.nc")
            files = sorted(glob.glob(pattern))
            file_list.extend(files)
        if len(file_list) == 0:
            print(f"No ELM files found for years {years} with pattern {pattern}", file=sys.stderr)
            return
        vars_to_get = ['QVEGE', 'QVEGT', 'QSOIL', 'lat', 'lon', 'time']
        ds_elm = xr.open_mfdataset(file_list, combine='by_coords', decode_times=True, use_cftime=True)
        for v in ['QVEGE', 'QVEGT', 'QSOIL']:
            if v not in ds_elm:
                raise KeyError(f"Variable {v} not found in ELM dataset.")
        # Restrict to years in case of extra
        if 'time' in ds_elm.dims:
            tsel = ds_elm.sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
            ds_elm = tsel
        elm_et_mms = ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']  # mm/s per instructions
        elm_et_mmd = elm_et_mms * 86400.0
        elm_et_mmd.attrs['units'] = 'mm/day'
        elm_et_mean = weighted_time_mean(elm_et_mmd)
        # Standardize longitudes
        elm_et_mean = standardize_longitude(elm_et_mean, lon_name='lon')
        # Ensure lat/lon ascending
        try:
            elm_et_mean = elm_et_mean.sortby('lat')
            elm_et_mean = elm_et_mean.sortby('lon')
        except Exception:
            pass
    except Exception as e:
        print(f"Error processing ELM ET: {e}", file=sys.stderr)
        return

    # 4) Regrid ELM ET mean to MODIS grid
    try:
        elm_on_obs = regrid_to_obs_grid(elm_et_mean, obs_lat, obs_lon, method='linear')
    except Exception as e:
        print(f"Error regridding ELM ET to MODIS grid: {e}", file=sys.stderr)
        return

    # 5) Compute bias (model - obs), global mean bias, and spatial correlation
    try:
        bias = elm_on_obs - obs_et_mean
        bias.name = 'bias'
        bias.attrs['long_name'] = 'ELM minus MODIS ET'
        bias.attrs['units'] = 'mm/day'
        # Mask to valid points
        valid = np.isfinite(bias.values) & np.isfinite(obs_et_mean.values) & np.isfinite(elm_on_obs.values)
        if valid.sum() == 0:
            print("No overlapping valid grid cells between model and observations.", file=sys.stderr)
            return
        # Area weights using cosine latitude
        weights = compute_area_weights(obs_lat, obs_lon)
        w_mask = np.where(valid, weights, np.nan)
        global_mean_bias = np.nansum(bias.values * w_mask) / np.nansum(w_mask)
        # Spatial correlation (unweighted Pearson)
        x = elm_on_obs.values[valid].ravel()
        y = obs_et_mean.values[valid].ravel()
        if len(x) < 2:
            spatial_corr = np.nan
            pval = np.nan
        else:
            spatial_corr, pval = pearsonr(x, y)
    except Exception as e:
        print(f"Error computing bias and statistics: {e}", file=sys.stderr)
        return

    # 6) Save datasets and metrics
    try:
        ds_out = xr.Dataset(
            {
                'obs_et_mean': obs_et_mean,
                'elm_et_mean_on_obs_grid': elm_on_obs,
                'bias': bias
            }
        )
        ds_out['obs_et_mean'].attrs['description'] = 'MODIS ET time-mean'
        ds_out['elm_et_mean_on_obs_grid'].attrs['description'] = 'ELM ET climatology (1985-1989) interpolated to MODIS grid'
        ds_out['bias'].attrs['description'] = 'ELM minus MODIS ET (mm/day)'
        nc_out = os.path.join(output_dir, "et_benchmark_fields.nc")
        ds_out.to_netcdf(nc_out)
        print(f"Saved fields to {nc_out}")
    except Exception as e:
        print(f"Error saving NetCDF output: {e}", file=sys.stderr)

    try:
        metrics = pd.DataFrame({
            'metric': ['global_mean_bias_mm_per_day', 'spatial_correlation', 'pearson_p_value', 'n_valid_points'],
            'value': [global_mean_bias, spatial_corr, pval, int(valid.sum())]
        })
        csv_out = os.path.join(output_dir, "metrics.csv")
        metrics.to_csv(csv_out, index=False)
        print(f"Saved metrics to {csv_out}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}", file=sys.stderr)

    # 7) Plot bias map
    try:
        fig = plt.figure(figsize=(12,6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.5)
        # Determine symmetric color limits based on robust percentiles
        abs_bias = np.abs(bias.values[np.isfinite(bias.values)])
        if abs_bias.size > 0:
            vmax = np.nanpercentile(abs_bias, 95)
        else:
            vmax = 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im = ax.pcolormesh(obs_lon.values, obs_lat.values, bias.values, transform=ccrs.PlateCarree(),
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        cb = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.8)
        cb.set_label('Bias (ELM - MODIS) [mm/day]')
        title = f"ELM ET Bias vs MODIS (1985-1989 climatology)\nGlobal mean bias={global_mean_bias:.3f} mm/day; Spatial corr={spatial_corr:.3f}"
        plt.title(title)
        png_out = os.path.join(output_dir, "bias_map.png")
        plt.savefig(png_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved bias map to {png_out}")
    except Exception as e:
        print(f"Error generating bias map: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()