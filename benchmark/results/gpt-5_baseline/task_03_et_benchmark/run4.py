import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
import urllib.request
from urllib.error import URLError, HTTPError

warnings.filterwarnings("ignore", category=FutureWarning)

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)

def download_file(url, local_path):
    try:
        if not os.path.exists(local_path):
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, local_path)
            print(f"Saved to: {local_path}")
        else:
            print(f"File exists, skipping download: {local_path}")
    except (URLError, HTTPError) as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}", file=sys.stderr)
        raise

def standardize_lon_to_360(da, lon_name='lon'):
    if lon_name not in da.coords:
        return da
    lon = da[lon_name]
    lon360 = (lon % 360 + 360) % 360
    da = da.assign_coords({lon_name: lon360})
    try:
        da = da.sortby(lon_name)
    except Exception:
        pass
    return da

def get_time_weights_by_days(ds_time):
    # ds_time: xarray DataArray of datetime64
    try:
        days = xr.DataArray(ds_time.dt.days_in_month, dims=['time'])
    except Exception:
        # Fallback: assume 30 days per month
        days = xr.DataArray(np.full(ds_time.shape[0], 30), dims=['time'])
    return days

def time_weighted_mean_mm_per_day(et_mm_per_day):
    # Weighted by days in month to get correct climatology
    if 'time' not in et_mm_per_day.dims:
        return et_mm_per_day
    days = get_time_weights_by_days(et_mm_per_day['time'])
    wsum = (et_mm_per_day * days).sum(dim='time', skipna=True)
    w = days.sum(dim='time')
    clim = wsum / w
    return clim

def area_weighted_mean(data, lat_name='lat'):
    # data: 2D DataArray (lat, lon); weights based on cos(lat)
    if lat_name not in data.coords:
        raise ValueError(f"Latitude coordinate '{lat_name}' not found.")
    lat = data[lat_name]
    # Broadcast weights to data shape
    weights = np.cos(np.deg2rad(lat))
    # Align along lat dimension
    weights_2d = xr.where(np.isfinite(data), 1.0, np.nan) * weights
    num = (data * weights_2d).sum(dim=data.dims, skipna=True)
    den = weights_2d.sum(dim=data.dims, skipna=True)
    return (num / den).item()

def spatial_correlation(a, b):
    # a, b: DataArray with same shape; compute Pearson over finite pairs
    a_flat = a.values.ravel()
    b_flat = b.values.ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 2:
        return np.nan
    r, _ = pearsonr(a_flat[mask], b_flat[mask])
    return float(r)

def main():
    # Output directory (as specified by the user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_03_et_benchmark/run4_output"
    ensure_dir(output_dir)

    # Paths and parameters
    case_name = "sample.v3.LR.historical"
    e3sm_lnd_dir = "./data/sample/e3sm/lnd"
    start_year, end_year = 1985, 1989
    ilamb_base = "https://www.ilamb.org/ILAMB-Data/DATA"
    ilamb_var = "evspsbl"
    ilamb_dataset = "MODIS"
    ilamb_filename = "et_0.5x0.5.nc"
    ilamb_url = f"{ilamb_base}/{ilamb_var}/{ilamb_dataset}/{ilamb_filename}"
    ilamb_local_path = os.path.join(output_dir, "MODIS_et_0.5x0.5.nc")

    # 1) Download MODIS ET
    try:
        download_file(ilamb_url, ilamb_local_path)
    except Exception:
        print("Failed to download MODIS ET; exiting.", file=sys.stderr)
        return

    # 2) Load E3SM ELM data (QVEGE + QVEGT + QSOIL), 1985-1989 monthly files
    file_list = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            f = os.path.join(
                e3sm_lnd_dir,
                f"{case_name}.elm.h0.{y:04d}-{m:02d}.nc"
            )
            if os.path.exists(f):
                file_list.append(f)
    if len(file_list) == 0:
        print("No E3SM ELM files found for 1985-1989. Check data path.", file=sys.stderr)
        return

    try:
        ds = xr.open_mfdataset(
            file_list,
            combine="by_coords",
            decode_times=True,
            mask_and_scale=True
        )
    except Exception as e:
        print(f"Error opening ELM dataset: {e}", file=sys.stderr)
        return

    # Ensure expected variables exist
    for v in ["QVEGE", "QVEGT", "QSOIL"]:
        if v not in ds.variables:
            print(f"Variable {v} not found in ELM dataset.", file=sys.stderr)
            return

    # 3) Compute ET (mm/day) and time-weighted climatology 1985-1989
    try:
        et = ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]  # mm/s
        # Convert to mm/day
        et_mmd = et * 86400.0
        et_clim_mmd = time_weighted_mean_mm_per_day(et_mmd)
        # Keep only lat/lon coordinates for clarity
        # Attempt to identify lat/lon coordinate names
        lat_name = "lat" if "lat" in et_clim_mmd.coords else ("latitude" if "latitude" in et_clim_mmd.coords else None)
        lon_name = "lon" if "lon" in et_clim_mmd.coords else ("longitude" if "longitude" in et_clim_mmd.coords else None)
        if lat_name is None or lon_name is None:
            print("Latitude/Longitude coordinates not found in ELM ET data.", file=sys.stderr)
            return
        # Standardize lon to [0,360)
        et_clim_mmd = et_clim_mmd.rename({lat_name: "lat", lon_name: "lon"})
        et_clim_mmd = standardize_lon_to_360(et_clim_mmd, lon_name="lon")
    except Exception as e:
        print(f"Error computing ELM ET climatology: {e}", file=sys.stderr)
        return
    finally:
        try:
            ds.close()
        except Exception:
            pass

    # 4) Load MODIS ET and compute time-mean if needed
    try:
        ds_obs = xr.open_dataset(ilamb_local_path)
    except Exception as e:
        print(f"Error opening MODIS ET file: {e}", file=sys.stderr)
        return

    obs_var_name = "et" if "et" in ds_obs.variables else None
    if obs_var_name is None:
        print("Variable 'et' not found in MODIS ET file.", file=sys.stderr)
        return

    try:
        et_obs = ds_obs[obs_var_name]
        # Ensure lat/lon exist
        if "lat" not in et_obs.coords or "lon" not in et_obs.coords:
            # Try common alternates
            if "latitude" in et_obs.coords:
                et_obs = et_obs.rename({"latitude": "lat"})
            if "longitude" in et_obs.coords:
                et_obs = et_obs.rename({"longitude": "lon"})
        if "lat" not in et_obs.coords or "lon" not in et_obs.coords:
            print("MODIS ET file missing lat/lon coordinates.", file=sys.stderr)
            return

        # Convert units if necessary
        units = et_obs.attrs.get("units", "").lower()
        # If in kg m-2 s-1, convert to mm/day (1 kg/m2 = 1 mm water)
        if "kg" in units and "s" in units:
            et_obs = et_obs * 86400.0
        # If in m/s, convert to mm/day
        elif units in ["m s-1", "m/s"]:
            et_obs = et_obs * 86400.0 * 1000.0
        # Otherwise assume mm/day

        # Compute time mean if there is a time dimension
        time_like_dims = [d for d in et_obs.dims if d.lower() in ["time", "t", "month", "months"]]
        if len(time_like_dims) > 0:
            # Simple mean across any time-like dims
            for td in time_like_dims:
                et_obs = et_obs.mean(dim=td, skipna=True)

        # Standardize longitude to [0, 360)
        et_obs = standardize_lon_to_360(et_obs, lon_name="lon")
    except Exception as e:
        print(f"Error processing MODIS ET data: {e}", file=sys.stderr)
        return
    finally:
        try:
            ds_obs.close()
        except Exception:
            pass

    # 5) Interpolate model ET to observation grid
    try:
        # Ensure both have dims (lat, lon)
        # If model or obs includes other dims, squeeze them
        et_clim_mmd = et_clim_mmd.squeeze()
        et_obs = et_obs.squeeze()

        # Sort coordinates to be increasing (required for interpolation)
        try:
            et_clim_mmd = et_clim_mmd.sortby('lat').sortby('lon')
        except Exception:
            pass
        try:
            et_obs = et_obs.sortby('lat').sortby('lon')
        except Exception:
            pass

        # Interpolate model ET to obs grid
        model_on_obs = et_clim_mmd.interp(
            lat=et_obs['lat'],
            lon=et_obs['lon'],
            method='linear'
        )
    except Exception as e:
        print(f"Error interpolating model ET to obs grid: {e}", file=sys.stderr)
        return

    # 6) Compute bias (model - obs)
    try:
        bias = model_on_obs - et_obs
    except Exception as e:
        print(f"Error computing bias: {e}", file=sys.stderr)
        return

    # 7) Compute global mean bias (area-weighted) and spatial correlation
    try:
        # Ensure 2D DataArrays
        if set(bias.dims) != set(['lat', 'lon']):
            # Attempt to select lat/lon dims
            bias = bias.squeeze()
            model_on_obs = model_on_obs.squeeze()
            et_obs = et_obs.squeeze()
            # If still extra dims exist, select first index
            for da in [bias, model_on_obs, et_obs]:
                pass

        # Area-weighted mean bias using cosine latitude weights
        gm_bias = area_weighted_mean(bias, lat_name='lat')

        # Spatial correlation across valid grid cells
        corr = spatial_correlation(model_on_obs, et_obs)
    except Exception as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)
        gm_bias, corr = np.nan, np.nan

    # 8) Save results to CSV
    try:
        summary_csv = os.path.join(output_dir, "et_benchmark_summary.csv")
        df = pd.DataFrame({
            "metric": ["global_mean_bias_mm_per_day", "spatial_correlation"],
            "value": [gm_bias, corr]
        })
        df.to_csv(summary_csv, index=False)
        print(f"Saved summary metrics to {summary_csv}")
    except Exception as e:
        print(f"Error saving summary CSV: {e}", file=sys.stderr)

    # 9) Save NetCDF of fields
    try:
        out_nc = os.path.join(output_dir, "et_fields.nc")
        ds_out = xr.Dataset(
            {
                "et_model_clim_mm_day": et_clim_mmd,
                "et_obs_mm_day": et_obs,
                "et_model_on_obs_mm_day": model_on_obs,
                "et_bias_mm_day": bias
            }
        )
        # Add some attributes
        ds_out["et_model_clim_mm_day"].attrs["long_name"] = "ELM ET climatology (1985-1989)"
        ds_out["et_model_clim_mm_day"].attrs["units"] = "mm/day"
        ds_out["et_obs_mm_day"].attrs["long_name"] = "MODIS ET"
        ds_out["et_obs_mm_day"].attrs["units"] = "mm/day"
        ds_out["et_model_on_obs_mm_day"].attrs["long_name"] = "ELM ET climatology interpolated to MODIS grid"
        ds_out["et_model_on_obs_mm_day"].attrs["units"] = "mm/day"
        ds_out["et_bias_mm_day"].attrs["long_name"] = "Bias (ELM - MODIS)"
        ds_out["et_bias_mm_day"].attrs["units"] = "mm/day"
        ds_out.attrs["description"] = "E3SM ELM ET vs MODIS ET benchmark (1985-1989 climatology)"
        ds_out.attrs["global_mean_bias_mm_per_day"] = gm_bias
        ds_out.attrs["spatial_correlation"] = corr
        ds_out.to_netcdf(out_nc)
        print(f"Saved fields to {out_nc}")
    except Exception as e:
        print(f"Error saving NetCDF output: {e}", file=sys.stderr)

    # 10) Plot bias map
    try:
        fig = plt.figure(figsize=(12, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray')

        # Determine symmetric color scale based on robust range
        try:
            vmax = np.nanpercentile(np.abs(bias.values), 98)
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 2.0
        except Exception:
            vmax = 2.0

        im = ax.pcolormesh(
            bias["lon"], bias["lat"], bias,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax
        )
        cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
        cb.set_label("Bias (ELM - MODIS) [mm/day]")

        title = f"ET Bias (ELM - MODIS), 1985-1989 climatology\nGlobal mean bias: {gm_bias:.3f} mm/day | Spatial corr: {corr:.3f}"
        ax.set_title(title, fontsize=12)

        fig_fn = os.path.join(output_dir, "et_bias_map.png")
        plt.savefig(fig_fn, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved bias map to {fig_fn}")
    except Exception as e:
        print(f"Error creating/saving plot: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()