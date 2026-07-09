import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import requests
from scipy.stats import pearsonr

# Directories and file patterns
case_name = "sample.v3.LR.historical"
data_root = "./data/sample/e3sm"
lnd_dir = os.path.join(data_root, "lnd")
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_03_et_benchmark/run3_output"
os.makedirs(output_dir, exist_ok=True)

# 1. Fetch MODIS ET dataset from ILAMB
modis_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
modis_file = os.path.join(output_dir, "modis_et_0.5x0.5.nc")
if not os.path.exists(modis_file):
    try:
        print("Downloading MODIS ET data...")
        resp = requests.get(modis_url, stream=True)
        resp.raise_for_status()
        with open(modis_file, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading MODIS data: {e}")

# 2. Load MODIS ET and compute time mean 1985-1989
try:
    ds_modis = xr.open_dataset(modis_file)
    if "time" in ds_modis.dims:
        et_obs = ds_modis["et"].sel(time=slice("1985-01-01", "1989-12-31"))
        et_obs_mean = et_obs.mean(dim="time")
    else:
        et_obs_mean = ds_modis["et"].mean(dim=[d for d in ds_modis["et"].dims if d != "lat" and d != "lon"])
    # Ensure lat/lon dims
    et_obs_mean = et_obs_mean.rename({"latitude": "lat"} if "latitude" in et_obs_mean.dims else {}).rename({"longitude": "lon"} if "longitude" in et_obs_mean.dims else {})
    print("MODIS ET mean extracted.")
except Exception as e:
    print(f"Error processing MODIS ET data: {e}")
    raise

# 3. Load E3SM ELM ET for 1985-1989 and compute climatological mean
elm_files = []
for year in range(1985, 1990):
    for month in range(1, 13):
        fn = os.path.join(lnd_dir, f"{case_name}.elm.h0.{year:04d}-{month:02d}.nc")
        if os.path.exists(fn):
            elm_files.append(fn)
elm_files = sorted(elm_files)
if not elm_files:
    raise FileNotFoundError("No ELM files found for 1985-1989")

try:
    ds_elm = xr.open_mfdataset(elm_files, combine='by_coords')
    # Sum QVEGE, QVEGT, QSOIL (mm/s) and convert to mm/day
    et_elm = (ds_elm["QVEGE"] + ds_elm["QVEGT"] + ds_elm["QSOIL"]) * 86400.0
    et_elm_clim = et_elm.mean(dim="time")
    print("E3SM ELM ET climatology computed.")
except Exception as e:
    print(f"Error processing ELM ET data: {e}")
    raise

# 4. Interpolate E3SM ET to MODIS grid
try:
    # Align lon conventions if necessary
    if et_elm_clim.lon.max() > 180:
        et_elm_clim = et_elm_clim.assign_coords(lon=(((et_elm_clim.lon + 180) % 360) - 180)).sortby('lon')
    if et_obs_mean.lon.max() > 180:
        et_obs_mean = et_obs_mean.assign_coords(lon=(((et_obs_mean.lon + 180) % 360) - 180)).sortby('lon')
    et_elm_on_obs = et_elm_clim.interp(lat=et_obs_mean.lat, lon=et_obs_mean.lon, method="linear")
    print("Regridding ELM ET to MODIS grid done.")
except Exception as e:
    print(f"Error regridding ELM ET: {e}")
    raise

# 5. Compute bias field and statistics
try:
    bias = et_elm_on_obs - et_obs_mean
    # Compute global mean bias with cos(lat) weighting
    weights_1d = xr.DataArray(np.cos(np.deg2rad(bias.lat)), coords={'lat': bias.lat}, dims=['lat'])
    weights_2d = weights_1d.broadcast_like(bias)
    global_mean_bias = (bias * weights_2d).sum(dim=("lat", "lon")) / weights_2d.sum(dim=("lat", "lon"))
    global_mean_bias = float(global_mean_bias.values)

    # Spatial correlation
    m_flat = et_elm_on_obs.values.flatten()
    o_flat = et_obs_mean.values.flatten()
    mask = np.isfinite(m_flat) & np.isfinite(o_flat)
    if mask.sum() > 0:
        spatial_corr = float(pearsonr(m_flat[mask], o_flat[mask])[0])
    else:
        spatial_corr = np.nan
    print(f"Global mean bias: {global_mean_bias:.4f} mm/day")
    print(f"Spatial correlation: {spatial_corr:.4f}")
except Exception as e:
    print(f"Error computing bias or statistics: {e}")
    raise

# 6. Save bias field to NetCDF
try:
    bias.name = "et_bias"
    bias.attrs["long_name"] = "Evapotranspiration Bias (E3SM - MODIS)"
    bias.attrs["units"] = "mm/day"
    bias.to_netcdf(os.path.join(output_dir, "et_bias_1985_1989.nc"))
    print("Bias field saved to NetCDF.")
except Exception as e:
    print(f"Error saving bias NetCDF: {e}")

# 7. Save statistics to CSV
try:
    df_stats = pd.DataFrame({
        "global_mean_bias_mm_day": [global_mean_bias],
        "spatial_correlation": [spatial_corr]
    })
    df_stats.to_csv(os.path.join(output_dir, "et_benchmark_stats.csv"), index=False)
    print("Statistics saved to CSV.")
except Exception as e:
    print(f"Error saving statistics CSV: {e}")

# 8. Plot bias map
try:
    bplot = bias.copy()
    # Prepare for map
    if bplot.lon.max() > 180:
        bplot = bplot.assign_coords(lon=(((bplot.lon + 180) % 360) - 180)).sortby('lon')
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(bplot.lon, bplot.lat, bplot,
                       cmap='RdBu', vmin=-np.nanmax(abs(bplot)), vmax=np.nanmax(abs(bplot)),
                       transform=ccrs.PlateCarree())
    ax.coastlines()
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label("ET Bias (mm/day)")
    ax.set_title("E3SM ELM ET Bias vs MODIS (1985-1989)")
    fig_path = os.path.join(output_dir, "et_bias_map.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Bias map saved to {fig_path}.")
except Exception as e:
    print(f"Error plotting bias map: {e}")