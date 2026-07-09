import os
import sys
import requests
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

def main():
    case_name = "sample.v3.LR.historical"
    elm_dir = "./data/sample/e3sm/lnd"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_03_et_benchmark/run1_debug/v2/output"
    os.makedirs(output_dir, exist_ok=True)

    # Download MODIS ET from ILAMB
    modis_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
    modis_file = os.path.join(output_dir, "et_0.5x0.5.nc")
    if not os.path.exists(modis_file):
        try:
            r = requests.get(modis_url, stream=True)
            r.raise_for_status()
            with open(modis_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print("Error downloading MODIS ET:", e)

    # Load MODIS ET and compute 1985-1989 mean
    try:
        ds_mod = xr.open_dataset(modis_file)
        et_mod = ds_mod["et"]
        et_mod = et_mod.sel(time=slice("1985-01-01", "1989-12-31"))
        et_mod_mean = et_mod.mean(dim="time", skipna=True)
    except Exception as e:
        print("Error loading MODIS ET data:", e)
        sys.exit(1)

    # Load E3SM ELM monthly data 1985-1989
    file_list = []
    for year in range(1985, 1990):
        for month in range(1, 13):
            fname = f"{case_name}.elm.h0.{year}-{month:02d}.nc"
            path = os.path.join(elm_dir, fname)
            if os.path.exists(path):
                file_list.append(path)
    if not file_list:
        print("No ELM files found, exiting.")
        sys.exit(1)
    try:
        ds_elm = xr.open_mfdataset(file_list, combine="by_coords")
        el_et = ds_elm["QVEGE"] + ds_elm["QVEGT"] + ds_elm["QSOIL"]
        el_et_mmday = el_et * 86400.0
        el_et_mean = el_et_mmday.mean(dim="time", skipna=True)
    except Exception as e:
        print("Error processing E3SM ELM data:", e)
        sys.exit(1)

    # Regrid MODIS mean to ELM grid via nearest-neighbor
    try:
        lat_mod = et_mod_mean["lat"].values
        lon_mod = et_mod_mean["lon"].values
        lon2d_mod, lat2d_mod = np.meshgrid(lon_mod, lat_mod)
        mod_coords = np.vstack((lat2d_mod.ravel(), lon2d_mod.ravel())).T
        tree = cKDTree(mod_coords)

        lat_var = ds_elm["lat"]
        lon_var = ds_elm["lon"]
        lat_model = lat_var.values
        lon_model = lon_var.values
        if lat_model.ndim == 1 and lon_model.ndim == 1:
            lon2d_model, lat2d_model = np.meshgrid(lon_model, lat_model)
        else:
            lat2d_model = lat_model
            lon2d_model = lon_model
        shape = lat2d_model.shape
        flat_lat = lat2d_model.ravel()
        flat_lon = lon2d_model.ravel()
        model_pts = np.vstack((flat_lat, flat_lon)).T
        _, idx = tree.query(model_pts, k=1)

        mod_flat = et_mod_mean.values.ravel()
        mod_on_model = mod_flat[idx].reshape(shape)
    except Exception as e:
        print("Error regridding MODIS to ELM grid:", e)
        sys.exit(1)

    # Compute bias and metrics
    model_mean = el_et_mean.values
    if model_mean.shape != mod_on_model.shape:
        print("Model grid and regridded MODIS grid shapes differ:", model_mean.shape, mod_on_model.shape)
        sys.exit(1)
    bias = model_mean - mod_on_model
    weights = np.cos(np.deg2rad(lat2d_model))
    mask = np.isfinite(bias)
    if np.any(mask):
        mean_bias = np.sum(bias[mask] * weights[mask]) / np.sum(weights[mask])
    else:
        mean_bias = np.nan

    flat_mod = mod_on_model.ravel()[mask.ravel()]
    flat_model = model_mean.ravel()[mask.ravel()]
    if flat_mod.size > 1:
        r_val, _ = pearsonr(flat_model, flat_mod)
    else:
        r_val = np.nan

    # Save statistics CSV
    try:
        df = pd.DataFrame({"mean_bias_mmday": [mean_bias], "spatial_correlation": [r_val]})
        stats_file = os.path.join(output_dir, "et_benchmark_stats.csv")
        df.to_csv(stats_file, index=False)
    except Exception as e:
        print("Error saving stats CSV:", e)

    # Save bias field to NetCDF
    try:
        dims2 = el_et_mean.dims
        coords2 = {
            dims2[0]: ds_elm.coords[dims2[0]].values,
            dims2[1]: ds_elm.coords[dims2[1]].values,
            "lat": (dims2, lat2d_model),
            "lon": (dims2, lon2d_model),
        }
        ds_out = xr.Dataset({"et_bias_mmday": (dims2, bias)}, coords=coords2)
        nc_file = os.path.join(output_dir, "et_bias.nc")
        ds_out.to_netcdf(nc_file)
    except Exception as e:
        print("Error saving bias NetCDF:", e)

    # Plot bias map
    try:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon2d_model, lat2d_model, bias,
                            transform=ccrs.PlateCarree(), cmap="RdBu_r", shading="auto")
        ax.coastlines()
        ax.set_global()
        ax.set_title(f"E3SM ET Bias (mm/day)\nMean bias={mean_bias:.2f}, Corr={r_val:.2f}")
        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("Bias (model - obs) [mm/day]")
        fig_file = os.path.join(output_dir, "et_bias_map.png")
        fig.savefig(fig_file, dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error plotting bias map:", e)

if __name__ == "__main__":
    main()