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
    # Settings
    case_name = "sample.v3.LR.historical"
    elm_dir = "./data/sample/e3sm/lnd"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_03_et_benchmark/run1_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download MODIS ET from ILAMB if not exists
    modis_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
    modis_file = os.path.join(output_dir, "et_0.5x0.5.nc")
    if not os.path.exists(modis_file):
        try:
            print("Downloading MODIS ET data...")
            r = requests.get(modis_url, stream=True)
            r.raise_for_status()
            with open(modis_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print("Downloaded MODIS ET to", modis_file)
        except Exception as e:
            print("Error downloading MODIS ET:", e)

    # 2. Load MODIS ET and compute 1985-1989 mean
    try:
        ds_mod = xr.open_dataset(modis_file)
        et_mod = ds_mod["et"]
        et_mod = et_mod.sel(time=slice("1985-01-01", "1989-12-31"))
        et_mod_mean = et_mod.mean(dim="time", skipna=True)
        print("Loaded and computed MODIS ET climatology")
    except Exception as e:
        print("Error loading MODIS ET data:", e)
        sys.exit(1)

    # 3. Load E3SM ELM monthly data 1985-1989
    file_list = []
    for year in range(1985, 1990):
        for month in range(1, 13):
            fname = f"{case_name}.elm.h0.{year}-{month:02d}.nc"
            path = os.path.join(elm_dir, fname)
            if os.path.exists(path):
                file_list.append(path)
            else:
                print("Missing ELM file:", path)
    if not file_list:
        print("No ELM files found, exiting.")
        sys.exit(1)
    try:
        ds_elm = xr.open_mfdataset(file_list, combine="by_coords")
        el_et = ds_elm["QVEGE"] + ds_elm["QVEGT"] + ds_elm["QSOIL"]
        # convert from mm/s (kg/m2/s) to mm/day
        el_et_mmday = el_et * 86400.0
        el_et_mean = el_et_mmday.mean(dim="time", skipna=True)
        print("Loaded E3SM ELM ET and computed climatology")
    except Exception as e:
        print("Error processing E3SM ELM data:", e)
        sys.exit(1)

    # 4. Regrid MODIS mean to ELM grid via nearest-neighbor
    try:
        lat_model = ds_elm["lat"].values
        lon_model = ds_elm["lon"].values
        shape = lat_model.shape
        flat_lat = lat_model.ravel()
        flat_lon = lon_model.ravel()

        lat_mod = et_mod_mean["lat"].values
        lon_mod = et_mod_mean["lon"].values
        lon2d_mod, lat2d_mod = np.meshgrid(lon_mod, lat_mod)
        mod_coords = np.vstack([lat2d_mod.ravel(), lon2d_mod.ravel()]).T
        tree = cKDTree(mod_coords)
        model_pts = np.vstack([flat_lat, flat_lon]).T
        _, idx = tree.query(model_pts, k=1)

        mod_flat = et_mod_mean.values.ravel()
        mod_on_model = mod_flat[idx].reshape(shape)
        print("Regridded MODIS ET to ELM grid")
    except Exception as e:
        print("Error regridding MODIS to ELM grid:", e)
        sys.exit(1)

    # 5. Compute bias and metrics
    model_mean = el_et_mean.values
    bias = model_mean - mod_on_model
    weights = np.cos(np.deg2rad(lat_model))
    mask = np.isfinite(bias)
    if np.any(mask):
        mean_bias = np.sum(bias[mask] * weights[mask]) / np.sum(weights[mask])
    else:
        mean_bias = np.nan

    # spatial correlation
    flat_mod = mod_on_model.ravel()
    flat_mod = flat_mod[mask.ravel()]
    flat_model = model_mean.ravel()
    flat_model = flat_model[mask.ravel()]
    if flat_mod.size > 0:
        r_val, _ = pearsonr(flat_model, flat_mod)
    else:
        r_val = np.nan
    print(f"Mean bias (mm/day): {mean_bias:.4f}, Spatial correlation: {r_val:.4f}")

    # 6. Save statistics CSV
    try:
        df = pd.DataFrame({"mean_bias_mmday": [mean_bias], "spatial_correlation": [r_val]})
        stats_file = os.path.join(output_dir, "et_benchmark_stats.csv")
        df.to_csv(stats_file, index=False)
        print("Saved stats to", stats_file)
    except Exception as e:
        print("Error saving stats CSV:", e)

    # 7. Save bias field to NetCDF
    try:
        dims = el_et_mean.dims  # e.g., ('nlat','nlon')
        coords = {dims[0]: ds_elm.coords[dims[0]].values,
                  dims[1]: ds_elm.coords[dims[1]].values,
                  "lat": (dims, lat_model),
                  "lon": (dims, lon_model)}
        ds_out = xr.Dataset({"et_bias_mmday": (dims, bias)}, coords=coords)
        nc_file = os.path.join(output_dir, "et_bias.nc")
        ds_out.to_netcdf(nc_file)
        print("Saved bias field to", nc_file)
    except Exception as e:
        print("Error saving bias NetCDF:", e)

    # 8. Plot bias map
    try:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon_model, lat_model, bias,
                            transform=ccrs.PlateCarree(), cmap="RdBu_r", shading="auto")
        ax.coastlines()
        ax.set_global()
        ax.set_title(f"E3SM ET Bias (mm/day)\nMean bias={mean_bias:.2f}, Corr={r_val:.2f}")
        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("Bias (model - obs) [mm/day]")
        fig_file = os.path.join(output_dir, "et_bias_map.png")
        fig.savefig(fig_file, dpi=300)
        plt.close(fig)
        print("Saved bias map to", fig_file)
    except Exception as e:
        print("Error plotting bias map:", e)

if __name__ == "__main__":
    main()