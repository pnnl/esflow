import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def main():
    # Paths and URLs
    data_dir = "./data/sample/e3sm"
    case_name = "sample.v3.LR.historical"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_03_et_benchmark/run4_output"
    os.makedirs(output_dir, exist_ok=True)
    ilamb_url = "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc"
    ilamb_file = os.path.join(output_dir, "et_MODIS_0.5x0.5.nc")

    # Download MODIS ET data if not present
    if not os.path.exists(ilamb_file):
        try:
            print("Downloading ILAMB MODIS ET data...")
            resp = requests.get(ilamb_url, stream=True)
            resp.raise_for_status()
            with open(ilamb_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading ILAMB data: {e}")

    # Load MODIS ET and compute climatology
    try:
        ds_modis = xr.open_dataset(ilamb_file)
        et_modis = ds_modis["et"]
        et_modis = et_modis.sel(time=slice("1985-01-01","1989-12-31"))
        et_modis_mean = et_modis.mean(dim="time", skipna=True)
    except Exception as e:
        print(f"Error loading MODIS ET data: {e}")
        return

    # Load E3SM ELM ET components and compute climatology
    elm_pattern = os.path.join(data_dir, "lnd", f"{case_name}.elm.h0.*.nc")
    try:
        ds_elm = xr.open_mfdataset(elm_pattern, combine="by_coords")
        et_elm = ds_elm["QVEGE"] + ds_elm["QVEGT"] + ds_elm["QSOIL"]
        # convert mm/s to mm/day
        et_elm = et_elm * 86400.0
        et_elm = et_elm.sel(time=slice("1985-01-01","1989-12-31"))
        et_elm_mean = et_elm.mean(dim="time", skipna=True)
    except Exception as e:
        print(f"Error loading ELM ET data: {e}")
        return

    # Regrid ELM ET to MODIS grid
    try:
        et_elm_on_modis = et_elm_mean.interp(lat=et_modis_mean.lat, lon=et_modis_mean.lon, method="linear")
    except Exception as e:
        print(f"Error regridding ELM ET: {e}")
        return

    # Compute bias
    bias = et_elm_on_modis - et_modis_mean

    # Compute global mean bias with area weights
    lat_rad = np.deg2rad(et_modis_mean.lat)
    weights = np.cos(lat_rad)
    # broadcast to 2D
    w2d = weights.broadcast_like(bias)
    mask = np.isfinite(bias) & np.isfinite(et_modis_mean)
    w = w2d.where(mask)
    x = bias.where(mask)
    y = et_modis_mean.where(mask)
    wsum = w.sum(dim=["lat","lon"])
    mean_bias_num = (x * w).sum(dim=["lat","lon"])
    global_mean_bias = mean_bias_num / wsum

    # Compute spatial correlation
    mean_x = (x * w).sum(dim=["lat","lon"]) / wsum
    mean_y = (y * w).sum(dim=["lat","lon"]) / wsum
    cov_xy = ((x - mean_x) * (y - mean_y) * w).sum(dim=["lat","lon"]) / wsum
    var_x = (((x - mean_x)**2) * w).sum(dim=["lat","lon"]) / wsum
    var_y = (((y - mean_y)**2) * w).sum(dim=["lat","lon"]) / wsum
    spatial_corr = cov_xy / np.sqrt(var_x * var_y)

    # Save bias netCDF
    try:
        bias.to_dataset(name="et_bias").to_netcdf(os.path.join(output_dir, "et_bias.nc"))
    except Exception as e:
        print(f"Error saving bias NetCDF: {e}")

    # Save statistics CSV
    try:
        df = pd.DataFrame({
            "global_mean_bias_mm_day": [float(global_mean_bias.values)],
            "spatial_correlation": [float(spatial_corr.values)]
        })
        df.to_csv(os.path.join(output_dir, "et_bias_stats.csv"), index=False)
    except Exception as e:
        print(f"Error saving stats CSV: {e}")

    # Plot bias map
    try:
        fig = plt.figure(figsize=(10,5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        bias_plot = bias.plot(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap="RdBu_r", vmin=-np.nanmax(np.abs(bias)), vmax=np.nanmax(np.abs(bias)),
            cbar_kwargs={"label": "ET bias (mm/day)"}
        )
        ax.coastlines()
        ax.set_title("E3SM ELM ET Bias (1985-1989) Model - MODIS")
        png_file = os.path.join(output_dir, "et_bias_map.png")
        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting bias map: {e}")

if __name__ == "__main__":
    main()