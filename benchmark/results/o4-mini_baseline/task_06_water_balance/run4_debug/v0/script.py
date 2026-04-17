#!/usr/bin/env python3
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point
import warnings

def main():
    # Settings
    case_name = "sample.v3.LR.historical"
    start_year = 1985
    end_year = 1989
    data_root = "./data/sample/e3sm"
    lnd_dir = os.path.join(data_root, "lnd")
    out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_06_water_balance/run4_output"
    os.makedirs(out_dir, exist_ok=True)

    # Basin info
    basin_ids = {
        3629000: "Amazon",
        4121801: "Missouri",
        4115200: "Columbia",
        6742900: "Danube",
        2969100: "Mekong",
        1159100: "Orange"
    }
    basin_geojson = "./data/sample/obs/basin_polygons.geojson"

    # Read and combine ELM monthly files
    pattern = os.path.join(lnd_dir, f"{case_name}.elm.h0.*.nc")
    elm_files = sorted(glob.glob(pattern))
    if not elm_files:
        raise FileNotFoundError(f"No ELM files found with pattern {pattern}")
    ds = xr.open_mfdataset(elm_files, combine="by_coords")
    # Select time period
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Compute components (units mm/s)
    P = ds["RAIN"] + ds["SNOW"]
    ET = ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]
    Q = ds["QRUNOFF"]

    # Time mean fields
    P_mean = P.mean(dim="time")
    ET_mean = ET.mean(dim="time")
    Q_mean = Q.mean(dim="time")
    residual = P_mean - ET_mean - Q_mean

    # Obtain area weights
    if "area" in ds:
        area = ds["area"]
    elif "area_t" in ds:
        area = ds["area_t"]
    elif "CELL_AREA" in ds:
        area = ds["CELL_AREA"]
    else:
        warnings.warn("Area variable not found; using equal weights")
        area = xr.ones_like(P_mean)

    # Compute global means (mm/s)
    total_area = area.sum()
    gm_P = (P_mean * area).sum() / total_area
    gm_ET = (ET_mean * area).sum() / total_area
    gm_Q = (Q_mean * area).sum() / total_area
    gm_res = (residual * area).sum() / total_area

    # Convert to mm/day for reporting
    sec_per_day = 86400.0
    gm_dict = {
        "P": float(gm_P.values) * sec_per_day,
        "ET": float(gm_ET.values) * sec_per_day,
        "Q": float(gm_Q.values) * sec_per_day,
        "Res": float(gm_res.values) * sec_per_day
    }
    global_means_df = pd.DataFrame({
        "variable": list(gm_dict.keys()),
        "global_mean_mm_per_day": list(gm_dict.values())
    })
    try:
        global_means_df.to_csv(os.path.join(out_dir, "global_means.csv"), index=False)
    except Exception as e:
        print(f"Error saving global means CSV: {e}")

    # Load basin polygons
    basin_gdf = gpd.read_file(basin_geojson)
    basin_gdf = basin_gdf[basin_gdf["grdc_no"].isin(basin_ids.keys())].copy()
    basin_gdf["name"] = basin_gdf["grdc_no"].map(basin_ids)

    # Prepare lat/lon grid
    # Assume P_mean.dims = ('lat','lon') or coords 'lat','lon'
    if "lat" in P_mean.dims and "lon" in P_mean.dims:
        lat = P_mean["lat"].values
        lon = P_mean["lon"].values
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lat2d = lat
            lon2d = lon
    else:
        # try coordinates
        lat2d = P_mean["latitude"].values
        lon2d = P_mean["longitude"].values

    shape = lat2d.shape
    points = np.array([lon2d.flatten(), lat2d.flatten()]).T

    # Compute per-basin means
    basin_means = []
    basin_fields = {}
    for idx, row in basin_gdf.iterrows():
        bname = row["name"]
        geom = row["geometry"]
        mask_flat = np.array([geom.contains(Point(x, y)) for x, y in points])
        mask = mask_flat.reshape(shape)
        # area mask
        area_masked = area.where(mask)
        # compute mean mm/s
        bp = (P_mean * area_masked).sum() / area_masked.sum()
        bet = (ET_mean * area_masked).sum() / area_masked.sum()
        bq = (Q_mean * area_masked).sum() / area_masked.sum()
        br = (residual * area_masked).sum() / area_masked.sum()
        # convert to mm/day
        basin_means.append({
            "basin": bname,
            "P_mm_per_day": float(bp.values) * sec_per_day,
            "ET_mm_per_day": float(bet.values) * sec_per_day,
            "Q_mm_per_day": float(bq.values) * sec_per_day,
            "Res_mm_per_day": float(br.values) * sec_per_day
        })
        # store clipped fields
        basin_fields[bname] = {
            "P": P_mean.where(mask),
            "ET": ET_mean.where(mask),
            "Q": Q_mean.where(mask),
            "Res": residual.where(mask)
        }

    basin_means_df = pd.DataFrame(basin_means)
    try:
        basin_means_df.to_csv(os.path.join(out_dir, "basin_means.csv"), index=False)
    except Exception as e:
        print(f"Error saving basin means CSV: {e}")

    # Save fields to NetCDF
    try:
        ds_out = xr.Dataset({
            "P_mean": P_mean,
            "ET_mean": ET_mean,
            "Q_mean": Q_mean,
            "residual": residual
        })
        ds_out.attrs["description"] = "Time-mean fields (1985-1989) from ELM: P, ET, Q, and residual (P-ET-Q)"
        ds_out.to_netcdf(os.path.join(out_dir, "water_balance_fields.nc"))
    except Exception as e:
        print(f"Error saving NetCDF fields: {e}")

    # Plotting
    try:
        fig = plt.figure(figsize=(14, 10))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])

        # Global residual map
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        res_daily = (residual * sec_per_day).transpose()
        im = ax_map.pcolormesh(lon2d, lat2d, res_daily.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, transform=ccrs.PlateCarree())
        for _, row in basin_gdf.iterrows():
            ax_map.plot(*row["geometry"].exterior.xy, color="k", linewidth=1, transform=ccrs.PlateCarree())
        ax_map.coastlines()
        ax_map.set_title("Global Water Balance Residual (P-ET-Q) mm/day, 1985-1989")
        cax = fig.add_axes([0.92, 0.6, 0.015, 0.3])
        plt.colorbar(im, cax=cax, label="mm/day")

        # Per-basin bar charts
        vars_order = ["P", "ET", "Q", "Res"]
        for i, basin in enumerate(basin_means_df["basin"]):
            ax = fig.add_subplot(gs[1 + i // 3, i % 3])
            vals = [
                basin_means_df.loc[basin_means_df["basin"] == basin, f"{v}_mm_per_day"].values[0]
                for v in vars_order
            ]
            ax.bar(vars_order, vals, color=["tab:blue", "tab:green", "tab:cyan", "tab:red"])
            ax.set_title(basin)
            ax.set_ylabel("mm/day")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        fig_path = os.path.join(out_dir, "water_balance.png")
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    main()