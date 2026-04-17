import os
import sys
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import json
from shapely.geometry import shape
from shapely.vectorized import contains

def main():
    # Paths and parameters
    case_name = "sample.v3.LR.historical"
    e3sm_root = "./data/sample/e3sm"
    lnd_dir = os.path.join(e3sm_root, "lnd")
    years = ("1985-01-01", "1989-12-31")
    basin_ids = {
        3629000: "Amazon",
        4121801: "Missouri",
        4115200: "Columbia",
        6742900: "Danube",
        2969100: "Mekong",
        1159100: "Orange"
    }
    basin_geojson = "./data/sample/obs/basin_polygons.geojson"
    out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_06_water_balance/run1_output"
    os.makedirs(out_dir, exist_ok=True)

    # Load ELM monthly data
    try:
        file_pattern = os.path.join(lnd_dir, f"{case_name}.elm.h0.*.nc")
        ds = xr.open_mfdataset(file_pattern, combine='by_coords', parallel=False)
        ds = ds.sel(time=slice(*years))
    except Exception:
        print("Error loading ELM data:")
        traceback.print_exc()
        sys.exit(1)

    # Compute P, ET, Q in mm/day
    try:
        # RAIN, SNOW in kg/m2/s -> mm/s; *86400 -> mm/day
        P = (ds["RAIN"] + ds["SNOW"]) * 86400.0
        ET = (ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]) * 86400.0
        Q = ds["QRUNOFF"] * 86400.0
        # Climatological mean
        P_mean = P.mean(dim="time")
        ET_mean = ET.mean(dim="time")
        Q_mean = Q.mean(dim="time")
        # Residual
        Residual = P_mean - ET_mean - Q_mean
    except Exception:
        print("Error computing climatologies:")
        traceback.print_exc()
        sys.exit(1)

    # Compute global area-weighted means
    try:
        # identify dims for weighting
        dims = P_mean.dims
        # get lat coordinate
        if "lat" in ds.coords:
            lat = ds["lat"]
        else:
            # try first coord
            lat = ds.coords[list(ds.coords)[0]]
        # build weights
        weights = np.cos(np.deg2rad(lat))
        # if 2D lat, weights 2D; else broadcast
        # Use xarray weighted
        gm_P = P_mean.weighted(weights).mean(dims).item()
        gm_ET = ET_mean.weighted(weights).mean(dims).item()
        gm_Q = Q_mean.weighted(weights).mean(dims).item()
        gm_Res = Residual.weighted(weights).mean(dims).item()
    except Exception:
        print("Error computing global means:")
        traceback.print_exc()
        sys.exit(1)

    # Read basin polygons
    try:
        with open(basin_geojson) as f:
            gj = json.load(f)
        basin_shapes = {}
        for feat in gj["features"]:
            gid = feat["properties"].get("grdc_no")
            if gid in basin_ids:
                basin_shapes[gid] = shape(feat["geometry"])
    except Exception:
        print("Error reading basin polygons:")
        traceback.print_exc()
        sys.exit(1)

    # Prepare lat/lon grid for masking
    try:
        lon = ds["lon"]
        lat = ds["lat"]
        # Ensure 2D arrays
        if lon.ndim == 1 and lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lon2d = lon.values
            lat2d = lat.values
    except Exception:
        print("Error preparing lon/lat grid:")
        traceback.print_exc()
        sys.exit(1)

    # Per-basin summary
    summary = []
    for gid, name in basin_ids.items():
        shape_poly = basin_shapes.get(gid)
        if shape_poly is None:
            continue
        try:
            mask = contains(shape_poly, lon2d, lat2d)
            mask_da = xr.DataArray(mask, coords=P_mean.coords, dims=P_mean.dims)
            w = weights
            # masked means
            P_b = (P_mean.where(mask_da) * w.where(mask_da)).sum(dims) / w.where(mask_da).sum(dims)
            ET_b = (ET_mean.where(mask_da) * w.where(mask_da)).sum(dims) / w.where(mask_da).sum(dims)
            Q_b = (Q_mean.where(mask_da) * w.where(mask_da)).sum(dims) / w.where(mask_da).sum(dims)
            R_b = (Residual.where(mask_da) * w.where(mask_da)).sum(dims) / w.where(mask_da).sum(dims)
            summary.append({
                "basin_id": gid,
                "basin_name": name,
                "P_mean": float(P_b.values),
                "ET_mean": float(ET_b.values),
                "Q_mean": float(Q_b.values),
                "Residual": float(R_b.values)
            })
        except Exception:
            print(f"Error processing basin {name}:")
            traceback.print_exc()

    df = pd.DataFrame(summary).set_index("basin_name")

    # Save summary CSV
    try:
        csv_path = os.path.join(out_dir, "basin_water_balance_summary.csv")
        df.to_csv(csv_path)
    except Exception:
        print("Error saving CSV:")
        traceback.print_exc()

    # Save NetCDF of fields
    try:
        out_ds = xr.Dataset({
            "P_mean": P_mean,
            "ET_mean": ET_mean,
            "Q_mean": Q_mean,
            "Residual": Residual
        })
        nc_path = os.path.join(out_dir, "water_balance_climatology.nc")
        out_ds.to_netcdf(nc_path)
    except Exception:
        print("Error saving NetCDF:")
        traceback.print_exc()

    # Plot composite figure
    try:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])
        # Top: global residual map
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax_map.coastlines()
        im = ax_map.pcolormesh(lon, lat, Residual, transform=ccrs.PlateCarree(), cmap="RdBu_r")
        cbar = fig.colorbar(im, ax=ax_map, orientation="horizontal", pad=0.05)
        cbar.set_label("Residual (mm/day)")
        ax_map.set_title("Global Water Balance Residual (P - ET - Q)")
        # overlay basin outlines
        for gid, name in basin_ids.items():
            poly = basin_shapes.get(gid)
            if poly is not None:
                ax_map.add_geometries([poly], crs=ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="black", linewidth=1)

        # Bottom: bar charts for each basin
        basins = list(df.index)
        for i, basin in enumerate(basins):
            row = 1 + i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            vals = df.loc[basin, ["P_mean", "ET_mean", "Q_mean", "Residual"]]
            vals.plot(kind="bar", ax=ax, color=["blue", "green", "cyan", "red"])
            ax.set_title(basin)
            ax.set_ylabel("mm/day")
            ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        png_path = os.path.join(out_dir, "water_balance_composite.png")
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
    except Exception:
        print("Error creating plot:")
        traceback.print_exc()

if __name__ == "__main__":
    main()