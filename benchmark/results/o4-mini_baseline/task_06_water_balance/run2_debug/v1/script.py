import os
import glob
import traceback

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point

def main():
    try:
        # Settings
        case = "sample.v3.LR.historical"
        elm_dir = "./data/sample/e3sm/lnd"
        basin_geojson = "./data/sample/obs/basin_polygons.geojson"
        basin_ids = {
            "Amazon": 3629000,
            "Missouri": 4121801,
            "Columbia": 4115200,
            "Danube": 6742900,
            "Mekong": 2969100,
            "Orange": 1159100
        }
        outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_06_water_balance/run2_debug/v1/output"
        os.makedirs(outdir, exist_ok=True)

        # 1. Load ELM monthly files for 1985-1989
        pattern = os.path.join(elm_dir, f"{case}.elm.h0.*.nc")
        all_files = sorted(glob.glob(pattern))
        files = []
        for f in all_files:
            try:
                ym = os.path.basename(f).split('.')[-2]
                year = int(ym.split('-')[0])
                if 1985 <= year <= 1989:
                    files.append(f)
            except:
                continue
        if not files:
            raise RuntimeError("No files found for 1985-1989")

        ds = xr.open_mfdataset(files, combine='by_coords')
        # Compute climatology
        # Units: RAIN,SNOW kg/m2/s -> mm/s; ET mm/s; Q mm/s. Convert to mm/day
        P = (ds["RAIN"] + ds["SNOW"]) * 86400.0
        ET = (ds["QVEGE"] + ds["QVEGT"] + ds["QSOIL"]) * 86400.0
        Q = ds["QRUNOFF"] * 86400.0

        P_clim = P.mean("time")
        ET_clim = ET.mean("time")
        Q_clim = Q.mean("time")
        residual = P_clim - ET_clim - Q_clim

        # Determine lat/lon grid
        if ("lat" in P_clim.dims) and ("lon" in P_clim.dims):
            lat = P_clim["lat"]
            lon = P_clim["lon"]
            if lat.ndim == 1 and lon.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon.values, lat.values)
            else:
                lat2d = lat.values
                lon2d = lon.values
        else:
            raise RuntimeError("Dataset does not have lat/lon dims")

        # Create weights
        weight = np.cos(np.deg2rad(lat2d))
        weight_da = xr.DataArray(weight, dims=P_clim.dims, coords=P_clim.coords)
        weight_norm = weight_da / weight_da.sum()

        # Global means
        global_P = float((P_clim * weight_norm).sum(P_clim.dims))
        global_ET = float((ET_clim * weight_norm).sum(ET_clim.dims))
        global_Q = float((Q_clim * weight_norm).sum(Q_clim.dims))
        global_res = float((residual * weight_norm).sum(residual.dims))

        print("Global climatological means (mm/day):")
        print(f"  P  = {global_P:.3f}")
        print(f"  ET = {global_ET:.3f}")
        print(f"  Q  = {global_Q:.3f}")
        print(f"  Residual = {global_res:.3e}")

        # Load and fix basins
        basins = gpd.read_file(basin_geojson)
        basins = basins[basins["grdc_no"].isin(list(basin_ids.values()))]
        basins["geometry"] = basins["geometry"].buffer(0)

        # Prepare per-basin results
        results = []
        for name, gid in basin_ids.items():
            row = basins[basins["grdc_no"] == gid]
            if row.empty:
                print(f"Warning: basin {name} ({gid}) not found in geojson")
                continue
            geom = row.geometry.values[0]
            # Build mask
            pts = np.column_stack((lon2d.ravel(), lat2d.ravel()))
            mask_flat = np.array([geom.contains(Point(x, y)) for x, y in pts])
            mask = mask_flat.reshape(lat2d.shape)
            mask_da = xr.DataArray(mask, dims=P_clim.dims, coords=P_clim.coords)

            # Basin mean for each field
            w = weight_da.where(mask_da, other=0.0)
            wsum = w.sum().item()
            if wsum <= 0:
                print(f"Warning: no grid cells found for basin {name}")
                bp = be = bq = br = np.nan
            else:
                bp = float((P_clim * w).sum(P_clim.dims) / wsum)
                be = float((ET_clim * w).sum(ET_clim.dims) / wsum)
                bq = float((Q_clim * w).sum(Q_clim.dims) / wsum)
                br = float((residual * w).sum(residual.dims) / wsum)
            results.append({
                "basin": name,
                "P": bp,
                "ET": be,
                "Q": bq,
                "residual": br
            })

        df = pd.DataFrame(results).set_index("basin")
        # Save CSV
        try:
            df.to_csv(os.path.join(outdir, "basin_water_balance.csv"))
        except Exception:
            print("Error saving CSV:")
            traceback.print_exc()

        # Save NetCDF of fields
        try:
            ds_out = xr.Dataset({
                "P": P_clim,
                "ET": ET_clim,
                "Q": Q_clim,
                "residual": residual
            })
            ds_out.to_netcdf(os.path.join(outdir, "water_balance_climatology.nc"))
        except Exception:
            print("Error saving NetCDF:")
            traceback.print_exc()

        # Plotting
        try:
            fig = plt.figure(figsize=(10, 12))
            # Map
            ax_map = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
            mesh = ax_map.pcolormesh(lon2d, lat2d, residual,
                                     transform=ccrs.PlateCarree(),
                                     cmap="RdBu_r", vmin=-1, vmax=1)
            ax_map.coastlines()
            ax_map.add_feature(cfeature.BORDERS, linestyle=":")
            # Overlay basin outlines
            for geom in basins.geometry:
                ax_map.add_geometries([geom], crs=ccrs.PlateCarree(),
                                      edgecolor="black", facecolor="none", linewidth=1.5)
            cb = fig.colorbar(mesh, ax=ax_map, orientation="horizontal", pad=0.05)
            cb.set_label("Residual (mm/day)")
            ax_map.set_title("Climatological Water Balance Residual (P - ET - Q)")

            # Bar chart
            ax_bar = fig.add_subplot(2, 1, 2)
            x = np.arange(len(df))
            width = 0.2
            ax_bar.bar(x - 1.5*width, df["P"], width, label="P")
            ax_bar.bar(x - 0.5*width, df["ET"], width, label="ET")
            ax_bar.bar(x + 0.5*width, df["Q"], width, label="Q")
            ax_bar.bar(x + 1.5*width, df["residual"], width, label="Residual")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(df.index, rotation=45)
            ax_bar.set_ylabel("mm/day")
            ax_bar.set_title("Basin Water Balance Components")
            ax_bar.legend(ncol=2, fontsize="small")

            plt.tight_layout()
            figfile = os.path.join(outdir, "water_balance_map_and_bars.png")
            plt.savefig(figfile, dpi=300)
            plt.close(fig)
        except Exception:
            print("Error creating or saving plot:")
            traceback.print_exc()

    except Exception:
        print("Fatal error in script:")
        traceback.print_exc()

if __name__ == "__main__":
    main()