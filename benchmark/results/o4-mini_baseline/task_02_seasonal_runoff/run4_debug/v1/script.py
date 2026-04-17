#!/usr/bin/env python3
import os
import sys
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def main():
    # Parameters
    base_dir = "./data/sample/e3sm"
    case = "sample.v3.LR.historical"
    start_year = 1985
    end_year = 1989
    varname = "QRUNOFF"  # mm/s in ELM output
    out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_02_seasonal_runoff/run4_debug/v1/output"
    os.makedirs(out_dir, exist_ok=True)

    # Build list of monthly files for ELM land component
    files = []
    periods = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq='MS')
    for dt in periods:
        fn = os.path.join(base_dir, "lnd", f"{case}.elm.h0.{dt.year}-{dt.month:02d}.nc")
        files.append(fn)

    # Load data
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
    except Exception as e:
        print("Error opening ELM files:")
        traceback.print_exc()
        sys.exit(1)

    if varname not in ds:
        print(f"Variable {varname} not found in dataset.")
        sys.exit(1)

    # Select time range and compute climatological mean
    try:
        da = ds[varname].sel(time=slice(f"{start_year}-01", f"{end_year}-12"))
        clim = da.mean(dim='time', keep_attrs=True)
    except Exception:
        print("Error computing climatological mean:")
        traceback.print_exc()
        sys.exit(1)

    # Retrieve grid cell area for weighting
    if "area" in ds:
        area = ds["area"]
    elif "cell_area" in ds:
        area = ds["cell_area"]
    else:
        print("Grid cell area variable not found (expected 'area' or 'cell_area').")
        sys.exit(1)

    # Compute area-weighted global mean (mm/s)
    try:
        w = area.where(~np.isnan(clim))
        global_mean = (clim * w).sum() / w.sum()
    except Exception:
        print("Error computing area-weighted global mean:")
        traceback.print_exc()
        sys.exit(1)

    # Convert to mm/day for reporting and plotting
    # 1 day = 86400 seconds
    clim_mmday = clim * 86400.0
    try:
        global_mean_mmday = float((global_mean * 86400.0).compute())
    except Exception:
        # fallback for non-dask arrays
        global_mean_mmday = float(global_mean * 86400.0)

    # Save climatology to NetCDF
    out_nc = os.path.join(out_dir, "mean_QRUNOFF_1985_1989.nc")
    try:
        clim_mmday.to_dataset(name="QRUNOFF_mm_per_day").to_netcdf(out_nc)
        print(f"Saved climatology NetCDF to {out_nc}")
    except Exception:
        print("Error saving climatology NetCDF:")
        traceback.print_exc()

    # Save global mean to CSV
    out_csv = os.path.join(out_dir, "global_mean_runoff.csv")
    try:
        df = pd.DataFrame({
            "start_year": [start_year],
            "end_year": [end_year],
            "global_mean_runoff_mm_per_day": [global_mean_mmday]
        })
        df.to_csv(out_csv, index=False)
        print(f"Saved global mean to CSV {out_csv}")
    except Exception:
        print("Error saving global mean CSV:")
        traceback.print_exc()

    # Plot the field
    try:
        if ("lat" in clim.coords) and ("lon" in clim.coords):
            lats = clim["lat"]
            lons = clim["lon"]
            Lon, Lat = np.meshgrid(lons, lats)
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            pcm = ax.pcolormesh(Lon, Lat, clim_mmday, cmap="viridis", shading="auto",
                                transform=ccrs.PlateCarree())
            ax.coastlines()
            cb = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05)
            cb.set_label("Runoff (mm/day)")
            title = f"E3SM QRUNOFF Climatology ({start_year}-{end_year})\nGlobal mean = {global_mean_mmday:.2f} mm/day"
            plt.title(title)
            out_png = os.path.join(out_dir, "mean_runoff_map.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved map PNG to {out_png}")
        else:
            print("Latitude/longitude coordinates not found; cannot plot map.")
    except Exception:
        print("Error during plotting:")
        traceback.print_exc()

if __name__ == "__main__":
    main()