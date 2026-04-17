import os
import sys
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def main():
    # Define paths
    input_dir = "./data/sample/e3sm/lnd"
    case_name = "sample.v3.LR.historical"
    file_pattern = os.path.join(input_dir, f"{case_name}.elm.h0.*.nc")
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_02_seasonal_runoff/run2_output"
    os.makedirs(output_dir, exist_ok=True)

    # Open ELM monthly data
    try:
        ds = xr.open_mfdataset(file_pattern, combine='by_coords')
    except Exception as e:
        print("Error opening ELM dataset:", e)
        sys.exit(1)

    # Select time period
    try:
        ds_period = ds.sel(time=slice("1985-01", "1989-12"))
    except Exception as e:
        print("Error selecting time slice:", e)
        sys.exit(1)

    # Extract QRUNOFF (mm/s) and compute climatological mean
    try:
        runoff = ds_period["QRUNOFF"]
        runoff_clim = runoff.mean(dim="time", skipna=True)
    except Exception as e:
        print("Error computing climatology:", e)
        sys.exit(1)

    # Compute area weights based on latitude
    try:
        lat = ds["lat"]
        # If lat is 2D with dims (lat, lon) or 1D
        weights = np.cos(np.deg2rad(lat))
        # Ensure weights has same dims as runoff_clim
        if set(weights.dims) != set(runoff_clim.dims):
            weights = weights.broadcast_like(runoff_clim)
    except Exception as e:
        print("Error computing weights:", e)
        sys.exit(1)

    # Compute global area-weighted mean runoff
    try:
        weighted = runoff_clim.weighted(weights)
        global_mean = weighted.mean(dim=("lat", "lon"), skipna=True).item()
    except Exception as e:
        print("Error computing global mean:", e)
        sys.exit(1)

    # Save climatological field to NetCDF
    nc_out = os.path.join(output_dir, "runoff_clim_qrunoff_1985_1989.nc")
    try:
        runoff_clim.to_netcdf(nc_out)
        print(f"Saved climatological runoff field to {nc_out}")
    except Exception as e:
        print("Error saving NetCDF:", e)

    # Save global mean to CSV
    csv_out = os.path.join(output_dir, "global_mean_runoff_1985_1989.csv")
    try:
        with open(csv_out, "w") as f:
            f.write("variable,period,global_mean_mm_per_s\n")
            f.write(f"QRUNOFF,1985-1989,{global_mean:.6e}\n")
        print(f"Saved global mean to {csv_out}")
    except Exception as e:
        print("Error saving CSV:", e)

    # Plot the climatological mean field
    try:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        runoff_clim_plot = runoff_clim
        cmap = plt.get_cmap("viridis")
        p = runoff_clim_plot.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            add_colorbar=True,
            cbar_kwargs={"label": "Runoff (mm/s)"},
        )
        ax.coastlines()
        ax.set_global()
        ax.set_title("Climatological Mean Runoff (QRUNOFF)\n1985-1989")

        # Overlay global mean statistic
        textstr = f"Global mean QRUNOFF: {global_mean:.4e} mm/s"
        ax.text(
            0.02,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        png_out = os.path.join(output_dir, "runoff_clim_map_1985_1989.png")
        fig.savefig(png_out, dpi=300)
        plt.close(fig)
        print(f"Saved plot to {png_out}")
    except Exception as e:
        print("Error during plotting or saving figure:", e)

if __name__ == "__main__":
    main()