import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Input and output directories
input_data_dir = "./data/sample/e3sm/"
elm_dir = os.path.join(input_data_dir, "lnd")
case_name = "sample.v3.LR.historical"
start_year, end_year = 1985, 1989
out_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_02_seasonal_runoff/run1_output"
os.makedirs(out_dir, exist_ok=True)

# Collect ELM monthly files for 1985-1989
file_list = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        fname = os.path.join(elm_dir, f"{case_name}.elm.h0.{year}-{month:02d}.nc")
        if os.path.exists(fname):
            file_list.append(fname)
if not file_list:
    raise FileNotFoundError("No ELM files found for the specified period.")

# Load and concatenate
ds = xr.open_mfdataset(file_list, combine="by_coords", parallel=False)
if "QRUNOFF" not in ds.variables:
    raise KeyError("QRUNOFF variable not found in the dataset.")

# Compute time mean for 1985-1989
qr = ds["QRUNOFF"]
qr_clim = qr.mean(dim="time", skipna=True)

# Compute area weights based on latitude
# Attempt to get lat and lon; assume coords named 'lat' and 'lon'
if "lat" in qr_clim.coords and "lon" in qr_clim.coords:
    lat = qr_clim["lat"]
    lon = qr_clim["lon"]
else:
    # fallback coordinate names
    lat = qr_clim.coords[list(qr_clim.coords)[-2]]
    lon = qr_clim.coords[list(qr_clim.coords)[-1]]

# broadcast weights
# If lat is 1D, make 2D
if lat.ndim == 1 and lon.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat)
    lat2d = xr.DataArray(lat2d, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
else:
    lat2d = lat

weights = np.cos(np.deg2rad(lat2d))
# Ensure weights have same dims as qr_clim
weights = weights / weights.mean()  # normalize to avoid overflow

# Identify spatial dimensions
spatial_dims = [d for d in qr_clim.dims if d != "time"]

# Compute global mean
global_mean = (qr_clim * weights).sum(dim=spatial_dims) / weights.sum(dim=spatial_dims)
global_mean_val = float(global_mean.values)

# Save climatology field to NetCDF
try:
    out_nc = os.path.join(out_dir, "QRUNOFF_clim_1985-1989.nc")
    qr_clim.to_dataset(name="QRUNOFF_climatology").to_netcdf(out_nc)
    print(f"Saved climatology NetCDF to {out_nc}")
except Exception as e:
    print(f"Error saving NetCDF: {e}")

# Save global mean to CSV
try:
    df = pd.DataFrame({
        "variable": ["QRUNOFF"],
        "period_start": [f"{start_year}-01"],
        "period_end": [f"{end_year}-12"],
        "global_mean_mm_per_s": [global_mean_val]
    })
    out_csv = os.path.join(out_dir, "QRUNOFF_global_mean_1985-1989.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved global mean CSV to {out_csv}")
except Exception as e:
    print(f"Error saving CSV: {e}")

# Plot map
try:
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # Prepare lon/lat for pcolormesh
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d = lon
        lat2d = lat
    pcm = ax.pcolormesh(lon2d, lat2d, qr_clim, transform=ccrs.PlateCarree(), cmap="viridis")
    cb = plt.colorbar(pcm, orientation="horizontal", pad=0.05, aspect=50)
    cb.set_label("QRUNOFF (mm/s)")
    title = f"Climatological Mean QRUNOFF 1985-1989\nGlobal Mean = {global_mean_val:.3e} mm/s"
    plt.title(title)
    out_png = os.path.join(out_dir, "QRUNOFF_clim_map_1985-1989.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved map PNG to {out_png}")
except Exception as e:
    print(f"Error creating or saving plot: {e}")