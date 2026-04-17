import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd

# Define paths and case name
data_dir = "./data/sample/e3sm/"
case_name = "sample.v3.LR.historical"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_02_seasonal_runoff/run3_debug/v2/output"
os.makedirs(output_dir, exist_ok=True)

# Define the time range
start_year = 1985
end_year = 1989

# Load ELM data for QRUNOFF and compute climatological mean
qr_runoff_files = [f"{data_dir}lnd/{case_name}.elm.h0.{year:04d}-01.nc" for year in range(start_year, end_year + 1)]
try:
    qr_runoff_ds_list = [xr.open_dataset(file) for file in qr_runoff_files]
    qr_runoff_ds = xr.concat(qr_runoff_ds_list, dim='time')
except Exception as e:
    print(f"Error loading ELM data: {e}")
    raise

# Extract QRUNOFF and compute area-weighted mean
try:
    qr_runoff = qr_runoff_ds['QRUNOFF']
    
    # Calculate grid cell areas using latitude bounds
    lat_bnds = np.deg2rad(qr_runoff.lat.values)
    lon_bnds = np.deg2rad(qr_runoff.lon.values)

    dlat = np.diff(lat_bnds, axis=0).mean()
    dlon = np.diff(lon_bnds, axis=1).mean() * np.cos(np.deg2rad(qr_runoff.lat))

    # Area in m^2
    earth_radius = 6371000  # Earth radius in meters
    dA = (earth_radius**2) * dlat[:, None] * dlon[None, :]

    # Compute area-weighted mean runoff
    weighted_mean_runoff = (qr_runoff * dA).mean(dim=['lat', 'lon', 'time'])
except Exception as e:
    print(f"Error processing QRUNOFF data: {e}")
    raise

# Save the climatological mean runoff to a NetCDF file
try:
    output_nc_path = os.path.join(output_dir, "climatology_mean_runoff.nc")
    weighted_mean_runoff.to_netcdf(output_nc_path)
except Exception as e:
    print(f"Error saving NetCDF: {e}")
    raise

# Plot the mean runoff field with global statistics
try:
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    qr_runoff.mean(dim='time').plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', cbar_kwargs={'label': 'Runoff (mm/s)'})
    ax.coastlines()
    ax.set_global()

    # Overlay global statistics
    global_mean = weighted_mean_runoff.values * 1e3  # Convert to m^3/s from mm/s
    plt.title(f"Global Mean Runoff: {global_mean:.2f} m³/s")
    plt.savefig(os.path.join(output_dir, "mean_runoff_map.png"))
except Exception as e:
    print(f"Error plotting mean runoff map: {e}")
    raise

# Save global statistics to a CSV file
try:
    stats_df = pd.DataFrame({
        'Global Mean Runoff (m³/s)': [global_mean]
    })
    stats_csv_path = os.path.join(output_dir, "global_runoff_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
except Exception as e:
    print(f"Error saving CSV: {e}")
    raise