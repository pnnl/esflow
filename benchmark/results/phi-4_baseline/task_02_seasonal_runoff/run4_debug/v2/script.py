import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Define paths and case name
data_dir = "./data/sample/e3sm/"
case_name = "sample.v3.LR.historical"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_02_seasonal_runoff/run4_debug/v2/output"
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
    
    # Compute grid cell area using latitude bounds
    lat_bnds = np.deg2rad(qr_runoff.lat.values)
    lon_bnds = np.deg2rad(qr_runoff.lon.values)
    
    dlat = np.diff(lat_bnds).mean()
    dlon = np.diff(lon_bnds[0]) * np.cos(np.mean(lat_bnds))
    
    # Convert from degrees to meters
    dx = dlon * 111.32e3
    dy = dlat * 110.54e3
    
    area = dx * dy

    # Compute the climatological mean for each grid cell over the period
    qr_climatology = qr_runoff.groupby('time.month').mean(dim='time')

    # Compute the global, area-weighted mean runoff
    weighted_mean_runoff = (qr_climatology * area).sum(dim=('lat', 'lon')) / area.sum()
except Exception as e:
    print(f"Error processing QRUNOFF data: {e}")
    raise

# Save the climatological mean runoff to a NetCDF file
try:
    output_nc_path = os.path.join(output_dir, "qr_climatology.nc")
    qr_climatology.to_netcdf(output_nc_path)
except Exception as e:
    print(f"Error saving NetCDF: {e}")

# Plot the map of mean runoff field with global statistics overlaid
try:
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    qr_climatology.mean(dim='month').plot(
        transform=ccrs.PlateCarree(), 
        cmap='Blues', 
        cbar_kwargs={'label': 'Mean Runoff (mm/s)'}
    )
    ax.coastlines()
    plt.title(f"Global Mean QRUNOFF Climatology ({start_year}-{end_year})")
    
    # Overlay global statistics
    stats_text = f"Global Mean: {weighted_mean_runoff.mean().values:.2f} mm/s"
    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    output_png_path = os.path.join(output_dir, "qr_climatology_map.png")
    plt.savefig(output_png_path)
except Exception as e:
    print(f"Error plotting map: {e}")

# Save global statistics to a CSV file
try:
    stats_df = weighted_mean_runoff.to_dataframe(name='Mean_Runoff_mm_s')
    output_csv_path = os.path.join(output_dir, "global_statistics.csv")
    stats_df.to_csv(output_csv_path)
except Exception as e:
    print(f"Error saving CSV: {e}")