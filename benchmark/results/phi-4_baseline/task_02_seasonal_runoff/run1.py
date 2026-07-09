import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Define paths and case name
data_dir = "./data/sample/e3sm/"
case_name = "sample.v3.LR.historical"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_02_seasonal_runoff/run1_output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the time range for climatology calculation
start_year = 1985
end_year = 1989

try:
    # Load ELM data and filter by years
    files = sorted([f for f in os.listdir(data_dir + 'lnd/') if case_name in f and f.endswith('.elm.h0.nc')])
    datasets = [xr.open_dataset(os.path.join(data_dir, 'lnd', file)) for file in files]
    
    # Filter datasets to include only the years 1985-1989
    datasets = [ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31')) for ds in datasets]

    # Concatenate all datasets into a single dataset
    combined_ds = xr.concat(datasets, dim='time')

    # Calculate the climatological mean of QRUNOFF over 1985-1989
    climatology = combined_ds['QRUNOFF'].groupby('time.month').mean(dim='time')

    # Compute area-weighted global mean runoff
    # Assuming a regular lat-lon grid, calculate cell areas
    lon = combined_ds.lon.values
    lat = combined_ds.lat.values

    dlon = np.diff(lon)[0]
    dlat = np.diff(lat)[0]

    # Calculate the area of each grid cell in square meters (assuming Earth radius ~ 6.371e6 m)
    earth_radius = 6.371e6
    dx = earth_radius * np.cos(np.deg2rad(lat)) * np.pi/180 * dlon
    dy = earth_radius * np.pi/180 * dlat

    # Calculate the area of each grid cell
    area = dx[:, None] * dy[None, :]

    # Convert QRUNOFF from mm/s to m3/s by multiplying with area and converting units
    runoff_m3s = climatology.values * (area / 1e6)  # convert mm to meters

    # Compute the global mean runoff
    global_mean_runoff = np.sum(runoff_m3s) / np.sum(area)

    # Save the climatological mean QRUNOFF field as a NetCDF file
    climatology.to_netcdf(os.path.join(output_dir, 'climatological_mean_QRUNOFF.nc'))

    # Plotting the map of mean runoff field with global statistics overlaid
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    climatology.mean(dim='month').plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', cbar_kwargs={'label': 'Runoff (mm/s)'})
    
    # Add global mean runoff as text on the plot
    plt.text(0.05, 0.95, f'Global Mean Runoff: {global_mean_runoff:.2f} m3/s',
             transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    plt.title('Climatological Mean Runoff (1985-1989)')
    plt.savefig(os.path.join(output_dir, 'mean_runoff_map.png'))
    plt.close()

except Exception as e:
    print(f"An error occurred: {e}")