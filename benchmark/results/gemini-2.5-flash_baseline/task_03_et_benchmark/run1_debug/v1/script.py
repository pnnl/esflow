import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import requests
import shutil
from scipy import stats

# --- Configuration ---
# Output directory as specified in the task
output_base_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_03_et_benchmark/run1_debug/v1/output"
output_dir = os.path.join(output_base_dir)

# E3SM configuration
e3sm_case_name = "sample.v3.LR.historical"
e3sm_component = "lnd"
e3sm_h_stream = "h0"
e3sm_years = range(1985, 1990) # 1985-1989 inclusive
e3sm_data_path = "./data/sample/e3sm/"

# ILAMB configuration for MODIS ET
ilamb_base_url = "https://www.ilamb.org/ILAMB-Data/DATA/"
modis_variable = "evspsbl"
modis_dataset = "MODIS"
modis_filename = "et_0.5x0.5.nc"
modis_et_var_name = "et" # Variable name inside the MODIS NetCDF file

# --- Create output directory ---
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
except Exception as e:
    print(f"Error creating output directory {output_dir}: {e}")
    # If output directory cannot be created, the script cannot proceed.
    exit()

# --- 1. Fetch MODIS ET Data ---
modis_url = f"{ilamb_base_url}{modis_variable}/{modis_dataset}/{modis_filename}"
modis_local_path = os.path.join(output_dir, modis_filename)

print(f"Attempting to download MODIS ET from: {modis_url}")
try:
    # Check if file already exists to avoid re-downloading
    if os.path.exists(modis_local_path):
        print(f"MODIS file already exists locally at {modis_local_path}. Skipping download.")
    else:
        with requests.get(modis_url, stream=True) as r:
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(modis_local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded MODIS ET to: {modis_local_path}")
except requests.exceptions.RequestException as e:
    print(f"Error downloading MODIS ET from {modis_url}: {e}")
    if not os.path.exists(modis_local_path):
        print("MODIS file not found locally and download failed. Exiting.")
        exit() # Cannot proceed without MODIS data
except Exception as e:
    print(f"An unexpected error occurred during MODIS download: {e}")
    if not os.path.exists(modis_local_path):
        print("MODIS file not found locally and download failed. Exiting.")
        exit() # Cannot proceed without MODIS data

# --- 2. Load and Process E3SM ELM Data ---
print("Loading E3SM ELM data...")
e3sm_file_pattern = os.path.join(
    e3sm_data_path,
    e3sm_component,
    f"{e3sm_case_name}.{e3sm_component}.{e3sm_h_stream}.{{year}}-{{month:02d}}.nc"
)

e3sm_files = []
for year in e3sm_years:
    for month in range(1, 13):
        e3sm_files.append(e3sm_file_pattern.format(year=year, month=month))

# Filter out files that don't exist (e.g., if sample data is incomplete)
e3sm_files_exist = [f for f in e3sm_files if os.path.exists(f)]
if not e3sm_files_exist:
    # Fix: Print the pattern string directly without trying to format with 'YYYY' and 'MM'
    print(f"No E3SM ELM files found for pattern: {e3sm_file_pattern}")
    exit()
else:
    print(f"Found {len(e3sm_files_exist)} E3SM ELM files.")

try:
    # Preprocess function to select variables and convert units
    def preprocess_e3sm(ds):
        # Select ET components and convert from kg/m2/s (mm/s) to mm/day
        # 1 kg/m2 = 1 mm depth of water
        # 1 day = 86400 seconds
        et_components = ['QVEGE', 'QVEGT', 'QSOIL']
        
        # Check if all components exist
        missing_components = [comp for comp in et_components if comp not in ds.data_vars]
        if missing_components:
            print(f"Warning: E3SM variables {missing_components} not found in dataset. Skipping file.")
            return None # Return None to skip this file

        ds_et = (ds['QVEGE'] + ds['QVEGT'] + ds['QSOIL']) * 86400.0
        ds_et = ds_et.to_dataset(name='ET_E3SM')
        
        # Ensure longitude is 0-360
        if 'lon' in ds_et.coords and ds_et['lon'].min() < 0: # Check if it contains negative longitudes
            ds_et = ds_et.assign_coords(lon=(ds_et['lon'] + 360) % 360)
            ds_et = ds_et.sortby('lon')
        
        return ds_et

    # Open multiple files as a single dataset
    ds_e3sm = xr.open_mfdataset(
        e3sm_files_exist,
        combine='by_coords',
        preprocess=preprocess_e3sm,
        parallel=True, # Use dask for parallel loading
        coords='minimal', # Only load coordinates that are necessary
        data_vars='minimal', # Only load data variables that are necessary
        compat='override' # Override conflicting attributes
    )
    
    # Compute climatological time-mean ET
    e3sm_et_mean = ds_e3sm['ET_E3SM'].mean(dim='time', skipna=True)
    print("E3SM ELM data loaded and climatological mean computed.")

except Exception as e:
    print(f"Error loading or processing E3SM ELM data: {e}")
    exit()

# --- 3. Load and Process MODIS ET Data ---
print("Loading MODIS ET data...")
try:
    ds_modis = xr.open_dataset(modis_local_path)
    modis_et = ds_modis[modis_et_var_name]

    # Ensure longitude is 0-360 for consistency
    if 'lon' in modis_et.coords and modis_et['lon'].min() < 0: # Check if it contains negative longitudes
        modis_et = modis_et.assign_coords(lon=(modis_et['lon'] + 360) % 360)
        modis_et = modis_et.sortby('lon')
    
    # Filter MODIS data to the specified years (1985-1989)
    # MODIS data might cover a different period, so we need to select.
    # Assuming 'time' dimension exists and is datetime-like.
    modis_et_filtered = modis_et.sel(time=slice(f'{e3sm_years[0]}-01-01', f'{e3sm_years[-1]}-12-31'))
    
    # Compute climatological time-mean ET
    modis_et_mean = modis_et_filtered.mean(dim='time', skipna=True)
    print("MODIS ET data loaded and climatological mean computed.")

    # Check MODIS units - ILAMB data is usually standardized to mm/day for ET
    if 'units' in modis_et.attrs and modis_et.attrs['units'] != 'mm day-1':
        print(f"Warning: MODIS ET units are {modis_et.attrs['units']}. Assuming mm day-1 for comparison.")
        # If units are different, a conversion would be needed here.
        # For this task, we assume ILAMB MODIS ET is already in mm/day.

except Exception as e:
    print(f"Error loading or processing MODIS ET data: {e}")
    exit()

# --- 4. Regrid E3SM to MODIS Grid ---
print("Regridding E3SM ET to MODIS grid...")
try:
    # Ensure both have the same coordinate names for interpolation
    # E3SM ELM h0 files typically have 'lat' and 'lon' as 1D coordinates
    # MODIS also has 'lat' and 'lon' as 1D coordinates
    
    # Interpolate E3SM data to the MODIS grid
    # Using linear interpolation
    e3sm_et_regridded = e3sm_et_mean.interp(
        lat=modis_et_mean['lat'],
        lon=modis_et_mean['lon'],
        method='linear',
        kwargs={"fill_value": np.nan} # Fill outside bounds with NaN
    )
    print("E3SM ET successfully regridded.")
except Exception as e:
    print(f"Error during regridding E3SM ET: {e}")
    exit()

# --- 5. Compute Bias and Statistics ---
print("Computing bias and statistics...")
try:
    # Calculate spatial bias (Model - Observations)
    et_bias = e3sm_et_regridded - modis_et_mean
    et_bias.name = 'ET_Bias'
    et_bias.attrs['units'] = 'mm day-1'
    et_bias.attrs['long_name'] = 'E3SM ELM ET Bias (Model - MODIS)'

    # Calculate global mean bias
    # Mask NaNs before computing mean
    global_mean_bias = et_bias.mean(skipna=True).item()

    # Calculate spatial correlation
    # Flatten arrays and remove NaNs for correlation calculation
    e3sm_flat = e3sm_et_regridded.values.flatten()
    modis_flat = modis_et_mean.values.flatten()

    # Create a mask for non-NaN values in both arrays
    valid_mask = ~np.isnan(e3sm_flat) & ~np.isnan(modis_flat)

    if np.sum(valid_mask) > 1: # Need at least 2 points for correlation
        correlation, _ = stats.pearsonr(e3sm_flat[valid_mask], modis_flat[valid_mask])
    else:
        correlation = np.nan
        print("Warning: Not enough valid data points to compute spatial correlation.")

    print(f"Global Mean Bias (E3SM - MODIS): {global_mean_bias:.2f} mm/day")
    print(f"Spatial Correlation (E3SM vs MODIS): {correlation:.2f}")

    # Save statistics to a text file
    stats_output_path = os.path.join(output_dir, "et_benchmark_stats.txt")
    with open(stats_output_path, 'w') as f:
        f.write(f"E3SM ELM ET Benchmark (1985-1989 Climatology)\n")
        f.write(f"------------------------------------------------\n")
        f.write(f"Global Mean Bias (E3SM - MODIS): {global_mean_bias:.2f} mm/day\n")
        f.write(f"Spatial Correlation (E3SM vs MODIS): {correlation:.2f}\n")
    print(f"Statistics saved to: {stats_output_path}")

except Exception as e:
    print(f"Error computing bias or statistics: {e}")
    exit()

# --- 6. Plotting Bias Map ---
print("Generating bias map...")
try:
    fig = plt.figure(figsize=(12, 8))
    # Use PlateCarree projection, centered on 180 longitude for a global view
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))

    # Define color levels for the bias plot
    bias_max_abs = np.nanmax(np.abs(et_bias.values))
    if np.isnan(bias_max_abs):
        print("Cannot plot bias map: Bias data contains only NaNs.")
    else:
        # Use symmetric levels around zero
        levels = np.linspace(-bias_max_abs, bias_max_abs, 21) # 21 levels for smoother gradient

        et_bias.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=levels,
            cmap='RdBu', # Red-Blue for bias (red for positive bias, blue for negative)
            cbar_kwargs={'label': 'ET Bias (mm/day)', 'extend': 'both'}
        )

        ax.coastlines()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax.set_title(f'E3SM ELM ET Bias (Model - MODIS) 1985-1989 Climatology\nGlobal Mean Bias: {global_mean_bias:.2f} mm/day, Spatial Corr: {correlation:.2f}')

        plot_output_path = os.path.join(output_dir, "et_bias_map.png")
        plt.savefig(plot_output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Bias map saved to: {plot_output_path}")

except Exception as e:
    print(f"Error generating bias map: {e}")

# --- 7. Save Bias Field as NetCDF ---
print("Saving bias field to NetCDF...")
try:
    bias_netcdf_path = os.path.join(output_dir, "et_bias_field.nc")
    et_bias.to_netcdf(bias_netcdf_path)
    print(f"Bias field saved to: {bias_netcdf_path}")
except Exception as e:
    print(f"Error saving bias field to NetCDF: {e}")

print("ET benchmark script finished.")