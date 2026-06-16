import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import requests
import io
from scipy.stats import pearsonr

# --- Configuration ---
# Output directory (provided by the user)
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_03_et_benchmark/run3_debug/v1/output"

# E3SM data paths
e3sm_data_root = "./data/sample/e3sm/"
e3sm_case_name = "sample.v3.LR.historical"
e3sm_component = "lnd"
e3sm_years = range(1985, 1990) # 1985-1989 inclusive

# ILAMB data configuration
ilamb_base_url = "https://www.ilamb.org/ILAMB-Data/DATA/"
modis_et_url_path = "evspsbl/MODIS/et_0.5x0.5.nc"
modis_et_var_name = "et" # Variable name inside the MODIS NetCDF file

# --- Create Output Directory ---
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
except Exception as e:
    print(f"Error creating output directory {output_dir}: {e}")
    # Exit or handle more gracefully if directory creation is critical
    exit()

# --- 1. Fetch MODIS ET Data ---
print("Fetching MODIS ET data from ILAMB...")
modis_et_ds = None
try:
    ilamb_url = os.path.join(ilamb_base_url, modis_et_url_path)
    response = requests.get(ilamb_url, stream=True)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    
    # Read content into a BytesIO object and open with xarray
    # The error indicates that for BytesIO, engine='scipy' or 'h5netcdf' is required.
    # 'scipy' is a good general-purpose choice for in-memory NetCDF files.
    with io.BytesIO(response.content) as f:
        modis_et_ds = xr.open_dataset(f, engine='scipy')
    print("MODIS ET data fetched successfully.")

    # Extract the 'et' variable
    modis_et = modis_et_ds[modis_et_var_name]

    # Ensure longitude is -180 to 180 for consistency if it's 0-360
    if modis_et.lon.max() > 180:
        modis_et = modis_et.assign_coords(lon=(((modis_et.lon + 180) % 360) - 180)).sortby('lon')
        print("MODIS longitude converted from 0-360 to -180-180.")

except requests.exceptions.RequestException as e:
    print(f"Error fetching MODIS ET data from ILAMB: {e}")
except Exception as e:
    print(f"Error processing MODIS ET data: {e}")

# --- 2. Process E3SM ELM Data ---
print("Processing E3SM ELM data...")
e3sm_et_mean = None
if modis_et_ds is not None: # Only proceed if MODIS data was fetched successfully
    try:
        e3sm_file_pattern = os.path.join(e3sm_data_root, e3sm_component, 
                                         f"{e3sm_case_name}.{e3sm_component}.h0.{{year}}-{{month:02d}}.nc")
        
        # Create a list of file paths for the specified years
        e3sm_files = []
        for year in e3sm_years:
            for month in range(1, 13):
                e3sm_files.append(e3sm_file_pattern.format(year=year, month=month))
        
        # Open multiple files as a single dataset
        # For local files, 'netcdf4' is generally the robust choice.
        with xr.open_mfdataset(e3sm_files, combine='by_coords', decode_times=True, engine='netcdf4') as ds_e3sm:
            # Select ET components
            qvege = ds_e3sm['QVEGE']
            qvegt = ds_e3sm['QVEGT']
            qsoil = ds_e3sm['QSOIL']

            # Calculate total ET (sum of components)
            # E3SM variables are kg/m2/s, which is equivalent to mm/s for water
            e3sm_et_mm_s = qvege + qvegt + qsoil

            # Convert E3SM ET from mm/s to mm/day (1 day = 86400 seconds)
            e3sm_et_mm_day = e3sm_et_mm_s * 86400
            e3sm_et_mm_day.name = 'ET'
            e3sm_et_mm_day.attrs['units'] = 'mm/day'
            e3sm_et_mm_day.attrs['long_name'] = 'Total Evapotranspiration'

            # Compute climatological time-mean ET for E3SM
            e3sm_et_mean = e3sm_et_mm_day.mean(dim='time', keep_attrs=True)
            print("E3SM ELM ET climatology computed.")

            # Ensure E3SM longitude is -180 to 180 (it usually is)
            if e3sm_et_mean.lon.max() > 180:
                e3sm_et_mean = e3sm_et_mean.assign_coords(lon=(((e3sm_et_mean.lon + 180) % 360) - 180)).sortby('lon')
                print("E3SM longitude converted from 0-360 to -180-180.")

    except FileNotFoundError as e:
        print(f"Error: E3SM file not found. Check path and file pattern: {e}")
    except Exception as e:
        print(f"Error processing E3SM ELM data: {e}")

# --- 3. Process MODIS ET Data (Climatology) ---
print("Processing MODIS ET data climatology...")
modis_et_mean = None
if modis_et_ds is not None:
    try:
        # Filter MODIS data for the specified years (1985-1989)
        # MODIS 'et_0.5x0.5.nc' is often a climatology or a long-term mean.
        # If it has a time dimension, filter it. If not, it's already the mean.
        if 'time' in modis_et.dims:
            # Ensure time dimension is properly decoded if it wasn't already
            if not pd.api.types.is_datetime64_any_dtype(modis_et['time']):
                # This line might be problematic if modis_et_ds was opened with engine='scipy'
                # and time decoding was not fully handled. Let's assume xarray handles it.
                # If issues persist, manual time decoding might be needed.
                pass 
            
            modis_et_filtered = modis_et.sel(time=slice(f'{e3sm_years[0]}-01-01', f'{e3sm_years[-1]}-12-31'))
            if modis_et_filtered.sizes['time'] > 0:
                modis_et_mean = modis_et_filtered.mean(dim='time', keep_attrs=True)
                print(f"MODIS ET climatology computed for {e3sm_years[0]}-{e3sm_years[-1]}.")
            else:
                print(f"Warning: No MODIS data found for years {e3sm_years[0]}-{e3sm_years[-1]}. Using full dataset mean if available.")
                modis_et_mean = modis_et.mean(dim='time', keep_attrs=True) if 'time' in modis_et.dims else modis_et.copy(deep=True)
        else:
            modis_et_mean = modis_et.copy(deep=True) # Assume it's already a mean/climatology
            print("MODIS ET data appears to be a climatology or single time slice, no time averaging performed.")
        
        modis_et_mean.name = 'ET'
        modis_et_mean.attrs['units'] = 'mm/day'
        modis_et_mean.attrs['long_name'] = 'MODIS Evapotranspiration'

    except Exception as e:
        print(f"Error computing MODIS ET climatology: {e}")

# --- 4. Regrid E3SM to MODIS Grid and Compute Bias/Correlation ---
global_mean_bias = np.nan
spatial_correlation = np.nan
bias_field = None

if e3sm_et_mean is not None and modis_et_mean is not None:
    print("Regridding E3SM data to MODIS grid and computing bias/correlation...")
    try:
        # Regrid E3SM data to the MODIS grid using linear interpolation
        e3sm_et_regridded = e3sm_et_mean.interp(lat=modis_et_mean.lat, lon=modis_et_mean.lon, method='linear')
        print("E3SM ET regridded to MODIS grid.")

        # Compute spatial bias (Model - Observations)
        bias_field = e3sm_et_regridded - modis_et_mean
        bias_field.name = 'ET_Bias'
        bias_field.attrs['units'] = 'mm/day'
        bias_field.attrs['long_name'] = 'ET Bias (E3SM - MODIS)'

        # Compute global mean bias
        global_mean_bias = bias_field.mean(skipna=True).item()
        print(f"Global Mean Bias (E3SM - MODIS): {global_mean_bias:.2f} mm/day")

        # Compute spatial correlation
        # Flatten the arrays, removing NaNs
        e3sm_flat = e3sm_et_regridded.values.flatten()
        modis_flat = modis_et_mean.values.flatten()

        # Create a mask for non-NaN values in both arrays
        valid_mask = ~np.isnan(e3sm_flat) & ~np.isnan(modis_flat)
        
        if np.sum(valid_mask) > 1: # Need at least 2 points for correlation
            spatial_correlation, _ = pearsonr(e3sm_flat[valid_mask], modis_flat[valid_mask])
            print(f"Spatial Correlation (E3SM vs MODIS): {spatial_correlation:.2f}")
        else:
            print("Not enough valid data points to compute spatial correlation.")

    except Exception as e:
        print(f"Error during regridding, bias, or correlation calculation: {e}")

# --- 5. Save Results ---
print("Saving results...")

# Save global mean bias and spatial correlation to a text file
try:
    results_file = os.path.join(output_dir, "et_benchmark_summary.txt")
    with open(results_file, 'w') as f:
        f.write(f"ET Benchmark Summary (1985-1989 Climatology)\n")
        f.write(f"---------------------------------------------\n")
        f.write(f"Global Mean Bias (E3SM - MODIS): {global_mean_bias:.2f} mm/day\n")
        f.write(f"Spatial Correlation (E3SM vs MODIS): {spatial_correlation:.2f}\n")
    print(f"Benchmark summary saved to {results_file}")
except Exception as e:
    print(f"Error saving benchmark summary: {e}")

# Save bias field as NetCDF
if bias_field is not None:
    try:
        bias_nc_file = os.path.join(output_dir, "et_bias_field.nc")
        bias_field.to_netcdf(bias_nc_file)
        print(f"Bias field saved to {bias_nc_file}")
    except Exception as e:
        print(f"Error saving bias field NetCDF: {e}")

# --- 6. Plotting ---
if bias_field is not None:
    print("Generating bias map...")
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Determine a symmetric color scale
        max_abs_bias = np.nanmax(np.abs(bias_field.values))
        vmin = -max_abs_bias
        vmax = max_abs_bias

        bias_field.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                   x='lon', y='lat', cmap='RdBu',
                                   vmin=vmin, vmax=vmax,
                                   cbar_kwargs={'label': 'ET Bias (mm/day)'})
        
        ax.coastlines()
        ax.set_title(f'E3SM ELM vs MODIS ET Bias (1985-1989 Climatology)\n'
                     f'Global Mean Bias: {global_mean_bias:.2f} mm/day, Spatial Corr: {spatial_correlation:.2f}')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        plot_file = os.path.join(output_dir, "et_bias_map.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Bias map saved to {plot_file}")
    except Exception as e:
        print(f"Error generating bias map: {e}")
else:
    print("Bias field not available, skipping plot generation.")

print("Script finished.")