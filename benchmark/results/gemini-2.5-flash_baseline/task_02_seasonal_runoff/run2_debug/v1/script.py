import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import glob

# --- Configuration ---
# Input data paths
E3SM_DATA_DIR = "./data/sample/e3sm/"
CASE_NAME = "sample.v3.LR.historical"
ELM_COMPONENT = "lnd" # This refers to the subdirectory name
ELM_STREAM = "h0" # Monthly output stream
ELM_VAR = "QRUNOFF" # Runoff variable (mm/s)
START_YEAR = 1985
END_YEAR = 1989

# Output directory
# This path is provided by the user and should be created if it doesn't exist.
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_02_seasonal_runoff/run2_debug/v1/output"

# --- Create output directory ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created or already exists: {OUTPUT_DIR}")
except Exception as e:
    print(f"Error creating output directory {OUTPUT_DIR}: {e}")
    # If directory creation fails, subsequent file operations will likely fail.
    # For a self-contained script, it's better to exit or raise here.
    raise SystemExit(f"Failed to create output directory: {e}")

# --- 1. Find and load ELM data ---
print(f"Loading ELM data for {ELM_VAR} from {START_YEAR}-{END_YEAR}...")
# Corrected file pattern: The component name in the filename is 'elm', not 'lnd' (ELM_COMPONENT)
# The ELM_COMPONENT variable is used for the subdirectory, but 'elm' is hardcoded in the filename.
elm_file_pattern = os.path.join(E3SM_DATA_DIR, ELM_COMPONENT, f"{CASE_NAME}.elm.{ELM_STREAM}.*.nc")
all_elm_files = sorted(glob.glob(elm_file_pattern))

# Filter files for the specified time period
target_files = []
for f in all_elm_files:
    try:
        # Extract YYYY-MM from filename: e.g., sample.v3.LR.historical.elm.h0.1985-01.nc
        date_str = f.split('.')[-2]
        year = int(date_str.split('-')[0])
        if START_YEAR <= year <= END_YEAR:
            target_files.append(f)
    except (IndexError, ValueError):
        print(f"Warning: Could not parse date from filename {f}. Skipping.")
        continue

if not target_files:
    print(f"Error: No ELM files found for {CASE_NAME} in {E3SM_DATA_DIR} for {START_YEAR}-{END_YEAR}.")
    raise SystemExit("No input files found.")

print(f"Found {len(target_files)} ELM files for the period.")

ds_elm = None
try:
    # Open multiple files as a single dataset
    # Use decode_times=True to ensure time dimension is parsed correctly
    ds_elm = xr.open_mfdataset(target_files, combine='by_coords', decode_times=True)
    print("ELM dataset loaded successfully.")
except Exception as e:
    print(f"Error loading ELM data: {e}")
    raise SystemExit(f"Failed to load ELM data: {e}")

# --- 2. Compute climatological mean runoff ---
print(f"Computing climatological mean for {ELM_VAR}...")
climatology_runoff = None
try:
    # Select the variable and compute the mean over the time dimension
    # QRUNOFF is in mm/s
    climatology_runoff = ds_elm[ELM_VAR].mean(dim='time', skipna=True)
    print("Climatological mean runoff computed.")
except KeyError:
    print(f"Error: Variable '{ELM_VAR}' not found in ELM dataset.")
    raise SystemExit(f"Variable '{ELM_VAR}' not found.")
except Exception as e:
    print(f"Error computing climatological mean runoff: {e}")
    raise SystemExit(f"Failed to compute climatological mean: {e}")

# --- 3. Compute area-weighted global mean ---
print("Computing area-weighted global mean runoff...")
global_mean_runoff_val = np.nan
try:
    # E3SM h0 files typically have 'area' (cell area in m^2)
    if 'area' in ds_elm.data_vars:
        cell_area = ds_elm['area']
    elif 'area' in ds_elm.coords:
        cell_area = ds_elm['area']
    else:
        raise ValueError("Required 'area' variable not found in ELM dataset for area-weighted mean calculation.")

    # Ensure cell_area has no time dimension if it somehow acquired one
    if 'time' in cell_area.dims:
        cell_area = cell_area.isel(time=0) # Take first time step if area has time dim

    # Mask out NaN values in runoff before weighting, as they represent ocean or missing data
    # and should not contribute to the sum. Apply the same mask to area.
    masked_runoff = climatology_runoff.where(~np.isnan(climatology_runoff))
    masked_area = cell_area.where(~np.isnan(climatology_runoff)) # Mask area based on runoff's land points

    # Calculate area-weighted mean
    weighted_runoff_sum = (masked_runoff * masked_area).sum(skipna=True)
    total_masked_area = masked_area.sum(skipna=True)

    if total_masked_area.item() > 0:
        global_mean_runoff = weighted_runoff_sum / total_masked_area
        global_mean_runoff_val = global_mean_runoff.item()
        print(f"Area-weighted global mean runoff: {global_mean_runoff_val:.4f} mm/s")
    else:
        print("Warning: Total masked area is zero, global mean cannot be computed.")

except Exception as e:
    print(f"Error computing area-weighted global mean runoff: {e}")

# --- 4. Produce a map of the mean runoff field ---
print("Generating map of climatological mean runoff...")
try:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Plot the climatological mean runoff field
    # xarray's plot.pcolormesh method is robust for structured lat/lon grids
    # and handles colorbar and projection well.
    plot_handle = climatology_runoff.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(), # Data coordinates are PlateCarree
        cmap='viridis',
        cbar_kwargs={'label': f'{ELM_VAR} (mm/s)'},
        add_colorbar=True,
        # Set vmin/vmax for consistent color scaling, handling potential all-NaN cases
        vmin=climatology_runoff.min().item() if not np.isnan(climatology_runoff.min().item()) else None,
        vmax=climatology_runoff.max().item() if not np.isnan(climatology_runoff.max().item()) else None,
    )

    ax.set_title(f'Climatological Mean {ELM_VAR} ({START_YEAR}-{END_YEAR})')
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Overlay global statistics
    if not np.isnan(global_mean_runoff_val):
        ax.text(0.02, 0.02,
                f'Global Mean: {global_mean_runoff_val:.4f} mm/s',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Save the figure
    output_map_path = os.path.join(OUTPUT_DIR, f'climatological_mean_{ELM_VAR}_{START_YEAR}-{END_YEAR}.png')
    plt.savefig(output_map_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Map saved to {output_map_path}")

except Exception as e:
    print(f"Error generating map: {e}")

# --- 5. Save other outputs ---
# Save the global mean value to a text file
try:
    output_stats_path = os.path.join(OUTPUT_DIR, f'global_mean_{ELM_VAR}_{START_YEAR}-{END_YEAR}.txt')
    with open(output_stats_path, 'w') as f:
        f.write(f"Area-weighted global mean {ELM_VAR} ({START_YEAR}-{END_YEAR}): {global_mean_runoff_val:.6f} mm/s\n")
    print(f"Global mean statistics saved to {output_stats_path}")
except Exception as e:
    print(f"Error saving global mean statistics: {e}")

# Save the climatological mean runoff field as a NetCDF file
try:
    output_nc_path = os.path.join(OUTPUT_DIR, f'climatological_mean_{ELM_VAR}_{START_YEAR}-{END_YEAR}.nc')
    if climatology_runoff is not None:
        climatology_runoff.name = ELM_VAR # Ensure the variable has a name for saving
        climatology_runoff.to_netcdf(output_nc_path)
        print(f"Climatological mean runoff field saved to {output_nc_path}")
    else:
        print("Warning: Climatological runoff data is not available to save to NetCDF.")
except Exception as e:
    print(f"Error saving climatological mean runoff NetCDF: {e}")

print("Script finished.")