import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os
import regionmask
import warnings

# Suppress specific warnings that might occur with xarray/dask/matplotlib
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*mean of empty slice.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The input coordinates to pcolormesh are not monotonically increasing")
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame.sum with axis=None is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="The 'warn' parameter is deprecated")


# --- Configuration ---
# Input data paths
E3SM_DATA_DIR = "./data/sample/e3sm/"
OBS_DATA_DIR = "./data/sample/obs/"
BASIN_POLYGONS_PATH = os.path.join(OBS_DATA_DIR, "basin_polygons.geojson")
GAUGE_METADATA_PATH = os.path.join(OBS_DATA_DIR, "gauge_metadata.csv") # Not directly used for this task, but good to keep for context

# E3SM specific
CASE_NAME = "sample.v3.LR.historical"
# The component *subdirectory* is 'lnd', but the component *identifier in the filename* is 'elm'.
# We need to use 'lnd' for the subdirectory path and 'elm' for the filename itself.
COMPONENT_SUBDIR = "lnd"
COMPONENT_FILENAME_IDENTIFIER = "elm"
STREAM = "h0" # Monthly output
START_YEAR = 1985
END_YEAR = 1989

# Output directory (will be created if it doesn't exist)
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_06_water_balance/run1_debug/v2/output"

# Selected basins (GRDC_NO from gauge_metadata.csv)
SELECTED_BASINS = {
    "Amazon": 3629000,
    "Missouri": 4121801,
    "Columbia": 4115200,
    "Danube": 6742900,
    "Mekong": 2969100,
    "Orange": 1159100,
}

# Unit conversion factor: mm/s to mm/day
MM_PER_S_TO_MM_PER_DAY = 86400


# --- Helper Functions ---
def create_output_dir(path):
    """Creates the output directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Output directory '{path}' ensured.")
    except Exception as e:
        print(f"Error creating output directory '{path}': {e}")
        raise

def get_elm_filepaths(data_dir, case_name, component_subdir, component_filename_identifier, stream, start_year, end_year):
    """Generates a list of ELM file paths for a given period.
    
    Args:
        data_dir (str): Base directory for E3SM data (e.g., "./data/sample/e3sm/").
        case_name (str): E3SM case name (e.g., "sample.v3.LR.historical").
        component_subdir (str): The subdirectory name for the component (e.g., "lnd").
        component_filename_identifier (str): The identifier used in the filename (e.g., "elm").
        stream (str): The output stream (e.g., "h0").
        start_year (int): Start year for file selection.
        end_year (int): End year for file selection.
    """
    filepaths = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Corrected filename format: {case_name}.{component_filename_identifier}.{stream}.YYYY-MM.nc
            filename = f"{case_name}.{component_filename_identifier}.{stream}.{year}-{month:02d}.nc"
            filepaths.append(os.path.join(data_dir, component_subdir, filename))
    return filepaths

def calculate_area_weighted_mean(da, area_da):
    """Calculates the area-weighted mean of an xarray DataArray."""
    # Ensure area_da has the same spatial dimensions as da
    # This handles cases where da might be masked (contain NaNs)
    weighted_sum = (da * area_da).sum(dim=['lat', 'lon'], skipna=True)
    total_area = area_da.where(~da.isnull()).sum(dim=['lat', 'lon'], skipna=True) # Sum only areas where data exists
    return weighted_sum / total_area


# --- Main Script ---
if __name__ == "__main__":
    create_output_dir(OUTPUT_DIR)

    # 1. Load ELM Data
    print("Loading ELM data...")
    elm_data_path = os.path.join(E3SM_DATA_DIR)
    elm_files = get_elm_filepaths(elm_data_path, CASE_NAME, COMPONENT_SUBDIR, COMPONENT_FILENAME_IDENTIFIER, STREAM, START_YEAR, END_YEAR)

    try:
        # Open multiple files as a single dataset
        ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True, parallel=True)
        print(f"Loaded ELM data from {len(elm_files)} files.")

        # Ensure longitude is -180 to 180 for consistency with cartopy/regionmask
        if ds_elm.lon.max() > 180:
            ds_elm = ds_elm.assign_coords(lon=(((ds_elm.lon + 180) % 360) - 180)).sortby('lon')
            print("Converted longitude from 0-360 to -180-180.")

        # Get area variable
        if 'area' in ds_elm.data_vars:
            area_da = ds_elm['area']
        else:
            print("Error: 'area' variable not found in ELM dataset. Cannot perform area weighting.")
            exit()

    except Exception as e:
        print(f"Error loading ELM data: {e}")
        exit()

    # 2. Calculate Water Balance Components (climatological time-mean)
    print("Calculating water balance components...")
    try:
        # Convert to mm/day and compute climatological mean
        P_clim = (ds_elm['RAIN'] + ds_elm['SNOW']).mean(dim='time') * MM_PER_S_TO_MM_PER_DAY
        ET_clim = (ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']).mean(dim='time') * MM_PER_S_TO_MM_PER_DAY
        Q_clim = ds_elm['QRUNOFF'].mean(dim='time') * MM_PER_S_TO_MM_PER_DAY

        P_clim.name = 'P_mm_per_day'
        ET_clim.name = 'ET_mm_per_day'
        Q_clim.name = 'Q_mm_per_day'

        # 3. Compute Area-Weighted Global Mean
        global_P = calculate_area_weighted_mean(P_clim, area_da).item()
        global_ET = calculate_area_weighted_mean(ET_clim, area_da).item()
        global_Q = calculate_area_weighted_mean(Q_clim, area_da).item()
        global_residual = global_P - global_ET - global_Q

        print(f"Global Area-Weighted Means (mm/day):")
        print(f"  P: {global_P:.2f}")
        print(f"  ET: {global_ET:.2f}")
        print(f"  Q: {global_Q:.2f}")
        print(f"  Residual (P - ET - Q): {global_residual:.2f}")

        # 4. Compute Water Balance Residual spatial field
        Residual_clim = P_clim - ET_clim - Q_clim
        Residual_clim.name = 'Residual_mm_per_day'

        # Save global climatological fields
        try:
            P_clim.to_netcdf(os.path.join(OUTPUT_DIR, "global_P_clim.nc"))
            ET_clim.to_netcdf(os.path.join(OUTPUT_DIR, "global_ET_clim.nc"))
            Q_clim.to_netcdf(os.path.join(OUTPUT_DIR, "global_Q_clim.nc"))
            Residual_clim.to_netcdf(os.path.join(OUTPUT_DIR, "global_Residual_clim.nc"))
            print("Saved global climatological fields to NetCDF.")
        except Exception as e:
            print(f"Error saving global climatological fields: {e}")

    except Exception as e:
        print(f"Error calculating water balance components: {e}")
        exit()

    # 5. Load Basin Polygons and Metadata
    print("Loading basin polygons and metadata...")
    try:
        gdf_basins = gpd.read_file(BASIN_POLYGONS_PATH)

        # Filter basins to the selected ones
        selected_grdc_nos = list(SELECTED_BASINS.values())
        gdf_selected_basins = gdf_basins[gdf_basins['grdc_no'].isin(selected_grdc_nos)].copy()

        # Add basin names to the GeoDataFrame for easier lookup
        grdc_to_name = {v: k for k, v in SELECTED_BASINS.items()}
        gdf_selected_basins['basin_name'] = gdf_selected_basins['grdc_no'].map(grdc_to_name)

        # Ensure the GeoDataFrame has a CRS and convert to WGS84 (EPSG:4326)
        if gdf_selected_basins.crs is None:
            gdf_selected_basins.set_crs("EPSG:4326", inplace=True)
        else:
            gdf_selected_basins = gdf_selected_basins.to_crs("EPSG:4326")

        print(f"Loaded {len(gdf_selected_basins)} selected basin polygons.")

    except Exception as e:
        print(f"Error loading basin data: {e}")
        exit()

    # 6. Clip Fields to Basins and Calculate Basin-Averaged Values
    print("Clipping fields to basins and calculating basin-averaged values...")
    basin_results = {} # To store P, ET, Q, Residual for each basin

    try:
        # Create a regionmask from the selected basins
        lons = ds_elm.lon.values
        lats = ds_elm.lat.values

        # The mask will have values corresponding to the index of the basin in gdf_selected_basins
        # or NaN if outside all basins.
        mask = regionmask.mask_geopandas(gdf_selected_basins, lons, lats, numbers=gdf_selected_basins.index)

        for idx, row in gdf_selected_basins.iterrows():
            basin_name = row['basin_name']
            grdc_no = row['grdc_no']
            print(f"  Processing basin: {basin_name} (GRDC_NO: {grdc_no})")

            # Create a boolean mask for the current basin
            basin_mask = (mask == idx)

            # Apply mask to P, ET, Q, Residual fields
            P_basin = P_clim.where(basin_mask)
            ET_basin = ET_clim.where(basin_mask)
            Q_basin = Q_clim.where(basin_mask)
            Residual_basin = Residual_clim.where(basin_mask)

            # Mask the area_da as well to ensure correct area weighting within the basin
            area_basin = area_da.where(basin_mask)

            basin_P_mean = calculate_area_weighted_mean(P_basin, area_basin).item()
            basin_ET_mean = calculate_area_weighted_mean(ET_basin, area_basin).item()
            basin_Q_mean = calculate_area_weighted_mean(Q_basin, area_basin).item()
            basin_Residual_mean = calculate_area_weighted_mean(Residual_basin, area_basin).item()

            basin_results[basin_name] = {
                'P': basin_P_mean,
                'ET': basin_ET_mean,
                'Q': basin_Q_mean,
                'Residual': basin_Residual_mean
            }
        
        df_basin_results = pd.DataFrame.from_dict(basin_results, orient='index')
        print("Basin-averaged results:")
        print(df_basin_results)

        # Save basin-averaged results
        try:
            df_basin_results.to_csv(os.path.join(OUTPUT_DIR, "basin_water_balance_summary.csv"))
            print("Saved basin-averaged water balance summary to CSV.")
        except Exception as e:
            print(f"Error saving basin-averaged results: {e}")

    except Exception as e:
        print(f"Error during basin clipping and averaging: {e}")
        exit()

    # 7. Generate Composite Figure
    print("Generating composite figure...")
    try:
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1]) # Top for map, bottom for bar charts

        # --- Top Panel: Global Residual Map with Basin Outlines ---
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.coastlines()
        ax_map.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.7)
        ax_map.add_feature(cfeature.LAKES, alpha=0.5)
        ax_map.add_feature(cfeature.RIVERS)

        # Plot global residual
        residual_plot = Residual_clim.plot.pcolormesh(
            ax=ax_map,
            transform=ccrs.PlateCarree(),
            cmap='RdBu',
            center=0, # Center colormap around 0 for residual
            cbar_kwargs={'label': 'Residual (P - ET - Q) [mm/day]'},
            extend='both',
            add_colorbar=True,
            add_labels=False # xarray adds labels, but we want to control them
        )
        ax_map.set_title(f'ELM Climatological Water Balance Residual ({START_YEAR}-{END_YEAR})', fontsize=16)

        # Overlay basin outlines
        gdf_selected_basins.geometry.boundary.plot(ax=ax_map, edgecolor='black', linewidth=1.5, alpha=0.8, transform=ccrs.PlateCarree())

        # --- Bottom Panel: Per-Basin Bar Charts ---
        ax_bars = fig.add_subplot(gs[1, 0])

        df_plot = df_basin_results[['P', 'ET', 'Q', 'Residual']]
        df_plot.plot(kind='bar', ax=ax_bars, rot=45, width=0.8,
                     color=['skyblue', 'lightcoral', 'lightgreen', 'grey'])
        ax_bars.set_title('Basin-Averaged Water Balance Components (mm/day)', fontsize=16)
        ax_bars.set_ylabel('Flux (mm/day)')
        ax_bars.set_xlabel('Basin')
        ax_bars.legend(title='Component')
        ax_bars.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the figure
        figure_path = os.path.join(OUTPUT_DIR, "water_balance_composite_figure.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Saved composite figure to {figure_path}")

    except Exception as e:
        print(f"Error generating composite figure: {e}")

    print("Script finished.")