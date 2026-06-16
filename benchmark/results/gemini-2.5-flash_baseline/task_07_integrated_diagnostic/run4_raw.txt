import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import requests
import geopandas as gpd
import rioxarray
from scipy.stats import wasserstein_distance
from scipy.spatial import KDTree
import warnings

# Suppress specific warnings that might be noisy
warnings.filterwarnings("ignore", category=UserWarning, module="rioxarray")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*mean of empty slice.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The 'unit' keyword argument is deprecated.*")

# --- Configuration ---
# Base directories
BASE_DATA_DIR = "./data/sample/"
# Output directory as specified in the prompt
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_07_integrated_diagnostic/run4_output"

# E3SM specific
E3SM_CASE_NAME = "sample.v3.LR.historical"
E3SM_LND_DIR = os.path.join(BASE_DATA_DIR, "e3sm/lnd")
E3SM_ROF_DIR = os.path.join(BASE_DATA_DIR, "e3sm/rof")

# Observation specific
OBS_STREAMFLOW_DIR = os.path.join(BASE_DATA_DIR, "obs/streamflow")
OBS_GAUGE_METADATA_PATH = os.path.join(BASE_DATA_DIR, "obs/gauge_metadata.csv")
OBS_BASIN_POLYGONS_PATH = os.path.join(BASE_DATA_DIR, "obs/basin_polygons.geojson")

# ILAMB data
ILAMB_DATA_DIR = os.path.join(OUTPUT_DIR, "ilamb_data") # Store downloaded ILAMB data here
ILAMB_URLS = {
    "pr_GPCC": "https://www.ilamb.org/ILAMB-Data/DATA/pr/GPCCv2018/pr.nc",
    "et_MODIS": "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc",
    "mrro_LORA": "https://www.ilamb.org/ILAMB-Data/DATA/mrro/LORA/LORA.nc",
}

# Time period
START_YEAR = 1985
END_YEAR = 1989
YEARS = range(START_YEAR, END_YEAR + 1)

# Target basins (name: GRDC_NO)
TARGET_BASINS = {
    "Amazon": 3629000,
    "Missouri": 4121801,
    "Columbia": 4115200,
    "Danube": 6742900,
    "Mekong": 2969100,
    "Orange": 1159100,
}

# --- Helper Functions ---

def ensure_dir(path):
    """Ensures a directory exists."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def download_ilamb_data(url, local_path):
    """Downloads a file from a URL if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"Downloading {os.path.basename(local_path)} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {local_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False
    else:
        print(f"File already exists: {local_path}")
    return True

def convert_units_to_mm_day(da, var_name):
    """Converts various units to mm/day."""
    # E3SM ELM variables are kg/m2/s or mm/s. 1 kg/m2 = 1 mm water depth.
    # So kg/m2/s is equivalent to mm/s.
    if var_name in ["RAIN", "SNOW", "QVEGE", "QVEGT", "QSOIL", "QRUNOFF"]:
        # mm/s to mm/day
        return da * 86400.0
    elif var_name == "pr_GPCC":
        # GPCC is already mm/day
        return da
    elif var_name == "et_MODIS":
        # MODIS ET is kg/m2/8day. 1 kg/m2 = 1 mm. So mm/8day.
        # Convert to mm/day
        return da / 8.0
    elif var_name == "mrro_LORA":
        # LORA is already mm/day
        return da
    else:
        print(f"Warning: Unknown variable {var_name} for unit conversion. Assuming mm/day.")
        return da

def calculate_fdc(discharge_series):
    """Calculates Flow Duration Curve (FDC) values."""
    if discharge_series.empty:
        return np.array([]), np.array([])
    sorted_discharge = discharge_series.sort_values(ascending=False).values
    exceedance_probability = np.arange(1, len(sorted_discharge) + 1) / (len(sorted_discharge) + 1)
    return sorted_discharge, exceedance_probability

# --- Main Script ---
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(ILAMB_DATA_DIR)

    # --- Part 1: Extract model fields (P, ET, Q from ELM) ---
    print("\n--- Part 1: Extracting E3SM ELM fields ---")
    elm_files = []
    for year in YEARS:
        for month in range(1, 13):
            filename = f"{E3SM_CASE_NAME}.elm.h0.{year}-{month:02d}.nc"
            filepath = os.path.join(E3SM_LND_DIR, filename)
            if os.path.exists(filepath):
                elm_files.append(filepath)
            else:
                print(f"Warning: ELM file not found: {filepath}")

    ds_elm = None
    model_P_clim, model_ET_clim, model_Q_runoff_clim = None, None, None
    model_area = None

    try:
        if elm_files:
            ds_elm = xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True)
            # Ensure time dimension is correctly parsed and sorted
            ds_elm = ds_elm.sortby('time')
            # Select only the desired time period
            ds_elm = ds_elm.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

            # Calculate P, ET, Q
            ds_elm['P'] = convert_units_to_mm_day(ds_elm['RAIN'] + ds_elm['SNOW'], "RAIN")
            ds_elm['ET'] = convert_units_to_mm_day(ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL'], "QVEGE")
            ds_elm['Q_runoff'] = convert_units_to_mm_day(ds_elm['QRUNOFF'], "QRUNOFF")

            # Compute climatological time-mean
            model_P_clim = ds_elm['P'].mean(dim='time')
            model_ET_clim = ds_elm['ET'].mean(dim='time')
            model_Q_runoff_clim = ds_elm['Q_runoff'].mean(dim='time')

            # Add CRS for rioxarray
            if 'lon' in model_P_clim.coords and 'lat' in model_P_clim.coords:
                model_P_clim = model_P_clim.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
                model_ET_clim = model_ET_clim.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
                model_Q_runoff_clim = model_Q_runoff_clim.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
            else:
                print("Warning: 'lon' or 'lat' not found in ELM dataset. Clipping might fail.")

            # Extract area for weighting
            if 'area' in ds_elm.data_vars:
                model_area = ds_elm['area'].isel(time=0).drop_vars('time', errors='ignore') # Area is static
                if 'lon' in model_area.coords and 'lat' in model_area.coords:
                    model_area = model_area.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
                else:
                    print("Warning: 'lon' or 'lat' not found in ELM area variable. Area weighting might be inaccurate.")
            else:
                print("Warning: 'area' variable not found in ELM dataset. Area weighting will not be applied.")

            print("E3SM ELM climatological means calculated.")
        else:
            print("Error: No ELM files found for the specified period. Skipping ELM processing.")
    except Exception as e:
        print(f"Error processing E3SM ELM data: {e}")

    # --- Part 2: Fetch and extract observation fields ---
    print("\n--- Part 2: Fetching and extracting observation fields ---")
    obs_data = {}
    for key, url in ILAMB_URLS.items():
        local_path = os.path.join(ILAMB_DATA_DIR, os.path.basename(url))
        if download_ilamb_data(url, local_path):
            try:
                ds_obs = xr.open_dataset(local_path)
                var_name = ""
                if key == "pr_GPCC":
                    var_name = "pr"
                    # GPCC uses 0-360 longitude, convert to -180 to 180
                    if (ds_obs['lon'] > 180).any():
                        ds_obs = ds_obs.assign_coords(lon=(((ds_obs['lon'] + 180) % 360) - 180)).sortby('lon')
                elif key == "et_MODIS":
                    var_name = "et" # Variable name inside the file is 'et'
                elif key == "mrro_LORA":
                    var_name = "mrro"

                if var_name and var_name in ds_obs:
                    # Select time period and compute climatological mean
                    ds_obs_sel = ds_obs[var_name].sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
                    obs_data[key] = convert_units_to_mm_day(ds_obs_sel, key).mean(dim='time')
                    # Add CRS for rioxarray
                    if 'lon' in obs_data[key].coords and 'lat' in obs_data[key].coords:
                        obs_data[key] = obs_data[key].rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
                    else:
                        print(f"Warning: 'lon' or 'lat' not found in {key} dataset. Clipping might fail.")
                    print(f"Observation {key} climatological mean calculated.")
                else:
                    print(f"Error: Variable '{var_name}' not found in {key} dataset.")
            except Exception as e:
                print(f"Error processing ILAMB data {key}: {e}")

    # --- Part 3: Clip to basin means ---
    print("\n--- Part 3: Clipping to basin means ---")
    basin_polygons = None
    try:
        basin_polygons = gpd.read_file(OBS_BASIN_POLYGONS_PATH)
        basin_polygons = basin_polygons.set_crs("EPSG:4326", allow_override=True)
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        return

    if basin_polygons is None:
        print("Error: Basin polygons not loaded. Exiting.")
        return

    basin_metrics = {} # Stores P, ET, Q for model and obs for each basin

    for basin_name, grdc_no in TARGET_BASINS.items():
        print(f"Processing basin: {basin_name} (GRDC: {grdc_no})")
        basin_metrics[basin_name] = {}
        try:
            basin_geom = basin_polygons[basin_polygons['grdc_no'] == grdc_no].geometry
            if basin_geom.empty:
                print(f"Warning: No polygon found for GRDC_NO {grdc_no}. Skipping basin.")
                continue

            # Model data clipping
            if model_P_clim is not None and model_ET_clim is not None and model_Q_runoff_clim is not None:
                try:
                    clipped_P = model_P_clim.rio.clip(basin_geom, drop=True)
                    clipped_ET = model_ET_clim.rio.clip(basin_geom, drop=True)
                    clipped_Q_runoff = model_Q_runoff_clim.rio.clip(basin_geom, drop=True)

                    if model_area is not None:
                        clipped_area = model_area.rio.clip(basin_geom, drop=True)
                        # Ensure dimensions match for multiplication and sum
                        if 'lat' in clipped_P.dims and 'lon' in clipped_P.dims and \
                           'lat' in clipped_area.dims and 'lon' in clipped_area.dims:
                            basin_metrics[basin_name]['model_P'] = (clipped_P * clipped_area).sum() / clipped_area.sum()
                            basin_metrics[basin_name]['model_ET'] = (clipped_ET * clipped_area).sum() / clipped_area.sum()
                            basin_metrics[basin_name]['model_Q_runoff'] = (clipped_Q_runoff * clipped_area).sum() / clipped_area.sum()
                        else:
                            print(f"Warning: Area weighting for model data in {basin_name} failed due to dimension mismatch. Using simple mean.")
                            basin_metrics[basin_name]['model_P'] = clipped_P.mean()
                            basin_metrics[basin_name]['model_ET'] = clipped_ET.mean()
                            basin_metrics[basin_name]['model_Q_runoff'] = clipped_Q_runoff.mean()
                    else:
                        # Fallback to simple mean if area variable is not available
                        basin_metrics[basin_name]['model_P'] = clipped_P.mean()
                        basin_metrics[basin_name]['model_ET'] = clipped_ET.mean()
                        basin_metrics[basin_name]['model_Q_runoff'] = clipped_Q_runoff.mean()

                    # Convert to scalar if xarray.DataArray
                    for k in ['model_P', 'model_ET', 'model_Q_runoff']:
                        if isinstance(basin_metrics[basin_name][k], xr.DataArray):
                            basin_metrics[basin_name][k] = basin_metrics[basin_name][k].item()

                except Exception as e:
                    print(f"Error clipping model data for {basin_name}: {e}")
                    basin_metrics[basin_name]['model_P'] = np.nan
                    basin_metrics[basin_name]['model_ET'] = np.nan
                    basin_metrics[basin_name]['model_Q_runoff'] = np.nan

            # Observation data clipping
            for obs_key, obs_da in obs_data.items():
                try:
                    clipped_obs = obs_da.rio.clip(basin_geom, drop=True)
                    # For observation data, assume equal area cells or use default rioxarray weighting
                    basin_metrics[basin_name][f'obs_{obs_key.split("_")[0]}'] = clipped_obs.mean().item()
                except Exception as e:
                    print(f"Error clipping observation data {obs_key} for {basin_name}: {e}")
                    basin_metrics[basin_name][f'obs_{obs_key.split("_")[0]}'] = np.nan

        except Exception as e:
            print(f"General error for basin {basin_name}: {e}")

    # --- Part 4: Streamflow FDC metrics ---
    print("\n--- Part 4: Calculating Streamflow FDC metrics ---")
    gauge_metadata = None
    try:
        gauge_metadata = pd.read_csv(OBS_GAUGE_METADATA_PATH, index_col='gauge_id')
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        # Fill with NaNs for streamflow metrics
        for basin_name in TARGET_BASINS:
            if basin_name in basin_metrics:
                basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                basin_metrics[basin_name]['wasserstein_distance'] = np.nan
        return

    mosart_daily_files = []
    for year in YEARS:
        # MOSART daily files are h1, e.g., sample.v3.LR.historical.mosart.h1.1985-01-02-00000.nc
        # We need to glob for all files in the year-month directories
        mosart_daily_files.extend(
            sorted([os.path.join(E3SM_ROF_DIR, f) for f in os.listdir(E3SM_ROF_DIR)
                    if f.startswith(f"{E3SM_CASE_NAME}.mosart.h1.{year}") and f.endswith(".nc")])
        )

    if not mosart_daily_files:
        print("Error: No MOSART daily files found for the specified period. Streamflow analysis skipped.")
        # Fill with NaNs for streamflow metrics
        for basin_name in TARGET_BASINS:
            if basin_name in basin_metrics:
                basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                basin_metrics[basin_name]['wasserstein_distance'] = np.nan
    else:
        ds_mosart = None
        try:
            # Open MOSART daily files
            ds_mosart = xr.open_mfdataset(mosart_daily_files, combine='by_coords', decode_times=True)
            ds_mosart = ds_mosart.sortby('time')
            ds_mosart = ds_mosart.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

            # Get MOSART grid coordinates for nearest neighbor search
            mosart_lats = ds_mosart['lat'].values
            mosart_lons = ds_mosart['lon'].values
            mosart_coords = np.column_stack((mosart_lats, mosart_lons))
            mosart_kdtree = KDTree(mosart_coords)

            for basin_name, grdc_no in TARGET_BASINS.items():
                if basin_name not in basin_metrics: # Skip if basin was not processed in Part 3
                    continue

                try:
                    gauge_info = gauge_metadata.loc[grdc_no]
                    gauge_lat, gauge_lon = gauge_info['lat'], gauge_info['lon']

                    # Find nearest MOSART grid cell
                    _, idx = mosart_kdtree.query([gauge_lat, gauge_lon])
                    mosart_grid_lat = mosart_lats[idx]
                    mosart_grid_lon = mosart_lons[idx]

                    # Extract simulated discharge
                    sim_discharge_da = ds_mosart['RIVER_DISCHARGE_OVER_LAND_LIQ'].sel(lat=mosart_grid_lat, lon=mosart_grid_lon, method='nearest')
                    sim_discharge = sim_discharge_da.to_series().dropna()
                    sim_discharge.index = pd.to_datetime(sim_discharge.index) # Ensure datetime index
                    sim_discharge = sim_discharge.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]

                    # Load observed discharge
                    obs_discharge_path = os.path.join(OBS_STREAMFLOW_DIR, f"{grdc_no}.csv")
                    obs_discharge_df = pd.read_csv(obs_discharge_path, parse_dates=['date'], index_col='date')
                    obs_discharge = obs_discharge_df['discharge_m3s'].loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"].dropna()

                    # Align indices (important for FDC and volume comparison)
                    common_dates = sim_discharge.index.intersection(obs_discharge.index)
                    sim_discharge_aligned = sim_discharge.loc[common_dates]
                    obs_discharge_aligned = obs_discharge.loc[common_dates]

                    if sim_discharge_aligned.empty or obs_discharge_aligned.empty:
                        print(f"Warning: No common discharge data for {basin_name}. Skipping FDC metrics.")
                        basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                        basin_metrics[basin_name]['wasserstein_distance'] = np.nan
                        continue

                    # Compute Volume Bias
                    sim_total_volume = sim_discharge_aligned.sum() # m3/s * days * seconds/day
                    obs_total_volume = obs_discharge_aligned.sum()
                    if obs_total_volume > 0:
                        volume_bias = (sim_total_volume - obs_total_volume) / obs_total_volume
                    else:
                        volume_bias = np.nan # Avoid division by zero
                    basin_metrics[basin_name]['streamflow_volume_bias'] = volume_bias

                    # Compute Wasserstein Distance for FDCs
                    sim_fdc_values, _ = calculate_fdc(sim_discharge_aligned)
                    obs_fdc_values, _ = calculate_fdc(obs_discharge_aligned)

                    if len(sim_fdc_values) > 0 and len(obs_fdc_values) > 0:
                        wd = wasserstein_distance(obs_fdc_values, sim_fdc_values)
                        basin_metrics[basin_name]['wasserstein_distance'] = wd
                    else:
                        basin_metrics[basin_name]['wasserstein_distance'] = np.nan

                    print(f"Streamflow metrics for {basin_name} calculated.")

                except KeyError:
                    print(f"Warning: Gauge {grdc_no} not found in metadata. Skipping streamflow for {basin_name}.")
                    basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                    basin_metrics[basin_name]['wasserstein_distance'] = np.nan
                except FileNotFoundError:
                    print(f"Warning: Observation streamflow file for {grdc_no} not found. Skipping streamflow for {basin_name}.")
                    basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                    basin_metrics[basin_name]['wasserstein_distance'] = np.nan
                except Exception as e:
                    print(f"Error processing streamflow for {basin_name}: {e}")
                    basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                    basin_metrics[basin_name]['wasserstein_distance'] = np.nan
        except Exception as e:
            print(f"Error processing MOSART data: {e}")
            for basin_name in TARGET_BASINS:
                if basin_name in basin_metrics:
                    basin_metrics[basin_name]['streamflow_volume_bias'] = np.nan
                    basin_metrics[basin_name]['wasserstein_distance'] = np.nan


    # --- Part 5: Combine and visualise ---
    print("\n--- Part 5: Combining and visualizing results ---")

    # Calculate biases and water balance residual
    summary_data = []
    for basin_name, metrics in basin_metrics.items():
        row = {'Basin': basin_name}

        # P, ET, Q values
        row['Model P (mm/day)'] = metrics.get('model_P', np.nan)
        row['Obs P (mm/day)'] = metrics.get('obs_pr', np.nan)
        row['Model ET (mm/day)'] = metrics.get('model_ET', np.nan)
        row['Obs ET (mm/day)'] = metrics.get('obs_et', np.nan)
        row['Model Q_runoff (mm/day)'] = metrics.get('model_Q_runoff', np.nan)
        row['Obs Q_runoff (mm/day)'] = metrics.get('obs_mrro', np.nan)

        # Biases
        if not np.isnan(row['Obs P (mm/day)']) and row['Obs P (mm/day)'] != 0:
            row['P Bias (%)'] = ((row['Model P (mm/day)'] - row['Obs P (mm/day)']) / row['Obs P (mm/day)']) * 100
        else:
            row['P Bias (%)'] = np.nan
        if not np.isnan(row['Obs ET (mm/day)']) and row['Obs ET (mm/day)'] != 0:
            row['ET Bias (%)'] = ((row['Model ET (mm/day)'] - row['Obs ET (mm/day)']) / row['Obs ET (mm/day)']) * 100
        else:
            row['ET Bias (%)'] = np.nan
        if not np.isnan(row['Obs Q_runoff (mm/day)']) and row['Obs Q_runoff (mm/day)'] != 0:
            row['Q_runoff Bias (%)'] = ((row['Model Q_runoff (mm/day)'] - row['Obs Q_runoff (mm/day)']) / row['Obs Q_runoff (mm/day)']) * 100
        else:
            row['Q_runoff Bias (%)'] = np.nan

        # Water Balance Residual (P - ET - Q)
        model_wb = row['Model P (mm/day)'] - row['Model ET (mm/day)'] - row['Model Q_runoff (mm/day)']
        obs_wb = row['Obs P (mm/day)'] - row['Obs ET (mm/day)'] - row['Obs Q_runoff (mm/day)']
        row['Model Water Balance Residual (mm/day)'] = model_wb
        row['Obs Water Balance Residual (mm/day)'] = obs_wb
        row['Water Balance Residual Bias (mm/day)'] = model_wb - obs_wb

        # Streamflow metrics
        row['Streamflow Volume Bias (%)'] = metrics.get('streamflow_volume_bias', np.nan) * 100
        row['Wasserstein Distance (m3/s)'] = metrics.get('wasserstein_distance', np.nan)

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Basin')

    # Save summary table
    try:
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "integrated_water_cycle_summary.csv"))
        print(f"Summary table saved to {os.path.join(OUTPUT_DIR, 'integrated_water_cycle_summary.csv')}")
    except Exception as e:
        print(f"Error saving summary table: {e}")

    # --- Plotting ---

    # Bar Chart: Model vs. Observation P, ET, Q
    try:
        fig, axes = plt.subplots(nrows=len(TARGET_BASINS), ncols=1, figsize=(10, 4 * len(TARGET_BASINS)), sharex=True)
        if len(TARGET_BASINS) == 1: # Handle single subplot case
            axes = [axes]

        width = 0.35
        x = np.arange(3) # P, ET, Q

        for i, basin_name in enumerate(TARGET_BASINS.keys()):
            ax = axes[i]
            model_values = [
                summary_df.loc[basin_name, 'Model P (mm/day)'],
                summary_df.loc[basin_name, 'Model ET (mm/day)'],
                summary_df.loc[basin_name, 'Model Q_runoff (mm/day)']
            ]
            obs_values = [
                summary_df.loc[basin_name, 'Obs P (mm/day)'],
                summary_df.loc[basin_name, 'Obs ET (mm/day)'],
                summary_df.loc[basin_name, 'Obs Q_runoff (mm/day)']
            ]

            rects1 = ax.bar(x - width/2, model_values, width, label='Model', color='skyblue')
            rects2 = ax.bar(x + width/2, obs_values, width, label='Observation', color='lightcoral')

            ax.set_ylabel('Flux (mm/day)')
            ax.set_title(f'{basin_name} - P, ET, Runoff')
            ax.set_xticks(x)
            ax.set_xticklabels(['Precipitation (P)', 'Evapotranspiration (ET)', 'Runoff (Q)'])
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "basin_P_ET_Q_barchart.png"))
        print(f"Bar chart saved to {os.path.join(OUTPUT_DIR, 'basin_P_ET_Q_barchart.png')}")
        plt.close(fig)
    except Exception as e:
        print(f"Error generating P, ET, Q bar chart: {e}")

    # Radar Chart: Multi-variable diagnostic
    try:
        # Metrics to include in radar chart (normalized)
        radar_metrics = [
            'P Bias (%)',
            'ET Bias (%)',
            'Q_runoff Bias (%)',
            'Streamflow Volume Bias (%)',
            'Water Balance Residual Bias (mm/day)',
            'Wasserstein Distance (m3/s)'
        ]

        # Normalize metrics for radar chart (0-1 scale, 1 is best)
        normalized_metrics = pd.DataFrame(index=summary_df.index, columns=radar_metrics)

        for metric in radar_metrics:
            values = summary_df[metric].abs()
            # For Wasserstein distance, a larger value means worse.
            # For biases, a larger absolute value means worse.
            # We want to normalize such that 0 is worst (0 score) and max_val is best (1 score).
            # Let's use a transformation that maps 0 (ideal) to 1, and larger deviations to smaller scores.
            # A simple approach: 1 - (abs_value / max_abs_value_across_all_basins)
            # Or, for biases, use a transformation like exp(-abs(bias)/scale)
            # For simplicity and interpretability, let's use 1 - (abs_value / max_abs_value_across_all_basins)
            # and cap at 0.
            
            # Determine a scaling factor for each metric.
            # For percentage biases, a 100% bias is very large.
            # For water balance residual, a few mm/day is significant.
            # For Wasserstein distance, it depends on the discharge magnitude.
            # To make them comparable on a radar chart, we need to scale them.
            # Let's define a "tolerance" or "max acceptable deviation" for each metric.
            # For now, we'll use the max observed absolute value across all basins for each metric.
            
            max_abs_val = values.max()
            if max_abs_val > 0:
                normalized_metrics[metric] = 1 - (values / max_abs_val)
            else:
                normalized_metrics[metric] = 1 # If all values are 0, it's perfect
            
            normalized_metrics[metric] = normalized_metrics[metric].fillna(0) # NaN values become 0 (worst score)
            normalized_metrics[metric] = normalized_metrics[metric].clip(lower=0) # Ensure values are not negative

        # Prepare data for radar chart
        num_vars = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # Complete the loop

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, polar=True)

        for basin_name in TARGET_BASINS.keys():
            values = normalized_metrics.loc[basin_name].tolist()
            values += values[:1] # Complete the loop
            ax.plot(angles, values, label=basin_name, linewidth=1.5)
            ax.fill(angles, values, alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, fontsize=10)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
        ax.set_ylim(0, 1)
        ax.set_title("Integrated Water Cycle Diagnostic (Normalized Scores)", va='bottom', fontsize=16)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "integrated_water_cycle_radar_chart.png"))
        print(f"Radar chart saved to {os.path.join(OUTPUT_DIR, 'integrated_water_cycle_radar_chart.png')}")
        plt.close(fig)
    except Exception as e:
        print(f"Error generating radar chart: {e}")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()