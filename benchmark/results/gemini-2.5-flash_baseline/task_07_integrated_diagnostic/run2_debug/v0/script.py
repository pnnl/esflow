import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import requests
import shutil
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d
import warnings

# Suppress specific warnings that might arise from data processing
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*mean of empty slice.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The input coordinates are not within the valid range.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The `xarray.DataArray.drop` method will be deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*CRS mismatch between the DataArray and the clip geometry.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The `drop` parameter is deprecated.*")


# --- Configuration ---
# Output directory (user-specified in the prompt)
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_07_integrated_diagnostic/run2_output"
DATA_ROOT = "./data/sample/"
E3SM_ROOT = os.path.join(DATA_ROOT, "e3sm")
OBS_ROOT = os.path.join(DATA_ROOT, "obs")
ILAMB_CACHE_DIR = os.path.join(DATA_ROOT, "ilamb_cache")

CASE_NAME = "sample.v3.LR.historical"
START_YEAR = 1985
END_YEAR = 1989
YEARS = range(START_YEAR, END_YEAR + 1)

# Target basins and their GRDC numbers (which match gauge_id)
TARGET_BASINS = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}

# ILAMB data URLs and local filenames
ILAMB_DATA_INFO = {
    "pr": {
        "variable": "pr",
        "dataset": "GPCCv2018",
        "filename": "pr.nc",
        "url": "https://www.ilamb.org/ILAMB-Data/DATA/pr/GPCCv2018/pr.nc",
        "unit_conversion_factor": 1, # mm/day (already correct)
    },
    "et": {
        "variable": "et", # variable name *inside* the file
        "dataset": "MODIS",
        "filename": "et_0.5x0.5.nc",
        "url": "https://www.ilamb.org/ILAMB-Data/DATA/evspsbl/MODIS/et_0.5x0.5.nc",
        "unit_conversion_factor": 1/8, # kg/m2/8day to mm/day (1 kg/m2 = 1 mm)
    },
    "mrro": {
        "variable": "mrro",
        "dataset": "LORA",
        "filename": "LORA.nc",
        "url": "https://www.ilamb.org/ILAMB-Data/DATA/mrro/LORA/LORA.nc",
        "unit_conversion_factor": 1/30.4375, # mm/month to mm/day (average days in month)
    },
}

# --- Helper Functions ---

def ensure_output_dir(path):
    """Ensures the output directory exists."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Output directory '{path}' ensured.")
    except Exception as e:
        print(f"Error creating output directory '{path}': {e}")

def download_ilamb_data(info, cache_dir):
    """Downloads ILAMB data if not already present."""
    local_path = os.path.join(cache_dir, info["filename"])
    if not os.path.exists(local_path):
        print(f"Downloading {info['filename']} from {info['url']}...")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with requests.get(info["url"], stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded to {local_path}")
        except Exception as e:
            print(f"Error downloading {info['filename']}: {e}")
            return None
    else:
        print(f"{info['filename']} already exists at {local_path}")
    return local_path

def convert_lon_360_to_180(ds):
    """Converts longitude from 0-360 to -180-180."""
    if 'lon' in ds.coords and ds['lon'].max() > 180:
        ds = ds.assign_coords(lon=(((ds['lon'] + 180) % 360) - 180))
        ds = ds.sortby('lon')
    return ds

def convert_lon_180_to_360(ds):
    """Converts longitude from -180-180 to 0-360."""
    if 'lon' in ds.coords and ds['lon'].min() < 0:
        ds = ds.assign_coords(lon=(ds['lon'] % 360))
        ds = ds.sortby('lon')
    return ds

def calculate_fdc(discharge_series, num_points=100):
    """Calculates Flow Duration Curve (FDC) values."""
    if discharge_series.empty or discharge_series.isnull().all():
        return np.full(num_points, np.nan)
    
    sorted_discharge = discharge_series.dropna().sort_values(ascending=False).values
    if len(sorted_discharge) < 2: # Not enough data to interpolate
        return np.full(num_points, np.nan)
    
    # Calculate exceedance probabilities (percentiles)
    exceedance_prob = np.linspace(0, 100, len(sorted_discharge), endpoint=True)
    
    # Interpolate to a fixed number of points for comparison
    interp_func = interp1d(exceedance_prob, sorted_discharge, kind='linear', fill_value="extrapolate")
    fdc_points = np.linspace(0, 100, num_points, endpoint=True)
    return interp_func(fdc_points)

# --- Main Script ---
if __name__ == "__main__":
    ensure_output_dir(OUTPUT_DIR)
    ensure_output_dir(ILAMB_CACHE_DIR)

    # --- Part 1: Extract Model Fields (P, ET, Q from ELM) ---
    print("\n--- Part 1: Extracting Model Fields (ELM) ---")
    elm_files = []
    for year in YEARS:
        elm_files.extend(glob.glob(os.path.join(E3SM_ROOT, "lnd", f"{CASE_NAME}.elm.h0.{year}-*.nc")))
    elm_files = sorted(list(set(elm_files))) # Remove duplicates and sort

    if not elm_files:
        print(f"Error: No ELM files found for {CASE_NAME} in {E3SM_ROOT}/lnd for years {START_YEAR}-{END_YEAR}.")
        exit()

    try:
        with xr.open_mfdataset(elm_files, combine='by_coords', decode_times=True, parallel=True) as ds_elm:
            # Ensure time dimension is correctly parsed and within the desired range
            ds_elm = ds_elm.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

            # Convert units from kg/m2/s or mm/s to mm/day
            # RAIN, SNOW: kg/m2/s -> mm/s -> mm/day (1 kg/m2 = 1 mm)
            # QVEGE, QVEGT, QSOIL, QRUNOFF: mm/s -> mm/day
            SECONDS_PER_DAY = 86400

            model_P = (ds_elm['RAIN'] + ds_elm['SNOW']) * SECONDS_PER_DAY
            model_ET = (ds_elm['QVEGE'] + ds_elm['QVEGT'] + ds_elm['QSOIL']) * SECONDS_PER_DAY
            model_Q = ds_elm['QRUNOFF'] * SECONDS_PER_DAY

            # Compute climatological time-mean
            model_P_clim = model_P.mean(dim='time')
            model_ET_clim = model_ET.mean(dim='time')
            model_Q_clim = model_Q.mean(dim='time')

            # Ensure consistent longitude convention (E3SM is usually 0-360)
            # No explicit conversion needed if all data is handled consistently (e.g., all to 0-360)
            # E3SM is typically 0-360, so we'll convert others to match.
            print("Model P, ET, Q climatological means calculated.")
    except Exception as e:
        print(f"Error processing ELM files: {e}")
        model_P_clim, model_ET_clim, model_Q_clim = None, None, None
        exit() # Cannot proceed without model data

    # --- Part 2: Fetch and Extract Observation Fields (P, ET, Q from ILAMB) ---
    print("\n--- Part 2: Fetching and Extracting Observation Fields (ILAMB) ---")
    obs_data = {}
    for var_name, info in ILAMB_DATA_INFO.items():
        local_path = download_ilamb_data(info, ILAMB_CACHE_DIR)
        if local_path:
            try:
                with xr.open_dataset(local_path) as ds_obs:
                    # Select the correct variable
                    obs_var = ds_obs[info["variable"]]

                    # Ensure time dimension is correctly parsed and within the desired range
                    if 'time' in obs_var.dims:
                        obs_var = obs_var.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
                        obs_var_clim = obs_var.mean(dim='time')
                    else:
                        obs_var_clim = obs_var # Assume it's already a climatology or static field

                    # Apply unit conversion
                    obs_var_clim = obs_var_clim * info["unit_conversion_factor"]

                    # Ensure consistent longitude convention (GPCC is 0-360, others might vary)
                    # Convert all to 0-360 for consistency with E3SM
                    obs_var_clim = convert_lon_180_to_360(obs_var_clim) 

                    obs_data[var_name] = obs_var_clim
                    print(f"Observation {var_name} climatological mean extracted.")
            except Exception as e:
                print(f"Error processing ILAMB data for {var_name}: {e}")
                obs_data[var_name] = None
        else:
            obs_data[var_name] = None

    if not all(obs_data.values()):
        print("Error: Not all observation datasets could be processed. Exiting.")
        exit()

    # --- Part 3: Clip to Basin Means ---
    print("\n--- Part 3: Clipping to Basin Means ---")
    basin_polygons_path = os.path.join(OBS_ROOT, "basin_polygons.geojson")
    try:
        basins_gdf = gpd.read_file(basin_polygons_path)
        # Ensure CRS is set for spatial operations
        if basins_gdf.crs is None:
            basins_gdf.set_crs("epsg:4326", inplace=True)
        else:
            basins_gdf = basins_gdf.to_crs("epsg:4326")
        print(f"Loaded {len(basins_gdf)} basin polygons.")
    except Exception as e:
        print(f"Error loading basin polygons from {basin_polygons_path}: {e}")
        exit()

    basin_metrics = {}

    for basin_name, grdc_no in TARGET_BASINS.items():
        print(f"\nProcessing basin: {basin_name} (GRDC No: {grdc_no})")
        basin_row = basins_gdf[basins_gdf['grdc_no'] == grdc_no]

        if basin_row.empty:
            print(f"Warning: Basin {basin_name} (GRDC No: {grdc_no}) not found in GeoJSON. Skipping.")
            continue

        basin_geom = basin_row.geometry.iloc[0]
        basin_metrics[basin_name] = {}

        # Function to clip and average
        def clip_and_average(da, geometry, name=""):
            try:
                # Ensure the DataArray has spatial dimensions and CRS set for rioxarray
                if 'lat' in da.dims and 'lon' in da.dims:
                    # Ensure lon is 0-360 for rioxarray if it's not already
                    if da['lon'].min() < 0:
                        da = convert_lon_180_to_360(da)

                    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                    da = da.rio.write_crs("epsg:4326", inplace=True)
                    
                    # Clip using the geometry
                    clipped_da = da.rio.clip([geometry], crs="epsg:4326", drop=True)
                    
                    # Calculate spatial mean, skipping NaNs
                    mean_val = clipped_da.mean(dim=['lat', 'lon'], skipna=True).item()
                    print(f"  {name} clipped and averaged: {mean_val:.2f} mm/day")
                    return mean_val
                else:
                    print(f"  Warning: {name} does not have 'lat' and 'lon' dimensions for clipping. Skipping.")
                    return np.nan
            except Exception as e:
                print(f"  Error clipping and averaging {name} for {basin_name}: {e}")
                return np.nan

        # Model P, ET, Q
        basin_metrics[basin_name]['model_P'] = clip_and_average(model_P_clim, basin_geom, "Model P")
        basin_metrics[basin_name]['model_ET'] = clip_and_average(model_ET_clim, basin_geom, "Model ET")
        basin_metrics[basin_name]['model_Q'] = clip_and_average(model_Q_clim, basin_geom, "Model Q")

        # Observation P, ET, Q
        basin_metrics[basin_name]['obs_P'] = clip_and_average(obs_data['pr'], basin_geom, "Obs P")
        basin_metrics[basin_name]['obs_ET'] = clip_and_average(obs_data['et'], basin_geom, "Obs ET")
        basin_metrics[basin_name]['obs_Q'] = clip_and_average(obs_data['mrro'], basin_geom, "Obs Q")

    # --- Part 4: Streamflow FDC Metrics ---
    print("\n--- Part 4: Streamflow FDC Metrics ---")
    gauge_metadata_path = os.path.join(OBS_ROOT, "gauge_metadata.csv")
    try:
        gauge_meta = pd.read_csv(gauge_metadata_path, index_col='gauge_id')
        print(f"Loaded {len(gauge_meta)} gauge metadata entries.")
    except Exception as e:
        print(f"Error loading gauge metadata from {gauge_metadata_path}: {e}")
        exit()

    mosart_h1_files = []
    for year in YEARS:
        mosart_h1_files.extend(glob.glob(os.path.join(E3SM_ROOT, "rof", f"{CASE_NAME}.mosart.h1.{year}-*-*-00000.nc")))
    mosart_h1_files = sorted(list(set(mosart_h1_files)))

    if not mosart_h1_files:
        print(f"Error: No MOSART h1 files found for {CASE_NAME} in {E3SM_ROOT}/rof for years {START_YEAR}-{END_YEAR}.")
        mosart_discharge = None # Set to None to skip streamflow processing
    else:
        try:
            # Open MOSART files, but only for the variable we need to save memory
            # Drop other variables if they exist to reduce memory footprint
            first_file_vars = list(xr.open_dataset(mosart_h1_files[0]).data_vars)
            drop_vars = [v for v in ['RIVER_DISCHARGE_OVER_LAND_LIQ_SUM', 'RIVER_DISCHARGE_OVER_LAND_LIQ_MAX'] if v in first_file_vars]

            with xr.open_mfdataset(mosart_h1_files, combine='by_coords', decode_times=True, parallel=True,
                                   drop_variables=drop_vars) as ds_mosart:
                ds_mosart = ds_mosart.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
                mosart_discharge = ds_mosart['RIVER_DISCHARGE_OVER_LAND_LIQ']
                print("MOSART h1 discharge data loaded.")
        except Exception as e:
            print(f"Error loading MOSART h1 files: {e}")
            mosart_discharge = None

    for basin_name, grdc_no in TARGET_BASINS.items():
        if basin_name not in basin_metrics: # Skip if basin was not found in GeoJSON
            continue

        print(f"  Processing streamflow for {basin_name} (GRDC No: {grdc_no})")
        gauge_id = grdc_no # GRDC No is the gauge_id
        
        if gauge_id not in gauge_meta.index:
            print(f"    Warning: Gauge {gauge_id} not found in metadata. Skipping streamflow for this basin.")
            basin_metrics[basin_name]['obs_streamflow_m3s'] = np.nan
            basin_metrics[basin_name]['model_streamflow_m3s'] = np.nan
            basin_metrics[basin_name]['streamflow_volume_bias_pct'] = np.nan
            basin_metrics[basin_name]['wasserstein_distance'] = np.nan
            continue

        gauge_lat = gauge_meta.loc[gauge_id, 'lat']
        gauge_lon = gauge_meta.loc[gauge_id, 'lon']

        # --- Observed Streamflow ---
        obs_streamflow_path = os.path.join(OBS_ROOT, "streamflow", f"{gauge_id}.csv")
        try:
            obs_df = pd.read_csv(obs_streamflow_path, parse_dates=['date'], index_col='date')
            obs_df = obs_df.loc[f"{START_YEAR}":f"{END_YEAR}"]
            obs_discharge = obs_df['discharge_m3s'].dropna()
            basin_metrics[basin_name]['obs_streamflow_m3s'] = obs_discharge.mean()
            print(f"    Observed streamflow loaded for {gauge_id}. Mean: {obs_discharge.mean():.2f} m3/s")
        except Exception as e:
            print(f"    Error loading observed streamflow for {gauge_id}: {e}")
            obs_discharge = pd.Series([], dtype=float)
            basin_metrics[basin_name]['obs_streamflow_m3s'] = np.nan

        # --- Simulated Streamflow ---
        if mosart_discharge is not None:
            try:
                # Find nearest MOSART grid cell
                # Convert gauge_lon to 0-360 if necessary for MOSART grid
                gauge_lon_360 = gauge_lon % 360 if gauge_lon < 0 else gauge_lon
                
                # Use xarray's nearest neighbor selection
                sim_discharge_da = mosart_discharge.sel(lat=gauge_lat, lon=gauge_lon_360, method='nearest')
                sim_discharge = sim_discharge_da.to_series().dropna()
                basin_metrics[basin_name]['model_streamflow_m3s'] = sim_discharge.mean()
                print(f"    Simulated streamflow extracted for {gauge_id}. Mean: {sim_discharge.mean():.2f} m3/s")
            except Exception as e:
                print(f"    Error extracting simulated streamflow for {gauge_id}: {e}")
                sim_discharge = pd.Series([], dtype=float)
                basin_metrics[basin_name]['model_streamflow_m3s'] = np.nan
        else:
            sim_discharge = pd.Series([], dtype=float)
            basin_metrics[basin_name]['model_streamflow_m3s'] = np.nan

        # --- FDC Metrics ---
        if not obs_discharge.empty and not sim_discharge.empty:
            # Volume Bias
            obs_total_volume = obs_discharge.sum() * 86400 # m3/s * seconds_in_day * num_days
            sim_total_volume = sim_discharge.sum() * 86400
            if obs_total_volume > 0:
                volume_bias_pct = ((sim_total_volume - obs_total_volume) / obs_total_volume) * 100
            else:
                volume_bias_pct = np.nan if sim_total_volume == 0 else np.inf # Handle zero observed volume
            basin_metrics[basin_name]['streamflow_volume_bias_pct'] = volume_bias_pct
            print(f"    Streamflow Volume Bias: {volume_bias_pct:.2f}%")

            # Wasserstein Distance
            # Calculate FDCs
            obs_fdc = calculate_fdc(obs_discharge)
            sim_fdc = calculate_fdc(sim_discharge)

            if not np.isnan(obs_fdc).all() and not np.isnan(sim_fdc).all():
                wd = wasserstein_distance(obs_fdc, sim_fdc)
                basin_metrics[basin_name]['wasserstein_distance'] = wd
                print(f"    Wasserstein Distance: {wd:.2f}")
            else:
                basin_metrics[basin_name]['wasserstein_distance'] = np.nan
                print("    Not enough data to compute Wasserstein Distance.")
        else:
            basin_metrics[basin_name]['streamflow_volume_bias_pct'] = np.nan
            basin_metrics[basin_name]['wasserstein_distance'] = np.nan
            print("    Not enough data for streamflow FDC metrics.")

    # --- Part 5: Combine and Visualise ---
    print("\n--- Part 5: Combining and Visualising ---")

    # Create summary table
    summary_data = []
    for basin_name, metrics in basin_metrics.items():
        # Calculate biases and water balance residual
        P_bias = ((metrics.get('model_P', np.nan) - metrics.get('obs_P', np.nan)) / metrics.get('obs_P', np.nan)) * 100 if metrics.get('obs_P', 0) != 0 else np.nan
        ET_bias = ((metrics.get('model_ET', np.nan) - metrics.get('obs_ET', np.nan)) / metrics.get('obs_ET', np.nan)) * 100 if metrics.get('obs_ET', 0) != 0 else np.nan
        Q_bias = ((metrics.get('model_Q', np.nan) - metrics.get('obs_Q', np.nan)) / metrics.get('obs_Q', np.nan)) * 100 if metrics.get('obs_Q', 0) != 0 else np.nan

        # Water balance residual: P - ET - Q (model and obs)
        model_WB_residual = metrics.get('model_P', np.nan) - metrics.get('model_ET', np.nan) - metrics.get('model_Q', np.nan)
        obs_WB_residual = metrics.get('obs_P', np.nan) - metrics.get('obs_ET', np.nan) - metrics.get('obs_Q', np.nan)
        WB_residual_diff = model_WB_residual - obs_WB_residual # Difference in residuals

        summary_data.append({
            'Basin': basin_name,
            'Model P (mm/day)': metrics.get('model_P', np.nan),
            'Obs P (mm/day)': metrics.get('obs_P', np.nan),
            'P Bias (%)': P_bias,
            'Model ET (mm/day)': metrics.get('model_ET', np.nan),
            'Obs ET (mm/day)': metrics.get('obs_ET', np.nan),
            'ET Bias (%)': ET_bias,
            'Model Q (mm/day)': metrics.get('model_Q', np.nan),
            'Obs Q (mm/day)': metrics.get('obs_Q', np.nan),
            'Q Bias (%)': Q_bias,
            'Model Streamflow (m3/s)': metrics.get('model_streamflow_m3s', np.nan),
            'Obs Streamflow (m3/s)': metrics.get('obs_streamflow_m3s', np.nan),
            'Streamflow Volume Bias (%)': metrics.get('streamflow_volume_bias_pct', np.nan),
            'Model WB Residual (mm/day)': model_WB_residual,
            'Obs WB Residual (mm/day)': obs_WB_residual,
            'WB Residual Diff (mm/day)': WB_residual_diff,
            'Wasserstein Distance': metrics.get('wasserstein_distance', np.nan),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_table_path = os.path.join(OUTPUT_DIR, "integrated_water_cycle_summary.csv")
    try:
        summary_df.to_csv(summary_table_path, index=False, float_format="%.2f")
        print(f"Summary table saved to {summary_table_path}")
    except Exception as e:
        print(f"Error saving summary table: {e}")

    # --- Bar Chart: P, ET, Q comparison ---
    try:
        # Filter out basins that might have been skipped
        basins_to_plot = [b for b in TARGET_BASINS.keys() if b in basin_metrics]
        if not basins_to_plot:
            print("No basins with valid data to plot bar chart.")
        else:
            fig, axes = plt.subplots(nrows=len(basins_to_plot), ncols=1, figsize=(10, 4 * len(basins_to_plot)), sharex=True)
            if len(basins_to_plot) == 1: # Handle single subplot case
                axes = [axes]

            width = 0.35
            x = np.arange(3) # P, ET, Q

            for i, basin_name in enumerate(basins_to_plot):
                metrics = basin_metrics[basin_name]
                
                model_vals = [metrics.get('model_P', np.nan), metrics.get('model_ET', np.nan), metrics.get('model_Q', np.nan)]
                obs_vals = [metrics.get('obs_P', np.nan), metrics.get('obs_ET', np.nan), metrics.get('obs_Q', np.nan)]

                ax = axes[i]
                ax.bar(x - width/2, model_vals, width, label='Model', color='skyblue')
                ax.bar(x + width/2, obs_vals, width, label='Observation', color='lightcoral')
                
                ax.set_ylabel('Flux (mm/day)')
                ax.set_title(f'{basin_name} - P, ET, Q Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(['Precipitation', 'Evapotranspiration', 'Runoff'])
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            bar_chart_path = os.path.join(OUTPUT_DIR, "basin_P_ET_Q_comparison.png")
            plt.savefig(bar_chart_path, dpi=300)
            print(f"Bar chart saved to {bar_chart_path}")
            plt.close(fig)
    except Exception as e:
        print(f"Error generating P, ET, Q bar chart: {e}")

    # --- Radar Chart: Multi-variable diagnostic ---
    try:
        # Metrics to include in radar chart (normalized)
        # Ensure summary_df is not empty and contains necessary columns
        if summary_df.empty or not all(col in summary_df.columns for col in ['P Bias (%)', 'ET Bias (%)', 'Q Bias (%)', 'Streamflow Volume Bias (%)', 'WB Residual Diff (mm/day)', 'Wasserstein Distance']):
            print("Summary DataFrame is empty or missing columns, cannot generate radar chart.")
        else:
            radar_metrics_raw = {
                'P Bias (%)': summary_df['P Bias (%)'].abs(),
                'ET Bias (%)': summary_df['ET Bias (%)'].abs(),
                'Q Bias (%)': summary_df['Q Bias (%)'].abs(),
                'Streamflow Volume Bias (%)': summary_df['Streamflow Volume Bias (%)'].abs(),
                'WB Residual Diff (mm/day)': summary_df['WB Residual Diff (mm/day)'].abs(),
                'Wasserstein Distance': summary_df['Wasserstein Distance'],
            }

            # Normalize metrics to 0-1 scale (higher is worse)
            radar_metrics_normalized = pd.DataFrame()
            for metric_name, series in radar_metrics_raw.items():
                # Fill NaNs with 0 for normalization, assuming NaN means "no data, treat as neutral/best" or "not applicable"
                # If NaN should represent "worst", fill with a large number before normalization.
                # For this context, 0 is safer as it won't skew the max value if a basin has no data.
                series_no_nan = series.fillna(0) 
                max_val = series_no_nan.max()
                if max_val > 0:
                    radar_metrics_normalized[metric_name] = series_no_nan / max_val
                else:
                    radar_metrics_normalized[metric_name] = 0 # All zeros if max_val is 0

            radar_metrics_normalized.index = summary_df['Basin']

            # Prepare for radar plot
            labels = list(radar_metrics_normalized.columns)
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1] # Complete the loop

            fig = plt.figure(figsize=(12, 12))
            # Adjust grid size based on number of basins
            num_basins_for_radar = len(radar_metrics_normalized.index)
            if num_basins_for_radar == 0:
                print("No basins with valid data for radar chart.")
                plt.close(fig)
                raise ValueError("No data for radar chart.")

            nrows = int(np.ceil(num_basins_for_radar / 2))
            ncols = 2 if num_basins_for_radar > 1 else 1
            gs = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.4) 

            for i, basin_name in enumerate(radar_metrics_normalized.index):
                ax = fig.add_subplot(gs[i // ncols, i % ncols], polar=True)
                
                values = radar_metrics_normalized.loc[basin_name].tolist()
                values += values[:1] # Complete the loop

                ax.plot(angles, values, linewidth=1, linestyle='solid', label=basin_name)
                ax.fill(angles, values, 'b', alpha=0.25)
                ax.set_title(basin_name, va='bottom')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, size=8)
                ax.set_ylim(0, 1) # Normalized scale
                ax.set_yticklabels([]) # Hide radial ticks for cleaner look
                ax.grid(True)

            plt.suptitle('Multi-variable Diagnostic (Radar Chart) - Normalized Biases (Higher is Worse)', y=1.02, fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
            radar_chart_path = os.path.join(OUTPUT_DIR, "basin_radar_chart.png")
            plt.savefig(radar_chart_path, dpi=300)
            print(f"Radar chart saved to {radar_chart_path}")
            plt.close(fig)

    except Exception as e:
        print(f"Error generating radar chart: {e}")

    print("\n--- Analysis Complete ---")