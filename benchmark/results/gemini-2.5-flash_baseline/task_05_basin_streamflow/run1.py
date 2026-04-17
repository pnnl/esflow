import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import geopandas as gpd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# --- Configuration ---
E3SM_ROOT = "./data/sample/e3sm/"
OBS_ROOT = "./data/sample/obs/"
CASE_NAME = "sample.v3.LR.historical"
MOSART_VAR = "RIVER_DISCHARGE_OVER_LAND_LIQ"
START_YEAR = 1985
END_YEAR = 1989
TARGET_BASINS = {
    "Amazon": 3629000,
    "Missouri": 4121801,
    "Columbia": 4115200,
    "Danube": 6742900,
    "Mekong": 2969100,
    "Orange": 1159100,
}
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_05_basin_streamflow/run1_output"

# --- Helper Functions for Metrics ---
def calculate_rmse(sim, obs):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(obs, sim))

def calculate_nse(sim, obs):
    """Calculate Nash-Sutcliffe Efficiency (NSE)."""
    numerator = np.sum((obs - sim)**2)
    denominator = np.sum((obs - np.mean(obs))**2)
    if denominator == 0:
        return np.nan # Avoid division by zero
    return 1 - (numerator / denominator)

def calculate_pbias(sim, obs):
    """Calculate Percent Bias (PBIAS)."""
    return np.sum(sim - obs) / np.sum(obs) * 100

def calculate_kge(sim, obs):
    """Calculate Kling-Gupta Efficiency (KGE)."""
    r = pearsonr(sim, obs)[0]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# --- Main Script ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading gauge metadata from {OBS_ROOT}gauge_metadata.csv")
    try:
        gauge_metadata = pd.read_csv(os.path.join(OBS_ROOT, "gauge_metadata.csv"))
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        return

    print(f"Loading basin polygons from {OBS_ROOT}basin_polygons.geojson")
    try:
        basin_polygons = gpd.read_file(os.path.join(OBS_ROOT, "basin_polygons.geojson"))
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        return

    # Load MOSART grid from a sample file to get lat/lon
    print("Loading MOSART grid information...")
    mosart_grid_file = os.path.join(E3SM_ROOT, "rof", f"{CASE_NAME}.mosart.h0.{START_YEAR}-01.nc")
    try:
        with xr.open_dataset(mosart_grid_file) as ds_grid:
            mosart_lats = ds_grid['lat'].values
            mosart_lons = ds_grid['lon'].values
            # Adjust longitude to -180 to 180 if necessary
            if np.any(mosart_lons > 180):
                mosart_lons = np.where(mosart_lons > 180, mosart_lons - 360, mosart_lons)
    except Exception as e:
        print(f"Error loading MOSART grid from {mosart_grid_file}: {e}")
        return

    # Prepare to load MOSART data
    mosart_files = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            mosart_files.append(os.path.join(E3SM_ROOT, "rof", f"{CASE_NAME}.mosart.h0.{year}-{month:02d}.nc"))

    print(f"Loading MOSART data for {len(mosart_files)} files...")
    try:
        ds_mosart = xr.open_mfdataset(mosart_files, combine='by_coords', decode_times=True)
        # Adjust longitude to -180 to 180 if necessary
        if 'lon' in ds_mosart.coords and np.any(ds_mosart['lon'].values > 180):
            ds_mosart = ds_mosart.assign_coords(lon=(((ds_mosart['lon'] + 180) % 360) - 180))
        ds_mosart = ds_mosart.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    except Exception as e:
        print(f"Error loading MOSART data: {e}")
        return

    results = []

    for basin_name, gauge_id in TARGET_BASINS.items():
        print(f"\n--- Processing {basin_name} (Gauge ID: {gauge_id}) ---")

        # 1. Get gauge info
        gauge_info = gauge_metadata[gauge_metadata['gauge_id'] == gauge_id]
        if gauge_info.empty:
            print(f"Gauge ID {gauge_id} not found in metadata. Skipping.")
            continue
        gauge_lat = gauge_info['lat'].values[0]
        gauge_lon = gauge_info['lon'].values[0]

        # 2. Match gauge to MOSART grid
        # Find nearest grid cell
        # Ensure MOSART lons are in -180 to 180 range for distance calculation
        mosart_lons_adjusted = np.where(mosart_lons > 180, mosart_lons - 360, mosart_lons)
        
        lat_idx = np.argmin(np.abs(mosart_lats - gauge_lat))
        lon_idx = np.argmin(np.abs(mosart_lons_adjusted - gauge_lon))
        
        mosart_grid_lat = mosart_lats[lat_idx]
        mosart_grid_lon = mosart_lons_adjusted[lon_idx] # Use adjusted for display
        
        print(f"Gauge ({gauge_lat:.2f}, {gauge_lon:.2f}) matched to MOSART grid cell ({mosart_grid_lat:.2f}, {mosart_grid_lon:.2f})")

        # 3. Extract simulated discharge
        try:
            sim_discharge_series = ds_mosart[MOSART_VAR].sel(lat=mosart_grid_lat, lon=mosart_grid_lon, method='nearest').to_series()
            sim_discharge_series.index = sim_discharge_series.index.to_period('D').to_timestamp('D') # Convert monthly to daily for consistency
            sim_discharge_series = sim_discharge_series.resample('D').ffill() # Fill daily values with monthly data
            sim_discharge_series = sim_discharge_series.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]
        except Exception as e:
            print(f"Error extracting simulated discharge for {basin_name}: {e}")
            continue

        # 4. Extract observed discharge
        obs_file = os.path.join(OBS_ROOT, "streamflow", f"{gauge_id}.csv")
        try:
            obs_discharge_df = pd.read_csv(obs_file, parse_dates=['date'], index_col='date')
            obs_discharge_series = obs_discharge_df['discharge_m3s']
            obs_discharge_series = obs_discharge_series.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]
        except Exception as e:
            print(f"Error loading observed discharge for {basin_name} from {obs_file}: {e}")
            continue

        # 5. Align time series (daily)
        # Ensure both series are daily and have the same time range
        common_index = sim_discharge_series.index.intersection(obs_discharge_series.index)
        if common_index.empty:
            print(f"No overlapping data for {basin_name}. Skipping metrics and plot.")
            continue

        sim_aligned = sim_discharge_series.loc[common_index]
        obs_aligned = obs_discharge_series.loc[common_index]

        # Drop NaNs from both series for metric calculation
        valid_data = pd.DataFrame({'sim': sim_aligned, 'obs': obs_aligned}).dropna()
        if valid_data.empty:
            print(f"No valid (non-NaN) overlapping data for {basin_name}. Skipping metrics and plot.")
            continue
        
        sim_valid = valid_data['sim']
        obs_valid = valid_data['obs']

        # 6. Compute validation metrics
        if len(sim_valid) > 1: # Need at least 2 points for some metrics
            rmse = calculate_rmse(sim_valid, obs_valid)
            nse = calculate_nse(sim_valid, obs_valid)
            kge = calculate_kge(sim_valid, obs_valid)
            pbias = calculate_pbias(sim_valid, obs_valid)
        else:
            rmse, nse, kge, pbias = np.nan, np.nan, np.nan, np.nan
            print(f"Not enough valid data points for {basin_name} to compute metrics.")

        metrics = {
            "basin": basin_name,
            "gauge_id": gauge_id,
            "RMSE": rmse,
            "NSE": nse,
            "KGE": kge,
            "PBIAS": pbias,
        }
        results.append(metrics)
        print(f"Metrics for {basin_name}: RMSE={rmse:.2f}, NSE={nse:.2f}, KGE={kge:.2f}, PBIAS={pbias:.2f}%")

        # 7. Produce per-basin figure
        try:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

            # Map subplot
            ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            ax0.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree()) # Global extent initially
            ax0.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax0.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax0.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
            ax0.add_feature(cfeature.OCEAN, facecolor='lightblue')

            # Plot basin polygon
            basin_poly = basin_polygons[basin_polygons['grdc_no'] == gauge_id]
            if not basin_poly.empty:
                basin_poly.geometry.plot(ax=ax0, facecolor='none', edgecolor='red', linewidth=2, transform=ccrs.PlateCarree(), label=f'{basin_name} Basin')
                # Zoom to basin extent
                minx, miny, maxx, maxy = basin_poly.total_bounds
                # Add some padding
                pad = 5
                ax0.set_extent([minx - pad, maxx + pad, miny - pad, maxy + pad], crs=ccrs.PlateCarree())
            else:
                print(f"Basin polygon for {basin_name} (ID: {gauge_id}) not found.")

            # Plot gauge location
            ax0.plot(gauge_lon, gauge_lat, 'o', color='blue', markersize=8, transform=ccrs.PlateCarree(), label='Gauge Location')
            # Plot MOSART grid cell
            ax0.plot(mosart_grid_lon, mosart_grid_lat, 'x', color='green', markersize=8, transform=ccrs.PlateCarree(), label='MOSART Grid Cell')

            ax0.set_title(f'{basin_name} Basin and Gauge Location')
            ax0.legend()
            ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

            # Time series subplot
            ax1 = fig.add_subplot(gs[1, 0])
            ax1.plot(obs_aligned.index, obs_aligned.values, label='Observed Discharge', color='blue', linewidth=1.5)
            ax1.plot(sim_aligned.index, sim_aligned.values, label='Simulated Discharge', color='red', linestyle='--', linewidth=1.5)
            ax1.set_title(f'{basin_name} Streamflow (1985-1989)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Discharge (m$^3$/s)')
            ax1.legend()
            ax1.grid(True)

            # Add metrics to plot
            metrics_text = (
                f"RMSE: {rmse:.2f} m$^3$/s\n"
                f"NSE: {nse:.2f}\n"
                f"KGE: {kge:.2f}\n"
                f"PBIAS: {pbias:.2f}%"
            )
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

            plt.tight_layout()
            fig_path = os.path.join(OUTPUT_DIR, f"{basin_name}_streamflow_analysis.png")
            plt.savefig(fig_path)
            plt.close(fig)
            print(f"Saved figure to {fig_path}")
        except Exception as e:
            print(f"Error generating plot for {basin_name}: {e}")

    # Save all metrics to a CSV
    try:
        metrics_df = pd.DataFrame(results)
        metrics_csv_path = os.path.join(OUTPUT_DIR, "basin_streamflow_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nSaved all metrics to {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

if __name__ == "__main__":
    main()