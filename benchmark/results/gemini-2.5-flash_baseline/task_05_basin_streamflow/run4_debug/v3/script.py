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
import cftime # Import cftime to handle its objects explicitly

# --- Configuration ---
E3SM_DIR = "./data/sample/e3sm/"
OBS_DIR = "./data/sample/obs/"
CASE_NAME = "sample.v3.LR.historical"
MOSART_VAR = "RIVER_DISCHARGE_OVER_LAND_LIQ"
START_YEAR = 1985
END_YEAR = 1989
TARGET_BASINS = {
    "Amazon": "3629000",
    "Missouri": "4121801",
    "Columbia": "4115200",
    "Danube": "6742900",
    "Mekong": "2969100",
    "Orange": "1159100",
}
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_05_basin_streamflow/run4_debug/v3/output"

# --- Create output directory ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception as e:
    print(f"Error creating output directory {OUTPUT_DIR}: {e}")

# --- Helper Functions for Metrics ---
def calculate_rmse(sim, obs):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(obs, sim))

def calculate_nse(sim, obs):
    """Calculate Nash-Sutcliffe Efficiency."""
    numerator = np.sum((obs - sim)**2)
    denominator = np.sum((obs - np.mean(obs))**2)
    if denominator == 0:
        return np.nan # Avoid division by zero
    return 1 - (numerator / denominator)

def calculate_pbias(sim, obs):
    """Calculate Percent Bias."""
    numerator = np.sum(sim - obs)
    denominator = np.sum(obs)
    if denominator == 0:
        return np.nan # Avoid division by zero
    return (numerator / denominator) * 100

def calculate_kge(sim, obs):
    """Calculate Kling-Gupta Efficiency."""
    r = pearsonr(sim, obs)[0]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# --- Load Data ---
print("Loading gauge metadata...")
try:
    gauge_metadata = pd.read_csv(os.path.join(OBS_DIR, "gauge_metadata.csv"))
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    exit()

print("Loading basin polygons...")
try:
    basin_polygons = gpd.read_file(os.path.join(OBS_DIR, "basin_polygons.geojson"))
except Exception as e:
    print(f"Error loading basin polygons: {e}")
    exit()

# --- Load MOSART grid to find nearest cell ---
print("Loading MOSART grid...")
mosart_grid_file = os.path.join(E3SM_DIR, "rof", f"{CASE_NAME}.mosart.h0.{START_YEAR}-01.nc")
try:
    with xr.open_dataset(mosart_grid_file) as ds_grid:
        mosart_lats = ds_grid['lat'].values
        mosart_lons = ds_grid['lon'].values
        # Adjust longitude to -180 to 180 if necessary for consistency
        if np.any(mosart_lons > 180):
            mosart_lons[mosart_lons > 180] -= 360
except Exception as e:
    print(f"Error loading MOSART grid from {mosart_grid_file}: {e}")
    exit()

# --- Process each target basin ---
results_list = []

for basin_name, gauge_id_str in TARGET_BASINS.items():
    print(f"\nProcessing basin: {basin_name} (Gauge ID: {gauge_id_str})")
    gauge_id = int(gauge_id_str)

    # 1. Find gauge metadata
    gauge_info = gauge_metadata[gauge_metadata['gauge_id'] == gauge_id]
    if gauge_info.empty:
        print(f"  Gauge {gauge_id} not found in metadata. Skipping.")
        continue
    gauge_lat = gauge_info['lat'].values[0]
    gauge_lon = gauge_info['lon'].values[0]
    # Adjust gauge longitude to -180 to 180 if necessary
    if gauge_lon > 180:
        gauge_lon -= 360

    # 2. Match gauge to MOSART grid
    lat_diff = np.abs(mosart_lats - gauge_lat)
    lon_diff = np.abs(mosart_lons - gauge_lon)
    
    mosart_lat_idx = np.argmin(lat_diff)
    mosart_lon_idx = np.argmin(lon_diff)
    
    print(f"  Gauge ({gauge_lat:.2f}, {gauge_lon:.2f}) matched to MOSART grid cell ({mosart_lats[mosart_lat_idx]:.2f}, {mosart_lons[mosart_lon_idx]:.2f})")

    # 3. Load observed discharge
    obs_file = os.path.join(OBS_DIR, "streamflow", f"{gauge_id_str}.csv")
    try:
        obs_df = pd.read_csv(obs_file, parse_dates=['date'], index_col='date')
        obs_df = obs_df.loc[f"{START_YEAR}":f"{END_YEAR}"]
        obs_discharge = obs_df['discharge_m3s'].resample('D').mean() # Ensure daily frequency
        # MOSART output is monthly, so we need to resample obs to monthly for comparison
        obs_discharge_monthly = obs_discharge.resample('ME').mean() # Use 'ME' for month end
        print(f"  Loaded {len(obs_discharge_monthly)} monthly observed discharge records.")
    except Exception as e:
        print(f"  Error loading observed discharge for {gauge_id_str}: {e}. Skipping.")
        continue

    # 4. Load simulated discharge
    sim_discharge_list = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            month_str = f"{month:02d}"
            mosart_file = os.path.join(E3SM_DIR, "rof", f"{CASE_NAME}.mosart.h0.{year}-{month_str}.nc")
            if not os.path.exists(mosart_file):
                print(f"  Warning: MOSART file not found for {year}-{month_str}. Skipping.")
                continue
            try:
                with xr.open_dataset(mosart_file) as ds_mosart:
                    # Ensure the longitude convention matches (-180 to 180) for .sel
                    ds_mosart_sel = ds_mosart.copy()
                    if 'lon' in ds_mosart_sel.coords and np.any(ds_mosart_sel['lon'].values > 180):
                        ds_mosart_sel['lon'] = (ds_mosart_sel['lon'] + 180) % 360 - 180
                    
                    # Select the nearest point
                    sim_data = ds_mosart_sel[MOSART_VAR].sel(lat=gauge_lat, lon=gauge_lon, method='nearest')
                    sim_discharge_list.append(sim_data)
            except Exception as e:
                print(f"  Error loading MOSART data for {year}-{month_str}: {e}")

    if not sim_discharge_list:
        print(f"  No simulated discharge data found for {basin_name}. Skipping.")
        continue

    sim_discharge = xr.concat(sim_discharge_list, dim='time')
    
    # Convert xarray DataArray to pandas Series. The index will be a CFTimeIndex.
    sim_discharge_series = sim_discharge.to_series()
    
    # Convert CFTimeIndex to pandas DatetimeIndex
    # pd.to_datetime cannot directly convert cftime objects like DatetimeNoLeap.
    # Convert cftime objects to ISO format strings first, then to DatetimeIndex.
    try:
        sim_discharge_series.index = pd.to_datetime([str(d) for d in sim_discharge_series.index])
    except Exception as e:
        print(f"  Error converting CFTimeIndex to DatetimeIndex for {basin_name}: {e}. Skipping.")
        continue
    
    # Resample to monthly mean. 'ME' (Month End) is used for consistency.
    sim_discharge_monthly = sim_discharge_series.resample('ME').mean()
    
    print(f"  Loaded {len(sim_discharge_monthly)} monthly simulated discharge records.")

    # 5. Align time series
    # Ensure both series have the same time index for comparison
    common_index = obs_discharge_monthly.index.intersection(sim_discharge_monthly.index)
    if common_index.empty:
        print(f"  No common time period between observed and simulated data for {basin_name}. Skipping.")
        continue

    obs_aligned = obs_discharge_monthly.loc[common_index]
    sim_aligned = sim_discharge_monthly.loc[common_index]

    if len(obs_aligned) < 12: # Require at least a year of data for meaningful metrics
        print(f"  Insufficient common data points ({len(obs_aligned)}) for {basin_name}. Skipping.")
        continue

    # 6. Compute validation metrics
    rmse = calculate_rmse(sim_aligned, obs_aligned)
    nse = calculate_nse(sim_aligned, obs_aligned)
    kge = calculate_kge(sim_aligned, obs_aligned)
    pbias = calculate_pbias(sim_aligned, obs_aligned)

    metrics = {
        "basin": basin_name,
        "gauge_id": gauge_id_str,
        "RMSE": rmse,
        "NSE": nse,
        "KGE": kge,
        "PBIAS": pbias,
    }
    results_list.append(metrics)
    print(f"  Metrics: RMSE={rmse:.2f}, NSE: {nse:.2f}, KGE: {kge:.2f}, PBIAS: {pbias:.2f}%")

    # 7. Produce per-basin figure
    try:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

        # Map subplot
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree()) # Global extent initially
        ax_map.add_feature(cfeature.LAND, color='lightgray')
        ax_map.add_feature(cfeature.OCEAN, color='lightblue')
        ax_map.add_feature(cfeature.COASTLINE)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.add_feature(cfeature.RIVERS)
        ax_map.add_feature(cfeature.LAKES, alpha=0.5)

        # Plot basin polygon
        basin_poly = basin_polygons[basin_polygons['grdc_no'] == gauge_id_str]
        if not basin_poly.empty:
            basin_poly.geometry.plot(ax=ax_map, facecolor='none', edgecolor='red', linewidth=2, transform=ccrs.PlateCarree(), label=f'{basin_name} Basin')
            # Zoom map to basin extent
            minx, miny, maxx, maxy = basin_poly.total_bounds
            # Add some padding
            pad = 5
            ax_map.set_extent([minx - pad, maxx + pad, miny - pad, maxy + pad], crs=ccrs.PlateCarree())
        else:
            print(f"  Warning: Basin polygon not found for {basin_name}.")

        # Plot gauge location
        ax_map.plot(gauge_lon, gauge_lat, 'o', color='blue', markersize=8, transform=ccrs.PlateCarree(), label='Gauge Location')
        ax_map.set_title(f'{basin_name} Basin and Gauge Location')
        ax_map.legend()
        ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Time series subplot
        ax_ts = fig.add_subplot(gs[1, 0])
        ax_ts.plot(obs_aligned.index, obs_aligned.values, label='Observed Discharge', color='blue')
        ax_ts.plot(sim_aligned.index, sim_aligned.values, label='Simulated Discharge', color='red', linestyle='--')
        ax_ts.set_title(f'{basin_name} Streamflow (1985-1989)\nRMSE: {rmse:.2f}, NSE: {nse:.2f}, KGE: {kge:.2f}, PBIAS: {pbias:.2f}%')
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel('Discharge (m$^3$/s)')
        ax_ts.legend()
        ax_ts.grid(True)

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, f"{basin_name.replace(' ', '_')}_streamflow_analysis.png")
        plt.savefig(fig_path)
        plt.close(fig)
        print(f"  Saved figure to {fig_path}")

    except Exception as e:
        print(f"  Error generating plot for {basin_name}: {e}")

# --- Save all metrics to a CSV ---
if results_list:
    metrics_df = pd.DataFrame(results_list)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "basin_streamflow_metrics.csv")
    try:
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nSaved all basin metrics to {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")
else:
    print("\nNo results to save.")

print("\nAnalysis complete.")