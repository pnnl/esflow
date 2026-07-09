import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import glob
import cartopy.crs as ccrs

# Define paths and case name
data_dir = "./data/sample/e3sm/"
obs_dir = "./data/sample/obs/streamflow/"
gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_04_streamflow_fdc/run1_debug/v1/output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load gauge metadata
gauge_metadata = pd.read_csv(gauge_metadata_path)
selected_gauges = gauge_metadata[gauge_metadata['grdc_no'].isin([3629000, 4121801, 4115200, 6742900, 2969100, 1159100])]

def load_mosart_discharge(case_name, start_year=1985, end_year=1989):
    discharge_data = {}
    for year in range(start_year, end_year + 1):
        daily_file_pattern = f"{data_dir}rof/{case_name}.mosart.h1.{year}-*.nc"
        files = sorted(glob.glob(daily_file_pattern))
        if not files:
            continue
        ds_daily = xr.open_mfdataset(files, combine='by_coords')
        monthly_file = f"{data_dir}rof/{case_name}.mosart.h0.{year:04d}-*.nc"
        ds_monthly = xr.open_mfdataset(sorted(glob.glob(monthly_file)), combine='by_coords')
        
        # Extract river discharge
        daily_discharge = ds_daily['RIVER_DISCHARGE_OVER_LAND_LIQ'].values * 86400  # Convert to m3/day
        monthly_discharge = ds_monthly['RIVER_DISCHARGE_OVER_LAND_LIQ'].mean(dim='time').values
        
        lat, lon = ds_daily.lat.values, ds_daily.lon.values
        discharge_data[year] = {'daily': daily_discharge, 'monthly': monthly_discharge, 'lat': lat, 'lon': lon}
    return discharge_data

def load_observation(gauge_id):
    obs_path = f"{obs_dir}{gauge_id}.csv"
    try:
        df = pd.read_csv(obs_path)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['discharge_m3s']
    except FileNotFoundError:
        print(f"Observation file for gauge {gauge_id} not found.")
        return None

def find_nearest_grid_point(lat, lon, grid_lats, grid_lons):
    idx = np.argmin((grid_lats - lat)**2 + (grid_lons - lon)**2)
    return np.unravel_index(idx, grid_lats.shape)

def compute_fdc_metrics(simulated, observed):
    # Sort data
    sim_sorted = np.sort(simulated)
    obs_sorted = np.sort(observed)
    
    # Compute FDC metrics
    volume_bias = np.sum(sim_sorted) / np.sum(obs_sorted) - 1
    wasserstein_dist = wasserstein_distance(sim_sorted, obs_sorted)
    quantiles_sim = np.percentile(sim_sorted, [10, 50, 90])
    quantiles_obs = np.percentile(obs_sorted, [10, 50, 90])
    q_ratios = quantiles_sim / quantiles_obs
    
    return volume_bias, wasserstein_dist, q_ratios

def plot_fdc_comparison(gauge_id, sim_discharge, obs_discharge):
    plt.figure(figsize=(12, 6))
    
    # Plot FDC
    plt.subplot(1, 2, 1)
    sim_sorted = np.sort(sim_discharge)
    obs_sorted = np.sort(obs_discharge)
    percentiles = np.linspace(0, 100, len(sim_sorted))
    plt.plot(percentiles, sim_sorted, label='Simulated')
    plt.plot(percentiles, obs_sorted, label='Observed')
    plt.xlabel('Exceedance Probability (%)')
    plt.ylabel('Discharge (m3/s)')
    plt.title(f'FDC for Gauge {gauge_id}')
    plt.legend()
    
    # Plot map with Wasserstein distance
    plt.subplot(1, 2, 2)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    gauge_lat = selected_gauges.loc[selected_gauges['grdc_no'] == gauge_id, 'lat'].values[0]
    gauge_lon = selected_gauges.loc[selected_gauges['grdc_no'] == gauge_id, 'lon'].values[0]
    plt.plot(gauge_lon, gauge_lat, 'ro', transform=ccrs.PlateCarree())
    ax.text(gauge_lon + 1, gauge_lat, f'Wasserstein: {wasserstein_dist:.2f}', 
            horizontalalignment='left', verticalalignment='bottom')
    
    plt.title('Gauge Locations and Wasserstein Distance')
    plt.show()

# Main analysis
case_name = "sample.v3.LR.historical"
mosart_discharge_data = load_mosart_discharge(case_name)

results = []

for _, gauge in selected_gauges.iterrows():
    gauge_id = gauge['grdc_no']
    obs_discharge = load_observation(gauge_id)
    
    if obs_discharge is None:
        continue
    
    # Find nearest grid point
    lat, lon = gauge['lat'], gauge['lon']
    for year in range(1985, 1990):
        if year not in mosart_discharge_data:
            continue
        daily_discharge = mosart_discharge_data[year]['daily']
        monthly_discharge = mosart_discharge_data[year]['monthly']
        grid_lats, grid_lons = mosart_discharge_data[year]['lat'], mosart_discharge_data[year]['lon']
        
        row_idx, col_idx = find_nearest_grid_point(lat, lon, grid_lats, grid_lons)
        sim_daily_discharge = daily_discharge[:, row_idx, col_idx]
        sim_monthly_discharge = monthly_discharge[row_idx, col_idx]
    
    # Interpolate simulated data to match observation dates
    sim_dates = pd.date_range(start='1985-01-01', end='1989-12-31')
    sim_daily_discharge_interp = np.interp(sim_dates.dayofyear, 
                                           np.arange(1, 367), 
                                           sim_daily_discharge.flatten())
    
    # Compute metrics
    volume_bias, wasserstein_dist, q_ratios = compute_fdc_metrics(sim_daily_discharge_interp, obs_discharge.values)
    
    results.append({
        'gauge_id': gauge_id,
        'volume_bias': volume_bias,
        'wasserstein_distance': wasserstein_dist,
        'q10_ratio': q_ratios[0],
        'q50_ratio': q_ratios[1],
        'q90_ratio': q_ratios[2]
    })
    
    # Save FDC data
    fdc_data = pd.DataFrame({
        'percentile': np.linspace(0, 100, len(sim_daily_discharge_interp)),
        'simulated': sim_daily_discharge_interp,
        'observed': obs_discharge.values
    })
    fdc_data.to_csv(f"{output_dir}/fdc_gauge_{gauge_id}.csv", index=False)
    
    # Plot FDC comparison
    plot_fdc_comparison(gauge_id, sim_daily_discharge_interp, obs_discharge.values)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/streamflow_metrics.csv", index=False)