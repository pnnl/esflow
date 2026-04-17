import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import shape, Point
import geopandas as gpd
from scipy.stats import pearsonr

# Define constants and paths
E3SM_DIR = "./data/sample/e3sm/"
OBS_DIR = "./data/sample/obs/"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_05_basin_streamflow/run2_debug/v2/output"
CASE_NAME = "sample.v3.LR.historical"
BASIN_IDS = [3629000, 4121801, 4115200, 6742900, 2969100, 1159100]
START_YEAR = 1985
END_YEAR = 1989

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mosart_discharge():
    """Load MOSART river discharge data."""
    try:
        files = sorted([f for f in os.listdir(E3SM_DIR + 'rof/') 
                        if f.startswith(CASE_NAME) and f.endswith('.mosart.h0.*.nc')])
        
        # Ensure there are files to process
        if not files:
            raise ValueError("No MOSART h0 files found.")
        
        ds_list = [xr.open_dataset(os.path.join(E3SM_DIR, 'rof', file)) for file in files]
        
        # Check if the list is empty after filtering
        if not ds_list:
            raise ValueError("No valid datasets loaded from MOSART h0 files.")
        
        ds = xr.concat(ds_list, dim='time')
        return ds['RIVER_DISCHARGE_OVER_LAND_LIQ']
    except Exception as e:
        print(f"Error loading MOSART discharge data: {e}")
        raise

def load_gauge_metadata():
    """Load gauge metadata."""
    try:
        return pd.read_csv(os.path.join(OBS_DIR, 'gauge_metadata.csv'))
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        raise

def load_basin_polygons():
    """Load basin polygons."""
    try:
        return gpd.read_file(os.path.join(OBS_DIR, 'basin_polygons.geojson'))
    except Exception as e:
        print(f"Error loading basin polygons: {e}")
        raise

def match_gauges_to_grid(ds_mosart, gauge_metadata):
    """Match gauges to the MOSART grid."""
    lat = ds_mosart.lat.values
    lon = ds_mosart.lon.values
    grid_points = np.array([lon.flatten(), lat.flatten()]).T
    
    def find_nearest(grid_points, point):
        distances = np.sqrt(((grid_points - point) ** 2).sum(axis=1))
        return np.argmin(distances)
    
    gauge_metadata['nearest_grid_idx'] = gauge_metadata.apply(
        lambda row: find_nearest(grid_points, [row.lon, row.lat]), axis=1
    )
    return gauge_metadata

def load_observed_discharge(gauge_id):
    """Load observed discharge for a given gauge."""
    try:
        obs_file = os.path.join(OBS_DIR, 'streamflow', f'{gauge_id}.csv')
        df = pd.read_csv(obs_file)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['discharge_m3s']
    except Exception as e:
        print(f"Error loading observed discharge for gauge {gauge_id}: {e}")
        raise

def compute_metrics(sim, obs):
    """Compute validation metrics."""
    sim_mean = np.nanmean(sim)
    obs_mean = np.nanmean(obs)
    
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
    r, _ = pearsonr(sim, obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + ((sim_mean / obs_mean) - 1) ** 2 + ((np.std(sim) / np.std(obs)) - 1) ** 2)
    pbias = (np.sum(sim - obs) / np.sum(obs)) * 100
    
    return rmse, nse, kge, pbias

def plot_basin_streamflow(basin_id, basin_polygons, gauge_metadata, ds_mosart, metrics):
    """Plot streamflow for a given basin."""
    try:
        # Filter data for the current basin
        basin_polygon = basin_polygons[basin_polygons['grdc_no'] == basin_id].geometry.iloc[0]
        gauges_in_basin = gauge_metadata[gauge_metadata.apply(lambda row: Point(row.lon, row.lat).within(basin_polygon), axis=1)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Basin {basin_id} Streamflow")
        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge (m³/s)")
        
        # Plot basin boundary
        gpd.GeoSeries([basin_polygon]).plot(ax=ax, edgecolor='black', facecolor='none')
        
        for _, gauge in gauges_in_basin.iterrows():
            sim_discharge = ds_mosart.isel(lat=gauge['nearest_grid_idx'] // len(ds_mosart.lon), 
                                           lon=gauge['nearest_grid_idx'] % len(ds_mosart.lon)).values
            obs_discharge = load_observed_discharge(gauge['gauge_id'])
            
            # Align time indices
            sim_time = pd.date_range(start=f"{START_YEAR}-01-01", end=f"{END_YEAR}-12-31", freq='D')
            common_dates = pd.Index(sim_time).intersection(obs_discharge.index)
            sim_aligned = sim_discharge[common_dates]
            obs_aligned = obs_discharge[common_dates].values
            
            # Plot time series
            ax.plot(common_dates, sim_aligned, label=f"Simulated {gauge['river_name']}")
            ax.plot(common_dates, obs_aligned, label=f"Observed {gauge['river_name']}", linestyle='--')
        
        plt.legend()
        plt.grid(True)
        
        # Save plot
        fig.savefig(os.path.join(OUTPUT_DIR, f'basin_{basin_id}_streamflow.png'))
        plt.close(fig)
        
        return metrics
    
    except Exception as e:
        print(f"Error plotting basin {basin_id} streamflow: {e}")
        raise

def main():
    # Load data
    try:
        ds_mosart = load_mosart_discharge()
    except ValueError as ve:
        print(ve)
        return
    
    gauge_metadata = load_gauge_metadata()
    basin_polygons = load_basin_polygons()
    
    # Match gauges to the MOSART grid
    gauge_metadata = match_gauges_to_grid(ds_mosart, gauge_metadata)
    
    metrics_results = {}
    
    for basin_id in BASIN_IDS:
        print(f"Processing Basin {basin_id}")
        
        # Compute metrics and plot streamflow
        metrics = []
        for _, gauge in gauge_metadata[gauge_metadata['grdc_no'] == basin_id].iterrows():
            sim_discharge = ds_mosart.isel(lat=gauge['nearest_grid_idx'] // len(ds_mosart.lon), 
                                           lon=gauge['nearest_grid_idx'] % len(ds_mosart.lon)).values
            obs_discharge = load_observed_discharge(gauge['gauge_id'])
            
            # Align time indices
            sim_time = pd.date_range(start=f"{START_YEAR}-01-01", end=f"{END_YEAR}-12-31", freq='D')
            common_dates = pd.Index(sim_time).intersection(obs_discharge.index)
            sim_aligned = sim_discharge[common_dates]
            obs_aligned = obs_discharge[common_dates].values
            
            # Compute metrics
            rmse, nse, kge, pbias = compute_metrics(sim_aligned, obs_aligned)
            metrics.append((rmse, nse, kge, pbias))
        
        # Average metrics for the basin
        avg_metrics = np.mean(metrics, axis=0)
        metrics_results[basin_id] = avg_metrics
        
        # Plot streamflow and save results
        plot_basin_streamflow(basin_id, basin_polygons, gauge_metadata, ds_mosart, avg_metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_results, index=['RMSE', 'NSE', 'KGE', 'PBIAS']).T
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'basin_streamflow_metrics.csv'))

if __name__ == "__main__":
    main()